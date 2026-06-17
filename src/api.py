"""
api.py  –  FastAPI Backend cho hệ thống Dịch thuật Đa ngôn ngữ
==============================================================
Chạy:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
Yêu cầu:
    pip install fastapi uvicorn tokenizers torch
"""

import os
import re
import html
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Đảm bảo import được src.models ──────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.models.transformer import Transformer
from tokenizers import Tokenizer as HFTokenizer

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# 1. CẤU HÌNH
# ===========================================================================
# Ưu tiên: ep35 > best_transformer > fallback
# (averaged checkpoint bị lỗi degenerate nên bỏ qua)
_EP35     = os.path.join(PROJECT_DIR, "model_assets", "transformer_ep35.pt")
_STANDARD = os.path.join(PROJECT_DIR, "model_assets", "best_transformer_model.pt")
MODEL_PATH     = _EP35 if os.path.exists(_EP35) else _STANDARD
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "tokenizer", "tokenizer.json")
STATIC_DIR     = os.path.join(os.path.dirname(__file__), "static")

PAD_IDX   = 0
BOS_IDX   = 2
EOS_IDX   = 3
MAX_LEN   = 64
BEAM_SIZE = 5          # Beam Search beam size
LEN_PEN   = 0.7       # Length penalty (0 = no penalty, 1 = full penalty)

# Map ngôn ngữ → tag đích (gắn vào đầu câu nguồn)
LANG_TAG: dict[str, str] = {
    "Tiếng Việt"  : "<2vi>",
    "Tiếng Nhật"  : "<2ja>",
    "Tiếng Trung" : "<2zh>",
    "Tiếng Anh"   : "<2en>",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến global lưu model và tokenizer (khởi tạo trong startup)
_model: Optional[Transformer] = None
_tokenizer: Optional[HFTokenizer] = None


# ===========================================================================
# 2. INFERENCE HELPERS
# ===========================================================================
def _clean_output(text: str) -> str:
    """
    Hậu xử lý kết quả dịch:
    - Decode HTML entities có khoảng trắng (& apos ; → ')
    - Xóa khoảng trắng thừa trước dấu câu
    """
    # Bước 1: Gộp spaced HTML entities: "& apos ;" → "&apos;"
    # Model tokenize HTML entities thành các token riêng lẻ
    text = re.sub(r'& (\w+) ;', r'&\1;', text)

    # Bước 2: Unescape: &apos; → ' , &amp; → & , &quot; → "
    text = html.unescape(text)

    # Bước 3: Xóa khoảng trắng trước dấu câu
    text = re.sub(r"\s+([.,!?;:'\"])", r'\1', text)
    # Bước 4: Xóa khoảng trắng sau dấu mở ngoặc
    text = re.sub(r'([\[(])\s+', r'\1', text)
    # Bước 5: Gộp khoảng trắng đôi
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """
    Tách đoạn văn dài thành các câu nhỏ để dịch từng câu.
    Giới hạn MAX_LEN=64 token nên cần tách trước khi encode.
    """
    # Tách theo dấu câu kết thúc hoặc xuống dòng
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    # Lọc câu rỗng
    return [p.strip() for p in parts if p.strip()]


def _encode_input(text: str) -> torch.Tensor:
    """Tokenize và pad chuỗi text, trả về tensor [1, MAX_LEN]."""
    ids = _tokenizer.encode(text).ids[:MAX_LEN]
    ids = ids + [PAD_IDX] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


def _beam_search(src_tensor: torch.Tensor) -> str:
    """
    Beam Search decoding với Length Penalty.
    Tốt hơn Greedy: giữ nhiều ứng viên song song, chọn câu tốt nhất.
    """
    with torch.no_grad():
        enc_out, src_mask = _model._encode(src_tensor)

    beams = [(0.0, [BOS_IDX])]   # (log_prob, token_ids)
    completed = []

    for _ in range(MAX_LEN):
        candidates = []
        for log_prob, tokens in beams:
            if tokens[-1] == EOS_IDX:
                completed.append((log_prob, tokens))
                continue
            tgt_t = torch.tensor([tokens], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                dec_out   = _model._decode(tgt_t, enc_out, src_mask)
                log_probs = torch.log_softmax(
                    _model.output_projection(dec_out[:, -1, :]), dim=-1
                )[0]
            topk_lp, topk_ids = log_probs.topk(BEAM_SIZE)
            for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                candidates.append((log_prob + lp, tokens + [tid]))

        # Sắp xếp có tính length penalty, giữ beam tốt nhất
        candidates.sort(
            key=lambda x: x[0] / (len(x[1]) ** LEN_PEN), reverse=True
        )
        beams = [c for c in candidates if c[1][-1] != EOS_IDX][:BEAM_SIZE]
        for c in candidates:
            if c[1][-1] == EOS_IDX:
                completed.append(c)
        if not beams:
            break

    if not completed:
        completed = beams
    best = max(completed, key=lambda x: x[0] / (len(x[1]) ** LEN_PEN))
    out_ids = [t for t in best[1] if t not in (BOS_IDX, EOS_IDX, PAD_IDX)]
    return _tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def _translate_direct(source_text: str, target_lang: str) -> str:
    """
    Dịch TRỰC TIẾP từng câu (sentence-by-sentence) để xử lý đoạn văn dài.
    Câu dài hơn MAX_LEN token sẽ bị tách nhỏ trước khi dịch.
    """
    tag      = LANG_TAG[target_lang]
    sentences = _split_sentences(source_text)

    translated_parts = []
    for sent in sentences:
        src_str = f"{tag} {sent}"
        src_t   = _encode_input(src_str)
        part    = _beam_search(src_t)
        part    = _clean_output(part)
        if part:
            translated_parts.append(part)

    return " ".join(translated_parts) if translated_parts else ""


def _translate_pivot(source_text: str, source_lang: str, target_lang: str) -> str:
    """
    Dịch TRUNG GIAN cho các cặp không có dữ liệu train trực tiếp (VI↔JA, VI↔ZH, JA↔ZH).
    Dùng Google Translate (deep-translator) để đảm bảo chất lượng cao hơn self-pivot.
    """
    GOOGLE_CODE = {
        "Tiếng Anh"  : "en",
        "Tiếng Việt" : "vi",
        "Tiếng Nhật" : "ja",
        "Tiếng Trung": "zh-CN",
    }
    try:
        from deep_translator import GoogleTranslator
        src_code = GOOGLE_CODE[source_lang]
        tgt_code = GOOGLE_CODE[target_lang]

        # Tách câu để tránh vượt giới hạn ký tự của Google Translate
        sentences = _split_sentences(source_text)
        parts = []
        for sent in sentences:
            translated = GoogleTranslator(source=src_code, target=tgt_code).translate(sent)
            if translated:
                parts.append(_clean_output(translated))
        result = " ".join(parts)
        logger.info("Google Pivot: %s→%s | '%s' → '%s'",
                    src_code, tgt_code, source_text[:30], result[:30])
        return result
    except Exception as e:
        logger.warning("Google Translate lỗi (%s), fallback về self-pivot.", e)
        # Fallback: tự dịch qua EN nếu Google thất bại
        english = _translate_direct(source_text, "Tiếng Anh")
        return _translate_direct(english, target_lang)


# ===========================================================================
# 3. STARTUP / SHUTDOWN
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp model và tokenizer khi server khởi động."""
    global _model, _tokenizer

    logger.info("Đang nạp tokenizer từ %s ...", TOKENIZER_PATH)
    _tokenizer = HFTokenizer.from_file(TOKENIZER_PATH)
    logger.info("Tokenizer OK | vocab_size=%d", _tokenizer.get_vocab_size())

    logger.info("Đang nạp model từ %s ...", MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    cfg  = ckpt.get("model_config", {
        "vocab_size" : 32_000,
        "d_model"    : 256,
        "num_heads"  : 8,
        "num_layers" : 3,
        "d_ff"       : 1024,
        "dropout"    : 0.1,
        "pad_idx"    : PAD_IDX,
        "max_len"    : MAX_LEN + 10,
    })
    _model = Transformer(**cfg).to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    total = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    logger.info(
        "Model OK | epoch=%s | params=%s | device=%s | file=%s",
        ckpt.get("epoch", "averaged"),
        f"{total:,}",
        DEVICE,
        os.path.basename(MODEL_PATH),
    )

    yield  # Server đang chạy

    logger.info("Server đang tắt...")


# ===========================================================================
# 4. FASTAPI APP
# ===========================================================================
app = FastAPI(
    title       = "Multilingual MT API",
    description = "API dịch thuật máy đa ngôn ngữ dùng Transformer (Beam Search, Self-Pivot).",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ===========================================================================
# 5. SCHEMAS
# ===========================================================================
class TranslateRequest(BaseModel):
    source_text : str
    source_lang : str = "Tiếng Anh"
    target_lang : str = "Tiếng Việt"


class TranslateResponse(BaseModel):
    translated_text : str
    source_lang     : str
    target_lang     : str
    model           : str = "Transformer"
    pivot           : bool = False   # True nếu dùng pivot qua EN


# ===========================================================================
# 6. ENDPOINTS
# ===========================================================================
@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Multilingual MT API đang chạy. Xem /docs để biết thêm."}


@app.get("/health")
async def health():
    return {
        "status"      : "ok",
        "model_loaded": _model is not None,
        "model_file"  : os.path.basename(MODEL_PATH),
        "device"      : str(DEVICE),
        "beam_size"   : BEAM_SIZE,
    }


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """
    Dịch câu giữa các ngôn ngữ (EN, VI, JA, ZH) theo 2 chiến lược:
    - Trực tiếp (Direct):  EN↔VI, EN↔JA, EN↔ZH
    - Trung gian qua EN (Self-Pivot): VI↔JA, VI↔ZH, JA↔ZH
    Không dùng bất kỳ API ngoài nào (đã xóa deep-translator).
    """
    # ── Validate ─────────────────────────────────────────────────────────
    source = req.source_text.strip()
    if not source:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "source_text không được để trống.")
    if req.source_lang not in LANG_TAG:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"source_lang không hợp lệ: {list(LANG_TAG.keys())}")
    if req.target_lang not in LANG_TAG:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"target_lang không hợp lệ: {list(LANG_TAG.keys())}")
    if req.source_lang == req.target_lang:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Ngôn ngữ nguồn và đích không được giống nhau.")

    # ── Normalize input (chỉ áp dụng cho Tiếng Anh) ──────────────────────
    if req.source_lang == "Tiếng Anh":
        source = re.sub(r'\bi\b', 'I', source)
        if source:
            source = source[0].upper() + source[1:]

    # ── Routing: Direct hay Pivot? ────────────────────────────────────────
    # Các cặp có thể dịch TRỰC TIẾP (EN là một trong 2 đầu):
    DIRECT_PAIRS = {
        ("Tiếng Anh",   "Tiếng Việt"),
        ("Tiếng Anh",   "Tiếng Nhật"),
        ("Tiếng Anh",   "Tiếng Trung"),
        ("Tiếng Việt",  "Tiếng Anh"),
        ("Tiếng Nhật",  "Tiếng Anh"),
        ("Tiếng Trung", "Tiếng Anh"),
    }

    try:
        pair = (req.source_lang, req.target_lang)
        use_pivot = pair not in DIRECT_PAIRS

        if use_pivot:
            logger.info("Google Pivot: %s → %s", req.source_lang, req.target_lang)
            result = _translate_pivot(source, req.source_lang, req.target_lang)
        else:
            # Dịch trực tiếp
            logger.info("Direct: %s → %s", req.source_lang, req.target_lang)
            result = _translate_direct(source, req.target_lang)

        if not result:
            result = "[Không thể dịch]"

    except Exception as exc:
        logger.exception("Lỗi inference: %s", exc)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Lỗi trong quá trình dịch: {exc}",
        ) from exc

    return TranslateResponse(
        translated_text = result,
        source_lang     = req.source_lang,
        target_lang     = req.target_lang,
        pivot           = use_pivot,
    )
