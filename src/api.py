"""
api.py  –  FastAPI Backend cho hệ thống Dịch thuật Đa ngôn ngữ
==============================================================
Chạy:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
Yêu cầu:
    pip install fastapi uvicorn tokenizers torch
"""

import os
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
MODEL_PATH     = os.path.join(PROJECT_DIR, "model_assets", "best_transformer_model.pt")
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "tokenizer", "tokenizer.json")
STATIC_DIR     = os.path.join(os.path.dirname(__file__), "static")

PAD_IDX = 0
BOS_IDX = 2
EOS_IDX = 3
MAX_LEN = 64

# Map ngôn ngữ → tag đích (gắn vào đầu câu nguồn)
LANG_TAG: dict[str, str] = {
    "Tiếng Việt"  : "<2vi>",
    "Tiếng Nhật"  : "<2ja>",
    "Tiếng Trung" : "<2zh>",
    "Tiếng Anh"   : "<2en>",
}

# Các ngôn ngữ nguồn hợp lệ
SOURCE_LANGS = ["Tiếng Anh", "Tiếng Việt", "Tiếng Nhật", "Tiếng Trung"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến global lưu model và tokenizer (khởi tạo trong startup)
_model: Optional[Transformer] = None
_tokenizer: Optional[HFTokenizer] = None


# ===========================================================================
# 2. STARTUP / SHUTDOWN
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp model và tokenizer khi server khởi động."""
    global _model, _tokenizer

    logger.info("Đang nạp tokenizer từ %s ...", TOKENIZER_PATH)
    _tokenizer = HFTokenizer.from_file(TOKENIZER_PATH)
    logger.info("Tokenizer OK | vocab_size=%d", _tokenizer.get_vocab_size())

    logger.info("Đang nạp model từ %s ...", MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    cfg  = ckpt.get("model_config", {
        "vocab_size" : 32_000,
        "d_model"    : 512,
        "num_heads"  : 8,
        "num_layers" : 6,
        "d_ff"       : 2048,
        "dropout"    : 0.1,
        "pad_idx"    : PAD_IDX,
        "max_len"    : MAX_LEN + 10,
    })
    _model = Transformer(**cfg).to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    total = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    logger.info(
        "Model OK | epoch=%s | params=%s | device=%s",
        ckpt.get("epoch", "?"), f"{total:,}", DEVICE,
    )

    yield  # Server đang chạy

    logger.info("Server đang tắt...")


# ===========================================================================
# 3. FASTAPI APP
# ===========================================================================
app = FastAPI(
    title       = "Multilingual MT API",
    description = "API dịch thuật máy đa ngôn ngữ dùng Transformer.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS – cho phép mọi origin (development)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Phục vụ file tĩnh (index.html)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ===========================================================================
# 4. SCHEMAS
# ===========================================================================
class TranslateRequest(BaseModel):
    source_text : str
    source_lang : str = "Tiếng Anh"   # ngôn ngữ nguồn
    target_lang : str = "Tiếng Việt"  # ngôn ngữ đích


class TranslateResponse(BaseModel):
    translated_text : str
    source_lang     : str = "Tiếng Anh"
    target_lang     : str
    model           : str = "Transformer"


# ===========================================================================
# 5. ENDPOINTS
# ===========================================================================
@app.get("/", include_in_schema=False)
async def root():
    """Phục vụ trang giao diện chính."""
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Multilingual MT API đang chạy. Xem /docs để biết thêm."}


@app.get("/health")
async def health():
    """Kiểm tra trạng thái server."""
    return {
        "status"      : "ok",
        "model_loaded": _model is not None,
        "device"      : str(DEVICE),
    }


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """
    Dịch câu từ tiếng Anh sang ngôn ngữ đích.

    Args:
        req.source_text : Câu tiếng Anh cần dịch.
        req.target_lang : Ngôn ngữ đích (xem LANG_TAG).

    Returns:
        TranslateResponse với trường translated_text.
    """
    # ── Validate đầu vào ─────────────────────────────────────────────────
    source = req.source_text.strip()
    if not source:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "source_text không được để trống.",
        )
    if req.target_lang not in LANG_TAG:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = f"target_lang không hợp lệ. Chọn một trong: {list(LANG_TAG.keys())}",
        )
    if req.source_lang == req.target_lang:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Ngôn ngữ nguồn và đích không được giống nhau.",
        )

    # ── Greedy Decoding ──────────────────────────────────────────────────
    try:
        import re
        # Normalize: "i" thuong → "I" hoa, viet hoa chu dau
        source = re.sub(r'\bi\b', 'I', source)
        source = source[0].upper() + source[1:]

        # ── PIVOT TRANSLATION ────────────────────────────────────────────
        # Model chi biet dich TU tieng Anh.
        # Neu nguon khong phai tieng Anh: dung Google Translate de chuyen
        # ve tieng Anh truoc, sau do dung model chinh de ra ngon ngu dich.
        GOOGLE_CODE = {
            "Tiếng Anh"  : "en",
            "Tiếng Việt" : "vi",
            "Tiếng Nhật" : "ja",
            "Tiếng Trung": "zh-CN",
        }

        english_text = source  # mac dinh: nguon da la tieng Anh

        if req.source_lang != "Tiếng Anh":
            # Buoc 1: Nguon → Tieng Anh (Google Translate lam trung gian)
            from deep_translator import GoogleTranslator
            src_code = GOOGLE_CODE[req.source_lang]
            english_text = GoogleTranslator(
                source=src_code, target="en"
            ).translate(source)
            logger.info("Pivot: '%s' (%s) → '%s' (en)",
                        source[:40], src_code, english_text[:40])

        # Buoc 2: Tieng Anh → Ngon ngu dich (Model Transformer cua minh)
        if req.target_lang == "Tiếng Anh":
            # Dich dau ra la tieng Anh → tra luon ket qua pivot
            result = english_text
        else:
            tag     = LANG_TAG[req.target_lang]
            src_str = f"{tag} {english_text}"

            ids = _tokenizer.encode(src_str).ids[:MAX_LEN]
            ids = ids + [PAD_IDX] * (MAX_LEN - len(ids))
            src_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)

            with torch.no_grad():
                out_ids = _model.translate_greedy(
                    src_tensor, BOS_IDX, EOS_IDX, MAX_LEN
                )
            result = _tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            if not result:
                result = "[Không thể dịch]"

    except Exception as exc:
        logger.exception("Lỗi inference: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Lỗi trong quá trình dịch: {exc}",
        ) from exc

    return TranslateResponse(
        translated_text = result,
        source_lang     = req.source_lang,
        target_lang     = req.target_lang,
    )
