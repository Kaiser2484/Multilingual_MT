import os
import torch
import math
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tokenizers import Tokenizer
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo App
app = FastAPI(title="Multilingual MT API", description="AI Translation Hub")

# Mount thư mục giao diện tĩnh
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Định nghĩa Pydantic Model cho Request
class TranslateRequest(BaseModel):
    text: str
    source_lang: str  # en, vi, ja, zh
    target_lang: str  # en, vi, ja, zh

# -------------------------------------------------------------
# 1. KHỞI TẠO VÀ LOAD MODEL & TOKENIZER
# -------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# Giả lập biến toàn cục để lưu model (sẽ load khi khởi động)
tokenizer = None
model = None
BOS_IDX, EOS_IDX, PAD_IDX = 0, 0, 0
MAX_LEN = 128

@app.on_event("startup")
def load_assets():
    global tokenizer, model, BOS_IDX, EOS_IDX, PAD_IDX
    
    # 1. Load Tokenizer
    tok_path = "tokenizer/tokenizer.json"
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
        BOS_IDX = tokenizer.token_to_id("[BOS]")
        EOS_IDX = tokenizer.token_to_id("[EOS]")
        PAD_IDX = tokenizer.token_to_id("[PAD]")
        logger.info("Tokenizer loaded successfully.")
    else:
        logger.warning("Tokenizer not found! API will run in Mock Mode.")
        
    # 2. Load Model Architecture (Import từ source nội bộ)
    try:
        from src.models.transformer import TransformerMT
        if tokenizer:
            model = TransformerMT(vocab_size=tokenizer.get_vocab_size(), pad_idx=PAD_IDX).to(DEVICE)
            
            # Cố gắng load checkpoint mới nhất nếu có
            ckpt_path = "model_assets/transformer_best.pt"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=DEVICE)
                model.load_state_dict(ckpt['model_state'])
                logger.info(f"Transformer Checkpoint loaded from {ckpt_path}.")
            else:
                logger.warning("No checkpoint found. Model will produce random gibberish.")
            model.eval()
    except ImportError as e:
        logger.error(f"Failed to import Model Architecture: {e}")

# -------------------------------------------------------------
# 2. HÀM TRANSLATE CỐT LÕI (GREEDY DECODING)
# -------------------------------------------------------------
def infer_transformer(src_text: str, target_tag: str) -> str:
    """Hàm nội bộ dịch 1 đoạn text bằng Model AI trực tiếp."""
    if not model or not tokenizer:
        return f"[MOCK] Dịch giả lập '{src_text}' sang {target_tag}"
    
    src_full = f"{target_tag} {src_text}"
    src_ids = tokenizer.encode(src_full).ids
    src_tensor = torch.tensor([BOS_IDX] + src_ids + [EOS_IDX], dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    tgt_indices = [BOS_IDX]
    
    with torch.no_grad():
        for i in range(MAX_LEN):
            tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
            output = model(src_tensor, tgt_tensor)
            next_token = output[0, -1, :].argmax().item()
            
            if next_token == EOS_IDX:
                break
            tgt_indices.append(next_token)
            
    # Decode text
    translated = tokenizer.decode(tgt_indices[1:])
    
    # Clean output để chống lỗi Unicode/HTML entities như & apos ;
    translated = translated.replace(" & apos ; ", "'").replace("&apos;", "'")
    return translated

# -------------------------------------------------------------
# 3. API ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return FileResponse("src/api/static/index.html")

@app.post("/api/translate")
def translate(req: TranslateRequest):
    if req.source_lang == req.target_lang:
        return {"translated_text": req.text, "method": "identity"}
        
    tag_map = {'vi': '<2vi>', 'ja': '<2ja>', 'zh': '<2zh>', 'en': '<2en>'}
    
    if req.target_lang not in tag_map or req.source_lang not in tag_map:
        raise HTTPException(status_code=400, detail="Ngôn ngữ không được hỗ trợ.")

    # CHIẾN LƯỢC DỊCH
    # Nếu liên quan tới Tiếng Anh (Chiều Direct: EN->X hoặc X->EN)
    if req.source_lang == 'en' or req.target_lang == 'en':
        tgt_tag = tag_map[req.target_lang]
        result = infer_transformer(req.text, tgt_tag)
        method = f"Direct AI ({req.source_lang.upper()}->{req.target_lang.upper()})"
    else:
        # Nếu là các chiều CHÉO (VD: VI -> JA)
        # Áp dụng Hybrid Self-Pivot qua Tiếng Anh: (VI -> EN -> JA)
        en_tag = tag_map['en']
        intermediate_en = infer_transformer(req.text, en_tag)
        
        tgt_tag = tag_map[req.target_lang]
        result = infer_transformer(intermediate_en, tgt_tag)
        method = f"Self-Pivot AI ({req.source_lang.upper()}->EN->{req.target_lang.upper()})"
        
    return {
        "translated_text": result,
        "method": method
    }
