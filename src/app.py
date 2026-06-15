"""
app.py  –  Giao dien Web Demo Dich thuat May Da ngon ngu
==========================================================
Chay:  python src/app.py
Yeu cau:  pip install gradio tokenizers torch
"""

import os
import sys

# Dam bao import duoc src.models.transformer
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import torch
import gradio as gr
from tokenizers import Tokenizer as HFTokenizer
from src.models.transformer import Transformer

# ===========================================================================
# 1. CAU HINH DUONG DAN VA TOKEN IDS
# ===========================================================================
# Dieu chinh duong dan cho phu hop voi may tinh cua ban
MODEL_PATH     = os.path.join(PROJECT_DIR, "model_assets", "best_transformer_model.pt")
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "tokenizer",    "tokenizer.json")

PAD_IDX = 0   # Token <pad>
BOS_IDX = 2   # Token <bos> / <sos>
EOS_IDX = 3   # Token <eos>
MAX_LEN = 64  # Do dai toi da cua chuoi dau ra

# Anh xa ten ngon ngu → tag che do dich
LANG_TAG = {
    "Tieng Viet (vi)"  : "<2vi>",
    "Tieng Nhat (ja)"  : "<2ja>",
    "Tieng Trung (zh)" : "<2zh>",
    "Tieng Anh (en)"   : "<2en>",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# 2. NAP MO HINH VA TOKENIZER (thuc hien 1 lan khi khoi dong server)
# ===========================================================================
def load_model_and_tokenizer():
    """
    Nap tokenizer BPE va mo hinh Transformer tu file checkpoint.
    Tra ve (model, tokenizer) da san sang.
    """
    # ── Nap tokenizer ────────────────────────────────────────────────────
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Khong tim thay tokenizer: {TOKENIZER_PATH}")
    tokenizer = HFTokenizer.from_file(TOKENIZER_PATH)
    print(f"[OK] Tokenizer loaded  | vocab_size={tokenizer.get_vocab_size()}")

    # ── Nap checkpoint ────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    # Doc cau hinh model tu checkpoint (neu co), fallback ve tham so mac dinh
    cfg = ckpt.get("model_config", {
        "vocab_size" : 32_000,
        "d_model"    : 512,
        "num_heads"  : 8,
        "num_layers" : 6,
        "d_ff"       : 2048,
        "dropout"    : 0.1,
        "pad_idx"    : PAD_IDX,
        "max_len"    : MAX_LEN + 10,
    })

    # ── Khoi tao va nap trong so ──────────────────────────────────────────
    model = Transformer(**cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()   # Tat Dropout va BatchNorm khi inference

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Model loaded      | epoch={ckpt.get('epoch','?')} "
          f"| params={total:,} | device={DEVICE}")
    return model, tokenizer


# Nap model va tokenizer vao bien global (chi nap 1 lan)
print("Dang khoi dong server...")
model, tokenizer = load_model_and_tokenizer()
print("San sang phuc vu!\n")


# ===========================================================================
# 3. HAM DICH CHINH (GREEDY DECODING)
# ===========================================================================
def translate_text(source_text: str, target_lang: str) -> str:
    """
    Dich cau source_text sang ngon ngu target_lang bang Greedy Decoding.

    Quy trinh:
      1. Them tag ngon ngu vao dau cau nguon
      2. Tokenize → padding → chuyen sang Tensor
      3. Encoder xu ly cau nguon 1 lan
      4. Decoder sinh tung token cho den khi gap <eos> hoac du max_len
      5. Decode danh sach ID → chuoi van ban

    Args:
        source_text : Cau can dich (tieng Anh)
        target_lang : Ngon ngu dich (key trong LANG_TAG)

    Returns:
        Chuoi ket qua dich
    """
    source_text = source_text.strip()
    if not source_text:
        return ""

    # ── Buoc 1: Gan tag ngon ngu vao dau cau ─────────────────────────────
    tag     = LANG_TAG.get(target_lang, "<2vi>")
    src_str = f"{tag} {source_text}"

    # ── Buoc 2: Tokenize va tao Tensor dau vao ────────────────────────────
    # encode() tra ve Encoding, .ids la danh sach so nguyen
    ids = tokenizer.encode(src_str).ids[:MAX_LEN]          # Cat neu qua dai

    # Padding de dua ve chieu MAX_LEN  (padding phai = 0)
    ids = ids + [PAD_IDX] * (MAX_LEN - len(ids))

    # Them chieu batch (B=1): [MAX_LEN] → [1, MAX_LEN]
    src_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    # ── Buoc 3–4: Greedy Decoding ─────────────────────────────────────────
    with torch.no_grad():
        # Su dung phuong thuc translate_greedy co san trong class Transformer
        # Tra ve danh sach ID (khong tinh BOS)
        out_ids = model.translate_greedy(
            src_tensor,
            bos_id  = BOS_IDX,
            eos_id  = EOS_IDX,
            max_len = MAX_LEN,
        )

    # ── Buoc 5: Giai ma ID → van ban ─────────────────────────────────────
    # skip_special_tokens=True: loai bo <pad>, <bos>, <eos>, cac tag <2vi>...
    result = tokenizer.decode(out_ids, skip_special_tokens=True)
    return result.strip() if result.strip() else "[Khong the dich]"


# ===========================================================================
# 4. XAY DUNG GIAO DIEN GRADIO
# ===========================================================================
# Vi du mau de hien thi trong Gradio Examples
EXAMPLES = [
    ["Hello, how are you?",              "Tieng Viet (vi)"],
    ["I love machine learning.",         "Tieng Viet (vi)"],
    ["The weather is beautiful today.",  "Tieng Nhat (ja)"],
    ["Thank you very much.",             "Tieng Trung (zh)"],
    ["Artificial intelligence is the future of humanity.", "Tieng Viet (vi)"],
    ["She went to the market yesterday.", "Tieng Nhat (ja)"],
]

# CSS tuy chinh giao dien
CUSTOM_CSS = """
#title { text-align: center; }
#description { text-align: center; color: #666; }
.gradio-container { max-width: 900px !important; margin: auto; }
"""

with gr.Blocks() as demo:

    # Tieu de va mo ta
    gr.Markdown(
        "# Dich thuat May Da ngon ngu",
        elem_id="title",
    )
    gr.Markdown(
        "**Mo hinh Transformer** tu code bang PyTorch (512d · 8 heads · 6 layers · 32k vocab)\n\n"
        "Nhap cau tieng Anh, chon ngon ngu dich va nhan **Dich ngay**.",
        elem_id="description",
    )

    with gr.Row():
        # Cot trai: Dau vao
        with gr.Column(scale=1):
            src_box = gr.Textbox(
                label       = "Van ban nguon (Tieng Anh)",
                placeholder = "Nhap cau tieng Anh can dich...",
                lines       = 5,
                elem_id     = "source_input",
            )
            lang_radio = gr.Radio(
                choices = list(LANG_TAG.keys()),
                value   = "Tieng Viet (vi)",
                label   = "Ngon ngu dich",
                elem_id = "lang_selector",
            )
            btn = gr.Button("Dich ngay", variant="primary")

        # Cot phai: Dau ra
        with gr.Column(scale=1):
            out_box = gr.Textbox(
                label    = "Ket qua dich",
                lines    = 5,
                interactive = False,
                elem_id  = "translation_output",
            )

    # Nhan nut hoac nhan Enter trong textbox deu kich hoat dich
    btn.click(
        fn      = translate_text,
        inputs  = [src_box, lang_radio],
        outputs = out_box,
    )
    src_box.submit(
        fn      = translate_text,
        inputs  = [src_box, lang_radio],
        outputs = out_box,
    )

    # Vi du mau
    gr.Examples(
        examples        = EXAMPLES,
        inputs          = [src_box, lang_radio],
        outputs         = out_box,
        fn              = translate_text,
        cache_examples  = False,
        label           = "Vi du mau",
    )

    # Thong tin ky thuat o cuoi trang
    gr.Markdown(
        "---\n"
        "**Thong tin mo hinh:** vocab=32k · d=512 · heads=8 · layers=6 · "
        "Greedy Decoding · max_len=64"
    )


# ===========================================================================
# 5. CHAY SERVER
# ===========================================================================
if __name__ == "__main__":
    demo.launch(
        share        = False,   # Doi thanh True neu muon link public (can internet)
        server_name  = "0.0.0.0",
        server_port  = 7860,
        show_error   = True,
        css          = CUSTOM_CSS,
        theme        = gr.themes.Soft(),
    )
