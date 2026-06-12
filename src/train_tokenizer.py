"""
train_tokenizer.py
------------------
Huấn luyện bộ từ vựng BPE đa ngôn ngữ bằng thư viện `tokenizers` của HuggingFace.

Yêu cầu: pip install tokenizers
"""

import logging
from pathlib import Path
from typing import Optional, List

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Chưa cài `tokenizers`. Chạy: pip install tokenizers")

# ── Token đặc biệt ────────────────────────────────────────────────────────────
SPECIAL_TOKENS: List[str] = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]",
    "__src_en__",
    "__tgt_vi__", "__tgt_ja__", "__tgt_zh__",
]
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "[PAD]", "[UNK]", "[BOS]", "[EOS]"


def train_bpe_tokenizer(
    corpus_file: str,
    save_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
) -> "Tokenizer":
    """
    Huấn luyện BPE tokenizer trên corpus đa ngôn ngữ và lưu ra đĩa.

    Args:
        corpus_file:    Đường dẫn file corpus (train_multilingual.txt)
        save_dir:       Thư mục lưu kết quả
        vocab_size:     Kích thước từ vựng
        min_frequency:  Tần suất tối thiểu của cặp BPE
        special_tokens: Token đặc biệt (dùng SPECIAL_TOKENS nếu None)

    Returns:
        Đối tượng Tokenizer đã huấn luyện
    """
    if not HF_AVAILABLE:
        raise ImportError("Cài đặt tokenizers: pip install tokenizers")

    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    logger.info("Huấn luyện BPE trên: %s (vocab_size=%d)", corpus_file, vocab_size)
    tokenizer.train(files=[str(corpus_file)], trainer=trainer)

    # Thêm [BOS] / [EOS] tự động
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    save_path = save_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    logger.info("Đã lưu tokenizer tại: %s (vocab=%d)", save_path, tokenizer.get_vocab_size())

    # Lưu metadata
    with open(save_dir / "tokenizer_meta.txt", "w", encoding="utf-8") as f:
        f.write(f"vocab_size={tokenizer.get_vocab_size()}\n")
        f.write(f"min_frequency={min_frequency}\n")
        f.write(f"corpus_file={corpus_file}\n")
        f.write(f"special_tokens={special_tokens}\n")

    return tokenizer


def load_tokenizer(save_dir: str) -> "Tokenizer":
    """Nạp tokenizer đã lưu từ thư mục."""
    if not HF_AVAILABLE:
        raise ImportError("Cài đặt tokenizers: pip install tokenizers")
    path = Path(save_dir) / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {path}")
    tok = Tokenizer.from_file(str(path))
    logger.info("Đã nạp tokenizer từ %s (vocab=%d)", path, tok.get_vocab_size())
    return tok


class BPETokenizerWrapper:
    """
    Wrapper bọc HuggingFace Tokenizer, tương thích với MultilingualDataset.

    Giao diện:
        .encode(text)  -> List[int]
        .decode(ids)   -> str
        .pad_id, .bos_id, .eos_id, .vocab_size
    """

    def __init__(self, tokenizer: "Tokenizer"):
        self._tok = tokenizer
        self.pad_id   = self._tok.token_to_id(PAD_TOKEN) or 0
        self.bos_id   = self._tok.token_to_id(BOS_TOKEN)
        self.eos_id   = self._tok.token_to_id(EOS_TOKEN)
        self.vocab_size = self._tok.get_vocab_size()

    def encode(self, text: str):
        return self._tok.encode(text).ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    @classmethod
    def from_file(cls, save_dir: str) -> "BPETokenizerWrapper":
        return cls(load_tokenizer(save_dir))


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Huấn luyện BPE Tokenizer đa ngôn ngữ")
    parser.add_argument("--corpus",     default="data/processed/train_multilingual.txt")
    parser.add_argument("--save_dir",   default="tokenizer/")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_freq",   type=int, default=2)
    args = parser.parse_args()

    train_bpe_tokenizer(
        corpus_file=args.corpus,
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
    )
