"""
prepare_data.py  –  Tuần 1: Chuẩn bị dữ liệu đa ngôn ngữ
===========================================================
Nhiệm vụ:
    1. Tải EN-JA và EN-ZH từ Helsinki-NLP/tatoeba_mt (HuggingFace datasets)
    2. Đọc EN-VI từ file local (data/raw/train.en & data/raw/train.vi)
    3. Tiền xử lý văn bản: xóa khoảng trắng thừa, dòng rỗng, ký tự đặc biệt
    4. Gộp 3 cặp ngôn ngữ, thêm Target_Token vào đầu mỗi dòng
    5. Xáo trộn ngẫu nhiên và ghi ra data/processed/train_multilingual.txt

Định dạng mỗi dòng đầu ra:
    <2vi> câu_nguồn_en\tcâu_đích_vi
    <2ja> câu_nguồn_en\tcâu_đích_ja
    <2zh> câu_nguồn_en\tcâu_đích_zh

Yêu cầu:
    pip install datasets

Chạy:
    python src/prepare_data.py
    python src/prepare_data.py --max_pairs 50000 --seed 42
"""

import argparse
import logging
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────────────────────────────────────
# Dataset HuggingFace (opus-100 dùng Parquet chuẩn, không có custom script)
# tatoeba_mt đã bị loại bỏ vì dùng custom script không còn được hỗ trợ
HF_DATASET_NAME = "Helsinki-NLP/opus-100"

# Ánh xạ ngôn ngữ đích → token kiểm soát
TARGET_TOKEN = {
    "vi": "<2vi>",
    "ja": "<2ja>",
    "zh": "<2zh>",
}

# Số dòng tối đa mỗi cặp ngôn ngữ (để cân bằng)
DEFAULT_MAX_PAIRS = 100_000

# Seed ngẫu nhiên
DEFAULT_SEED = 42

# Đường dẫn mặc định EN-VI (file đã tải về bằng data.py)
DEFAULT_EN_VI_SRC = "data/raw/train.en-vi.en"
DEFAULT_EN_VI_TGT = "data/raw/train.en-vi.vi"
DEFAULT_OUTPUT    = "data/processed/train_multilingual.txt"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Tiền xử lý văn bản
# ──────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Làm sạch một chuỗi văn bản:
      - Chuẩn hóa Unicode về dạng NFC (xử lý ký tự đặc biệt đa ngôn ngữ)
      - Xóa ký tự điều khiển (control characters)
      - Chuẩn hóa dấu nháy và gạch ngang đặc biệt về dạng ASCII
      - Xóa khoảng trắng thừa (đầu, cuối, giữa câu)

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Chuỗi văn bản đã làm sạch; chuỗi rỗng nếu sau xử lý không còn nội dung.
    """
    if not text:
        return ""

    # Bước 1: Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Bước 2: Loại bỏ ký tự điều khiển (trừ \t và \n để tránh nhầm)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf"} or ch in ("\t", "\n")
    )

    # Bước 3: Chuẩn hóa dấu nháy đặc biệt → ASCII
    quote_map = {
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        "\u00ab": '"',  # «
        "\u00bb": '"',  # »
    }
    for src_char, dst_char in quote_map.items():
        text = text.replace(src_char, dst_char)

    # Bước 4: Chuẩn hóa gạch ngang đặc biệt → ASCII hyphen
    text = re.sub(r"[\u2012\u2013\u2014\u2015]", "-", text)

    # Bước 5: Xóa khoảng trắng thừa
    text = re.sub(r"[ \t]+", " ", text)   # nhiều space/tab → 1 space
    text = text.strip()

    return text


def is_valid_pair(src: str, tgt: str) -> bool:
    """
    Kiểm tra tính hợp lệ của một cặp câu sau tiền xử lý.
      - Cả hai phải không rỗng
      - Câu nguồn không được bắt đầu bằng '#' (comment)
      - Độ dài ký tự phải trong khoảng hợp lý [3, 500]

    Args:
        src: Câu nguồn (tiếng Anh).
        tgt: Câu đích.

    Returns:
        True nếu cặp hợp lệ.
    """
    if not src or not tgt:
        return False
    if src.startswith("#") or tgt.startswith("#"):
        return False
    if not (3 <= len(src) <= 500 and 3 <= len(tgt) <= 500):
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tải dữ liệu từ HuggingFace (EN-JA, EN-ZH)
# ──────────────────────────────────────────────────────────────────────────────
def load_from_huggingface(
    lang_pair: str,
    tgt_lang: str,
    max_pairs: int,
) -> List[Tuple[str, str]]:
    """
    Tải dữ liệu song ngữ từ Helsinki-NLP/opus-100 trên HuggingFace.

    Args:
        lang_pair: Chuỗi config name của opus-100, ví dụ "en-ja" hoặc "en-zh".
        tgt_lang:  Mã ngôn ngữ đích ngắn ("ja" hoặc "zh"), dùng để log.
        max_pairs: Số cặp tối đa cần lấy.

    Returns:
        Danh sách (câu_en, câu_đích) đã tiền xử lý và hợp lệ.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Thư viện `datasets` chưa được cài đặt.\n"
            "Chạy: pip install datasets"
        )

    logger.info("──────────────────────────────────────────")
    logger.info("Đang tải %s từ HuggingFace (%s) ...", lang_pair, HF_DATASET_NAME)

    # opus-100 lưu dạng Parquet, không dùng custom script
    # Thử split "train" trước, fallback về "test"
    dataset = None
    for split in ("train", "test"):
        try:
            dataset = load_dataset(
                HF_DATASET_NAME,  # "Helsinki-NLP/opus-100"
                lang_pair,        # config: "en-ja" hoặc "en-zh"
                split=split,
            )
            logger.info("  └─ Dùng split '%s' (%d mẫu tổng)", split, len(dataset))
            break
        except Exception as e:
            logger.debug("  split='%s' không khả dụng: %s", split, e)
            continue

    if dataset is None:
        logger.error("Không thể tải dataset cho cặp %s. Bỏ qua.", lang_pair)
        return []

    pairs: List[Tuple[str, str]] = []
    raw_count  = 0
    skip_count = 0

    for row in dataset:
        if len(pairs) >= max_pairs:
            break
        raw_count += 1

        # opus-100 luôn có cột "translation" dạng dict {"en": ..., "ja": ...}
        trans = row.get("translation", {})
        src_raw = trans.get("en", "")       # câu tiếng Anh
        tgt_raw = trans.get(tgt_lang, "")   # câu đích (ja / zh)

        src_clean = preprocess_text(src_raw)
        tgt_clean = preprocess_text(tgt_raw)

        if is_valid_pair(src_clean, tgt_clean):
            pairs.append((src_clean, tgt_clean))
        else:
            skip_count += 1

        if len(pairs) % 10_000 == 0 and len(pairs) > 0:
            logger.info("  ... đã xử lý %d dòng hợp lệ", len(pairs))

    logger.info(
        "  ✓ EN-%s: đọc %d dòng thô | hợp lệ %d | bỏ qua %d",
        tgt_lang.upper(), raw_count, len(pairs), skip_count,
    )
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 3. Đọc dữ liệu EN-VI từ file local
# ──────────────────────────────────────────────────────────────────────────────
def load_local_envi(
    src_path: str,
    tgt_path: str,
    max_pairs: int,
) -> List[Tuple[str, str]]:
    """
    Đọc cặp câu EN-VI từ 2 file plain-text song song.

    Args:
        src_path: Đường dẫn file tiếng Anh (mỗi dòng 1 câu).
        tgt_path: Đường dẫn file tiếng Việt (mỗi dòng 1 câu, song song với src).
        max_pairs: Số cặp tối đa cần lấy.

    Returns:
        Danh sách (câu_en, câu_vi) đã tiền xử lý và hợp lệ.
    """
    logger.info("──────────────────────────────────────────")
    logger.info("Đang đọc EN-VI từ file local ...")
    logger.info("  src: %s", src_path)
    logger.info("  tgt: %s", tgt_path)

    src_path = Path(src_path)
    tgt_path = Path(tgt_path)

    if not src_path.exists():
        logger.error("  ✗ Không tìm thấy file nguồn: %s", src_path)
        return []
    if not tgt_path.exists():
        logger.error("  ✗ Không tìm thấy file đích: %s", tgt_path)
        return []

    pairs: List[Tuple[str, str]] = []
    raw_count  = 0
    skip_count = 0

    with open(src_path, "r", encoding="utf-8") as f_src, \
         open(tgt_path, "r", encoding="utf-8") as f_tgt:

        for src_line, tgt_line in zip(f_src, f_tgt):
            if len(pairs) >= max_pairs:
                break
            raw_count += 1

            src_clean = preprocess_text(src_line)
            tgt_clean = preprocess_text(tgt_line)

            if is_valid_pair(src_clean, tgt_clean):
                pairs.append((src_clean, tgt_clean))
            else:
                skip_count += 1

    logger.info(
        "  ✓ EN-VI: đọc %d dòng thô | hợp lệ %d | bỏ qua %d",
        raw_count, len(pairs), skip_count,
    )
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gộp, xáo trộn và ghi file
# ──────────────────────────────────────────────────────────────────────────────
def merge_and_save(
    pairs_by_lang: dict,
    output_path: str,
    seed: int,
) -> None:
    """
    Gộp tất cả các cặp câu từ nhiều ngôn ngữ, thêm Target_Token, xáo trộn
    ngẫu nhiên rồi ghi ra file.

    Định dạng mỗi dòng:
        <2xx> câu_nguồn\tcâu_đích\n

    Args:
        pairs_by_lang: Dict ánh xạ mã ngôn ngữ → List[Tuple[str, str]].
                       Ví dụ: {"vi": [...], "ja": [...], "zh": [...]}
        output_path:   Đường dẫn file đầu ra.
        seed:          Seed cho random.shuffle để tái tạo được kết quả.
    """
    logger.info("──────────────────────────────────────────")
    logger.info("Đang gộp dữ liệu ...")

    all_lines: List[str] = []

    for lang, pairs in pairs_by_lang.items():
        token = TARGET_TOKEN[lang]
        count = 0
        for src, tgt in pairs:
            # Đảm bảo không có tab ẩn trong câu (sẽ làm vỡ định dạng TSV)
            src = src.replace("\t", " ")
            tgt = tgt.replace("\t", " ")
            all_lines.append(f"{token} {src}\t{tgt}\n")
            count += 1
        logger.info("  └─ %-4s : %d dòng (token: %s)", lang.upper(), count, token)

    logger.info("Tổng số dòng trước khi xáo trộn: %d", len(all_lines))

    # Xáo trộn ngẫu nhiên có seed
    random.seed(seed)
    random.shuffle(all_lines)
    logger.info("Đã xáo trộn với seed=%d", seed)

    # Ghi ra file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.writelines(all_lines)

    logger.info("──────────────────────────────────────────")
    logger.info("✅ Đã ghi %d dòng vào: %s", len(all_lines), output_path)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Pipeline chính
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    max_pairs: int,
    seed: int,
    en_vi_src: str,
    en_vi_tgt: str,
    output_path: str,
) -> None:
    """
    Chạy toàn bộ pipeline chuẩn bị dữ liệu đa ngôn ngữ:
        EN-JA (HuggingFace) + EN-ZH (HuggingFace) + EN-VI (local)
        → gộp với Target_Token → shuffle → lưu file.

    Args:
        max_pairs:   Số cặp tối đa mỗi ngôn ngữ.
        seed:        Seed ngẫu nhiên.
        en_vi_src:   Đường dẫn file EN (tiếng Anh cho cặp EN-VI).
        en_vi_tgt:   Đường dẫn file VI (tiếng Việt).
        output_path: Đường dẫn file đầu ra.
    """
    logger.info("==========================================")
    logger.info("  Multilingual MT – Chuẩn bị dữ liệu     ")
    logger.info("  max_pairs=%d | seed=%d                  ", max_pairs, seed)
    logger.info("==========================================")

    pairs_by_lang = {}

    # ── EN-JA ────────────────────────────────────────────────────────────────
    # Config name trong opus-100: "en-ja"
    pairs_ja = load_from_huggingface(
        lang_pair="en-ja",
        tgt_lang="ja",
        max_pairs=max_pairs,
    )
    if pairs_ja:
        pairs_by_lang["ja"] = pairs_ja

    # ── EN-ZH ────────────────────────────────────────────────────────────────
    # Config name trong opus-100: "en-zh"
    pairs_zh = load_from_huggingface(
        lang_pair="en-zh",
        tgt_lang="zh",
        max_pairs=max_pairs,
    )
    if pairs_zh:
        pairs_by_lang["zh"] = pairs_zh

    # ── EN-VI ────────────────────────────────────────────────────────────────
    pairs_vi = load_local_envi(
        src_path=en_vi_src,
        tgt_path=en_vi_tgt,
        max_pairs=max_pairs,
    )
    if pairs_vi:
        pairs_by_lang["vi"] = pairs_vi

    if not pairs_by_lang:
        logger.error("Không có dữ liệu nào để gộp. Thoát.")
        return

    # ── Gộp & Lưu ────────────────────────────────────────────────────────────
    merge_and_save(
        pairs_by_lang=pairs_by_lang,
        output_path=output_path,
        seed=seed,
    )

    # ── Thống kê cuối ────────────────────────────────────────────────────────
    logger.info("==========================================")
    logger.info("  Tóm tắt kết quả:")
    total = 0
    for lang, pairs in pairs_by_lang.items():
        logger.info("    EN-%-2s : %6d cặp câu", lang.upper(), len(pairs))
        total += len(pairs)
    logger.info("    Tổng  : %6d cặp câu", total)
    logger.info("    Output: %s", output_path)
    logger.info("==========================================")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tuần 1 – Chuẩn bị dữ liệu đa ngôn ngữ cho Multilingual MT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Số cặp câu tối đa mỗi ngôn ngữ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed ngẫu nhiên cho shuffle",
    )
    parser.add_argument(
        "--en_vi_src",
        type=str,
        default=DEFAULT_EN_VI_SRC,
        help="Đường dẫn file tiếng Anh của cặp EN-VI",
    )
    parser.add_argument(
        "--en_vi_tgt",
        type=str,
        default=DEFAULT_EN_VI_TGT,
        help="Đường dẫn file tiếng Việt của cặp EN-VI",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Đường dẫn file đầu ra (train_multilingual.txt)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        max_pairs=args.max_pairs,
        seed=args.seed,
        en_vi_src=args.en_vi_src,
        en_vi_tgt=args.en_vi_tgt,
        output_path=args.output,
    )
