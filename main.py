"""
main.py  –  Multilingual Machine Translation
=============================================
Entry point chinh cua du an.

Trang thai hien tai:
    [Done] data/raw/          – Du lieu thu truc tiep tu HuggingFace + local
    [Done] data/processed/    – train_multilingual.txt (300k cap cau)
                                train.txt / val.txt / test.txt (85/10/5)
    [Done] tokenizer/         – BPE tokenizer 32k vocab

Buoc tiep theo:
    - Tuan 2: Xay dung mo hinh Transformer (src/layers.py, src/attention.py)
    - Tuan 3: Vong lap huan luyen + danh gia BLEU

Chay:
    python main.py            # Chia du lieu va kiem tra DataLoader
    python main.py --skip_split   # Bo qua chia neu da co train/val/test.txt
"""

import argparse
import logging
from pathlib import Path

from src.data_utils import split_dataset, get_dataloaders

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Duong dan mac dinh
# ---------------------------------------------------------------------------
MULTILINGUAL_FILE = "data/processed/train_multilingual.txt"
PROCESSED_DIR     = "data/processed"
TOKENIZER_PATH    = "tokenizer/tokenizer.json"
TRAIN_FILE        = "data/processed/train.txt"
VAL_FILE          = "data/processed/val.txt"
TEST_FILE         = "data/processed/test.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multilingual MT – Pipeline chuan bi du lieu"
    )
    parser.add_argument(
        "--skip_split",
        action="store_true",
        help="Bo qua buoc chia dataset neu train/val/test.txt da ton tai",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Kich thuoc batch (mac dinh: 64)",
    )
    parser.add_argument(
        "--max_len", type=int, default=64,
        help="Do dai chuan cua chuoi token (mac dinh: 64)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 55)
    logger.info("  Multilingual MT – Data Pipeline")
    logger.info("=" * 55)

    # --- Buoc 1: Chia dataset ---
    already_split = (
        Path(TRAIN_FILE).exists()
        and Path(VAL_FILE).exists()
        and Path(TEST_FILE).exists()
    )

    if args.skip_split and already_split:
        logger.info("Bo qua chia dataset (file da ton tai).")
    else:
        logger.info("Dang chia dataset 85 / 10 / 5 ...")
        split_dataset(
            src_file=MULTILINGUAL_FILE,
            out_dir=PROCESSED_DIR,
        )

    # --- Buoc 2: Kiem tra DataLoader ---
    logger.info("Khoi tao DataLoader ...")
    train_dl, val_dl, test_dl = get_dataloaders(
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        test_file=TEST_FILE,
        tokenizer_path=TOKENIZER_PATH,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    batch = next(iter(train_dl))
    logger.info(
        "Batch mau – src: %s | tgt: %s | dtype: %s",
        tuple(batch["src"].shape),
        tuple(batch["tgt"].shape),
        batch["src"].dtype,
    )
    logger.info("Pipeline hoan tat. San sang cho buoc huan luyen.")


if __name__ == "__main__":
    main()
