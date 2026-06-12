"""
data_utils.py
=============
Pipeline du lieu cho he thong dich may da ngon ngu (EN-JA-ZH-VI).

Nhiem vu:
    1. split_dataset()        – Chia train_multilingual.txt → train/val/test
    2. MultilingualDataset    – PyTorch Dataset doc file .txt + tokenize
    3. get_dataloaders()      – Tra ve 3 DataLoader san sang cho vong lap huan luyen

Dinh dang moi dong trong file .txt:
    <2vi> cau_nguon_en<TAB>cau_dich_vi
    <2ja> cau_nguon_en<TAB>cau_dich_ja
    <2zh> cau_nguon_en<TAB>cau_dich_zh

Quy uoc Token ID dac biet (theo shared_tokenizer.json):
    PAD_ID = 0   [PAD]  – them vao khi chuoi ngan hon max_len
    EOS_ID = 2   [EOS]  – buoc ket thuc chuoi; phai la token cuoi sau cat

Quy uoc ky hieu Tensor Shape trong toan file:
    B       = batch_size  (so cau trong 1 mini-batch)
    T       = max_len     (do dai chuan sau pad_or_trim, mac dinh 64)
    V       = vocab_size  (kich thuoc tu dien BPE)
"""

import logging
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hang so mac dinh
# ---------------------------------------------------------------------------
PAD_ID  = 0          # ID cua token [PAD]
EOS_ID  = 2          # ID cua token [EOS]
MAX_LEN = 64         # Do dai chuan cua moi chuoi sau pad_or_trim


# ===========================================================================
# 1. CHIA TAP DU LIEU
# ===========================================================================
def split_dataset(
    src_file: str = "data/processed/train_multilingual.txt",
    out_dir:  str = "data/processed",
    train_ratio: float = 0.85,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.05,
    seed: int = 42,
) -> Tuple[str, str, str]:
    """
    Doc file goc, xao tron ngau nhien, chia thanh 3 tap va luu ra dia.

    Ti le mac dinh: Train 85% | Val 10% | Test 5%
    Tong ti le buoc phai bang 1.0.

    Args:
        src_file     : Duong dan file gop da xu ly.
        out_dir      : Thu muc luu 3 file dau ra.
        train_ratio  : Ti le tap Train.
        val_ratio    : Ti le tap Validation.
        test_ratio   : Ti le tap Test.
        seed         : Seed cho random.shuffle (de tai tao ket qua).

    Returns:
        Tuple (train_path, val_path, test_path) – duong dan cac file da luu.

    Raises:
        FileNotFoundError : Neu src_file khong ton tai.
        ValueError        : Neu tong ti le != 1.0 hoac file rong.
    """
    # --- Kiem tra ti le ---
    total = round(train_ratio + val_ratio + test_ratio, 6)
    if abs(total - 1.0) > 1e-5:
        raise ValueError(
            f"Tong ti le phai bang 1.0, hien tai = {total:.4f}"
        )

    src_path = Path(src_file)
    if not src_path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {src_path}")

    # --- Doc toan bo dong ---
    logger.info("Dang doc: %s", src_path)
    with open(src_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]   # bo dong rong

    if not lines:
        raise ValueError(f"File rong: {src_path}")

    total_lines = len(lines)
    logger.info("Tong so dong hop le: %d", total_lines)

    # --- Xao tron ngau nhien ---
    random.seed(seed)
    random.shuffle(lines)

    # --- Tinh nguong chia ---
    n_train = int(total_lines * train_ratio)
    n_val   = int(total_lines * val_ratio)
    # Test lay phan con lai de dam bao khong mat dong nao
    n_test  = total_lines - n_train - n_val

    train_lines = lines[:n_train]
    val_lines   = lines[n_train : n_train + n_val]
    test_lines  = lines[n_train + n_val :]

    logger.info(
        "Chia tap: Train=%d | Val=%d | Test=%d",
        len(train_lines), len(val_lines), len(test_lines),
    )

    # --- Luu ra dia ---
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = str(out_path / "train.txt")
    val_path   = str(out_path / "val.txt")
    test_path  = str(out_path / "test.txt")

    def _save(path: str, data: List[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(data)
        logger.info("  -> Da luu %d dong vao: %s", len(data), path)

    _save(train_path, train_lines)
    _save(val_path,   val_lines)
    _save(test_path,  test_lines)

    return train_path, val_path, test_path


# ===========================================================================
# 2. HAM BO TRO: pad_or_trim
# ===========================================================================
def pad_or_trim(ids: List[int], max_len: int = MAX_LEN) -> List[int]:
    """
    Ep mot chuoi Token ID ve dung do dai max_len.

    Quy tac:
        - Neu len(ids) < max_len : them PAD_ID vao cuoi cho du max_len
        - Neu len(ids) > max_len : cat bot, thay token cuoi bang EOS_ID
        - Neu len(ids) == max_len: giu nguyen (nhung van dam bao cuoi la EOS)

    Tai sao bat buoc ket thuc bang EOS_ID?
        Model can biet diem ket thuc cua chuoi. Neu cat, token cuoi cuoi
        co the la mot subword bat ky → Model se bi nham. Thay bang EOS
        dam bao tin hieu "ket thuc chuoi" luon co mat.

    Args:
        ids     : Danh sach Token ID (output cua tokenizer.encode).
        max_len : Do dai muc tieu.

    Returns:
        List[int] co dung max_len phan tu, ket thuc bang EOS_ID.

    Vi du:
        ids     = [2, 15, 37, 3]   (BOS, w1, w2, EOS)
        max_len = 6
        output  = [2, 15, 37, 3, 0, 0]   (pad them 2 vi tri)

        ids     = [2, 15, 37, 99, 102, 55, 201, 3]   (8 token)
        max_len = 6
        output  = [2, 15, 37, 99, 102, 2]   (cat + them EOS o cuoi)
    """
    if len(ids) < max_len:
        # Padding: them PAD_ID vao cuoi
        # Truoc: ids          = [t0, t1, ..., tn]        do dai n+1
        # Sau:   ids + padding = [t0, t1, ..., tn, 0, 0, ...]  do dai max_len
        padding = [PAD_ID] * (max_len - len(ids))
        return ids + padding                  # do dai = max_len

    else:
        # Cat bot va buoc EOS o vi tri cuoi
        # Lay max_len-1 token dau, them EOS vao cuoi
        truncated = ids[: max_len - 1]        # do dai = max_len - 1
        truncated.append(EOS_ID)              # do dai = max_len
        return truncated                      # do dai = max_len


# ===========================================================================
# 3. PYTORCH DATASET
# ===========================================================================
class MultilingualDataset(Dataset):
    """
    PyTorch Dataset doc file cac cap cau da xu ly.

    Moi dong trong file co dinh dang:
        <2xx> cau_nguon<TAB>cau_dich

    Quy trinh xu ly 1 mau (__getitem__):
        dong  →  tach TAB  →  tokenize  →  pad_or_trim  →  Tensor

    Dau ra cua __getitem__:
        {
          "src" : LongTensor [T]   – Token ID chuoi nguon (co tag ngon ngu)
          "tgt" : LongTensor [T]   – Token ID chuoi dich
        }

    Dau ra cua DataLoader (sau collate mac dinh):
        {
          "src" : LongTensor [B, T]
          "tgt" : LongTensor [B, T]
        }

    Args:
        file_path      : Duong dan den file .txt (train/val/test).
        tokenizer_path : Duong dan den shared_tokenizer.json.
        max_len        : Do dai chuan cua chuoi (mac dinh MAX_LEN=64).
    """

    def __init__(
        self,
        file_path: str,
        tokenizer_path: str,
        max_len: int = MAX_LEN,
    ):
        super().__init__()

        self.max_len = max_len

        # --- Nap tokenizer tu file JSON ---
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "Chua cai thu vien `tokenizers`.\n"
                "Chay: pip install tokenizers"
            )

        tok_path = Path(tokenizer_path)
        if not tok_path.exists():
            raise FileNotFoundError(f"Khong tim thay tokenizer: {tok_path}")

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        logger.info(
            "Nap tokenizer: %s (vocab=%d)",
            tok_path, self.tokenizer.get_vocab_size(),
        )

        # --- Doc va kiem tra file du lieu ---
        data_path = Path(file_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Khong tim thay file du lieu: {data_path}")

        self.samples: List[Tuple[str, str]] = []
        skip_count = 0

        with open(data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # bo dong rong

                # Moi dong: "<2vi> cau_nguon\tcau_dich"
                # Tach theo dau TAB dau tien
                parts = line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    # Dong khong hop le: thieu TAB phan cach
                    skip_count += 1
                    logger.debug(
                        "  Bo qua dong %d (khong co TAB): %s",
                        line_no, line[:60],
                    )
                    continue

                src_text, tgt_text = parts[0].strip(), parts[1].strip()
                if not src_text or not tgt_text:
                    skip_count += 1
                    continue

                self.samples.append((src_text, tgt_text))

        if not self.samples:
            raise ValueError(f"File rong hoac khong co dong hop le: {data_path}")

        logger.info(
            "Dataset '%s': %d mau hop le, bo qua %d dong.",
            data_path.name, len(self.samples), skip_count,
        )

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # -----------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        """
        Lay 1 cap cau, tokenize va pad/trim ve max_len.

        Returns:
            dict voi 2 key:
                "src" : torch.LongTensor  shape [T]
                "tgt" : torch.LongTensor  shape [T]
        """
        src_text, tgt_text = self.samples[idx]

        # --- Tokenize ---
        # tokenizer.encode() tra ve Encoding object
        # .ids la List[int] – danh sach Token ID
        src_ids: List[int] = self.tokenizer.encode(src_text).ids
        tgt_ids: List[int] = self.tokenizer.encode(tgt_text).ids
        # src_ids, tgt_ids : List[int]  do dai tuy y (chua chuan hoa)

        # --- Pad hoac cat ve max_len ---
        src_ids = pad_or_trim(src_ids, self.max_len)  # List[int], do dai = T
        tgt_ids = pad_or_trim(tgt_ids, self.max_len)  # List[int], do dai = T

        # --- Chuyen sang Tensor ---
        # torch.long = int64, can thiet cho nn.Embedding
        src_tensor = torch.tensor(src_ids, dtype=torch.long)  # [T]
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)  # [T]

        return {"src": src_tensor, "tgt": tgt_tensor}
        # Tensor shape khi dung DataLoader:
        #   src: [B, T]  voi B = batch_size, T = max_len
        #   tgt: [B, T]


# ===========================================================================
# 4. TAO DATALOADERS
# ===========================================================================
def get_dataloaders(
    train_file: str,
    val_file:   str,
    test_file:  str,
    tokenizer_path: str,
    max_len:    int = MAX_LEN,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Khoi tao 3 DataLoader tu 3 file train/val/test.

    Cau hinh:
        - Train : shuffle=True   (xao tron moi epoch de tranh hoc theo thu tu)
        - Val   : shuffle=False  (giu thu tu co dinh de ket qua val nhat quan)
        - Test  : shuffle=False

    Args:
        train_file     : Duong dan file train.txt.
        val_file       : Duong dan file val.txt.
        test_file      : Duong dan file test.txt.
        tokenizer_path : Duong dan shared_tokenizer.json.
        max_len        : Do dai chuan chuoi.
        batch_size     : So mau trong 1 batch (mac dinh 64).
        num_workers    : So tien trinh doc du lieu song song (0 = doc o main).

    Returns:
        (train_loader, val_loader, test_loader)

    Luu y Tensor Shape qua DataLoader:
        Moi batch la dict:
            batch["src"] : LongTensor [B, T]  – Token ID chuoi nguon
            batch["tgt"] : LongTensor [B, T]  – Token ID chuoi dich
        Voi B = batch_size (batch cuoi co the nho hon neu het du lieu),
             T = max_len.
    """
    logger.info("Khoi tao DataLoader (batch_size=%d, max_len=%d) ...", batch_size, max_len)

    train_ds = MultilingualDataset(train_file, tokenizer_path, max_len)
    val_ds   = MultilingualDataset(val_file,   tokenizer_path, max_len)
    test_ds  = MultilingualDataset(test_file,  tokenizer_path, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # bat buoc shuffle o tap Train
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # tang toc chuyen sang GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,          # giu nguyen thu tu
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "Hoan tat:\n"
        "  Train : %6d mau | %4d batch\n"
        "  Val   : %6d mau | %4d batch\n"
        "  Test  : %6d mau | %4d batch",
        len(train_ds), len(train_loader),
        len(val_ds),   len(val_loader),
        len(test_ds),  len(test_loader),
    )

    return train_loader, val_loader, test_loader


# ===========================================================================
# Self-test (chay: python -m src.data_utils)
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    SRC_FILE       = "data/processed/train_multilingual.txt"
    TOKENIZER_PATH = "tokenizer/tokenizer.json"   # dung file san co

    print("=" * 60)
    print("BUOC 1: Chia tap du lieu")
    print("=" * 60)
    train_f, val_f, test_f = split_dataset(
        src_file=SRC_FILE,
        out_dir="data/processed",
    )

    print()
    print("=" * 60)
    print("BUOC 2: Kiem tra MultilingualDataset")
    print("=" * 60)
    ds = MultilingualDataset(train_f, TOKENIZER_PATH, max_len=64)
    print(f"  So mau trong tap Train : {len(ds):,}")

    sample = ds[0]
    print(f"  Mau 0 – src shape : {tuple(sample['src'].shape)}")  # (64,)
    print(f"  Mau 0 – tgt shape : {tuple(sample['tgt'].shape)}")  # (64,)
    print(f"  src IDs (10 dau)  : {sample['src'][:10].tolist()}")
    print(f"  tgt IDs (10 dau)  : {sample['tgt'][:10].tolist()}")

    # Kiem tra token cuoi khong phai PAD
    last_src = sample['src'][-1].item()
    print(f"  Token cuoi src    : {last_src}  (EOS={EOS_ID} neu cat, PAD={PAD_ID} neu pad)")

    print()
    print("=" * 60)
    print("BUOC 3: Kiem tra get_dataloaders")
    print("=" * 60)
    train_dl, val_dl, test_dl = get_dataloaders(
        train_file=train_f,
        val_file=val_f,
        test_file=test_f,
        tokenizer_path=TOKENIZER_PATH,
        max_len=64,
        batch_size=64,
    )

    batch = next(iter(train_dl))
    print(f"  Batch src shape : {tuple(batch['src'].shape)}")  # [64, 64] = [B, T]
    print(f"  Batch tgt shape : {tuple(batch['tgt'].shape)}")  # [64, 64] = [B, T]
    print(f"  dtype           : {batch['src'].dtype}")          # torch.int64

    print()
    print("Tat ca kiem tra pass – du lieu san sang cho vong lap huan luyen!")
