"""
train_baseline.py  –  Vong lap huan luyen mo hinh Baseline LSTM Seq2Seq
========================================================================
Muc dich : Huan luyen va luu mo hinh LSTMBaseline lam moc so sanh
           cho mo hinh Transformer chinh cua du an.

Cau hinh thi nghiem (ghi vao bao cao):
    ┌────────────────────────────┬────────────────────────────┐
    │ Sieu tham so               │ Gia tri                    │
    ├────────────────────────────┼────────────────────────────┤
    │ Mo hinh                    │ LSTM Seq2Seq (no Attention) │
    │ vocab_size                 │ 32 000                     │
    │ embed_dim                  │ 256                        │
    │ hidden_dim                 │ 512                        │
    │ num_layers                 │ 2                          │
    │ dropout                    │ 0.3                        │
    │ Ham mat mat (Loss)         │ CrossEntropyLoss           │
    │   ignore_index             │ 0  (bo qua token PAD)      │
    │ Bo toi uu (Optimizer)      │ Adam                       │
    │   learning_rate            │ 1e-3                       │
    │   betas                    │ (0.9, 0.999)               │
    │   weight_decay             │ 1e-5                       │
    │ Gradient Clipping          │ max_norm = 1.0             │
    │ So epoch                   │ 10                         │
    │ batch_size                 │ 64                         │
    │ max_len (chuoi token)      │ 64                         │
    │ Chien luoc Decoder         │ Teacher Forcing (100%)     │
    └────────────────────────────┴────────────────────────────┘

Checkpoint:
    - Luu sau moi epoch: model_assets/checkpoint_epoch_N.pt
    - Luu mo hinh tot nhat: model_assets/best_baseline_model.pt

Chay:
    python src/train_baseline.py
    python src/train_baseline.py --epochs 5 --batch_size 32
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Them thu muc goc vao sys.path de import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import split_dataset, get_dataloaders
from src.models.baseline_lstm import LSTMBaseline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ===========================================================================
# CAU HINH THI NGHIEM
# (Thay doi cac hang so nay khi lam bao cao, khong can sua logic)
# ===========================================================================

# --- Du lieu ---
MULTILINGUAL_FILE = "data/processed/train_multilingual.txt"
TRAIN_FILE        = "data/processed/train.txt"
VAL_FILE          = "data/processed/val.txt"
TEST_FILE         = "data/processed/test.txt"
TOKENIZER_PATH    = "tokenizer/tokenizer.json"

# --- Tham so mo hinh ---
VOCAB_SIZE  = 32_000   # Phu hop voi BPE tokenizer da huan luyen
EMBED_DIM   = 256      # Chieu vector nhung
HIDDEN_DIM  = 512      # Chieu trang thai an cua LSTM
NUM_LAYERS  = 2        # So tang LSTM chong nhau
DROPOUT     = 0.3      # Ti le Dropout (chi ap khi NUM_LAYERS > 1)
PAD_IDX     = 0        # ID cua token [PAD]

# --- Tham so huan luyen ---
NUM_EPOCHS    = 10      # Tong so epoch
BATCH_SIZE    = 64      # Kich thuoc mini-batch
MAX_LEN       = 64      # Do dai chuan chuoi sau pad_or_trim
LEARNING_RATE = 1e-3    # Toc do hoc ban dau cho Adam
WEIGHT_DECAY  = 1e-5    # L2 Regularization, giam Overfitting
GRAD_CLIP     = 1.0     # Nguong cat gradient (Gradient Clipping)

# --- Duong dan luu mo hinh ---
CHECKPOINT_DIR  = "model_assets"
BEST_MODEL_PATH = "model_assets/best_baseline_model.pt"


# ===========================================================================
# HAM TINH LOSS TREN MOT EPOCH
# ===========================================================================
def run_epoch(
    model:      LSTMBaseline,
    dataloader: torch.utils.data.DataLoader,
    criterion:  nn.CrossEntropyLoss,
    optimizer:  torch.optim.Optimizer,
    device:     torch.device,
    is_training: bool,
) -> float:
    """
    Chay 1 epoch (training hoac validation).

    Teacher Forcing trong training:
        - src : [B, T]      → Encoder
        - tgt : [B, T]      → Decoder nhan tgt[:, :-1]
        - logits: [B, T-1, V]
        - target: tgt[:, 1:]  → [B, T-1]

        Vi sao tgt[:,:-1] va tgt[:,1:]?
            tgt = [BOS, w1, w2, ..., wn, EOS, PAD, ...]
            dau vao Decoder: [BOS, w1, ..., wn]   (bo EOS)
            nhan so sanh   : [w1,  w2, ..., wn, EOS]  (bo BOS)
        Mo hinh hoc du doan token tiep theo dua tren token hien tai.

    Args:
        model        : LSTMBaseline
        dataloader   : DataLoader cung cap batch {"src", "tgt"}
        criterion    : Ham mat mat CrossEntropyLoss
        optimizer    : Optimizer (chi dung khi is_training=True)
        device       : CPU / CUDA
        is_training  : True = cap nhat trong so; False = chi tinh loss

    Returns:
        float – loss trung binh tren toan bo epoch
    """
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    # Tat gradient khi validation de tiet kiem VRAM va tang toc
    context_manager = torch.enable_grad() if is_training else torch.no_grad()

    with context_manager:
        for batch in dataloader:
            # Chuyen batch len device (GPU neu co)
            src = batch["src"].to(device)   # [B, T]  LongTensor
            tgt = batch["tgt"].to(device)   # [B, T]  LongTensor

            # --- Forward Pass ---
            # logits: [B, T-1, V]
            logits = model(src, tgt)

            # --- Chuan bi target de tinh Loss ---
            # tgt[:, 1:] : bo token dau (BOS), lay tu vi tri 1 tro di
            # Shape: [B, T-1]
            target = tgt[:, 1:]              # [B, T-1]

            # CrossEntropyLoss yeu cau:
            #   input  : [N, C] hoac [N, C, d1, ...]
            #   target : [N]    hoac [N, d1, ...]
            # Reshape logits : [B, T-1, V] → [B*T-1, V]  = [N, C]
            # Reshape target : [B, T-1]   → [B*T-1]      = [N]
            B, T_minus1, V = logits.shape
            loss = criterion(
                logits.reshape(B * T_minus1, V),   # [B*(T-1), V]
                target.reshape(B * T_minus1),      # [B*(T-1)]
            )

            if is_training:
                optimizer.zero_grad()   # Xoa gradient cu
                loss.backward()         # Tinh gradient moi

                # Gradient Clipping: cap gradient de tranh bung no
                # Neu chuan L2 cua gradient > GRAD_CLIP thi scale xuong
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()        # Cap nhat trong so

            total_loss   += loss.item()
            total_batches += 1

    average_loss = total_loss / total_batches if total_batches > 0 else float("inf")
    return average_loss


# ===========================================================================
# HAM LUU CHECKPOINT
# ===========================================================================
def save_checkpoint(
    model:      LSTMBaseline,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    val_loss:   float,
    path:       str,
) -> None:
    """
    Luu trang thai huan luyen vao file .pt.

    Noi dung checkpoint:
        - epoch       : Epoch hien tai
        - model_state : Trong so mo hinh
        - optim_state : Trang thai optimizer (de tiep tuc huan luyen)
        - val_loss    : Val loss cua epoch nay
        - model_config: Sieu tham so mo hinh (de tai tao)

    Args:
        model     : Mo hinh can luu.
        optimizer : Optimizer can luu.
        epoch     : So epoch hien tai.
        val_loss  : Val loss tai epoch nay.
        path      : Duong dan file luu (.pt).
    """
    checkpoint = {
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "val_loss":     val_loss,
        "model_config": model.config,   # luu cau hinh de tai tao mo hinh
    }
    torch.save(checkpoint, path)
    logger.info("  -> Da luu checkpoint: %s", path)


# ===========================================================================
# VONG LAP HUAN LUYEN CHINH
# ===========================================================================
def train(args: argparse.Namespace) -> None:
    """
    Ham huan luyen chinh.

    Quy trinh:
        1. Chon device (GPU / CPU)
        2. Chuan bi DataLoader (doc lai neu da co train/val/test.txt)
        3. Khoi tao mo hinh, loss, optimizer
        4. Lap 10 epoch: train → tinh val loss → luu best model
    """
    # -----------------------------------------------------------------
    # Buoc 1: Chon device
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Su dung device: %s", device.type.upper())

    # -----------------------------------------------------------------
    # Buoc 2: Chuan bi du lieu
    # -----------------------------------------------------------------
    # Chia du lieu neu chua co file train/val/test.txt
    already_split = (
        Path(TRAIN_FILE).exists()
        and Path(VAL_FILE).exists()
        and Path(TEST_FILE).exists()
    )
    if not already_split:
        logger.info("Chua co file chia tap – thuc hien split_dataset ...")
        split_dataset(src_file=MULTILINGUAL_FILE, out_dir="data/processed")

    logger.info("Nap DataLoader ...")
    train_loader, val_loader, _ = get_dataloaders(
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        test_file=TEST_FILE,
        tokenizer_path=TOKENIZER_PATH,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    # -----------------------------------------------------------------
    # Buoc 3: Khoi tao mo hinh, ham mat mat, bo toi uu
    # -----------------------------------------------------------------
    logger.info("Khoi tao mo hinh LSTMBaseline ...")
    model = LSTMBaseline(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Tong tham so co the huan luyen: %s", f"{total_params:,}")

    # Ham mat mat: CrossEntropyLoss
    # ignore_index=PAD_IDX: khong tinh loss cho cac vi tri token [PAD]
    # → Tranh mo hinh hoc cach "du doan PAD" thay vi noi dung that
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Bo toi uu Adam:
    # - lr          : toc do hoc
    # - betas       : he so tinh trung binh dong luc bac 1 va bac 2
    # - weight_decay: L2 regularization de giam Overfitting
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------------------
    # Buoc 4: Vong lap epoch
    # -----------------------------------------------------------------
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")   # theo doi val loss tot nhat
    history = []                   # luu lich su de xem lai

    logger.info("=" * 60)
    logger.info("  BAT DAU HUAN LUYEN – %d EPOCH", args.epochs)
    logger.info(
        "  Config: lr=%.0e | batch=%d | max_len=%d | hidden=%d",
        args.lr, args.batch_size, args.max_len, HIDDEN_DIM,
    )
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Training ---
        train_loss = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_training=True,
        )

        # --- Validation ---
        val_loss = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_training=False,
        )

        elapsed = time.time() - t0
        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f | Val Loss: %.4f | %.1fs",
            epoch, args.epochs, train_loss, val_loss, elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        # --- Luu checkpoint moi epoch ---
        ckpt_path = str(Path(CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch:02d}.pt")
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)

        # --- Luu best model neu val loss giam ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, BEST_MODEL_PATH)
            logger.info(
                "  [*] Val Loss giam xuong %.4f – da luu best model!", val_loss
            )

    # -----------------------------------------------------------------
    # Ket thuc: Tom tat
    # -----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  HUAN LUYEN HOAN TAT")
    logger.info("  Best Val Loss : %.4f", best_val_loss)
    logger.info("  Best model    : %s", BEST_MODEL_PATH)
    logger.info("=" * 60)
    logger.info("Lich su Loss:")
    logger.info("  %5s | %10s | %10s", "Epoch", "Train", "Val")
    for h in history:
        logger.info(
            "  %5d | %10.4f | %10.4f", h["epoch"], h["train_loss"], h["val_loss"]
        )


# ===========================================================================
# CLI
# ===========================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huan luyen mo hinh Baseline LSTM Seq2Seq",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs",     type=int,   default=NUM_EPOCHS,
        help="Tong so epoch huan luyen",
    )
    parser.add_argument(
        "--batch_size", type=int,   default=BATCH_SIZE,
        help="Kich thuoc mini-batch",
    )
    parser.add_argument(
        "--max_len",    type=int,   default=MAX_LEN,
        help="Do dai chuan cua chuoi token",
    )
    parser.add_argument(
        "--lr",         type=float, default=LEARNING_RATE,
        help="Toc do hoc (learning rate) cho Adam",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    train(parse_args())
