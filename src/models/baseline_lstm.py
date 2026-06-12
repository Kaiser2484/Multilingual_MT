"""
baseline_lstm.py  –  Mo hinh Baseline: Seq2Seq LSTM khong Attention
====================================================================
Kien truc: Encoder-Decoder thuan LSTM (Sutskever et al., 2014)
Muc dich : Lam moc so sanh (baseline) cho mo hinh Transformer chinh.

Thanh phan:
    EncoderLSTM   – Ma hoa chuoi nguon thanh vector ngu canh
    DecoderLSTM   – Giai ma tung buoc dua tren trang thai Encoder
    LSTMBaseline  – Lop tong the ket noi Encoder va Decoder

Ky hieu Tensor Shape dung trong toan file:
    B   = batch_size        (so cau trong 1 mini-batch)
    T   = seq_len / max_len (do dai chuan cua chuoi, mac dinh 64)
    E   = embed_dim         (chieu vector nhung)
    H   = hidden_dim        (chieu trang thai an cua LSTM)
    V   = vocab_size        (kich thuoc tu dien BPE, mac dinh 32000)
    L   = num_layers        (so tang LSTM chong nhau)

Tham khao bao cao:
    - Mo hinh Baseline: LSTM Seq2Seq, khong co co che Attention
    - So sanh voi mo hinh chinh: Transformer da ngon ngu
"""

import torch
import torch.nn as nn


# ===========================================================================
# 1. ENCODER LSTM
# ===========================================================================
class EncoderLSTM(nn.Module):
    """
    Ma hoa chuoi nguon thanh mot cap trang thai an (hidden, cell).

    Quy trinh:
        Token ID [B, T]
            |
            v  nn.Embedding
        Embedding [B, T, E]
            |
            v  nn.LSTM (batch_first=True, bidirectional=False)
        outputs   [B, T, H]    – output tai moi buoc thoi gian
        hidden    [L, B, H]    – trang thai an cuoi cung cua tat ca cac tang
        cell      [L, B, H]    – trang thai o cuoi cung  cua tat ca cac tang
            |
            v  (tra ve hidden, cell lam gia tri khoi tao cho Decoder)

    Tai sao dung hidden va cell?
        LSTM dung 2 vector trang thai:
          - hidden (h): tom tat thong tin ngan han
          - cell   (c): luu thong tin dai han qua cac buoc thoi gian
        Chuyen ca hai sang Decoder de giu nguyen bo nho.

    Args:
        vocab_size (int): Kich thuoc tu dien BPE.
        embed_dim  (int): Chieu cua vector nhung (Embedding dimension).
        hidden_dim (int): So unit trong moi tang LSTM.
        num_layers (int): So tang LSTM chong nhau (stacked LSTM).
        dropout    (float): Ti le Dropout giua cac tang LSTM (chi ap khi L > 1).
        pad_idx    (int): ID cua token [PAD]; gradient cua no se bang 0.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        num_layers: int,
        dropout:    float,
        pad_idx:    int = 0,
    ):
        super().__init__()

        # Tang nhung tu: ID → vector day du ngha
        # Tham so bieu dien: [vocab_size, embed_dim]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,   # vector PAD luon la 0, khong cap nhat
        )

        # Mang LSTM:
        #   input_size  = embed_dim  (chieu vector nhung dau vao)
        #   hidden_size = hidden_dim (chieu trang thai an)
        #   num_layers  = L (chong nhau, ket qua tang truoc la dau vao tang sau)
        #   batch_first = True → chieu thu nhat la B, khong phai T
        #   dropout     = ap giua cac tang (khong ap o tang cuoi)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        src: torch.Tensor,   # [B, T]  – ID chuoi nguon
    ) -> tuple:
        """
        Args:
            src : LongTensor [B, T]

        Returns:
            hidden : FloatTensor [L, B, H]  – trang thai an cuoi cung
            cell   : FloatTensor [L, B, H]  – trang thai o cuoi cung
        """
        # src : [B, T]
        embedded = self.embedding(src)    # → [B, T, E]

        # LSTM xu ly toan bo chuoi mot lan (parallel tren T)
        # outputs: [B, T, H]  – khong dung truc tiep trong Seq2Seq khong Attention
        # hidden : [L, B, H]
        # cell   : [L, B, H]
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs chua output tai moi buoc thoi gian (khong dung o day)

        # Chi tra ve trang thai an cuoi cung lam gia tri khoi tao Decoder
        return hidden, cell   # [L, B, H], [L, B, H]


# ===========================================================================
# 2. DECODER LSTM
# ===========================================================================
class DecoderLSTM(nn.Module):
    """
    Giai ma tung buoc thoi gian de sinh chuoi dich.

    Nhan trang thai khoi tao tu Encoder (hidden, cell) thay vi bat dau
    tu trang thai zero → biet ngon ngu dich nho vao Target Token <2vi>, <2ja>.

    Quy trinh (Teacher Forcing – dung trong training):
        Token ID [B, T-1]           (chuoi dich bo token cuoi)
            |
            v  nn.Embedding
        Embedding [B, T-1, E]
            |
            v  nn.LSTM (bat dau tu hidden/cell cua Encoder)
        outputs [B, T-1, H]
            |
            v  nn.Linear(H → V)
        logits  [B, T-1, V]        (phan phoi xac suat tren tu dien)

    Args:
        vocab_size (int): Kich thuoc tu dien BPE.
        embed_dim  (int): Chieu vector nhung (nen bang Encoder.embed_dim).
        hidden_dim (int): Chieu trang thai an (bat buoc bang Encoder.hidden_dim).
        num_layers (int): So tang LSTM (bat buoc bang Encoder.num_layers).
        dropout    (float): Ti le Dropout.
        pad_idx    (int): ID cua token [PAD].
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        hidden_dim: int,
        num_layers: int,
        dropout:    float,
        pad_idx:    int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Tang chieu tuyen tinh: H → V
        # Chuyen trang thai an thanh phan phoi xac suat tren tu dien
        # Khong dung Softmax o day vi CrossEntropyLoss tinh tich hop
        self.output_projection = nn.Linear(
            in_features=hidden_dim,
            out_features=vocab_size,
        )

    def forward(
        self,
        tgt:    torch.Tensor,  # [B, T-1]  – ID chuoi dich (bo token cuoi)
        hidden: torch.Tensor,  # [L, B, H] – tu Encoder
        cell:   torch.Tensor,  # [L, B, H] – tu Encoder
    ) -> torch.Tensor:
        """
        Args:
            tgt    : LongTensor [B, T-1]   – chuoi dich vao (teacher forcing)
            hidden : FloatTensor [L, B, H]
            cell   : FloatTensor [L, B, H]

        Returns:
            logits : FloatTensor [B, T-1, V]  – chua ap Softmax
        """
        # tgt : [B, T-1]
        embedded = self.embedding(tgt)            # → [B, T-1, E]

        # LSTM nhan trang thai khoi tao tu Encoder
        # outputs: [B, T-1, H]  – output tai moi buoc thoi gian
        outputs, _ = self.lstm(embedded, (hidden, cell))
        # _ la (hidden_n, cell_n) cuoi – khong can trong training

        # Chieu output → phan phoi tu dien
        logits = self.output_projection(outputs)  # → [B, T-1, V]

        return logits   # [B, T-1, V]


# ===========================================================================
# 3. MO HINH TONG THE: LSTMBaseline
# ===========================================================================
class LSTMBaseline(nn.Module):
    """
    Mo hinh Baseline Seq2Seq LSTM khong Attention.

    Ket noi EncoderLSTM va DecoderLSTM:
        src [B, T]  →  Encoder  →  (hidden, cell) [L, B, H]
                                          |
        tgt [B, T]  →  Decoder(tgt[:,:-1], hidden, cell)  →  logits [B, T-1, V]

    Chien luoc Teacher Forcing (trong training):
        - Dua token dich DUNG vao Decoder tai moi buoc (tgt[:, 0..T-2])
        - So sanh logit dau ra voi token TIEP THEO thuc su (tgt[:, 1..T-1])
        - Giup mo hinh hoi tu nhanh hon trong giai doan dau

    Cau hinh thi nghiem (ghi vao bao cao):
        ┌──────────────────────┬──────────┐
        │ Tham so              │ Gia tri  │
        ├──────────────────────┼──────────┤
        │ vocab_size           │ 32 000   │
        │ embed_dim            │ 256      │
        │ hidden_dim           │ 512      │
        │ num_layers           │ 2        │
        │ dropout              │ 0.3      │
        │ Kien truc            │ LSTM Seq2Seq (no Attention) │
        └──────────────────────┴──────────┘

    Args:
        vocab_size (int): Kich thuoc tu dien BPE.
        embed_dim  (int): Chieu vector nhung.
        hidden_dim (int): Chieu trang thai an LSTM.
        num_layers (int): So tang LSTM chong nhau.
        dropout    (float): Ti le Dropout.
        pad_idx    (int): ID token PAD.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int  = 256,
        hidden_dim: int  = 512,
        num_layers: int  = 2,
        dropout:    float = 0.3,
        pad_idx:    int  = 0,
    ):
        super().__init__()

        self.encoder = EncoderLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.decoder = DecoderLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # Luu cau hinh de tien ghi vao checkpoint
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim":  embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout":    dropout,
            "pad_idx":    pad_idx,
        }

    def forward(
        self,
        src: torch.Tensor,   # [B, T]  – chuoi nguon
        tgt: torch.Tensor,   # [B, T]  – chuoi dich (bao gom BOS, khong EOS)
    ) -> torch.Tensor:
        """
        Thuc hien Teacher Forcing mot lan (forward toan bo batch).

        Buoc 1: Encoder xu ly src → (hidden, cell)
        Buoc 2: Decoder nhan tgt[:, :-1] → logits [B, T-1, V]

        Args:
            src : LongTensor [B, T]
            tgt : LongTensor [B, T]

        Returns:
            logits : FloatTensor [B, T-1, V]
                     So sanh voi tgt[:, 1:] bang CrossEntropyLoss
        """
        # Buoc 1: Ma hoa
        hidden, cell = self.encoder(src)         # [L,B,H], [L,B,H]

        # Buoc 2: Giai ma voi Teacher Forcing
        # tgt[:, :-1] : bo token cuoi (EOS hoac PAD cuoi)
        # → Decoder nhan token 0..T-2 va du doan token 1..T-1
        logits = self.decoder(tgt[:, :-1], hidden, cell)  # [B, T-1, V]

        return logits   # [B, T-1, V]


# ===========================================================================
# Self-test
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    torch.manual_seed(42)
    B, T, V = 4, 64, 32_000

    model = LSTMBaseline(vocab_size=V)
    total = sum(p.numel() for p in model.parameters())
    print(f"Tong tham so mo hinh : {total:,}")

    src = torch.randint(1, V, (B, T))
    tgt = torch.randint(1, V, (B, T))
    logits = model(src, tgt)

    print(f"src shape   : {tuple(src.shape)}")      # (4, 64)
    print(f"tgt shape   : {tuple(tgt.shape)}")      # (4, 64)
    print(f"logits shape: {tuple(logits.shape)}")   # (4, 63, 32000)
    print("Self-test PASSED!")
