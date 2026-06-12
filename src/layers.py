"""
layers.py  –  Tuần 2 | Thành viên 1
=====================================
Các lớp nền tảng đầu vào và Feed-Forward của Transformer đa ngôn ngữ.

Lớp được triển khai:
    1. TransformerEmbedding      – Token ID → vector nhúng đã chuẩn hóa
    2. PositionalEncoding        – Cộng thêm thông tin vị trí sin/cos
    3. PositionWiseFeedForward   – FFN(x) = max(0, xW1+b1)W2+b2

Quy ước ký hiệu tensor dùng trong toàn file:
    B   = batch_size   (số câu trong một mini-batch)
    T   = seq_len      (độ dài chuỗi, số token)
    D   = d_model      (chiều không gian embedding)
    Dff = d_ff         (chiều ẩn của Feed-Forward, thường = 4*D)
"""

import math
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# 1. TransformerEmbedding
# ══════════════════════════════════════════════════════════════════════════════
class TransformerEmbedding(nn.Module):
    """
    Chuyển đổi chuỗi Token ID thành ma trận vector nhúng và nhân với
    sqrt(d_model) để cân bằng biên độ trước khi cộng Positional Encoding.

    Lý do nhân sqrt(d_model):
        - nn.Embedding khởi tạo trọng số ~ N(0,1)
        - PositionalEncoding có biên độ trong khoảng [-1, 1]
        - Nếu không nhân hệ số, PE sẽ "át" thông tin embedding khi D lớn
        - Nhân sqrt(D) giúp hai nguồn thông tin đóng góp cân bằng nhau

    Args:
        vocab_size (int): Số lượng token trong từ điển (ví dụ: 32 000).
        d_model    (int): Chiều không gian embedding (ví dụ: 256).
        pad_idx    (int): ID của token [PAD]; gradient của nó sẽ bằng 0.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0):
        super().__init__()

        # Bảng tra cứu embedding: shape tham số = [vocab_size, d_model]
        # padding_idx=pad_idx → vector của [PAD] luôn là 0 và không được cập nhật
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        # Lưu hệ số nhân để dùng trong forward
        self.scale = math.sqrt(d_model)   # scalar: √D

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids : LongTensor [B, T]   – chuỗi các Token ID

        Returns:
            Tensor    [B, T, D]             – vector nhúng đã nhân √D
        """
        # token_ids : [B, T]
        # ↓ tra cứu bảng embedding
        x = self.embedding(token_ids)      # → [B, T, D]

        # Nhân √D để cân bằng biên độ với PositionalEncoding
        x = x * self.scale                 # → [B, T, D]  (scalar broadcast)

        return x                           # [B, T, D]


# ══════════════════════════════════════════════════════════════════════════════
# 2. PositionalEncoding
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """
    Cộng thông tin vị trí vào ma trận Embedding bằng hàm sin/cos cố định.

    Công thức (Vaswani et al., 2017 – "Attention Is All You Need"):
        PE(pos, 2i)   = sin( pos / 10000^(2i/D) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i/D) )

    Trong đó:
        pos ∈ [0, max_len)  – chỉ số vị trí của token
        i   ∈ [0, D/2)      – chỉ số chiều (đôi)
        D                   – d_model

    Ma trận PE được tính một lần và lưu vào buffer (không phải tham số học),
    tức là không được cập nhật bởi optimizer.

    Args:
        d_model  (int):   Chiều embedding.
        dropout  (float): Tỉ lệ dropout áp sau khi cộng PE.
        max_len  (int):   Độ dài chuỗi tối đa được hỗ trợ (mặc định 5 000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # ── Bước 1: Tạo tensor vị trí pos ──────────────────────────────────
        # pos : [max_len, 1]  (cột vector, để broadcast với chiều D)
        pos = torch.arange(0, max_len).unsqueeze(1)     # [max_len, 1]

        # ── Bước 2: Tạo vector mẫu số (denominator) ─────────────────────────
        # i chạy qua các chỉ số chẵn: 0, 2, 4, ..., D-2
        # div_term[i] = 1 / 10000^(2i/D) = exp(-2i * ln(10000) / D)
        # Trick dùng exp-log để ổn định số học, tránh tràn số khi D lớn.
        i = torch.arange(0, d_model, step=2)            # [D/2]
        div_term = torch.exp(
            i * (-math.log(10_000.0) / d_model)
        )                                               # [D/2]

        # ── Bước 3: Tính ma trận PE ─────────────────────────────────────────
        # pe : [max_len, D]  – khởi tạo bằng 0
        pe = torch.zeros(max_len, d_model)              # [max_len, D]

        # Chiều chẵn (0, 2, 4, ...) ← sin
        # pos * div_term : [max_len, 1] * [D/2] → [max_len, D/2]
        pe[:, 0::2] = torch.sin(pos * div_term)         # [max_len, D/2]

        # Chiều lẻ (1, 3, 5, ...) ← cos
        pe[:, 1::2] = torch.cos(pos * div_term)         # [max_len, D/2]

        # ── Bước 4: Thêm chiều batch để broadcast ───────────────────────────
        # pe : [max_len, D] → [1, max_len, D]
        pe = pe.unsqueeze(0)                            # [1, max_len, D]

        # Đăng ký pe là buffer (không tham gia training, nhưng tự động
        # chuyển device cùng model khi gọi .to(device))
        self.register_buffer("pe", pe)                  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor [B, T, D]  – embedding đầu vào (đã nhân √D)

        Returns:
              Tensor [B, T, D]  – embedding + positional encoding, sau dropout
        """
        # x             : [B, T, D]
        # self.pe       : [1, max_len, D]
        # self.pe[:, :T]: [1, T, D]  – cắt đúng độ dài thực tế của chuỗi

        T = x.size(1)                                   # lấy seq_len thực tế

        # Cộng PE vào embedding; broadcast [1, T, D] → [B, T, D]
        x = x + self.pe[:, :T, :]                       # [B, T, D]

        # Dropout để tránh overfitting trong quá trình học
        x = self.dropout(x)                             # [B, T, D]

        return x                                        # [B, T, D]


# ══════════════════════════════════════════════════════════════════════════════
# 3. PositionWiseFeedForward
# ══════════════════════════════════════════════════════════════════════════════
class PositionWiseFeedForward(nn.Module):
    """
    Tầng Feed-Forward vị trí-độc-lập (Position-wise FFN).

    Mỗi vị trí token được xử lý độc lập bằng cùng một mạng 2 lớp tuyến tính:
        FFN(x) = max(0, x @ W1 + b1) @ W2 + b2

    Trong đó:
        W1 : [D, Dff]   – chiếu lên không gian ẩn rộng hơn
        W2 : [Dff, D]   – chiếu về lại không gian D
        Dff = d_ff thường = 4 * d_model  (theo paper gốc: 512 → 2048)

    "Position-wise" nghĩa là:
        - Cùng một W1, W2 áp dụng cho MỌI vị trí trong chuỗi
        - Các vị trí KHÔNG tương tác với nhau ở tầng này
        - Tương tác xảy ra ở tầng Attention, không phải ở FFN

    Args:
        d_model  (int):   Chiều embedding đầu vào / đầu ra.
        d_ff     (int):   Chiều không gian ẩn bên trong (thường 4*d_model).
        dropout  (float): Tỉ lệ dropout áp sau ReLU.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Lớp tuyến tính 1: D → Dff   (mở rộng không gian đặc trưng)
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ff)
        # Trọng số W1 : [D, Dff], bias b1 : [Dff]

        # Hàm kích hoạt ReLU: max(0, ·)
        self.relu = nn.ReLU()

        # Dropout giữa hai lớp (áp sau kích hoạt, trước W2)
        self.dropout = nn.Dropout(p=dropout)

        # Lớp tuyến tính 2: Dff → D  (thu hẹp về chiều gốc)
        self.linear2 = nn.Linear(in_features=d_ff, out_features=d_model)
        # Trọng số W2 : [Dff, D], bias b2 : [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor [B, T, D]  – đầu ra từ tầng Attention (hoặc Add&Norm)

        Returns:
              Tensor [B, T, D]  – đã biến đổi qua FFN
        """
        # ── Lớp 1: chiếu lên không gian ẩn Dff ────────────────────────────
        # x                : [B, T, D]
        # linear1 áp dụng matmul theo chiều cuối: D → Dff
        x = self.linear1(x)         # → [B, T, Dff]

        # ── Kích hoạt ReLU: max(0, ·) ─────────────────────────────────────
        x = self.relu(x)            # → [B, T, Dff]  (âm → 0, dương giữ nguyên)

        # ── Dropout (chỉ bật trong training, tự tắt khi .eval()) ──────────
        x = self.dropout(x)         # → [B, T, Dff]

        # ── Lớp 2: chiếu về lại không gian D ──────────────────────────────
        x = self.linear2(x)         # → [B, T, D]

        return x                    # [B, T, D]


# ══════════════════════════════════════════════════════════════════════════════
# Kiểm tra nhanh (chạy: python src/layers.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)

    # ── Siêu tham số thử nghiệm ───────────────────────────────────────────
    BATCH   = 2        # B
    SEQ_LEN = 10       # T
    D_MODEL = 256      # D
    D_FF    = 1024     # Dff = 4 * D_MODEL
    VOCAB   = 32_000
    PAD_IDX = 0

    print("=" * 60)
    print("Kiểm tra TransformerEmbedding")
    print("=" * 60)
    # Tạo batch Token ID giả: [B, T]
    token_ids = torch.randint(low=1, high=VOCAB, size=(BATCH, SEQ_LEN))
    token_ids[0, -2:] = PAD_IDX          # giả lập token [PAD] ở cuối câu 1
    print(f"  Đầu vào  token_ids : {tuple(token_ids.shape)}")

    emb_layer = TransformerEmbedding(vocab_size=VOCAB, d_model=D_MODEL, pad_idx=PAD_IDX)
    emb_out   = emb_layer(token_ids)
    print(f"  Đầu ra   embedding  : {tuple(emb_out.shape)}")   # (2, 10, 256)
    print(f"  Token PAD → all-zero: {emb_out[0, -1].abs().sum().item() == 0.0}")

    print()
    print("=" * 60)
    print("Kiểm tra PositionalEncoding")
    print("=" * 60)
    pe_layer = PositionalEncoding(d_model=D_MODEL, dropout=0.0)  # dropout=0 để test
    pe_out   = pe_layer(emb_out)
    print(f"  Đầu vào  embedding  : {tuple(emb_out.shape)}")
    print(f"  Đầu ra   emb + PE   : {tuple(pe_out.shape)}")     # (2, 10, 256)

    # Giá trị PE tại vị trí 0, chiều 0 phải là sin(0) = 0
    raw_pe = pe_layer.pe[0]   # [max_len, D]
    print(f"  PE[pos=0, dim=0]  = sin(0)  = {raw_pe[0, 0].item():.4f}  (kỳ vọng: 0.0)")
    # Giá trị PE tại vị trí 0, chiều 1 phải là cos(0) = 1
    print(f"  PE[pos=0, dim=1]  = cos(0)  = {raw_pe[0, 1].item():.4f}  (kỳ vọng: 1.0)")

    print()
    print("=" * 60)
    print("Kiểm tra PositionWiseFeedForward")
    print("=" * 60)
    ffn_layer = PositionWiseFeedForward(d_model=D_MODEL, d_ff=D_FF, dropout=0.1)
    ffn_out   = ffn_layer(pe_out)
    print(f"  Đầu vào  x         : {tuple(pe_out.shape)}")
    print(f"  Đầu ra   FFN(x)    : {tuple(ffn_out.shape)}")     # (2, 10, 256)

    # Kiểm tra số tham số của FFN
    total_params = sum(p.numel() for p in ffn_layer.parameters())
    print(f"  Tổng tham số FFN   : {total_params:,}")
    # W1: D*Dff + Dff = 256*1024+1024 = 263168
    # W2: Dff*D + D   = 1024*256+256  = 262400
    # Tổng = 525568

    print()
    print("✅ Tất cả kiểm tra đều pass – shape đầu ra đúng [B, T, D]!")
