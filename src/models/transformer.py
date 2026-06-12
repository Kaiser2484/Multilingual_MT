"""
transformer.py  –  Transformer tu dinh nghia tu so 0
=====================================================
Ky hieu Tensor Shape trong toan file:
    B   = batch_size
    T_s = seq_len nguon (source)
    T_t = seq_len dich  (target)
    T   = seq_len bat ky
    D   = d_model
    H   = num_heads
    Dk  = d_model // num_heads  (chieu moi head)
    V   = vocab_size
    Dff = d_ff  (chieu FFN an)

Cau hinh thi nghiem:
    vocab_size=32000 | d_model=256 | num_heads=8
    num_layers=3     | d_ff=1024   | dropout=0.1
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. POSITIONAL ENCODING
# ===========================================================================
class PositionalEncoding(nn.Module):
    """
    Them thong tin vi tri vao Embedding bang cong thuc sin/cos co dinh.

    Cong thuc (Vaswani et al., 2017):
        PE[pos, 2i]   = sin(pos / 10000^(2i/D))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/D))

    Ma tran PE duoc dang ky la buffer (khong huan luyen, khong thay doi).

    Input  : x  [B, T, D]
    Output : x + PE[:, :T, :]  [B, T, D]  (cung kich thuoc voi dau vao)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # pe : [max_len, D]
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()          # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                              # [D/2]
        pe[:, 0::2] = torch.sin(pos * div)   # vi tri chan
        pe[:, 1::2] = torch.cos(pos * div)   # vi tri le

        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : [B, T, D]
        # pe  : [1, T, D]  (cat lai cho vua T)
        x = x + self.pe[:, : x.size(1)]   # [B, T, D]
        return self.dropout(x)             # [B, T, D]


# ===========================================================================
# 2. MULTI-HEAD ATTENTION
# ===========================================================================
class MultiHeadAttention(nn.Module):
    """
    Co che Scaled Dot-Product Attention da dau.

    Cong thuc:
        Attention(Q,K,V) = Softmax( Q @ K^T / sqrt(Dk) ) @ V

    Moi head xu ly khong gian con [B, T, Dk] doc lap.
    Sau do ghep lai (concat) va chieu qua W_o.

    Args:
        d_model   : Chieu embedding toan cuc D
        num_heads : So dau attention H (D phai chia het cho H)
        dropout   : Dropout ap len attention weight

    Input (forward):
        query  : [B, T_q, D]
        key    : [B, T_k, D]
        value  : [B, T_k, D]
        mask   : [B, 1, 1, T_k]  hoac  [B, 1, T_q, T_k]  (True = che, dien -inf)

    Output:
        out    : [B, T_q, D]
        weights: [B, H, T_q, T_k]   (attention weight – dung de visualize)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phai chia het cho num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # Dk = D / H

        # 4 ma tran chieu tuyen tinh (khong dung bias theo paper goc)
        self.W_q = nn.Linear(d_model, d_model, bias=False)   # [D, D]
        self.W_k = nn.Linear(d_model, d_model, bias=False)   # [D, D]
        self.W_v = nn.Linear(d_model, d_model, bias=False)   # [D, D]
        self.W_o = nn.Linear(d_model, d_model, bias=False)   # [D, D]

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tach D thanh H x Dk.
        Input : [B, T, D]
        Output: [B, H, T, Dk]
        """
        B, T, D = x.shape
        # view: [B, T, H, Dk]  →  transpose: [B, H, T, Dk]
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _scaled_dot_product(
        self,
        Q: torch.Tensor,            # [B, H, T_q, Dk]
        K: torch.Tensor,            # [B, H, T_k, Dk]
        V: torch.Tensor,            # [B, H, T_k, Dk]
        mask: torch.Tensor = None,  # [B, 1, T_q, T_k]  hoac broadcast tuong duong
    ):
        """
        Attention(Q,K,V) = Softmax(Q @ K^T / sqrt(Dk)) @ V

        Returns:
            context : [B, H, T_q, Dk]
            weights : [B, H, T_q, T_k]
        """
        # scores : [B, H, T_q, T_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Tai nhung vi tri mask=True, dat -inf de Softmax → 0
            scores = scores.masked_fill(mask, float("-1e9"))

        # weights : [B, H, T_q, T_k]  (tong theo T_k = 1.0)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # context : [B, H, T_q, Dk]
        context = torch.matmul(weights, V)
        return context, weights

    def forward(
        self,
        query: torch.Tensor,          # [B, T_q, D]
        key:   torch.Tensor,          # [B, T_k, D]
        value: torch.Tensor,          # [B, T_k, D]
        mask:  torch.Tensor = None,
    ):
        B = query.size(0)

        # Chieu tuyen tinh roi tach thanh H head
        Q = self._split_heads(self.W_q(query))  # [B, H, T_q, Dk]
        K = self._split_heads(self.W_k(key))    # [B, H, T_k, Dk]
        V = self._split_heads(self.W_v(value))  # [B, H, T_k, Dk]

        # Tinh attention
        context, weights = self._scaled_dot_product(Q, K, V, mask)
        # context : [B, H, T_q, Dk]
        # weights : [B, H, T_q, T_k]

        # Ghep cac head lai (Concatenate)
        # [B, H, T_q, Dk] → transpose → [B, T_q, H, Dk] → view → [B, T_q, D]
        context = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # Chieu output cuoi
        out = self.W_o(context)   # [B, T_q, D]
        return out, weights


# ===========================================================================
# 3. POSITION-WISE FEED FORWARD
# ===========================================================================
class PositionWiseFeedForward(nn.Module):
    """
    Ap dung doc lap tai moi vi tri trong chuoi.

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Input : [B, T, D]
    Output: [B, T, D]   (giu nguyen shape)

    Luu y: D_ff thuong = 4 * D (mo rong truoc, thu hep lai)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # [D, Dff]
        self.linear2 = nn.Linear(d_ff, d_model)    # [Dff, D]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x     : [B, T, D]
        x = self.linear1(x)        # [B, T, Dff]
        x = F.relu(x)              # [B, T, Dff]
        x = self.dropout(x)
        x = self.linear2(x)        # [B, T, D]
        return x                   # [B, T, D]


# ===========================================================================
# 4. ENCODER LAYER
# ===========================================================================
class EncoderLayer(nn.Module):
    """
    1 tang Encoder = Self-Attention + FFN + Residual + LayerNorm.

    Luong xu ly:
        x → [Self-Attn] → [Add & Norm] → [FFN] → [Add & Norm] → out

    Input : [B, T_s, D]
    Output: [B, T_s, D]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,         # [B, T_s, D]
        src_mask: torch.Tensor = None,  # [B, 1, 1, T_s]
    ) -> torch.Tensor:
        # --- Self-Attention + Residual + Norm ---
        attn_out, _ = self.self_attn(x, x, x, mask=src_mask)  # [B, T_s, D]
        x = self.norm1(x + self.dropout(attn_out))             # [B, T_s, D]

        # --- FFN + Residual + Norm ---
        ffn_out = self.ffn(x)                                  # [B, T_s, D]
        x = self.norm2(x + self.dropout(ffn_out))              # [B, T_s, D]

        return x   # [B, T_s, D]


# ===========================================================================
# 5. DECODER LAYER
# ===========================================================================
class DecoderLayer(nn.Module):
    """
    1 tang Decoder = Masked Self-Attn + Cross-Attn + FFN + Residual + Norm.

    Luong xu ly:
        x → [Masked Self-Attn] → [Add&Norm]
          → [Cross-Attn(Q=x, K=V=enc_out)] → [Add&Norm]
          → [FFN] → [Add&Norm] → out

    Input : x       [B, T_t, D]
            enc_out [B, T_s, D]
    Output: [B, T_t, D]

    2 loai mask:
        tgt_mask : causal + padding mask cho chuoi dich  [B,1,T_t,T_t]
        src_mask : padding mask cho chuoi nguon          [B,1,1,T_s]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn       = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn              = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.norm3  = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,          # [B, T_t, D]
        enc_out:  torch.Tensor,          # [B, T_s, D]
        tgt_mask: torch.Tensor = None,   # [B, 1, T_t, T_t]
        src_mask: torch.Tensor = None,   # [B, 1, 1,  T_s]
    ) -> torch.Tensor:

        # --- Masked Self-Attention (che tuong lai) ---
        sa_out, _ = self.masked_self_attn(x, x, x, mask=tgt_mask)  # [B, T_t, D]
        x = self.norm1(x + self.dropout(sa_out))                    # [B, T_t, D]

        # --- Cross-Attention (Q tu Decoder, K/V tu Encoder) ---
        ca_out, _ = self.cross_attn(x, enc_out, enc_out, mask=src_mask)  # [B, T_t, D]
        x = self.norm2(x + self.dropout(ca_out))                         # [B, T_t, D]

        # --- FFN ---
        x = self.norm3(x + self.dropout(self.ffn(x)))   # [B, T_t, D]

        return x   # [B, T_t, D]


# ===========================================================================
# 6. HAM TAO MASK
# ===========================================================================
def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tao padding mask: True tai cac vi tri la token [PAD].
    Input : [B, T]
    Output: [B, 1, 1, T]  (broadcastable cho [B, H, T_q, T_k])
    """
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)   # [B, 1, 1, T]


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Tao causal mask (tam giac tren = True, che cac vi tri tuong lai).
    Output: [1, 1, T, T]  (broadcastable)
    """
    # torch.triu lay tam giac tren (tru duong cheo chinh)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask.unsqueeze(0).unsqueeze(0)   # [1, 1, T, T]


def make_decoder_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Gop padding mask va causal mask cho Decoder.
    Cong thuc: tgt_mask = padding_mask OR causal_mask

    Input : [B, T_t]
    Output: [B, 1, T_t, T_t]
    """
    pad_m   = make_padding_mask(tgt, pad_idx)              # [B, 1, 1,  T_t]
    causal  = make_causal_mask(tgt.size(1), tgt.device)    # [1, 1, T_t, T_t]
    return pad_m | causal                                  # [B, 1, T_t, T_t]


# ===========================================================================
# 7. MO HINH TONG THE: Transformer
# ===========================================================================
class Transformer(nn.Module):
    """
    Transformer Encoder-Decoder day du tu dinh nghia tu so 0.

    Luong du lieu (Teacher Forcing trong training):
        src [B, T_s]  ─►  Encoder Stack  ─►  enc_out [B, T_s, D]
                                                     │
        tgt [B, T_t]  ─►  Decoder Stack (tgt[:,:-1], enc_out)
                       ─►  dec_out [B, T_t-1, D]
                       ─►  Linear(D → V)
                       ─►  logits  [B, T_t-1, V]

    So sanh logits voi tgt[:,1:] bang CrossEntropyLoss (Teacher Forcing):
        input  Decoder : tgt[:,  :-1]  = [BOS, w1, w2, ..., wn]
        target (label) : tgt[:, 1:  ]  = [w1,  w2, ..., wn, EOS]

    Cau hinh thi nghiem (ghi vao bao cao):
    ┌──────────────────────┬──────────┐
    │ Tham so              │ Gia tri  │
    ├──────────────────────┼──────────┤
    │ vocab_size           │ 32 000   │
    │ d_model   (D)        │ 256      │
    │ num_heads (H)        │ 8        │
    │ num_layers(N)        │ 3        │
    │ d_ff      (Dff)      │ 1 024    │
    │ dropout              │ 0.1      │
    │ max_len              │ 64       │
    └──────────────────────┴──────────┘
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 256,
        num_heads:   int   = 8,
        num_layers:  int   = 3,
        d_ff:        int   = 1024,
        dropout:     float = 0.1,
        pad_idx:     int   = 0,
        max_len:     int   = 512,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # --- Embedding + Positional Encoding (dung chung 1 embedding) ---
        # Nhan voi sqrt(D) de can bang bien do voi PE (theo paper goc)
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe = PositionalEncoding(d_model, dropout, max_len)
        self.tgt_pe = PositionalEncoding(d_model, dropout, max_len)

        # --- Encoder Stack: N tang EncoderLayer ---
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # --- Decoder Stack: N tang DecoderLayer ---
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # --- Tang chieu output: D → V ---
        # Weight Tying: dung lai trong so embedding cho output projection
        # (giam tham so, tang on dinh – Inan et al., 2017)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.src_embedding.weight

        # --- Luu cau hinh de ghi vao checkpoint ---
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model,
            num_heads=num_heads,   num_layers=num_layers,
            d_ff=d_ff,             dropout=dropout,
            pad_idx=pad_idx,       max_len=max_len,
        )

        # --- Khoi tao trong so Xavier Uniform ---
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -----------------------------------------------------------------------
    def _encode(
        self,
        src: torch.Tensor,          # [B, T_s]
    ):
        """
        Ma hoa chuoi nguon thanh bieu dien ngu canh.

        Returns:
            enc_out  : [B, T_s, D]
            src_mask : [B, 1, 1, T_s]
        """
        # Tao padding mask cho chuoi nguon
        src_mask = make_padding_mask(src, self.pad_idx)   # [B, 1, 1, T_s]

        # Embedding + scale + PE
        x = self.src_embedding(src) * math.sqrt(self.d_model)  # [B, T_s, D]
        x = self.src_pe(x)                                      # [B, T_s, D]

        # Qua N tang EncoderLayer
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)   # [B, T_s, D]

        enc_out = self.encoder_norm(x)         # [B, T_s, D]
        return enc_out, src_mask

    # -----------------------------------------------------------------------
    def _decode(
        self,
        tgt_in:   torch.Tensor,   # [B, T_t-1]
        enc_out:  torch.Tensor,   # [B, T_s, D]
        src_mask: torch.Tensor,   # [B, 1, 1, T_s]
    ) -> torch.Tensor:
        """
        Giai ma ket hop voi bieu dien Encoder.

        Returns:
            dec_out : [B, T_t-1, D]
        """
        # Tao causal + padding mask cho chuoi dich
        tgt_mask = make_decoder_mask(tgt_in, self.pad_idx)   # [B, 1, T_t-1, T_t-1]

        # Embedding + scale + PE
        x = self.tgt_embedding(tgt_in) * math.sqrt(self.d_model)  # [B, T_t-1, D]
        x = self.tgt_pe(x)                                         # [B, T_t-1, D]

        # Qua N tang DecoderLayer
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, src_mask=src_mask)

        dec_out = self.decoder_norm(x)   # [B, T_t-1, D]
        return dec_out

    # -----------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,   # [B, T_s]  – ID chuoi nguon
        tgt: torch.Tensor,   # [B, T_t]  – ID chuoi dich (co BOS o dau)
    ) -> torch.Tensor:
        """
        Teacher Forcing forward pass.

        Buoc 1: Encoder xu ly src → enc_out [B, T_s, D]
        Buoc 2: Decoder xu ly tgt[:,:-1] + enc_out → dec_out [B, T_t-1, D]
        Buoc 3: Linear(D → V) → logits [B, T_t-1, V]

        So sanh logits voi tgt[:,1:] bang CrossEntropyLoss.

        Returns:
            logits : FloatTensor [B, T_t-1, V]
        """
        enc_out, src_mask = self._encode(src)               # [B,T_s,D], [B,1,1,T_s]
        dec_out = self._decode(tgt[:, :-1], enc_out, src_mask)  # [B, T_t-1, D]
        logits  = self.output_projection(dec_out)           # [B, T_t-1, V]
        return logits

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def translate_greedy(
        self,
        src:     torch.Tensor,   # [1, T_s]
        bos_id:  int,
        eos_id:  int,
        max_len: int = 64,
    ) -> list:
        """
        Greedy decoding: chon token co xac suat cao nhat tai moi buoc.
        Chi dung khi inference (khong phai training).

        Returns: List[int] – chuoi token ID da dich (khong co BOS)
        """
        self.eval()
        enc_out, src_mask = self._encode(src)   # [1, T_s, D]
        tgt_ids = [bos_id]

        for _ in range(max_len):
            tgt_tensor = torch.tensor(
                [tgt_ids], dtype=torch.long, device=src.device
            )
            dec_out    = self._decode(tgt_tensor, enc_out, src_mask)  # [1, len, D]
            next_token = self.output_projection(dec_out[:, -1, :]).argmax(-1).item()
            tgt_ids.append(next_token)
            if next_token == eos_id:
                break

        return tgt_ids[1:]   # bo [BOS], giu nguyen [EOS] neu co


# ===========================================================================
# SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(42)

    B, T, V = 2, 64, 32_000
    model = Transformer(vocab_size=V)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tong tham so: {total:,}")

    src = torch.randint(1, V, (B, T))
    tgt = torch.randint(1, V, (B, T))

    logits = model(src, tgt)
    print(f"src    shape : {tuple(src.shape)}")      # (2, 64)
    print(f"tgt    shape : {tuple(tgt.shape)}")      # (2, 64)
    print(f"logits shape : {tuple(logits.shape)}")   # (2, 63, 32000)

    # Test greedy
    ids = model.translate_greedy(src[0:1], bos_id=2, eos_id=3, max_len=20)
    print(f"Greedy ({len(ids)} tokens): {ids[:8]}...")
    print("PASSED!")
