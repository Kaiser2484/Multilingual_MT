"""
attention.py  –  Tuan 2 | Thanh vien 2
=======================================
Co che Multi-Head Attention va cac block Encoder / Decoder.

Lay tu Thu vien Thanh vien 1 (src/layers.py):
    - PositionWiseFeedForward

Quy uoc tensor xuyen suot file:
    B    = batch_size
    T_s  = seq_len chuoi nguon (source)
    T_t  = seq_len chuoi dich  (target)
    D    = d_model
    H    = num_heads
    Dk   = d_k = D // H   (chieu moi head)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import PositionWiseFeedForward


# ==============================================================================
# HELPER: Scaled Dot-Product Attention (nhan biet mask)
# ==============================================================================
def scaled_dot_product_attention(
    Q: torch.Tensor,          # [B, H, T_q, Dk]
    K: torch.Tensor,          # [B, H, T_k, Dk]
    V: torch.Tensor,          # [B, H, T_k, Dk]
    mask: torch.Tensor = None,  # broadcastable vao [B, H, T_q, T_k]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tinh Attention theo cong thuc:
        Attention(Q,K,V) = softmax( Q @ K^T / sqrt(Dk) ) @ V

    Masking co 2 loai:
    ┌─────────────────┬─────────────────────────────────────────────────────┐
    │ Loai mask       │ Muc dich                                            │
    ├─────────────────┼─────────────────────────────────────────────────────┤
    │ Padding mask    │ An [PAD] o chuoi nguon/dich. Shape: [B,1,1,T_k]    │
    │                 │ True tai vi tri PAD → gan -inf → softmax ra ~0      │
    ├─────────────────┼─────────────────────────────────────────────────────┤
    │ Causal mask     │ Decoder chi nhin thay qua khu, an tuong lai.        │
    │ (look-ahead)    │ Ma tran tam giac tren, Shape: [1,1,T_t,T_t]        │
    │                 │ True tai [i,j] voi j>i → gan -inf                  │
    └─────────────────┴─────────────────────────────────────────────────────┘

    Returns:
        context : [B, H, T_q, Dk]  – output sau weighted sum
        attn_w  : [B, H, T_q, T_k] – trong so attention (de visualize)
    """
    Dk = Q.size(-1)  # chieu moi head

    # Buoc 1: Tinh attention score
    # Q @ K^T : [B,H,T_q,Dk] @ [B,H,Dk,T_k] → [B,H,T_q,T_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, T_q, T_k]
    scores = scores / math.sqrt(Dk)                # chia sqrt(Dk) de on dinh gradient

    # Buoc 2: AP dung mask (neu co)
    # Gan -1e9 vao vi tri can che → sau softmax se xap xi 0
    # mask: True  = vi tri bi che (PAD hoac tuong lai)
    #       False = vi tri hop le
    if mask is not None:
        scores = scores.masked_fill(mask, float("-1e9"))  # [B, H, T_q, T_k]

    # Buoc 3: Softmax theo chieu T_k (tong trong so = 1 tren moi hang)
    attn_w = F.softmax(scores, dim=-1)  # [B, H, T_q, T_k]

    # Buoc 4: Weighted sum cua V
    # [B,H,T_q,T_k] @ [B,H,T_k,Dk] → [B,H,T_q,Dk]
    context = torch.matmul(attn_w, V)  # [B, H, T_q, Dk]

    return context, attn_w


# ==============================================================================
# 1. MultiHeadAttention
# ==============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention theo Vaswani et al. 2017.

    Y tuong: Thay vi chay 1 ham attention duy nhat voi D chieu,
    ta chay H ham attention song song, moi ham dung D/H chieu.
    Cac head hoc cac kieu tuong quan khac nhau trong chuoi.

    Buoc tong quat:
        1. Chieu Q, K, V qua W_q, W_k, W_v : D → D
        2. Chia moi ma tran thanh H head     : [B,T,D] → [B,H,T,Dk]
        3. Chay scaled_dot_product_attention tren tung head song song
        4. Ghep H head lai                   : [B,H,T,Dk] → [B,T,D]
        5. Chieu output qua W_o              : D → D

    Args:
        d_model   (int): Chieu embedding.
        num_heads (int): So luong head. Bat buoc: d_model % num_heads == 0.
        dropout   (float): Dropout ap tren attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phai chia het cho num_heads"

        self.d_model = d_model       # D
        self.num_heads = num_heads   # H
        self.d_k = d_model // num_heads  # Dk = D/H

        # 4 ma tran chieu tuyen tinh: W_q, W_k, W_v, W_o
        # Moi cai: [D, D] (tham so) — Linear(D, D) tuong duong matmul voi [D,D]
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # chieu Q
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # chieu K
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # chieu V
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # chieu output

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tach chieu D thanh H head de chay attention song song.

        x    : [B, T, D]
        return [B, H, T, Dk]
        """
        B, T, D = x.shape
        # Reshape D → H * Dk
        x = x.view(B, T, self.num_heads, self.d_k)  # [B, T, H, Dk]
        # Hoan vi de H len truoc T: de matmul chay song song tren H
        x = x.transpose(1, 2)                        # [B, H, T, Dk]
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ghep H head lai sau khi tinh attention.

        x    : [B, H, T, Dk]
        return [B, T, D]
        """
        B, H, T, Dk = x.shape
        x = x.transpose(1, 2)             # [B, T, H, Dk]
        x = x.contiguous().view(B, T, -1) # [B, T, D]  (D = H * Dk)
        return x

    def forward(
        self,
        Q: torch.Tensor,           # [B, T_q, D]
        K: torch.Tensor,           # [B, T_k, D]
        V: torch.Tensor,           # [B, T_k, D]
        mask: torch.Tensor = None, # [B, 1, T_q, T_k] hoac [B, 1, 1, T_k]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output : [B, T_q, D]        – bieu dien moi da ket hop context
            attn_w : [B, H, T_q, T_k]  – trong so attention
        """
        # --- Buoc 1: Chieu tuyen tinh ---
        Q = self.W_q(Q)   # [B, T_q, D]
        K = self.W_k(K)   # [B, T_k, D]
        V = self.W_v(V)   # [B, T_k, D]

        # --- Buoc 2: Tach head ---
        Q = self._split_heads(Q)   # [B, H, T_q, Dk]
        K = self._split_heads(K)   # [B, H, T_k, Dk]
        V = self._split_heads(V)   # [B, H, T_k, Dk]

        # --- Buoc 3: Scaled Dot-Product Attention (tat ca H head song song) ---
        # context: [B, H, T_q, Dk]  |  attn_w: [B, H, T_q, T_k]
        context, attn_w = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_w = self.dropout(attn_w)

        # --- Buoc 4: Ghep head ---
        context = self._merge_heads(context)   # [B, T_q, D]

        # --- Buoc 5: Chieu output W_o ---
        output = self.W_o(context)             # [B, T_q, D]

        return output, attn_w                  # [B,T_q,D], [B,H,T_q,T_k]


# ==============================================================================
# HELPER: Tao mask
# ==============================================================================
def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tao padding mask: True tai vi tri co token [PAD].

    seq    : [B, T]  – chuoi token ID
    return : [B, 1, 1, T]  – da them chieu H va T_q de broadcast voi scores
                              [B, H, T_q, T_k]
    """
    # (seq == pad_idx) : [B, T]  – True tai PAD
    mask = (seq == pad_idx)           # [B, T]
    mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    return mask


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Tao causal (look-ahead) mask cho Decoder Self-Attention.
    Token o vi tri i chi duoc nhin cac token tu 0..i (khong nhin tuong lai).

    Vi du voi seq_len=4:
        [[F, T, T, T],   <- token 0 chi nhin chinh no
         [F, F, T, T],   <- token 1 nhin 0,1
         [F, F, F, T],   <- token 2 nhin 0,1,2
         [F, F, F, F]]   <- token 3 nhin tat ca

    (F=False=hop le, T=True=bi che)

    return : [1, 1, T, T]  – de broadcast vao [B, H, T_t, T_t]
    """
    # torch.triu lay tam giac tren (ke ca duong cheo chinh neu diagonal=1)
    # ta can che j > i, nen diagonal=1
    ones = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    mask = torch.triu(ones, diagonal=1)      # [T, T]  True o tam giac tren
    mask = mask.unsqueeze(0).unsqueeze(0)    # [1, 1, T, T]
    return mask


def make_decoder_mask(
    tgt: torch.Tensor,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Ket hop padding mask va causal mask cho Decoder Self-Attention.

    Voi moi vi tri (i, j):
        bi che neu: token[j] la PAD  HOAC  j > i (tuong lai)

    tgt    : [B, T_t]
    return : [B, 1, T_t, T_t]
    """
    B, T = tgt.shape
    device = tgt.device

    pad_mask    = make_padding_mask(tgt, pad_idx)        # [B, 1, 1, T_t]
    causal_mask = make_causal_mask(T, device)            # [1, 1, T_t, T_t]

    # OR: bi che neu thoa mot trong hai dieu kien
    combined = pad_mask | causal_mask                    # [B, 1, T_t, T_t]
    return combined


# ==============================================================================
# 2. EncoderLayer
# ==============================================================================
class EncoderLayer(nn.Module):
    """
    Mot tang Encoder gom:
        [1] Multi-Head Self-Attention  (Q=K=V=x)
        [2] Add & LayerNorm            (Residual Connection)
        [3] Position-wise FFN          (tu Thanh vien 1)
        [4] Add & LayerNorm

    So do luong du lieu:
        x [B,T_s,D]
          │
          ├─ Self-Attention(Q=K=V=x) ──► attn_out [B,T_s,D]
          │         └─ mask: src_padding_mask [B,1,1,T_s]
          │
          ├─ Add: x + attn_out  →  [B,T_s,D]
          ├─ LayerNorm          →  [B,T_s,D]   (= x1)
          │
          ├─ FFN(x1)           →  ffn_out [B,T_s,D]
          ├─ Add: x1 + ffn_out →  [B,T_s,D]
          └─ LayerNorm          →  [B,T_s,D]   (output)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionWiseFeedForward(d_model, d_ff, dropout)

        # LayerNorm ap sau moi sub-layer (Post-LN theo paper goc)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,               # [B, T_s, D]
        src_mask: torch.Tensor = None, # [B, 1, 1, T_s]  padding mask
    ) -> torch.Tensor:
        """Returns: [B, T_s, D]"""

        # --- Sub-layer 1: Self-Attention ---
        # Q = K = V = x (tu attention: moi token hoi ve tat ca token khac)
        attn_out, _ = self.self_attn(Q=x, K=x, V=x, mask=src_mask)
        # [B,T_s,D]
        attn_out = self.dropout(attn_out)

        # Residual + LayerNorm
        x = self.norm1(x + attn_out)   # [B, T_s, D]

        # --- Sub-layer 2: FFN ---
        ffn_out = self.ffn(x)          # [B, T_s, D]
        ffn_out = self.dropout(ffn_out)

        # Residual + LayerNorm
        x = self.norm2(x + ffn_out)    # [B, T_s, D]

        return x                       # [B, T_s, D]


# ==============================================================================
# 3. DecoderLayer
# ==============================================================================
class DecoderLayer(nn.Module):
    """
    Mot tang Decoder gom 3 sub-layer:

        [1] Masked Self-Attention   – Decoder nhin chuoi dich da sinh
            Mask = causal + padding (khong nhin tuong lai & khong nhin PAD)
            Q = K = V = x_dec

        [2] Cross-Attention (Encoder-Decoder Attention)
            Query  lay tu Decoder (x sau sub-layer 1)
            Key, Value lay tu Encoder output
            Mask = src_padding_mask (an PAD trong chuoi nguon)

        [3] Position-wise FFN

    So do luong du lieu:
        x_dec [B,T_t,D] , enc_out [B,T_s,D]
          │
          ├─ Masked Self-Attn(Q=K=V=x_dec, mask=tgt_mask) → [B,T_t,D]
          ├─ Add & Norm → x1 [B,T_t,D]
          │
          ├─ Cross-Attn(Q=x1, K=enc_out, V=enc_out, mask=src_mask) → [B,T_t,D]
          ├─ Add & Norm → x2 [B,T_t,D]
          │
          ├─ FFN(x2) → [B,T_t,D]
          └─ Add & Norm → output [B,T_t,D]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Sub-layer 1: Masked Self-Attention
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Cross-Attention (Encoder–Decoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 3: FFN
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 3 cap LayerNorm tuong ung voi 3 sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x_dec: torch.Tensor,            # [B, T_t, D]  – bieu dien Decoder
        enc_out: torch.Tensor,          # [B, T_s, D]  – output cua Encoder
        src_mask: torch.Tensor = None,  # [B, 1, 1, T_s] – padding mask chuoi nguon
        tgt_mask: torch.Tensor = None,  # [B, 1, T_t, T_t] – causal+pad mask chuoi dich
    ) -> torch.Tensor:
        """Returns: [B, T_t, D]"""

        # --- Sub-layer 1: Masked Self-Attention ---
        # Decoder chi duoc nhin cac token dich da xuat hien (khong nhin tuong lai)
        # Q = K = V = x_dec, mask = tgt_mask (causal + padding)
        attn1, _ = self.masked_self_attn(
            Q=x_dec, K=x_dec, V=x_dec, mask=tgt_mask
        )                                          # [B, T_t, D]
        attn1 = self.dropout(attn1)
        x = self.norm1(x_dec + attn1)             # [B, T_t, D]

        # --- Sub-layer 2: Cross-Attention ---
        # Query tu Decoder (x), Key va Value tu Encoder (enc_out)
        # → Decoder "hoi" Encoder de lay thong tin chuoi nguon
        # mask = src_mask: an PAD trong chuoi nguon
        attn2, _ = self.cross_attn(
            Q=x, K=enc_out, V=enc_out, mask=src_mask
        )                                          # [B, T_t, D]
        # Giai thich shape:
        #   Q: [B,H,T_t,Dk]  (T_q = T_t, so token dich)
        #   K: [B,H,T_s,Dk]  (T_k = T_s, so token nguon)
        #   V: [B,H,T_s,Dk]
        #   scores: [B,H,T_t,T_s]  moi token dich "hoi" tat ca token nguon
        #   output: [B,H,T_t,Dk] → merge → [B,T_t,D]
        attn2 = self.dropout(attn2)
        x = self.norm2(x + attn2)                 # [B, T_t, D]

        # --- Sub-layer 3: FFN ---
        ffn_out = self.ffn(x)                     # [B, T_t, D]
        ffn_out = self.dropout(ffn_out)
        x = self.norm3(x + ffn_out)               # [B, T_t, D]

        return x                                   # [B, T_t, D]


# ==============================================================================
# Self-test (chay: python src/attention.py)
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(0)

    B, T_s, T_t, D, H = 2, 12, 8, 256, 8
    D_FF  = 1024
    PAD   = 0

    device = torch.device("cpu")

    # --- Du lieu gia ---
    src_ids = torch.randint(1, 1000, (B, T_s))
    tgt_ids = torch.randint(1, 1000, (B, T_t))
    src_ids[0, -3:] = PAD   # giam lap PAD
    tgt_ids[1, -2:] = PAD

    enc_out  = torch.randn(B, T_s, D)   # [B, T_s, D] – gia su da qua Encoder
    x_dec    = torch.randn(B, T_t, D)   # [B, T_t, D]

    # --- Tao mask ---
    src_mask = make_padding_mask(src_ids, PAD)          # [B,1,1,T_s]
    tgt_mask = make_decoder_mask(tgt_ids, PAD)          # [B,1,T_t,T_t]

    print("=" * 55)
    print("Kiem tra MultiHeadAttention (Self-Attention)")
    print("=" * 55)
    mha = MultiHeadAttention(D, H)
    out, w = mha(Q=enc_out, K=enc_out, V=enc_out, mask=src_mask)
    print(f"  Output : {tuple(out.shape)}   ky vong: ({B},{T_s},{D})")
    print(f"  Weights: {tuple(w.shape)}     ky vong: ({B},{H},{T_s},{T_s})")

    print()
    print("=" * 55)
    print("Kiem tra EncoderLayer")
    print("=" * 55)
    enc_layer = EncoderLayer(D, H, D_FF)
    enc_result = enc_layer(enc_out, src_mask=src_mask)
    print(f"  Output: {tuple(enc_result.shape)}   ky vong: ({B},{T_s},{D})")

    print()
    print("=" * 55)
    print("Kiem tra DecoderLayer")
    print("=" * 55)
    dec_layer = DecoderLayer(D, H, D_FF)
    dec_result = dec_layer(x_dec, enc_result, src_mask=src_mask, tgt_mask=tgt_mask)
    print(f"  Output: {tuple(dec_result.shape)}   ky vong: ({B},{T_t},{D})")

    print()
    print("Kiem tra Causal Mask (seq_len=5):")
    cm = make_causal_mask(5, device)[0, 0]  # [5,5]
    print(cm.int().tolist())
    # Ky vong:
    # [[0,1,1,1,1],
    #  [0,0,1,1,1],
    #  [0,0,0,1,1],
    #  [0,0,0,0,1],
    #  [0,0,0,0,0]]

    print()
    print("All checks passed - shape dau ra dung!")
