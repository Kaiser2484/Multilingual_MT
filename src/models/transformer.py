import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        # self.pe shape: [max_len, 1, d_model]
        # transpose -> [1, max_len, d_model] để cộng broadcast
        pe_batch_first = self.pe.transpose(0, 1)
        x = x + pe_batch_first[:, :x.size(1), :]
        return self.dropout(x)

class TransformerMT(nn.Module):
    """
    Kiến trúc Transformer Base hoàn chỉnh hỗ trợ Multilingual.
    Sử dụng PyTorch nn.Transformer nội bộ để tối ưu tốc độ (FlashAttention hỗ trợ tự động ở PyTorch 2.0+).
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Sử dụng batch_first=True giúp dễ xử lý shape [batch, seq_len]
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz: int, device: torch.device):
        # Mask cho decoder để không nhìn thấy từ tương lai
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, src.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        """
        src shape: [batch_size, src_seq_len]
        tgt shape: [batch_size, tgt_seq_len]
        """
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        
        # Scale embedding theo bài báo gốc Attention is All You Need
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        outs = self.transformer(
            src_emb, tgt_emb, 
            src_mask=src_mask, tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.fc_out(outs)
