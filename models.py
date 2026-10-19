"""Attention and Transformer components exported from the source notebook."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, causal_mask=False):

    d_k = query.shape[-1]
    seq_len = query.shape[-2]


    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)


    if causal_mask:
        # mask[i,j] = True  means position i is NOT allowed to attend to j
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

    # softmax over the key dimension
    attention_weights = F.softmax(scores, dim=-1)

    # weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

def positional_encoding(seq_len, d_model):

    position = np.arange(seq_len)[:, np.newaxis]          # (seq_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )                                                       # (d_model/2,)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # even dimensions
    pe[:, 1::2] = np.cos(position * div_term)  # odd  dimensions
    return pe

class TransformerDecoderBlock(nn.Module):


    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                                  dropout=dropout, batch_first=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),          # dropout inside FFN
            nn.Linear(d_ff, d_model)
        )
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)   # after attention
        self.dropout2 = nn.Dropout(dropout)   # after FFN

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(0)

        # Build causal mask: upper triangular (seq_len × seq_len)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()


        attn_output, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x

class DecoderOnlyTransformer(nn.Module):


    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=d_ff, dropout=dropout,
                batch_first=False
            )
            for _ in range(num_layers)
        ])
        self.ln         = nn.LayerNorm(d_model)
        # Final linear projection: maps d_model → vocab_size
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model)  — pre-embedded tokens
        Returns:
            logits: (seq_len, batch_size, vocab_size)
        """
        seq_len = x.size(0)
        # Causal mask prevents attending to future positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        for layer in self.layers:
            # Causal self-attention without an unmasked cross-attention path.
            x = layer(x, src_mask=causal_mask)

        x = self.ln(x)
        logits = self.lm_head(x)   # project to vocabulary
        return logits

class SimpleLLM(nn.Module):


    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln    = nn.LayerNorm(d_model)
        self.fc    = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.fc(self.ln(self.embed(x)))
