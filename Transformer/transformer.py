import torch
import torch.nn as nn


# =========================================================
# Multi-Head Self Attention
# =========================================================
class SelfAttention(nn.Module):
    """
    Multi-Head Self-Attention layer used in both Encoder and Decoder.
    Works on CPU and GPU without any modification.
    """

    def __init__(self, embed_size: int, heads: int):
        super().__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Ensure embedding can be split across heads
        assert self.head_dim * heads == embed_size, "embed_size must be divisible by heads"

        # Linear projections for Query, Key, Value
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)

        # Final projection after concatenating heads
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Linear projections
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Split embedding into multiple heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Attention score calculation (scaled dot-product)
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

        # Apply mask if available (important for padding + decoder)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        # Scale and normalize attention scores
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        # Multiply attention weights with values
        out = torch.einsum("nhqk,nkhd->nqhd", attention, values)

        # Merge heads back
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        return self.fc_out(out)


# =========================================================
# Transformer Block (Attention + Feed Forward)
# =========================================================
class TransformerBlock(nn.Module):
    """
    Core building block of Transformer:
    1. Multi-Head Attention
    2. Add & Norm
    3. Feed Forward
    4. Add & Norm
    """

    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()

        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Position-wise feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),  # better than ReLU for modern transformers
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Residual connection + normalization
        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)

        # Second residual connection
        out = self.dropout(self.norm2(forward + x))

        return out


# =========================================================
# Encoder
# =========================================================
class Encoder(nn.Module):
    """
    Encoder = Embedding + Positional Encoding + N Transformer Blocks
    """

    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape

        # Create positional indices
        positions = torch.arange(0, seq_length, device=x.device).expand(N, seq_length)

        # Token embedding + positional embedding
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Pass through stacked encoder blocks
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


# =========================================================
# Decoder Block
# =========================================================
class DecoderBlock(nn.Module):
    """
    Decoder block has:
    1. Masked self-attention
    2. Encoder-Decoder attention
    3. Feed Forward block
    """

    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super().__init__()

        self.self_attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):

        # Masked self-attention (decoder cannot see future words)
        attention = self.self_attention(x, x, x, trg_mask)

        query = self.dropout(self.norm(attention + x))

        # Encoder-decoder attention
        out = self.transformer_block(value, key, query, src_mask)

        return out


# =========================================================
# Decoder
# =========================================================
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length, device=x.device).expand(N, seq_length)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        return self.fc_out(x)


# =========================================================
# Full Transformer Model
# =========================================================
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.1,
        max_length=100,
    ):
        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    # Padding mask (used in encoder + decoder attention)
    def make_src_mask(self, src):
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask (decoder cannot see future tokens)
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device))
        return mask.expand(N, 1, trg_len, trg_len)

    def forward(self, src, trg):

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out


# =========================================================
# Quick test (CPU + GPU friendly)
# =========================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0],
                      [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)

    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0],
                        [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    model = Transformer(
        src_vocab_size=10,
        trg_vocab_size=10,
        src_pad_idx=0,
        trg_pad_idx=0
    ).to(device)

    out = model(x, trg[:, :-1])
    print("Output shape:", out.shape)