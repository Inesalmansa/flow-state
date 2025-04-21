import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        # Multi-head self-attention layer. No positional encodings are added.
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Two layer norm layers for the residual connections.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # A simple feed-forward network (FFN) with dropout.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Expected input shape: (batch_size, num_tokens, embed_dim)
        # Transformer layers in PyTorch expect (num_tokens, batch_size, embed_dim)
        # x_t = x.transpose(0, 1)
        attn_output, _ = self.self_attn(x, x, x)
        # Residual connection + layer norm.
        x_attn = self.norm1(x + attn_output)
        # Feed-forward network with another residual connection and layer norm.
        ffn_output = self.ffn(x_attn)
        x_ffn = self.norm2(x_attn + ffn_output)
        # Return to original shape.
        return x_ffn

class TransformerNet(nn.Module):
    def __init__(
        self,
        in_features,        # Dimension of each input token.
        out_features,       # Desired output dimension.
        embed_dim,          # Embedding dimension for the transformer.
        num_heads,          # Number of attention heads.                 # Hidden dimension in the feed-forward network.
        num_layers,
        ff_hidden_dim = 512,  # Number of transformer blocks.
        dropout=0.1,
        preprocessing=None  # Optional preprocessing (e.g., periodic feature mapping).
    ):
        super().__init__()
        self.preprocessing = preprocessing
        # Initial linear embedding layer to map input features to embed_dim.
        self.input_layer = nn.Linear(in_features, embed_dim)
        # Stack of transformer blocks.
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        # Final output layer.
        self.output_layer = nn.Linear(embed_dim, out_features)

    def forward(self, inputs, context=None):
        # inputs shape: (batch_size, num_tokens, in_features)
        # 'context' is not used here because the transformer processes all tokens together.
        if self.preprocessing is not None:
            inputs = self.preprocessing(inputs)
        x = self.input_layer(inputs)
        for layer in self.layers:
            x = layer(x)
        outputs = self.output_layer(x)
        return outputs