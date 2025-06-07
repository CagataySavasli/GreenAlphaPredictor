# hybrid_transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Sinüzoidal pozisyon kodlaması (Vaswani et al. 2017).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Pozisyon ve boyut matrisleri oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for feature fusion between price and ESG contexts."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # x, residual: [batch, d_model]
        x1 = self.elu(self.fc1(x))
        x2 = self.fc2(x1)
        g = self.sigmoid(self.gate(x))
        # gated blending of new info and residual
        return g * x2 + (1 - g) * residual


class CrossModalTransformer(nn.Module):
    """
    Cross-Modal Transformer integrating ESG signals via cross-attention into price prediction.

    Args:
        input_size (int): total number of features (price + ESG)
        price_feature_size (int): number of price-only features
        d_model (int): transformer hidden dimension
        nhead (int): number of attention heads
        num_encoder_layers (int): encoder layer count
        dim_feedforward (int): width of feed-forward layers
        dropout (float): dropout rate
    """
    def __init__(
        self,
        input_size: int,
        price_feature_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.price_size = price_feature_size
        self.esg_size = input_size - price_feature_size

        # Separate projections for each modality
        self.price_proj = nn.Linear(self.price_size, d_model)
        self.esg_proj   = nn.Linear(self.esg_size, d_model)

        # Shared positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Independent transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.price_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.esg_encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Cross-attention block: price queries attend to ESG keys/values
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Fusion via gated residual network
        self.grn = GatedResidualNetwork(d_model)

        #Normalize fused features
        self.final_norm = nn.LayerNorm(d_model)

        # Final regression head
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_size]
        returns: [batch, 1]
        """
        # Split price vs ESG
        price_feats = x[..., :self.price_size]
        esg_feats   = x[..., self.price_size:]

        # Project & add positional encodings
        price_emb = self.pos_encoder(self.price_proj(price_feats))
        esg_emb   = self.pos_encoder(self.esg_proj(esg_feats))

        # Encode each modality
        enc_price = self.price_encoder(price_emb)
        enc_esg   = self.esg_encoder(esg_emb)

        # Take last time step from price as query
        query = enc_price[:, -1:, :]
        # Cross-attention over ESG sequence
        att_out, _ = self.cross_attn(query, enc_esg, enc_esg)
        att_out = att_out.squeeze(1)

        # Price context from last step
        price_last = enc_price[:, -1, :]

        # Fuse ESG-informed context with price context
        fused = self.grn(att_out, price_last)

        normed = self.final_norm(fused)

        # Regression output
        out = self.decoder(normed)
        return out
