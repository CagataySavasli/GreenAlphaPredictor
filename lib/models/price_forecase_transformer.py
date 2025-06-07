import math
import torch
import torch.nn as nn


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


class PriceForecastTransformer(nn.Module):
    def __init__(
            self,
            input_size: int,
            d_model: int = 64,
            nhead: int = 4,
            num_encoder_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1
    ):
        """
        Transformer tabanlı fiyat tahmin modeli.

        Args:
            input_size (int): Girdi özellik sayısı (fiyat + ESG).
            d_model (int): Transformer model boyutu.
            nhead (int): Çok başlıklı dikkat (multi-head attention) sayısı.
            num_encoder_layers (int): Encoder katmanı sayısı.
            dim_feedforward (int): Feed-forward katman genişliği.
            dropout (float): Dropout oranı.
        """
        super(PriceForecastTransformer, self).__init__()
        # Girdi boyutunu d_model'e projekte etmek için linear
        self.input_proj = nn.Linear(input_size, d_model)
        # Pozisyon kodlaması
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # Transformer encoder katmanı
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # x: [batch, seq, feature]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        # Son katman: d_model → 1 (regresyon)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            out: [batch_size, 1]
        """
        # 1) Özellikleri d_model boyutuna projekte et
        x = self.input_proj(x)
        # 2) Pozisyon kodlaması ekle
        x = self.pos_encoder(x)
        # 3) Transformer encoder
        encoded = self.transformer_encoder(x)
        # 4) Son zaman adımından çıktı alıp lineer katmandan geçir
        out = self.decoder(encoded[:, -1, :])
        return out
