"""
ml/model.py — Architecture LSTM + Transformer hybride

Structure :
  - Branche LSTM    : 2 couches (128->64), Dropout 0.2
  - Branche Transformer : Multi-head attention (4 heads) + FFN
  - Fusion          : Concatenate + Dense(64) + Dense(3)
  - Sortie          : Softmax -> P(HOLD), P(BUY), P(SELL)
"""

import torch
import torch.nn as nn
import math

MODEL_PATH  = "ml/model.pt"
SCALER_PATH = "ml/scaler.pkl"


# ==========================
# BRANCHE LSTM
# ==========================

class LSTMBranch(nn.Module):
    def __init__(self, input_size: int, hidden1: int = 128, hidden2: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1  = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2  = nn.Dropout(dropout)

    def forward(self, x):
        # x : (batch, seq, features)
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, (h, _) = self.lstm2(out)
        out = self.drop2(h[-1])   # derniere etape cachee
        return out   # (batch, 64)


# ==========================
# POSITIONAL ENCODING
# ==========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ==========================
# BRANCHE TRANSFORMER
# ==========================

class TransformerBranch(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64,
                 nhead: int = 4, ffn_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Projection des features vers d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)   # moyenne sur la sequence

    def forward(self, x):
        # x : (batch, seq, features)
        x = self.input_proj(x)         # (batch, seq, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)            # (batch, seq, d_model)
        # pooling sur la dimension temporelle
        x = x.permute(0, 2, 1)        # (batch, d_model, seq)
        x = self.pool(x).squeeze(-1)  # (batch, d_model)
        return x


# ==========================
# MODELE HYBRIDE COMPLET
# ==========================

class HybridModel(nn.Module):
    """
    LSTM + Transformer fusionnes pour prediction buy/sell/hold.

    Args:
        input_size : nombre de features (len(FEATURE_COLS))
        num_classes: 3 (HOLD=0, BUY=1, SELL=2)
    """
    def __init__(self, input_size: int, num_classes: int = 3,
                 lstm_hidden: int = 64, d_model: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm_branch        = LSTMBranch(input_size, 128, lstm_hidden, dropout)
        self.transformer_branch = TransformerBranch(input_size, d_model,
                                                     nhead=4, ffn_dim=128,
                                                     dropout=dropout)
        fusion_dim = lstm_hidden + d_model  # 64 + 64 = 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out  = self.lstm_branch(x)        # (batch, 64)
        trans_out = self.transformer_branch(x) # (batch, 64)
        fused = torch.cat([lstm_out, trans_out], dim=1)  # (batch, 128)
        logits = self.fusion(fused)            # (batch, 3)
        return logits   # pas de softmax ici (CrossEntropyLoss l'integre)

    def predict_proba(self, x):
        """Retourne les probabilites softmax pour l'inference."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# ==========================
# UTILITAIRES
# ==========================

def build_model(input_size: int, device: str = 'cpu') -> HybridModel:
    model = HybridModel(input_size=input_size)
    return model.to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: HybridModel, path: str = "ml/model.pt"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[model] Sauvegarde -> {path}")


def load_model(input_size: int, path: str = "ml/model.pt",
               device: str = 'cpu') -> HybridModel:
    model = build_model(input_size, device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"[model] Chargement -> {path}")
    return model