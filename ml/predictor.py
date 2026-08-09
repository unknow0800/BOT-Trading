"""
ml/predictor.py — Inference en temps reel dans la boucle du bot

Usage dans main.py :
    predictor = DLPredictor()
    signal = predictor.predict(df_30m)
    # signal = 'buy' | 'sell' | 'hold' | 'uncertain'
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features import prepare_last_sequence, FEATURE_COLS, WINDOW
from ml.model    import load_model, MODEL_PATH

CONFIDENCE_THRESHOLD = 0.60   # seuil minimum pour agir
CLASS_NAMES = {0: 'hold', 1: 'buy', 2: 'sell'}


class DLPredictor:
    """
    Wrapper pour l'inference DL en temps reel.
    Charge le modele une seule fois, puis predict() est appele a chaque bougie.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model  = None
        self.n_features = len(FEATURE_COLS)
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            print("[predictor] Modele non trouve. Lance d'abord : python ml/train.py")
            return
        if not os.path.exists("ml/scaler.pkl"):
            print("[predictor] Scaler non trouve. Lance d'abord : python ml/train.py")
            return
        try:
            self.model = load_model(self.n_features, MODEL_PATH, self.device)
            self.model.eval()
            print(f"[predictor] Modele charge ({MODEL_PATH})")
        except Exception as e:
            print(f"[predictor] Erreur chargement : {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, df_raw) -> dict:
        """
        Prend le DataFrame brut 30M et retourne :
        {
          'signal'    : 'buy' | 'sell' | 'hold' | 'uncertain',
          'confidence': float (0-1),
          'proba'     : {'hold': float, 'buy': float, 'sell': float}
        }
        """
        if not self.is_ready():
            return {'signal': 'uncertain', 'confidence': 0.0,
                    'proba': {'hold': 0, 'buy': 0, 'sell': 0}}

        try:
            x = prepare_last_sequence(df_raw).to(self.device)
            proba = self.model.predict_proba(x)[0].cpu().numpy()

            best_class = int(np.argmax(proba))
            confidence = float(proba[best_class])

            if confidence < CONFIDENCE_THRESHOLD:
                signal = 'uncertain'
            else:
                signal = CLASS_NAMES[best_class]

            return {
                'signal'    : signal,
                'confidence': round(confidence, 3),
                'proba'     : {
                    'hold': round(float(proba[0]), 3),
                    'buy' : round(float(proba[1]), 3),
                    'sell': round(float(proba[2]), 3),
                }
            }

        except Exception as e:
            print(f"[predictor] Erreur inference : {e}")
            return {'signal': 'uncertain', 'confidence': 0.0,
                    'proba': {'hold': 0, 'buy': 0, 'sell': 0}}

    def reload(self):
        """Recharge le modele (apres un re-entrainement)."""
        self._load()
