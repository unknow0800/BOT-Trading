"""
ml/train.py — Entrainement du modele LSTM + Transformer

Usage :
    python ml/train.py --symbol BTCUSDT --interval 30m --start 2024-01-01 --epochs 50

Etapes :
  1. Telecharge l'historique Binance testnet
  2. Prepare les features et labels
  3. Entraine le modele avec early stopping
  4. Sauvegarde model.pt + scaler.pkl + rapport de performance
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features import prepare_data, FEATURE_COLS, WINDOW
from ml.model    import build_model, save_model, count_params, MODEL_PATH

import requests
import pandas as pd

BASE_URL = "https://api.binance.com/api"


# ==========================
# TELECHARGEMENT HISTORIQUE
# ==========================

def download_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    print(f"[train] Telechargement {symbol} {interval} ({limit} bougies)...")
    r = requests.get(f"{BASE_URL}/v3/klines",
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=15)
    klines = r.json()
    df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','qav','trades','tbbase','tbquote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.set_index('timestamp', inplace=True)
    print(f"[train] {len(df)} bougies recues.")
    return df


# ==========================
# ENTRAINEMENT
# ==========================

def train(symbol: str = 'BTCUSDT',
          interval: str = '30m',
          limit: int    = 1000,
          epochs: int   = 60,
          batch_size: int = 32,
          lr: float     = 1e-3,
          patience: int = 10,
          device: str   = 'cpu'):

    os.makedirs('ml', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # --- 1. Donnees ---
    df = download_klines(symbol, interval, limit)
    X, y, scaler, cols = prepare_data(df, fit_scaler=True)

    # --- 2. Split train/val (80/20, pas de shuffle pour respecter la temporalite) ---
    if len(X) < 10:
        print(f"[train] Pas assez de sequences ({len(X)}). Augmente --limit ou reduis WINDOW.")
        return

    split = int(len(X) * 0.8)
    if split == 0:
        split = max(1, len(X) - 2)

    # --- Oversampling des classes minoritaires (BUY et SELL) sur le train set ---
    X_tr_raw, y_tr_raw = X[:split], y[:split]
    counts = np.bincount(y_tr_raw, minlength=3)
    max_count = counts.max()
    X_balanced, y_balanced = [], []
    for cls in range(3):
        idx = np.where(y_tr_raw == cls)[0]
        if len(idx) == 0:
            continue
        repeats = int(np.ceil(max_count / len(idx)))
        idx_over = np.tile(idx, repeats)[:max_count]
        np.random.shuffle(idx_over)
        X_balanced.append(X_tr_raw[idx_over])
        y_balanced.append(y_tr_raw[idx_over])
    X_train = np.concatenate(X_balanced)
    y_train = np.concatenate(y_balanced)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    X_val, y_val = X[split:], y[split:]
    print(f"[train] Apres oversampling -> Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"[train] Classes train: HOLD={np.sum(y_train==0)} BUY={np.sum(y_train==1)} SELL={np.sum(y_train==2)}")
    print(f"[train] Split brut: {split} train | {len(X)-split} val")

    class_weights = torch.ones(3, dtype=torch.float32).to(device)
    print(f"[train] Oversampling actif - class weights uniformes")

    # --- 3. DataLoaders ---
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val), torch.tensor(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # --- 4. Modele ---
    model = build_model(input_size=len(cols), device=device)
    print(f"[train] Modele: {count_params(model):,} parametres")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # --- 5. Boucle d'entrainement ---
    best_val_loss  = float('inf')
    patience_count = 0
    history = []

    print(f"\n[train] Debut entrainement ({epochs} epochs)...")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(xb)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total   += len(xb)

        # -- Validation --
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss    += loss.item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += len(xb)

        tl = train_loss / train_total
        ta = train_correct / train_total
        vl = val_loss / val_total
        va = val_correct / val_total

        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tl, 'val_loss': vl,
                        'train_acc': ta, 'val_acc': va})

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train loss: {tl:.4f} acc: {ta:.2%} | "
                  f"Val loss: {vl:.4f} acc: {va:.2%}")

        # -- Early stopping --
        if vl < best_val_loss - 1e-4:
            best_val_loss = vl
            save_model(model, MODEL_PATH)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[train] Early stopping a l'epoch {epoch}.")
                break

    # --- 6. Evaluation finale ---
    print("\n" + "=" * 55)
    print("[train] Evaluation sur le set de validation :")

    # Recharger le meilleur modele
    from ml.model import load_model
    model = load_model(len(cols), MODEL_PATH, device)

    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    print(classification_report(
        all_true, all_preds,
        target_names=['HOLD', 'BUY', 'SELL'],
        zero_division=0
    ))

    cm = confusion_matrix(all_true, all_preds)
    print("Matrice de confusion (HOLD / BUY / SELL) :")
    print(cm)

    # Sauvegarde du rapport
    report = {
        'symbol'    : symbol,
        'interval'  : interval,
        'n_train'   : len(X_train),
        'n_val'     : len(X_val),
        'best_val_loss': best_val_loss,
        'history'   : history[-10:]
    }
    with open('logs/train_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n[train] Rapport sauvegarde -> logs/train_report.json")
    print(f"[train] Modele -> {MODEL_PATH}")
    print(f"[train] Scaler -> ml/scaler.pkl")
    print("=" * 55)


# ==========================
# CLI
# ==========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol',   default='BTCUSDT')
    parser.add_argument('--interval', default='30m')
    parser.add_argument('--limit',    type=int, default=5000)
    parser.add_argument('--epochs',   type=int, default=150)
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--lr',       type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] Device : {device}")

    train(
        symbol    = args.symbol,
        interval  = args.interval,
        limit     = args.limit,
        epochs    = args.epochs,
        batch_size= args.batch,
        lr        = args.lr,
        patience  = args.patience,
        device    = device
    )