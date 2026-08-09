# 🤖 Bot de Trading Expert — Binance Testnet

## Architecture

```
bot_expert/
├── config.py        → Tous les paramètres centralisés
├── data.py          → Données multi-timeframe (4H, 30M, 5M)
├── indicators.py    → RSI, SMA, ATR, OBV, VWAP, Volume spike, Swing levels
├── strategy.py      → Signal multi-timeframe + filtre horaire
├── risk.py          → Sizing ATR, trailing stop, scale-out, breakeven, drawdown guard
├── execution.py     → Ordres Binance (buy/sell/scale-out)
├── logger.py        → Journal CSV des trades + métriques JSON (Sharpe, DD, win rate)
├── backtest.py      → Backtest vectorisé (vectorbt) ou simple
├── main.py          → Boucle principale
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration Binance Testnet

1. Va sur https://testnet.binance.vision
2. Connecte-toi avec GitHub
3. Génère tes clés API testnet
4. Configure les variables d'environnement :

```bash
export BINANCE_TESTNET_API_KEY="ta_cle_testnet"
export BINANCE_TESTNET_API_SECRET="ton_secret_testnet"
```

---

## Lancer le bot

```bash
cd bot_expert
python main.py
```

---

## Lancer le backtest

```bash
# Backtest simple (sans vectorbt)
python backtest.py --symbol BTCUSDT --interval 30m --start 2023-01-01 --simple

# Backtest avancé (vectorbt requis)
python backtest.py --symbol BTCUSDT --interval 30m --start 2023-01-01 --cash 10000
```

---

## Stratégie

### Analyse multi-timeframe

| Timeframe | Rôle |
|-----------|------|
| **4H** | Filtre de tendance macro (SMA50/200, OBV) |
| **30M** | Signal d'entrée (RSI, VWAP, Volume spike) |
| **5M** | Futur : scalping/entrée précise (réservé) |

### Conditions d'entrée (toutes requises)

- ✅ Tendance 4H haussière : `close > SMA50 > SMA200` + OBV haussier
- ✅ RSI(14) croise au-dessus de 30 sur 30M
- ✅ Prix au-dessus du VWAP
- ✅ OBV > signal OBV (pression acheteuse)
- ✅ Heure UTC valide (pas entre 22h-06h)

### Gestion de la position

| Événement | Action |
|-----------|--------|
| +1R atteint | SL déplacé au breakeven |
| TP1 = entry + 4×ATR | Vente de 50% (scale-out) |
| TP2 = entry + 8×ATR | Fermeture totale |
| SL = entry - 2×ATR | Stop-loss adaptatif |
| Trailing activé | Suit le prix après breakeven |
| RSI > 70 ou prix < SMA50 | Sortie stratégique |

### Protection globale

- **Max drawdown** : bot s'arrête si DD > 10%
- **Filtre horaire** : inactif 22h-06h UTC
- **1 trade max** à la fois

---

## Fichiers de logs

```
logs/
├── bot.log                    → Log temps réel (console + fichier)
├── trades.csv                 → Journal de chaque trade
├── performance.json           → Métriques (Sharpe, win rate, drawdown...)
├── backtest_trades.csv        → Trades du backtest
└── backtest_equity.csv        → Equity curve du backtest
```

---

## Paramètres modifiables (config.py)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `RISK_PER_TRADE_PCT` | 1% | Risque par trade |
| `ATR_SL_MULTIPLIER` | 2.0 | Multiplicateur ATR pour le SL |
| `ATR_TP_MULTIPLIER` | 4.0 | Multiplicateur ATR pour TP1 |
| `MAX_DRAWDOWN_PCT` | 10% | Drawdown maximum avant arrêt |
| `SCALE_OUT_PCT` | 50% | % vendu au TP1 |
| `AVOID_HOURS_UTC` | 22h-06h | Heures inactives |

---

## ⚠️ Avertissement

Ce bot est conçu **exclusivement pour le paper trading** sur Binance Testnet.
Le trading de cryptomonnaies comporte des risques élevés.
Ne jamais utiliser ce code en production sans tests approfondis.
