# Bot de Trading Algorithmique

**Bot de trading quantitatif multi-marchés combinant analyse technique multi-timeframe, gestion du risque avancée et un modèle de deep learning hybride (LSTM + Transformer).**

Deux moteurs distincts : un bot crypto sur **Binance Testnet** (BTC/USDT) et un bot actions/ETF sur **Alpaca Paper Trading**, couvrant plusieurs secteurs US (tech, semi-conducteurs, banques, santé, consommation...).

## Fonctionnalités principales

- **Stratégie multi-timeframe** : tendance macro, signal d'entrée et exécution sur trois horizons distincts (ex. 1H / 30M / 5M)
- **Indicateurs techniques** : RSI, SMA (croisement rapide/lent), ATR, OBV, VWAP, détection de pics de volume, niveaux de swing
- **Gestion du risque** : sizing basé sur l'ATR, stop-loss et take-profit dynamiques, trailing stop, scale-out partiel, passage à breakeven, garde-fou de drawdown maximum
- **Filtre horaire** : évite les périodes de faible liquidité (nuit UTC)
- **Modèle de deep learning hybride** (PyTorch) : architecture LSTM + Transformer entraînée sur les features techniques pour affiner les signaux
- **Backtesting** : moteur simple ou vectorisé (vectorbt) avec calcul de métriques (Sharpe ratio, drawdown, win rate)
- **Journalisation complète** : trades en CSV, métriques de performance en JSON, logs d'exécution
- **Module Alpaca avancé** : intelligence de marché et de portefeuille, ranking de setups, revue de trades, watchlist dynamique

## Stack technique

| Composant | Technologie |
|---|---|
| Trading crypto | python-binance (Testnet) |
| Trading actions | alpaca-py (Paper Trading) |
| Analyse technique | ta, pandas, numpy |
| Deep learning | PyTorch (LSTM + Transformer) |
| Backtesting | vectorbt |
| Scoring additionnel | tradingview-ta |

## Architecture

```
Bot de finance/
├── config.py              # Paramètres centralisés (risque, indicateurs, timeframes)
├── data.py                  # Récupération des données multi-timeframe
├── indicators.py            # Calcul des indicateurs techniques
├── strategy.py               # Génération de signaux multi-timeframe + filtre horaire
├── risk.py                    # Sizing, stops, trailing, scale-out, drawdown guard
├── execution.py                # Passage d'ordres (achat/vente/scale-out)
├── logger.py                    # Journal CSV des trades + métriques JSON
├── backtest.py                   # Backtest vectorisé ou simple
├── bot.py / main.py               # Boucle principale du bot crypto
│
├── ml/
│   ├── features.py            # Extraction des features pour le modèle
│   ├── model.py                 # Architecture LSTM + Transformer
│   ├── predictor.py              # Inférence du modèle entraîné
│   └── train.py                   # Entraînement du modèle
│
└── alpaca/
    ├── config_alpaca.py        # Configuration Alpaca (univers d'actifs, secrets)
    ├── data_alpaca.py            # Données multi-actifs Alpaca
    ├── strategy_alpaca.py         # Stratégie adaptée actions/ETF
    ├── market_intelligence.py      # Analyse de contexte marché
    ├── portfolio_intelligence.py    # Analyse du portefeuille
    ├── decision_logger.py            # Journalisation des décisions
    └── main_alpaca.py                # Boucle principale du bot Alpaca
```

## Installation

```bash
git clone https://github.com/unknow0800/BOT-Trading.git
cd "Bot de finance"
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Configuration

Aucune clé n'est codée en dur dans le code : tout passe par variables d'environnement.

**Binance Testnet** — clés à générer sur https://testnet.binance.vision :
```bash
setx BINANCE_TESTNET_API_KEY "ta_cle_testnet"
setx BINANCE_TESTNET_API_SECRET "ton_secret_testnet"
```

**Alpaca Paper Trading** — clés à générer sur https://app.alpaca.markets/paper/dashboard/overview :
```bash
setx ALPACA_API_KEY "ta_cle_alpaca"
setx ALPACA_API_SECRET "ton_secret_alpaca"
```

## Utilisation

**Bot crypto (Binance Testnet)**
```bash
python main.py
```

**Bot actions (Alpaca Paper Trading)**
```bash
cd alpaca
python main_alpaca.py
```

**Backtest**
```bash
# Version simple
python backtest.py --symbol BTCUSDT --interval 30m --start 2023-01-01 --simple

# Version avancée (vectorbt requis)
python backtest.py --symbol BTCUSDT --interval 30m --start 2023-01-01 --cash 10000
```

**Entraînement du modèle de deep learning**
```bash
python ml/train.py
```

## Gestion du risque

- Risque maximum par trade : paramétrable (`RISK_PER_TRADE_PCT`)
- Sizing dynamique basé sur l'ATR
- Stop-loss et take-profit calculés à partir de multiples d'ATR
- Passage automatique à breakeven après un seuil de R multiple atteint
- Scale-out partiel à mi-parcours
- Garde-fou de drawdown maximum pour stopper le bot en cas de perte excessive
- Nombre de positions ouvertes simultanées limité

## Avertissement

Ce projet est un outil d'expérimentation personnelle et pédagogique. Il fonctionne exclusivement en **environnement testnet / paper trading** — aucun ordre réel n'est passé avec de l'argent réel. Il ne constitue en aucun cas un conseil en investissement.

## Sécurité

Aucune clé API réelle n'est présente dans ce dépôt. Tous les secrets (clés Binance, Alpaca, Finnhub, FMP) sont exclus via `.gitignore` et doivent être fournis via variables d'environnement ou fichier local non commité.

## Licence

MIT
