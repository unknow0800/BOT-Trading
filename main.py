"""
Bot de trading expert - Binance Testnet
=======================================
Architecture multi-timeframe avec :
  - Analyse tendance 4H (SMA, OBV)
  - Signal entrée 30M (RSI, VWAP, volume spike)
  - Gestion risque ATR (trailing stop, scale-out, breakeven)
  - Journal des trades + métriques (Sharpe, drawdown, win rate)
  - Protection drawdown global
  - Filtre horaire (évite les heures creuses)
"""

import time
from datetime import datetime
from binance.client import Client

from config import (
    API_KEY, API_SECRET, TESTNET, TESTNET_BASE_URL,
    SYMBOL, SLEEP_SECONDS
)
from data import (
    get_multi_timeframe_data, get_account_balance,
    get_btc_balance, get_symbol_filters, get_current_price
)
from strategy import get_trend_bias, get_entry_signal, get_market_context
from risk import Position, calc_position_size, update_position, DrawdownGuard
from execution import open_long, close_long, get_fill_price, place_buy_order, place_sell_order
from logger import BotLogger
from ml.predictor import DLPredictor


# ==========================
# INITIALISATION
# ==========================

def init_client() -> Client:
    if TESTNET:
        client = Client(API_KEY, API_SECRET, testnet=True, tld='vision')
        print("🔧 Mode TESTNET activé (Binance Testnet Spot)")
    else:
        client = Client(API_KEY, API_SECRET)
        print("🚀 Mode LIVE activé")

    # ✅ Fix erreur -1021 : décalage horloge PC vs serveur Binance
    from data import sync_time
    offset = sync_time(client)
    client.timestamp_offset = offset
    print(f"🕐 Sync temps : offset = {offset} ms")

    return client


# ==========================
# BOUCLE PRINCIPALE
# ==========================

def main_loop():
    logger = BotLogger()
    client = init_client()

    # Récupération des filtres du symbole
    lot_step, min_notional = get_symbol_filters(client, SYMBOL)
    logger.log_info(f"Filtres {SYMBOL} | LotStep: {lot_step} | MinNotional: {min_notional}")

    # État global
    position: Position | None = None
    entry_time: datetime | None = None

    # Equity initiale = USDT disponible (coherent avec suivi boucle)
    _init_usdt   = get_account_balance(client, "USDT")
    _init_btc    = get_btc_balance(client)
    equity_history = [_init_usdt]
    drawdown_guard = DrawdownGuard(_init_usdt)

    # Predictor Deep Learning (optionnel : si modele pas encore entraine, mode technique seul)
    dl_predictor = DLPredictor()
    if dl_predictor.is_ready():
        logger.log_info("[DL] Modele LSTM+Transformer charge et pret.")
    else:
        logger.log_info("[DL] Modele non entraine. Signal technique seul active.")

    logger.log_info(f"Capital initial : {_init_usdt:.2f} USDT | {_init_btc:.5f} BTC")
    logger.log_info("="*60)
    logger.log_info("Bot démarré. Ctrl+C pour arrêter.")
    logger.log_info("="*60)

    while True:
        try:
            print(f"\n{'='*60}")
            print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC]")

            # --- 1. Données multi-timeframe ---
            data = get_multi_timeframe_data(client, SYMBOL)
            df_trend  = data['1h']
            df_30m = data['30m']

            current_price = get_current_price(client, SYMBOL)

            # --- 2. Contexte de marché ---
            ctx = get_market_context(df_trend, df_30m)
            logger.log_context(ctx)

            # --- 3. Vérification drawdown global ---
            current_equity = get_account_balance(client, "USDT")
            equity_history.append(current_equity)
            total_equity = current_equity

            if drawdown_guard.update(total_equity):
                logger.log_error(
                    f"🛑 DRAWDOWN MAX ATTEINT ({drawdown_guard.current_drawdown*100:.1f}%) → BOT STOPPÉ"
                )
                break

            # --- 4. Gestion de la position active ---
            if position is not None:
                action_result = update_position(position, current_price)
                action = action_result['action']
                reason = action_result['reason']
                qty    = action_result['qty_to_sell']

                logger.log_info(
                    f"📍 Position | Entry: {position.entry_price:.2f} | "
                    f"SL: {position.trailing_stop:.2f} | "
                    f"TP1: {position.take_profit_1:.2f} | "
                    f"R: {position.r_multiple:.2f}"
                )

                if action in ('exit_sl', 'exit_tp2'):
                    success = close_long(client, SYMBOL, qty, lot_step, reason, logger)
                    if success:
                        logger.log_trade(
                            symbol=SYMBOL, side='sell',
                            quantity=qty, price=current_price,
                            entry_price=position.entry_price,
                            entry_time=entry_time,
                            r_multiple=position.r_multiple,
                            reason=reason
                        )
                        position = None
                        entry_time = None

                elif action == 'scale_out':
                    success = close_long(client, SYMBOL, qty, lot_step, reason, logger)
                    if success:
                        logger.log_trade(
                            symbol=SYMBOL, side='sell',
                            quantity=qty, price=current_price,
                            entry_price=position.entry_price,
                            entry_time=entry_time,
                            r_multiple=position.r_multiple,
                            reason=reason
                        )
                        # Position reste ouverte (remaining_qty)

                elif action == 'move_breakeven':
                    logger.log_info(f"🔒 {reason}")

            # --- 5. Recherche de nouveaux signaux ---
            # --- Signal Deep Learning ---
            dl_result = dl_predictor.predict(df_30m) if dl_predictor.is_ready() else {'signal': 'uncertain', 'confidence': 0.0, 'proba': {}}
            logger.log_info(
                f"[DL] Signal: {dl_result['signal'].upper()} | "
                f"Confiance: {dl_result['confidence']:.0%} | "
                f"P(buy)={dl_result['proba'].get('buy',0):.2f} "
                f"P(hold)={dl_result['proba'].get('hold',0):.2f} "
                f"P(sell)={dl_result['proba'].get('sell',0):.2f}"
            )

            if position is None:
                trend = ctx['trend_4h']
                signal = get_entry_signal(df_30m, trend, in_position=False)
                logger.log_info(f"📡 Signal : {signal}")

                # Confirmation DL : on entre seulement si DL confirme BUY (ou modele pas dispo)
                dl_confirms_buy = (
                    not dl_predictor.is_ready() or
                    dl_result['signal'] == 'buy'
                )
                if signal in ('enter_long', 'enter_long_strong') and dl_confirms_buy:
                    atr = ctx['atr_30m']
                    equity_usdt = get_account_balance(client)
                    qty = calc_position_size(
                        entry_price=current_price,
                        atr=atr,
                        equity_usdt=equity_usdt,
                        lot_step=lot_step
                    )

                    logger.log_info(
                        f"🟢 Tentative achat {qty:.5f} BTC à ~{current_price:.2f} USDT "
                        f"({'FORT' if signal == 'enter_long_strong' else 'normal'})"
                    )

                    success = open_long(
                        client=client,
                        symbol=SYMBOL,
                        quantity=qty,
                        min_notional=min_notional,
                        lot_step=lot_step,
                        entry_price=current_price,
                        logger=logger
                    )

                    if success:
                        position = Position(
                            symbol=SYMBOL,
                            entry_price=current_price,
                            quantity=qty,
                            atr_at_entry=atr
                        )
                        entry_time = datetime.utcnow()
                        logger.log_trade(
                            symbol=SYMBOL, side='buy',
                            quantity=qty, price=current_price,
                            reason=signal
                        )
                        logger.log_info(
                            f"Position ouverte | SL: {position.stop_loss:.2f} | "
                            f"TP1: {position.take_profit_1:.2f} | "
                            f"TP2: {position.take_profit_2:.2f}"
                        )

            else:
                # On est en position → vérification exit stratégique
                signal = get_entry_signal(df_30m, ctx['trend_4h'], in_position=True)
                dl_confirms_sell = (
                    not dl_predictor.is_ready() or
                    dl_result['signal'] in ('sell', 'uncertain')
                )
                if signal == 'exit_long' and position is not None and dl_confirms_sell:
                    reason = "Signal stratégique : exit_long"
                    success = close_long(
                        client, SYMBOL, position.remaining_qty,
                        lot_step, reason, logger
                    )
                    if success:
                        logger.log_trade(
                            symbol=SYMBOL, side='sell',
                            quantity=position.remaining_qty,
                            price=current_price,
                            entry_price=position.entry_price,
                            entry_time=entry_time,
                            r_multiple=position.r_multiple,
                            reason=reason
                        )
                        position = None
                        entry_time = None

            # --- 6. Mise à jour des métriques ---
            logger.update_performance(equity_history)

        except KeyboardInterrupt:
            logger.log_info("⛔ Bot arrêté manuellement.")
            break
        except Exception as e:
            logger.log_error(f"Exception inattendue : {e}")
            import traceback
            traceback.print_exc()

        logger.log_info(f"⏳ Pause {SLEEP_SECONDS//60} min...")
        time.sleep(SLEEP_SECONDS)


# ==========================
# ENTRY POINT
# ==========================

if __name__ == "__main__":
    main_loop()
