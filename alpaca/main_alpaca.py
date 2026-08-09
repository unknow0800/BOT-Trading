"""
main_alpaca.py — Bot de trading actions US (GAFA) via Alpaca paper trading

Usage :
    python alpaca/main_alpaca.py

Stratégie :
  - Univers : AAPL, MSFT, GOOG, AMZN, META, NVDA, AMD, SPY, QQQ
  - Timeframe tendance : Daily
  - Timeframe signal   : Hourly
  - Risk : 1% par trade, SL ATR x2, TP ATR x4, max 3 positions
  - Confirmation DL si modèle disponible
"""

import sys, os, time, json, csv
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from datetime import datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from config_alpaca import (ALL_SYMBOLS, SLEEP_SECONDS, LOG_DIR,
                       TRADES_LOG_FILE, PERF_LOG_FILE,
                       MAX_POSITIONS, MAX_DRAWDOWN_PCT,
                       ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
                       RISK_PER_TRADE_PCT, POSITION_SIZE_CAP,
                       MANAGE_POSITIONS_OUTSIDE_MARKET_HOURS,
                       ENABLE_NO_OVERNIGHT, NO_OVERNIGHT_EXIT_TIME_ET,
                       ENABLE_MARKET_REGIME_FILTER, MARKET_REGIME_SYMBOLS,
                       LOSS_COOLDOWN_HOURS, COOLDOWN_STATE_FILE,
                       REQUIRE_BACKTEST_APPROVAL, BACKTEST_APPROVAL_FILE,
                       ENABLE_MARKET_INTELLIGENCE,
                       MIN_TECHNICAL_SCORE_TO_BUY, WATCHLIST_TOP_N)
from data_alpaca import (get_clients, get_account_info, get_bars,
                         get_latest_price, get_open_positions,
                         place_market_order, place_bracket_order, close_position,
                         get_news_client)
from market_intelligence import build_market_contexts
from strategy_alpaca import (get_trend_bias_daily, get_signal_hourly,
                              calc_position_size, is_market_hours,
                              get_market_status, analyze_hourly_signal)
from indicators import add_all_indicators
from decision_logger import init_decisions_csv, log_decision
from portfolio_intelligence import (
    combined_setup_score,
    context_score,
    log_setup_ranking,
    log_trade_review,
    log_watchlist,
    portfolio_block_reason,
    rank_setups,
    symbol_group,
    update_adaptive_threshold,
)

os.makedirs(LOG_DIR, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("AlpacaBot")


# ==========================
# JOURNAL DES TRADES
# ==========================

def init_trades_csv():
    if not os.path.exists(TRADES_LOG_FILE):
        with open(TRADES_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'date', 'symbol', 'side', 'qty', 'price',
                'pnl_usd', 'pnl_pct', 'reason'
            ])


def log_trade(symbol, side, qty, price, entry_price=0, reason=''):
    pnl_usd = pnl_pct = 0
    if side == 'sell' and entry_price > 0:
        pnl_usd = (price - entry_price) * qty
        pnl_pct = (price - entry_price) / entry_price * 100
    with open(TRADES_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            symbol, side, round(qty, 4), round(price, 2),
            round(pnl_usd, 2), round(pnl_pct, 3), reason
        ])
    if side == 'sell':
        sign = '+' if pnl_usd >= 0 else ''
        log.info(f"[TRADE] {symbol} ferme | PnL: {sign}{pnl_usd:.2f}$ ({sign}{pnl_pct:.2f}%)")


def _load_cooldown_until() -> datetime | None:
    if not os.path.exists(COOLDOWN_STATE_FILE):
        return None
    try:
        with open(COOLDOWN_STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f).get("cooldown_until_utc")
        return datetime.fromisoformat(raw) if raw else None
    except Exception:
        return None


def _save_cooldown_until(until_utc: datetime, reason: str):
    os.makedirs(os.path.dirname(COOLDOWN_STATE_FILE), exist_ok=True)
    with open(COOLDOWN_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "cooldown_until_utc": until_utc.isoformat(),
            "reason": reason,
        }, f, indent=2)


def _activate_loss_cooldown(symbol: str, now_utc: datetime):
    if LOSS_COOLDOWN_HOURS <= 0:
        return
    until = now_utc + timedelta(hours=LOSS_COOLDOWN_HOURS)
    _save_cooldown_until(until, f"loss:{symbol}")
    log.warning(f"[COOLDOWN] Pause nouveaux trades jusqu'a {until:%Y-%m-%d %H:%M UTC} apres perte {symbol}")


def _cooldown_active(now_utc: datetime) -> tuple[bool, str]:
    until = _load_cooldown_until()
    if not until:
        return False, ""
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    if now_utc < until:
        return True, f"cooldown_perte jusqu'a {until:%Y-%m-%d %H:%M UTC}"
    return False, ""


def _parse_et_time(value: str) -> dt_time:
    hour, minute = value.split(":", 1)
    return dt_time(int(hour), int(minute))


def _is_no_overnight_exit_window(now_utc: datetime) -> bool:
    if not ENABLE_NO_OVERNIGHT:
        return False
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    return now_et.time() >= _parse_et_time(NO_OVERNIGHT_EXIT_TIME_ET)


def _market_regime_allows_long(bars_daily: dict, bars_hourly: dict) -> tuple[bool, str]:
    if not ENABLE_MARKET_REGIME_FILTER:
        return True, "regime_filter_off"
    reasons = []
    for symbol in MARKET_REGIME_SYMBOLS:
        df_d = bars_daily.get(symbol)
        df_h = bars_hourly.get(symbol)
        if df_d is None or df_h is None or len(df_h) < 50:
            return False, f"regime_data_missing:{symbol}"
        trend = get_trend_bias_daily(df_d)
        df_h_ind = add_all_indicators(df_h.copy())
        if df_h_ind.empty:
            return False, f"regime_indicators_missing:{symbol}"
        if "sma_slow" not in df_h_ind.columns:
            df_h_ind["sma_slow"] = df_h_ind["close"].rolling(min(50, len(df_h_ind))).mean()
        clean = df_h_ind.dropna(subset=["sma_slow"])
        if clean.empty:
            return False, f"regime_sma_missing:{symbol}"
        last = clean.iloc[-1]
        above_slow = float(last["close"]) > float(last["sma_slow"])
        if trend != "bull" or not above_slow:
            return False, f"regime_block:{symbol} trend={trend} above_sma50={above_slow}"
        reasons.append(f"{symbol}=bull")
    return True, " ".join(reasons)


def _backtest_approval_allows_trading() -> tuple[bool, str]:
    if not REQUIRE_BACKTEST_APPROVAL:
        return True, "backtest_gate_off"
    if not os.path.exists(BACKTEST_APPROVAL_FILE):
        return False, f"backtest_required:{BACKTEST_APPROVAL_FILE}"
    try:
        with open(BACKTEST_APPROVAL_FILE, "r", encoding="utf-8") as f:
            approval = json.load(f)
    except Exception as exc:
        return False, f"backtest_approval_invalid:{exc}"
    if not approval.get("approved"):
        return False, "backtest_not_approved"
    min_trades = int(approval.get("min_trades", 30))
    profit_factor = float(approval.get("profit_factor", 0))
    max_drawdown = float(approval.get("max_drawdown_pct", 100))
    if int(approval.get("trades", 0)) < min_trades:
        return False, f"backtest_sample_too_small:{approval.get('trades', 0)}/{min_trades}"
    if profit_factor < 1.15:
        return False, f"backtest_pf_too_low:{profit_factor:.2f}"
    if max_drawdown > 8.0:
        return False, f"backtest_dd_too_high:{max_drawdown:.1f}%"
    return True, f"backtest_ok pf={profit_factor:.2f} dd={max_drawdown:.1f}%"


# ==========================
# ÉTAT DES POSITIONS OUVERTES
# ==========================

class PositionState:
    """Suit les SL/TP de chaque position ouverte."""
    def __init__(self):
        self.positions = {}  # symbol -> {entry, sl, tp1, tp2, qty, atr}

    def open(self, symbol, entry, atr, qty, setup=None):
        sl  = entry - ATR_SL_MULTIPLIER * atr
        tp1 = entry + ATR_TP_MULTIPLIER * atr
        tp2 = entry + ATR_TP_MULTIPLIER * 2 * atr
        self.positions[symbol] = {
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2,
            'qty': qty, 'atr': atr, 'scaled': False,
            'trailing_sl': sl, 'setup': setup or {}
        }
        log.info(f"[POS] {symbol} ouvert | entry={entry:.2f} SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f}")

    def close(self, symbol):
        self.positions.pop(symbol, None)

    def update_trailing(self, symbol, price):
        """Met à jour le trailing SL et retourne l'action à faire."""
        if symbol not in self.positions:
            return 'hold'
        p = self.positions[symbol]
        if p.get('setup', {}).get('broker_bracket'):
            return 'hold'

        # SL touché
        if price <= p['trailing_sl']:
            return 'exit_sl'

        # TP2 touché
        if price >= p['tp2']:
            return 'exit_tp2'

        # TP1 : scale-out 50%
        if not p['scaled'] and price >= p['tp1']:
            p['scaled'] = True
            p['trailing_sl'] = max(p['trailing_sl'], p['entry'])  # breakeven
            return 'scale_out'

        # Trailing stop (après TP1)
        if p['scaled']:
            new_trail = price - ATR_SL_MULTIPLIER * p['atr']
            if new_trail > p['trailing_sl']:
                p['trailing_sl'] = new_trail

        return 'hold'

    def has(self, symbol):
        return symbol in self.positions

    def count(self):
        return len(self.positions)


# ==========================
# BOUCLE PRINCIPALE
# ==========================

def main_loop():
    log.info("=" * 60)
    log.info("Bot Alpaca GAFA - Paper Trading")
    log.info("=" * 60)

    trading_client, data_client = get_clients()
    news_client = get_news_client() if ENABLE_MARKET_INTELLIGENCE else None
    acc = get_account_info(trading_client)
    log.info(f"Compte : ${acc['equity']:,.2f} | Cash : ${acc['cash']:,.2f}")

    init_trades_csv()
    init_decisions_csv()
    pos_state  = PositionState()
    peak_equity = acc['equity']
    adaptive_state = update_adaptive_threshold()
    log.info(
        f"[ADAPT] Offset seuil technique: {adaptive_state.get('offset', 0.0):+.2f} | "
        f"{adaptive_state.get('reason', '')}"
    )

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            log.info(f"\n{'='*60}")
            log.info(f"[{now_utc.strftime('%Y-%m-%d %H:%M')} UTC]")

            # Compte
            acc = get_account_info(trading_client)
            equity = acc['equity']
            log.info(f"Equity: ${equity:,.2f} | PnL: {'+' if acc['pnl']>=0 else ''}{acc['pnl']:.2f}$")
            adaptive_state = update_adaptive_threshold()
            effective_buy_score = MIN_TECHNICAL_SCORE_TO_BUY + adaptive_state.get("offset", 0.0)

            # Drawdown guard
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd >= MAX_DRAWDOWN_PCT:
                log.error(f"[DD] Drawdown max atteint ({dd:.1%}) - arret du bot")
                break

            # Recuperer les donnees
            log.info(f"Telechargement des donnees ({len(ALL_SYMBOLS)} symboles)...")
            bars_daily  = get_bars(data_client, ALL_SYMBOLS, 'Day',  days_back=160, logger=log)
            bars_hourly = get_bars(data_client, ALL_SYMBOLS, 'Hour', days_back=20, logger=log)
            log.info(
                f"Donnees recues: Daily {len(bars_daily)}/{len(ALL_SYMBOLS)} | "
                f"Hourly {len(bars_hourly)}/{len(ALL_SYMBOLS)}"
            )
            market_contexts = {}
            if ENABLE_MARKET_INTELLIGENCE:
                market_contexts = build_market_contexts(
                    news_client, ALL_SYMBOLS, bars_daily, bars_hourly, logger=log
                )

            # Positions actuelles sur Alpaca
            open_pos = get_open_positions(trading_client)

            # Sync pos_state avec positions reelles
            for sym in list(pos_state.positions.keys()):
                if sym not in open_pos:
                    log.info(f"[SYNC] {sym} fermé en dehors du bot")
                    state = pos_state.positions[sym]
                    try:
                        price = get_latest_price(data_client, sym)
                    except Exception:
                        price = state['entry']
                    qty = state['qty']
                    entry = state['entry']
                    pnl_usd = (price - entry) * qty
                    pnl_pct = (price - entry) / entry * 100 if entry else 0.0
                    reason = "broker_bracket_or_external"
                    log_trade(sym, 'sell', qty, price, entry, reason)
                    log_trade_review(sym, "close", qty, price, entry, pnl_usd, pnl_pct, state.get('setup', {}), reason)
                    if pnl_usd < 0:
                        _activate_loss_cooldown(sym, now_utc)
                    pos_state.close(sym)
            for sym, broker_pos in open_pos.items():
                if sym in pos_state.positions or sym not in bars_hourly:
                    continue
                try:
                    df_h_ind = add_all_indicators(bars_hourly[sym].copy())
                    atr = float(df_h_ind['atr'].iloc[-1]) if not df_h_ind.empty else broker_pos['avg_price'] * 0.02
                    pos_state.open(sym, broker_pos['avg_price'], atr, broker_pos['qty'])
                    log.info(f"[SYNC] {sym} position Alpaca reprise dans le suivi local")
                except Exception as sync_exc:
                    log.warning(f"[SYNC] Impossible de reprendre {sym}: {sync_exc}")

            market_open, market_reason = get_market_status()

            if market_open and _is_no_overnight_exit_window(now_utc) and pos_state.count() > 0:
                log.warning(f"[NO_OVERNIGHT] Fermeture forcee avant cloture ET ({NO_OVERNIGHT_EXIT_TIME_ET})")
                for symbol in list(pos_state.positions.keys()):
                    try:
                        price = get_latest_price(data_client, symbol)
                    except Exception:
                        price = pos_state.positions[symbol]['entry']
                    ok = close_position(trading_client, symbol)
                    if ok:
                        entry = pos_state.positions[symbol]['entry']
                        qty = pos_state.positions[symbol]['qty']
                        setup = pos_state.positions[symbol].get('setup', {})
                        log_trade(symbol, 'sell', qty, price, entry, 'no_overnight')
                        pnl_usd = (price - entry) * qty
                        pnl_pct = (price - entry) / entry * 100 if entry else 0.0
                        log_trade_review(symbol, "close", qty, price, entry, pnl_usd, pnl_pct, setup, "no_overnight")
                        pos_state.close(symbol)

            # --- GESTION DES POSITIONS OUVERTES ---
            if not market_open and not MANAGE_POSITIONS_OUTSIDE_MARKET_HOURS and pos_state.count() > 0:
                log.info(f"[FILTRE] Hors heures de marche US - gestion positions suspendue | {market_reason}")
            for symbol in list(pos_state.positions.keys()):
                if symbol not in bars_hourly:
                    continue
                if not market_open and not MANAGE_POSITIONS_OUTSIDE_MARKET_HOURS:
                    continue
                price = get_latest_price(data_client, symbol)
                action = pos_state.update_trailing(symbol, price)

                if action == 'exit_sl':
                    log.info(f"[SL] {symbol} stop-loss atteint a ${price:.2f}")
                    ok = close_position(trading_client, symbol)
                    if ok:
                        entry = pos_state.positions[symbol]['entry']
                        qty   = pos_state.positions[symbol]['qty']
                        setup = pos_state.positions[symbol].get('setup', {})
                        log_trade(symbol, 'sell', qty, price, entry, 'stop_loss')
                        pnl_usd = (price - entry) * qty
                        pnl_pct = (price - entry) / entry * 100 if entry else 0.0
                        log_trade_review(symbol, "close", qty, price, entry, pnl_usd, pnl_pct, setup, "stop_loss")
                        _activate_loss_cooldown(symbol, now_utc)
                        pos_state.close(symbol)

                elif action == 'exit_tp2':
                    log.info(f"[TP2] {symbol} take-profit 2 atteint a ${price:.2f}")
                    ok = close_position(trading_client, symbol)
                    if ok:
                        entry = pos_state.positions[symbol]['entry']
                        qty   = pos_state.positions[symbol]['qty']
                        setup = pos_state.positions[symbol].get('setup', {})
                        log_trade(symbol, 'sell', qty, price, entry, 'tp2')
                        pnl_usd = (price - entry) * qty
                        pnl_pct = (price - entry) / entry * 100 if entry else 0.0
                        log_trade_review(symbol, "close", qty, price, entry, pnl_usd, pnl_pct, setup, "tp2")
                        pos_state.close(symbol)

                elif action == 'scale_out':
                    qty_sell = round(pos_state.positions[symbol]['qty'] * 0.5, 4)
                    log.info(f"[TP1] {symbol} scale-out 50% a ${price:.2f}")
                    ok = place_market_order(trading_client, symbol, 'sell', qty_sell)
                    if ok:
                        entry = pos_state.positions[symbol]['entry']
                        setup = pos_state.positions[symbol].get('setup', {})
                        log_trade(symbol, 'sell', qty_sell, price, entry, 'tp1_scaleout')
                        pnl_usd = (price - entry) * qty_sell
                        pnl_pct = (price - entry) / entry * 100 if entry else 0.0
                        log_trade_review(symbol, "scale_out", qty_sell, price, entry, pnl_usd, pnl_pct, setup, "tp1")
                        pos_state.positions[symbol]['qty'] -= qty_sell

            # --- RECHERCHE DE NOUVEAUX SIGNAUX ---
            if not market_open:
                log.info(f"[FILTRE] Hors heures de marche US - pas de nouveaux trades | {market_reason}")
            elif _is_no_overnight_exit_window(now_utc):
                log.info(f"[FILTRE] Fenetre no-overnight active - pas de nouveaux trades")
            else:
                if pos_state.count() >= MAX_POSITIONS:
                    log.info(f"[FILTRE] Max positions atteint ({MAX_POSITIONS}) - analyse maintenue, achats bloques")
                cooldown_on, cooldown_reason = _cooldown_active(now_utc)
                regime_ok, regime_reason = _market_regime_allows_long(bars_daily, bars_hourly)
                backtest_ok, backtest_reason = _backtest_approval_allows_trading()
                if cooldown_on:
                    log.info(f"[FILTRE] {cooldown_reason} - achats bloques")
                if not regime_ok:
                    log.info(f"[FILTRE] Regime marche defensif - achats bloques | {regime_reason}")
                if not backtest_ok:
                    log.info(f"[FILTRE] Backtest non valide - achats bloques | {backtest_reason}")
                candidates = []
                for symbol in ALL_SYMBOLS:
                    if pos_state.has(symbol):
                        continue
                    if symbol not in bars_daily or symbol not in bars_hourly:
                        continue

                    df_d = bars_daily[symbol]
                    df_h = bars_hourly[symbol]

                    if len(df_h) < 10:
                        continue

                    # Indicateurs pour ATR
                    df_h_ind = add_all_indicators(df_h.copy())
                    if df_h_ind.empty:
                        continue

                    atr   = df_h_ind['atr'].iloc[-1]
                    price = df_h['close'].iloc[-1]

                    trend  = get_trend_bias_daily(df_d)
                    analysis = analyze_hourly_signal(df_h, trend,
                                                     in_position=False,
                                                     symbol=symbol,
                                                     min_score_to_buy=effective_buy_score)
                    signal = analysis["signal"]
                    context = market_contexts.get(symbol)
                    ctx_score = context_score(context)
                    combined_score = combined_setup_score(analysis, context)
                    blocked, block_reason, corr = portfolio_block_reason(symbol, pos_state, bars_hourly)
                    context_blocked = bool(context and not context.allow_long)
                    tradable = (
                        signal == 'enter_long'
                        and not blocked
                        and not context_blocked
                        and not cooldown_on
                        and regime_ok
                        and backtest_ok
                        and pos_state.count() < MAX_POSITIONS
                    )
                    candidate = {
                        "symbol": symbol,
                        "group": symbol_group(symbol),
                        "price": price,
                        "atr": atr,
                        "trend": trend,
                        "signal": signal,
                        "technical_score": analysis["score"],
                        "context_score": ctx_score,
                        "combined_score": combined_score,
                        "analysis_reason": analysis["reason"],
                        "context_reason": context.reason if context else "",
                        "context": context,
                        "analysis": analysis,
                        "tradable": tradable,
                        "decision": "watch",
                        "block_reason": (
                            block_reason if blocked else
                            "context_filter" if context_blocked else
                            cooldown_reason if cooldown_on else
                            regime_reason if not regime_ok else
                            backtest_reason if not backtest_ok else
                            ""
                        ),
                        "corr_to_positions": corr,
                    }
                    candidates.append(candidate)
                    ctx_msg = ""
                    if context:
                        ctx_msg = (
                            f" | Ctx: {context.score:+.2f} "
                            f"conf={context.confidence:.2f} risk={context.risk_flag}"
                        )

                    log.info(
                        f"{symbol} | ${price:.2f} | Tendance: {trend} | "
                        f"Signal: {signal} | Tech: {analysis['score']:.2f} "
                        f"{analysis['grade']} | Combo: {combined_score:.2f} | "
                        f"{analysis['reason']} | Bloc: {candidate['block_reason'] or 'ok'}{ctx_msg}"
                    )

                for candidate in candidates:
                    if candidate["tradable"]:
                        candidate["decision"] = "eligible"
                    elif candidate["signal"] == "enter_long":
                        candidate["decision"] = "blocked"
                    else:
                        candidate["decision"] = "watch"
                ranked = rank_setups(candidates)
                log_setup_ranking(ranked)
                log_watchlist(ranked)
                if ranked:
                    top_msg = " | ".join(
                        f"#{i+1} {row['symbol']} {row['combined_score']:.2f}"
                        for i, row in enumerate(ranked[:WATCHLIST_TOP_N])
                    )
                    log.info(f"[WATCHLIST] {top_msg}")

                for candidate in ranked:
                    symbol = candidate["symbol"]
                    context = candidate["context"]
                    signal = candidate["signal"]
                    trend = candidate["trend"]
                    price = candidate["price"]
                    atr = candidate["atr"]

                    if signal != 'enter_long':
                        log_decision(symbol, price, trend, signal, "refused",
                                     candidate["analysis_reason"], context)
                        continue
                    if pos_state.count() >= MAX_POSITIONS:
                        log_decision(symbol, price, trend, signal, "refused",
                                     "max_positions", context)
                        continue
                    if candidate["block_reason"]:
                        log_decision(symbol, price, trend, signal, "refused",
                                     f"{candidate['block_reason']} | {candidate['analysis_reason']}", context)
                        continue

                    qty = calc_position_size(
                        equity=acc['cash'],
                        price=price,
                        atr=atr,
                        risk_pct=RISK_PER_TRADE_PCT,
                        atr_mult=ATR_SL_MULTIPLIER,
                        position_cap=POSITION_SIZE_CAP
                    )
                    if qty * price < 1:
                        log_decision(symbol, price, trend, signal, "refused",
                                     f"notional_too_small | {candidate['analysis_reason']}", context)
                        log.info(f"[SKIP] {symbol} notional trop faible")
                        continue

                    log_decision(symbol, price, trend, signal, "accepted",
                                 f"ranked_buy | combo={candidate['combined_score']:.2f} | {candidate['analysis_reason']}",
                                 context)
                    log.info(
                        f"[BUY] #{ranked.index(candidate)+1} {symbol} | "
                        f"combo={candidate['combined_score']:.2f} tech={candidate['technical_score']:.2f} | "
                        f"qty={qty:.4f} | ~${qty*price:.2f}"
                    )
                    stop_price = max(0.01, price - ATR_SL_MULTIPLIER * atr)
                    take_profit_price = price + ATR_TP_MULTIPLIER * atr
                    order = place_bracket_order(
                        trading_client,
                        symbol,
                        qty,
                        take_profit_price=take_profit_price,
                        stop_price=stop_price,
                        logger=log,
                    )
                    if order:
                        setup = {
                            "technical_score": candidate["technical_score"],
                            "context_score": candidate["context_score"],
                            "combined_score": candidate["combined_score"],
                            "analysis_reason": candidate["analysis_reason"],
                            "broker_bracket": True,
                        }
                        pos_state.open(symbol, price, atr, qty, setup=setup)
                        log_trade(symbol, 'buy', qty, price, reason=signal)
                        log_trade_review(symbol, "open", qty, price, price, 0.0, 0.0, setup, "entry")
                        acc = get_account_info(trading_client)  # refresh

            # --- RÉSUMÉ ---
            log.info(f"\nPositions ouvertes : {list(pos_state.positions.keys())}")
            log.info(f"Equity: ${acc['equity']:,.2f} | DD: {dd:.2%}")

        except KeyboardInterrupt:
            log.info("Bot arrêté manuellement.")
            break
        except Exception as e:
            log.error(f"Exception: {e}")
            import traceback
            traceback.print_exc()

        log.info(f"Pause {SLEEP_SECONDS//60} min...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main_loop()
