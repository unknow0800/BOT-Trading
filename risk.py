import math
from dataclasses import dataclass, field
from typing import Optional
from config import (
    RISK_PER_TRADE_PCT, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    BREAKEVEN_TRIGGER_R, SCALE_OUT_PCT, MAX_DRAWDOWN_PCT
)


# ==========================
# ÉTAT DE LA POSITION
# ==========================

@dataclass
class Position:
    """Représente une position ouverte avec toutes ses métriques de risque."""
    symbol: str
    entry_price: float
    quantity: float               # Quantité totale achetée
    atr_at_entry: float           # ATR au moment de l'entrée
    stop_loss: float = 0.0        # SL ATR initial
    take_profit_1: float = 0.0    # TP1 : scale-out partiel
    take_profit_2: float = 0.0    # TP2 : fermeture totale
    trailing_stop: float = 0.0    # SL trailing (mis à jour)
    breakeven_moved: bool = False  # SL déplacé au breakeven ?
    scaled_out: bool = False       # 50% vendu au TP1 ?
    remaining_qty: float = 0.0    # Quantité restante après scale-out
    r_multiple: float = 0.0       # Gain actuel en unités de R

    def __post_init__(self):
        if self.stop_loss == 0.0:
            self.stop_loss = self.entry_price - ATR_SL_MULTIPLIER * self.atr_at_entry
        if self.take_profit_1 == 0.0:
            self.take_profit_1 = self.entry_price + ATR_TP_MULTIPLIER * self.atr_at_entry
        if self.take_profit_2 == 0.0:
            self.take_profit_2 = self.entry_price + (ATR_TP_MULTIPLIER * 2) * self.atr_at_entry
        if self.trailing_stop == 0.0:
            self.trailing_stop = self.stop_loss
        if self.remaining_qty == 0.0:
            self.remaining_qty = self.quantity

    @property
    def initial_risk(self) -> float:
        """Risque initial en USDT (1R)."""
        return self.entry_price - self.stop_loss

    @property
    def is_open(self) -> bool:
        return self.quantity > 0


# ==========================
# CALCUL DE LA TAILLE
# ==========================

def calc_position_size(entry_price: float,
                       atr: float,
                       equity_usdt: float,
                       lot_step: float = 0.00001) -> float:
    """
    Position size basée sur le risque ATR :
    qty = (equity * risk_pct) / (ATR_multiplier * ATR)

    Arrondit selon le stepSize Binance.
    """
    if equity_usdt <= 0 or atr <= 0:
        return 0.0

    risk_amount = equity_usdt * RISK_PER_TRADE_PCT
    per_unit_risk = ATR_SL_MULTIPLIER * atr

    qty = risk_amount / per_unit_risk
    qty = round_step_size(qty, lot_step)
    return qty


def round_step_size(quantity: float, step_size: float) -> float:
    """Arrondit à la précision du stepSize Binance."""
    if step_size <= 0:
        return quantity
    precision = max(0, int(round(-math.log10(step_size), 0)))
    return float(f"{quantity:.{precision}f}")


# ==========================
# GESTION DYNAMIQUE DU SL
# ==========================

def update_position(position: Position, current_price: float) -> dict:
    """
    Met à jour le trailing stop et détecte les événements de gestion.

    Retourne un dict d'actions à effectuer :
    {
        'action': 'hold' | 'scale_out' | 'move_breakeven' | 'exit_sl' | 'exit_tp2',
        'reason': str,
        'qty_to_sell': float
    }
    """
    p = position
    result = {'action': 'hold', 'reason': '', 'qty_to_sell': 0.0}

    # --- Calcul du R multiple actuel ---
    if p.initial_risk > 0:
        p.r_multiple = (current_price - p.entry_price) / p.initial_risk

    # --- 1. Stop-loss atteint ---
    if current_price <= p.trailing_stop:
        result['action'] = 'exit_sl'
        result['reason'] = f"SL trailing atteint ({current_price:.2f} <= {p.trailing_stop:.2f})"
        result['qty_to_sell'] = p.remaining_qty
        return result

    # --- 2. TP2 atteint (fermeture totale) ---
    if current_price >= p.take_profit_2:
        result['action'] = 'exit_tp2'
        result['reason'] = f"TP2 atteint ({current_price:.2f} >= {p.take_profit_2:.2f})"
        result['qty_to_sell'] = p.remaining_qty
        return result

    # --- 3. TP1 atteint : scale-out 50% ---
    if not p.scaled_out and current_price >= p.take_profit_1:
        qty_to_sell = round_step_size(p.quantity * SCALE_OUT_PCT, 0.00001)
        p.remaining_qty = p.quantity - qty_to_sell
        p.scaled_out = True
        result['action'] = 'scale_out'
        result['reason'] = f"TP1 atteint - vente de {qty_to_sell:.5f} BTC (50%)"
        result['qty_to_sell'] = qty_to_sell
        # Après scale-out, on déplace le SL au breakeven
        p.trailing_stop = max(p.trailing_stop, p.entry_price)
        p.breakeven_moved = True
        return result

    # --- 4. Déplacement breakeven à +1R ---
    if not p.breakeven_moved and p.r_multiple >= BREAKEVEN_TRIGGER_R:
        p.trailing_stop = max(p.trailing_stop, p.entry_price)
        p.breakeven_moved = True
        result['action'] = 'move_breakeven'
        result['reason'] = f"Breakeven atteint (+{p.r_multiple:.1f}R) → SL déplacé à {p.entry_price:.2f}"
        return result

    # --- 5. Trailing stop dynamique (après breakeven) ---
    if p.breakeven_moved:
        new_trail = current_price - (ATR_SL_MULTIPLIER * p.atr_at_entry)
        if new_trail > p.trailing_stop:
            p.trailing_stop = new_trail

    return result


# ==========================
# DRAWDOWN GLOBAL
# ==========================

class DrawdownGuard:
    """Surveille le drawdown global du compte et coupe le bot si nécessaire."""

    def __init__(self, initial_equity: float):
        self.peak_equity = initial_equity
        self.initial_equity = initial_equity

    def update(self, current_equity: float) -> bool:
        """
        Met à jour le peak et vérifie le drawdown.
        Retourne True si on doit STOPPER le bot.
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown >= MAX_DRAWDOWN_PCT:
            return True  # Stopper le bot
        return False

    @property
    def current_drawdown(self) -> float:
        return (self.peak_equity - self.initial_equity) / self.peak_equity
