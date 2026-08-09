import math
from binance.client import Client
from data import place_order
from risk import round_step_size
from logger import BotLogger


def place_buy_order(client, symbol, quantity, logger):
    try:
        result = place_order(client, symbol, 'BUY', quantity)
        if 'orderId' in result:
            logger.log_info(f"ACHAT {quantity} {symbol} | OrderID: {result['orderId']}")
            return result
        else:
            logger.log_error(f"Erreur achat : {result}")
            return None
    except Exception as e:
        logger.log_error(f"Exception achat : {e}")
        return None


def place_sell_order(client, symbol, quantity, logger):
    try:
        result = place_order(client, symbol, 'SELL', quantity)
        if 'orderId' in result:
            logger.log_info(f"VENTE {quantity} {symbol} | OrderID: {result['orderId']}")
            return result
        else:
            logger.log_error(f"Erreur vente : {result}")
            return None
    except Exception as e:
        logger.log_error(f"Exception vente : {e}")
        return None


def open_long(client, symbol, quantity, min_notional, lot_step, entry_price, logger):
    notional = quantity * entry_price
    if notional < min_notional:
        logger.log_warning(f"Notional trop faible ({notional:.2f} < {min_notional:.2f})")
        return False
    quantity = round_step_size(quantity, lot_step)
    if quantity <= 0:
        logger.log_warning("Quantité nulle après arrondi")
        return False
    order = place_buy_order(client, symbol, quantity, logger)
    return order is not None


def close_long(client, symbol, quantity, lot_step, reason, logger):
    quantity = round_step_size(quantity, lot_step)
    if quantity <= 0:
        logger.log_warning("Qty nulle lors fermeture")
        return False
    logger.log_info(f"Fermeture {quantity} BTC | Raison : {reason}")
    order = place_sell_order(client, symbol, quantity, logger)
    return order is not None


def get_fill_price(order: dict) -> float:
    if order is None:
        return 0.0
    fills = order.get('fills', [])
    if not fills:
        return float(order.get('price', 0))
    total_qty = sum(float(f['qty']) for f in fills)
    total_val = sum(float(f['price']) * float(f['qty']) for f in fills)
    return total_val / total_qty if total_qty > 0 else 0.0
