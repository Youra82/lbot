# src/lbot/utils/trade_manager.py
import logging
import time
import ccxt
import numpy as np

from lbot.utils.telegram import send_message
from lbot.utils.lstm_model import create_ann_features

def babysit_open_position(exchange, params, db_get_state, db_set_state, telegram_config, logger):
    symbol = params['market']['symbol']
    account_name = exchange.account.get('name', 'Standard-Account')
    strategy_id = f"{symbol.replace('/', '').replace(':', '')}_{params['market']['timeframe']}"
    
    logger.info(f"[Babysitter] Prüfe {symbol}...")
    try:
        position = exchange.fetch_open_positions(symbol)
        position = position[0] if position else None
        failure_key = f"babysitter_failure_count_{strategy_id}"

        if position:
            trigger_orders = exchange.fetch_open_trigger_orders(symbol)
            if not trigger_orders:
                failure_count = int(db_get_state(account_name, symbol, params['market']['timeframe'], failure_key, default=0))
                if failure_count >= 3:
                    message = f"🚨 *KRITISCHER DAUERALARM* für *{account_name}*\n\nPosition für *{symbol}* ist weiterhin UNGESCHÜTZT.\n\n*MANUELLES EINGREIFEN ERFORDERLICH!*"
                    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
                    return
                try:
                    manage_existing_position(exchange, position, params, logger)
                    db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, 0)
                    message = f"✅ *Sicherheitsnetz Wiederhergestellt*\n\nDer Babysitter hat eine ungeschützte Position für *{symbol}* gefunden und SL/TP erfolgreich neu platziert."
                    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
                except Exception as e:
                    failure_count += 1
                    db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, failure_count)
                    if failure_count < 3: message = f"🟡 *Babysitter WARNUNG* ({account_name})\n\nKonnte SL/TP für *{symbol}* nicht setzen (Versuch {failure_count}/3).\n\nFehler: _{e}_"
                    else: message = f"🚨 *KRITISCHER ALARM* ({account_name})\n\nKonnte SL/TP für *{symbol}* nach 3 Versuchen nicht setzen. Position ist UNGESCHÜTZT.\n\n*MANUELLES EINGREIFEN ERFORDERLICH!*"
                    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
            else:
                db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, 0)
        else:
            db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, 0)
    except Exception as e:
        logger.error(f"🚨 [Babysitter] Schwerer Fehler bei der Überwachung von {symbol}: {e}")

def housekeeper_routine(exchange, symbol, logger):
    logger.info(f"Starte radikale Aufräum-Routine für {symbol}...")
    try:
        trigger_orders = exchange.fetch_open_trigger_orders(symbol)
        if trigger_orders:
            for order in trigger_orders:
                try: exchange.cancel_trigger_order(order['id'], symbol)
                except Exception: pass
    except Exception: pass
    logger.info("Aufräumen abgeschlossen.")

def manage_existing_position(exchange, position, params, logger):
    logger.info(f"Position ({position['side']}) gefunden. Platziere/aktualisiere SL/TP...")
    symbol = position['symbol']
    p = params['risk']
    sl_pct = (p['risk_per_trade_pct'] / 100) / p['leverage']
    sl_distance = float(position['entryPrice']) * sl_pct
    if position['side'] == 'long':
        stop_loss_price, take_profit_price = float(position['entryPrice']) - sl_distance, float(position['entryPrice']) + sl_distance * p['risk_reward_ratio']
        close_side = 'sell'
    else:
        stop_loss_price, take_profit_price = float(position['entryPrice']) + sl_distance, float(position['entryPrice']) - sl_distance * p['risk_reward_ratio']
        close_side = 'buy'
    stop_loss_price_rounded = float(exchange.exchange.price_to_precision(symbol, stop_loss_price))
    take_profit_price_rounded = float(exchange.exchange.price_to_precision(symbol, take_profit_price))
    exchange.place_trigger_market_order(symbol, close_side, float(position['contracts']), take_profit_price_rounded, {'reduceOnly': True})
    exchange.place_trigger_market_order(symbol, close_side, float(position['contracts']), stop_loss_price_rounded, {'reduceOnly': True})
    logger.info(f"✅ SL/TP platziert/aktualisiert auf SL: {stop_loss_price_rounded}, TP: {take_profit_price_rounded}.")

def check_and_open_new_position(exchange, model, scaler, params, current_balance, db_set_state, telegram_config, logger):
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    account_name = exchange.account.get('name', 'Standard-Account')
    
    logger.info("Keine Position offen. Suche nach neuen Signalen...")
    
    SEQUENCE_LENGTH = 24
    data = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=SEQUENCE_LENGTH + 50)
    data_with_features = create_ann_features(data[['open', 'high', 'low', 'close', 'volume']])

    if len(data_with_features) < SEQUENCE_LENGTH:
        logger.warning(f"Nicht genug Daten ({len(data_with_features)}) für eine Sequenz von {SEQUENCE_LENGTH}.")
        return
        
    feature_columns_to_use = data_with_features.columns.drop('close', errors='ignore')
    latest_sequence_unscaled = data_with_features.iloc[-SEQUENCE_LENGTH:][feature_columns_to_use]
    
    if latest_sequence_unscaled.isnull().values.any():
        logger.warning("Daten für Vorhersage enthalten NaN-Werte.")
        return

    latest_sequence_scaled = scaler.transform(latest_sequence_unscaled)
    input_data = np.expand_dims(latest_sequence_scaled, axis=0)
    prediction = model.predict(input_data, verbose=0)[0][0]
    logger.info(f"LSTM-Modell-Vorhersage: {prediction:.3f}")
    
    pred_threshold = params['strategy']['prediction_threshold']
    side = 'buy' if prediction >= pred_threshold and params['behavior']['use_longs'] else None
    
    if side:
        p = params['risk']
        ticker = exchange.fetch_ticker(symbol)
        entry_price = ticker['last']
        risk_amount_usd = current_balance * (p['risk_per_trade_pct'] / 100)
        sl_distance_pct = 0.01
        notional_value = risk_amount_usd / sl_distance_pct
        market_info = exchange.markets.get(symbol, {})
        min_cost = market_info.get('limits', {}).get('cost', {}).get('min', 5.0)
        min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0.001)
        if notional_value < min_cost: notional_value = min_cost * 1.05
        amount = notional_value / entry_price
        if amount < min_amount: amount = min_amount * 1.05
        stop_loss_price = entry_price * (1 - sl_distance_pct) if side == 'buy' else entry_price * (1 + sl_distance_pct)
        take_profit_price = entry_price + (entry_price - stop_loss_price) * p['risk_reward_ratio'] if side == 'buy' else entry_price - (stop_loss_price - entry_price) * p['risk_reward_ratio']
        stop_loss_price_rounded = float(exchange.exchange.price_to_precision(symbol, stop_loss_price))
        take_profit_price_rounded = float(exchange.exchange.price_to_precision(symbol, take_profit_price))
        exchange.set_margin_mode(symbol, p['margin_mode'])
        exchange.set_leverage(symbol, p['leverage'])
        exchange.create_market_order(symbol, side, amount, params={'leverage': p['leverage'], 'marginMode': p['margin_mode']})
        time.sleep(2)
        exchange.place_trigger_market_order(symbol, 'sell' if side == 'buy' else 'buy', amount, take_profit_price_rounded, {'reduceOnly': True})
        exchange.place_trigger_market_order(symbol, 'sell' if side == 'buy' else 'buy', amount, stop_loss_price_rounded, {'reduceOnly': True})
        db_set_state(account_name, symbol, timeframe, 'initial_stop_loss', stop_loss_price_rounded)
        db_set_state(account_name, symbol, timeframe, 'trailing_stop_active', 'False')
        message = f"🧠 LSTM Signal für *{account_name}* ({symbol}, {side.upper()})\n- Entry @ Market (≈${entry_price:.4f})\n- SL: ${stop_loss_price_rounded:.4f}\n- TP: ${take_profit_price_rounded:.4f}"
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)

def full_trade_cycle(exchange, model, scaler, params, current_balance, db_set_state, telegram_config, logger):
    symbol = params['market']['symbol']
    account_name = exchange.account.get('name', 'Standard-Account')
    try:
        position = exchange.fetch_open_positions(symbol)
        position = position[0] if position else None
        if position:
             manage_existing_position(exchange, position, params, logger)
        else:
            housekeeper_routine(exchange, symbol, logger)
            check_and_open_new_position(exchange, model, scaler, params, current_balance, db_set_state, telegram_config, logger)
    except ccxt.InsufficientFunds as e:
        message = f"🟡 *Kein Guthaben für Trade* ({account_name}, {symbol})"
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
    except Exception as e:
        message = f"🚨 *Kritischer Fehler* ({account_name}, {symbol})\n\n_{e}_"
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
