# src/jaegerbot/utils/trade_manager.py
import logging
import time
import ccxt

from jaegerbot.utils.telegram import send_message
from jaegerbot.utils.ann_model import create_ann_features

def babysit_open_position(exchange, params, db_get_state, db_set_state, telegram_config, logger):
    """
    Überwacht eine offene Position. Eskaliert an den Nutzer, wenn eine Reparatur
    mehrfach fehlschlägt (Circuit Breaker Pattern).
    """
    symbol = params['market']['symbol']
    account_name = exchange.account.get('name', 'Standard-Account')
    strategy_id = f"{symbol.replace('/', '').replace(':', '')}_{params['market']['timeframe']}"
    
    logger.info(f"[Babysitter] Prüfe {symbol} auf ungeschützte Positionen...")

    try:
        position = exchange.fetch_open_positions(symbol)
        position = position[0] if position else None
        failure_key = f"babysitter_failure_count_{strategy_id}"

        if position:
            trigger_orders = exchange.fetch_open_trigger_orders(symbol)
            if not trigger_orders:
                failure_count = int(db_get_state(account_name, symbol, params['market']['timeframe'], failure_key, default=0))
                
                if failure_count >= 3:
                    logger.critical(f"🚨 [Babysitter] QUARANTÄNE: {symbol} ist ungeschützt. Sende Daueralarm.")
                    message = f"🚨 *KRITISCHER DAUERALARM* für *{account_name}*\n\nPosition für *{symbol}* ist weiterhin UNGESCHÜTZT. Bot kann SL/TP nicht setzen.\n\n*MANUELLES EINGREIFEN ERFORDERLICH!*"
                    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
                    return

                logger.critical(f"🚨 [Babysitter] KRITISCH: Position für {symbol} OHNE SL/TP gefunden! Versuche Reparatur (Versuch #{failure_count + 1}).")
                try:
                    manage_existing_position(exchange, position, params, logger)
                    db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, 0)
                    logger.info(f"✅ [Babysitter] Sicherheitsnetz für {symbol} erfolgreich wiederhergestellt.")
                    message = f"✅ *Sicherheitsnetz Wiederhergestellt*\n\nDer Babysitter hat eine ungeschützte Position für *{symbol}* gefunden und SL/TP erfolgreich neu platziert."
                    send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)

                except Exception as e:
                    failure_count += 1
                    db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, failure_count)
                    logger.error(f"🚨 [Babysitter] Reparatur für {symbol} fehlgeschlagen (Versuch #{failure_count}): {e}")
                    
                    if failure_count < 3:
                        message = f"🟡 *Babysitter WARNUNG* für *{account_name}*\n\nKonnte fehlenden SL/TP für *{symbol}* nicht setzen (Versuch {failure_count}/3).\n\nFehler: _{e}_"
                        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
                    else:
                        message = f"🚨 *KRITISCHER ALARM* für *{account_name}*\n\nKonnte SL/TP für *{symbol}* nach 3 Versuchen nicht setzen. Position ist jetzt UNGESCHÜTZT.\n\n*MANUELLES EINGREIFEN ERFORDERLICH!*"
                        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
            else:
                logger.info(f"[Babysitter] Position für {symbol} ist korrekt durch SL/TP geschützt.")
                db_set_state(account_name, symbol, params['market']['timeframe'], failure_key, 0)
        else:
            logger.info(f"[Babysitter] Keine offene Position für {symbol} gefunden. Alles ok.")
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
        stop_loss_price = float(position['entryPrice']) - sl_distance
        take_profit_price = float(position['entryPrice']) + sl_distance * p['risk_reward_ratio']
        close_side = 'sell'
    else:
        stop_loss_price = float(position['entryPrice']) + sl_distance
        take_profit_price = float(position['entryPrice']) - sl_distance * p['risk_reward_ratio']
        close_side = 'buy'

    stop_loss_price_rounded = float(exchange.exchange.price_to_precision(symbol, stop_loss_price))
    take_profit_price_rounded = float(exchange.exchange.price_to_precision(symbol, take_profit_price))

    exchange.place_trigger_market_order(symbol, close_side, float(position['contracts']), take_profit_price_rounded, {'reduceOnly': True})
    exchange.place_trigger_market_order(symbol, close_side, float(position['contracts']), stop_loss_price_rounded, {'reduceOnly': True})
    logger.info(f"✅ SL/TP erfolgreich platziert/aktualisiert auf SL: {stop_loss_price_rounded}, TP: {take_profit_price_rounded}.")

def check_and_open_new_position(exchange, model, scaler, params, current_balance, db_set_state, telegram_config, logger):
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    account_name = exchange.account.get('name', 'Standard-Account')
    
    logger.info("Keine Position offen. Suche nach neuen Signalen...")
    data = exchange.fetch_recent_ohlcv(symbol, timeframe)
    data_with_features = create_ann_features(data.copy())
    
    feature_cols = ['bb_width', 'obv', 'rsi', 'macd_diff', 'hour', 'day_of_week', 'returns_lag1', 'returns_lag2']
    latest_features = data_with_features.iloc[-1:][feature_cols]

    if latest_features.isnull().values.any():
        logger.warning("Daten unvollständig, überspringe."); return

    prediction = model.predict(scaler.transform(latest_features), verbose=0)[0][0]
    logger.info(f"Modell-Vorhersage: {prediction:.3f}")
    
    pred_threshold = params['strategy']['prediction_threshold']
    side = 'buy' if prediction >= pred_threshold and params['behavior']['use_longs'] else 'sell' if prediction <= (1 - pred_threshold) and params['behavior']['use_shorts'] else None
    
    if side:
        logger.info(f"[Flugschreiber] Gültiges Signal '{side.upper()}' erkannt. Beginne Trade-Eröffnungsprozess...")
        p = params['risk']
        ticker = exchange.fetch_ticker(symbol)
        entry_price = ticker['last']
        
        logger.info("[Flugschreiber] Schritt 1/8: Berechne Positionsgröße...")
        risk_amount_usd = current_balance * (p['risk_per_trade_pct'] / 100)
        sl_distance_pct = 0.01
        notional_value = risk_amount_usd / sl_distance_pct
        
        market_info = exchange.markets.get(symbol, {})
        min_cost = market_info.get('limits', {}).get('cost', {}).get('min', 5.0)
        min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0.001)

        if notional_value < min_cost: notional_value = min_cost * 1.05
        amount = notional_value / entry_price
        if amount < min_amount: amount = min_amount * 1.05
        
        logger.info("[Flugschreiber] Schritt 2/8: Berechne SL/TP Preise (ungerundet)...")
        stop_loss_price = entry_price * (1 - sl_distance_pct) if side == 'buy' else entry_price * (1 + sl_distance_pct)
        take_profit_price = entry_price + (entry_price - stop_loss_price) * p['risk_reward_ratio'] if side == 'buy' else entry_price - (stop_loss_price - entry_price) * p['risk_reward_ratio']
        
        logger.info("[Flugschreiber] Schritt 3/8: Runde Preise auf Börsen-Genauigkeit...")
        stop_loss_price_rounded = float(exchange.exchange.price_to_precision(symbol, stop_loss_price))
        take_profit_price_rounded = float(exchange.exchange.price_to_precision(symbol, take_profit_price))
        
        logger.info("[Flugschreiber] Schritt 4/8: Setze Margin-Modus und Hebel...")
        exchange.set_margin_mode(symbol, p['margin_mode'])
        exchange.set_leverage(symbol, p['leverage'])
        
        logger.info("[Flugschreiber] Schritt 5/8: SENDE MARKET ORDER AN DIE BÖRSE...")
        exchange.create_market_order(symbol, side, amount, params={'leverage': p['leverage'], 'marginMode': p['margin_mode']})
        logger.info("[Flugschreiber] Market Order gesendet. Warte 2 Sekunden...")
        time.sleep(2)

        logger.info("[Flugschreiber] Schritt 6/8: Platziere TAKE PROFIT Order...")
        exchange.place_trigger_market_order(symbol, 'sell' if side == 'buy' else 'buy', amount, take_profit_price_rounded, {'reduceOnly': True})
        
        logger.info("[Flugschreiber] Schritt 7/8: Platziere STOP LOSS Order...")
        exchange.place_trigger_market_order(symbol, 'sell' if side == 'buy' else 'buy', amount, stop_loss_price_rounded, {'reduceOnly': True})
        
        logger.info("[Flugschreiber] Schritt 8/8: Schreibe Zustand in Datenbank und sende Telegram-Nachricht...")
        db_set_state(account_name, symbol, timeframe, 'initial_stop_loss', stop_loss_price_rounded)
        db_set_state(account_name, symbol, timeframe, 'trailing_stop_active', 'False')

        message = f"🧠 ANN Signal für *{account_name}* ({symbol}, {side.upper()})\n- Entry @ Market (≈${entry_price:.4f})\n- SL: ${stop_loss_price_rounded:.4f}\n- TP: ${take_profit_price_rounded:.4f}"
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
        logger.info("[Flugschreiber] Trade-Eröffnungsprozess vollständig abgeschlossen.")

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
        logger.error(f"Fehler: Nicht genügend Guthaben. {e}")
        message = f"🟡 *Kein Guthaben für Trade*\n\nBot für *{account_name}* ({symbol}) wollte einen Trade eröffnen, aber das Guthaben reicht nicht aus."
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler im Handelszyklus: {e}", exc_info=True)
        message = f"🚨 *Kritischer Fehler*\n\nEin unerwarteter Fehler ist im Bot für *{account_name}* ({symbol}) aufgetreten."
        send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
