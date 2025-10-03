# src/lbot/utils/trade_manager.py
import numpy as np
from .lstm_model import create_ann_features
from .telegram import send_message

def get_rounded_price(price, market):
    """ Rundet einen Preis auf die von der Börse geforderte Genauigkeit. """
    if 'precision' in market and 'price' in market['precision']:
        return float(market['precision']['price'] * round(price / market['precision']['price']))
    # Fallback, falls Präzisionsdaten nicht verfügbar sind
    return round(price, 8)

def full_trade_cycle(exchange, model, scaler, params, settings, current_balance, get_state, set_state, telegram_config, logger):
    """
    Führt den kompletten Zyklus für eine mögliche Trade-Eröffnung im Live-Modus durch.
    """
    account_name = exchange.account.get('name', 'Standard')
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    
    position_status = get_state(account_name, symbol, timeframe, 'position_status', 'closed')
    if position_status == 'open':
        logger.info("Position ist bereits offen. Überspringe Trade-Eröffnung.")
        return

    try:
        model_conf = settings.get('model_settings', {})
        filter_conf = settings.get('strategy_filters', {})
        
        sequence_length = model_conf.get('sequence_length', 24)
        ema_period = filter_conf.get('ema_period', 200)
        atr_period = filter_conf.get('atr_period', 14)
        
        # Lade genug Daten für Features, EMA, ATR und die Sequenzlänge
        history_limit = sequence_length + ema_period + 50
        ohlcv = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=history_limit)
        ticker = exchange.fetch_ticker(symbol)
        
        if ohlcv.empty or ticker is None or 'last' not in ticker:
            logger.warning("Konnte keine vollständigen OHLCV-Daten oder Ticker-Infos abrufen.")
            return
        current_price = ticker['last']
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Marktdaten: {e}")
        return

    # Erstelle alle Features, inklusive der Indikatoren für die Filter
    data_with_features = create_ann_features(ohlcv, ema_period=ema_period, atr_period=atr_period)
    
    if data_with_features.empty:
        logger.warning("Nicht genügend Daten nach Feature-Erstellung vorhanden. Überspringe.")
        return

    # --- ANWENDUNG DER FILTER ---
    if filter_conf.get('use_trend_filter', False):
        latest_ema = data_with_features[f'ema_{ema_period}'].iloc[-1]
        if current_price < latest_ema:
            logger.info(f"TRADE VERHINDERT (Trend): Preis ({current_price}) < EMA-{ema_period} ({latest_ema:.4f}).")
            return
            
    if filter_conf.get('use_volatility_filter', False):
        natr_col = f'natr_{atr_period}'
        min_natr = params['strategy'].get('min_natr', 0)
        max_natr = params['strategy'].get('max_natr', 999)
        current_natr = data_with_features[natr_col].iloc[-1]
        
        if not (min_natr <= current_natr <= max_natr):
            logger.info(f"TRADE VERHINDERT (Vola): Aktueller nATR ({current_natr:.2f}) außerhalb des Fensters ({min_natr:.2f} - {max_natr:.2f}).")
            return

    # --- VORHERSAGE TREFFEN ---
    latest_sequence_unscaled = data_with_features.iloc[-sequence_length:]
    
    # Stelle sicher, dass die Spaltennamen exakt übereinstimmen
    feature_columns_to_scale = scaler.get_feature_names_out()
    
    try:
        scaled_values = scaler.transform(latest_sequence_unscaled[feature_columns_to_scale])
    except Exception as e:
        logger.error(f"Fehler beim Skalieren der Live-Daten: {e}. Spalten: {latest_sequence_unscaled.columns.tolist()}")
        return

    input_data = np.expand_dims(scaled_values, axis=0)
    prediction = model.predict(input_data, verbose=0)[0][0]
    logger.info(f"Modell-Vorhersage für {symbol}: {prediction:.4f}")

    # --- TRADE-ENTSCHEIDUNG UND AUSFÜHRUNG ---
    pred_threshold = params['strategy']['prediction_threshold']
    if prediction >= pred_threshold and params['behavior'].get('use_longs', False):
        logger.info(f"Einstiegssignal! Vorhersage ({prediction:.4f}) > Schwellenwert ({pred_threshold}).")
        
        risk = params['risk']
        leverage = risk['leverage']
        risk_per_trade_pct = risk['risk_per_trade_pct'] / 100
        rr_ratio = risk['risk_reward_ratio']
        
        position_size_usd = current_balance * risk_per_trade_pct * leverage
        amount = position_size_usd / current_price
        sl_pct = risk_per_trade_pct / leverage
        
        market = exchange.exchange.market(symbol)
        stop_loss_price = get_rounded_price(current_price * (1 - sl_pct), market)
        take_profit_price = get_rounded_price(current_price * (1 + (sl_pct * rr_ratio)), market)

        try:
            exchange.set_leverage(symbol, leverage)
            exchange.set_margin_mode(symbol, risk.get('margin_mode', 'isolated'))
            
            logger.info(f"Öffne LONG-Position: {amount:.4f} {market['base']} im Wert von {position_size_usd:.2f} USD.")
            order = exchange.create_market_order(symbol, 'buy', amount)
            
            # Warte kurz, damit die Position an der Börse registriert wird
            time.sleep(5)
            
            logger.info(f"Platziere Stop-Loss bei {stop_loss_price} und Take-Profit bei {take_profit_price}.")
            sl_order = exchange.place_trigger_market_order(symbol, 'sell', amount, stop_loss_price, {'reduceOnly': True})
            tp_order = exchange.place_trigger_market_order(symbol, 'sell', amount, take_profit_price, {'reduceOnly': True})
            
            if not sl_order or not tp_order or 'id' not in sl_order or 'id' not in tp_order:
                raise Exception("Fehler beim Platzieren der SL/TP-Orders. Keine Order-ID erhalten.")

            set_state(account_name, symbol, timeframe, 'position_status', 'open')
            set_state(account_name, symbol, timeframe, 'sl_order_id', sl_order['id'])
            set_state(account_name, symbol, timeframe, 'tp_order_id', tp_order['id'])

            msg = (f"✅ *L-Bot Trade Eröffnet*\n\n"
                   f"*{symbol} ({timeframe})*\n"
                   f"Seite: LONG\n"
                   f"Größe: {position_size_usd:.2f} USDT\n"
                   f"Einstiegspreis: ~{current_price:.4f}\n"
                   f"Take Profit: {take_profit_price}\n"
                   f"Stop Loss: {stop_loss_price}")
            send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)

        except Exception as e:
            logger.critical(f"FEHLER BEI TRADE-AUSFÜHRUNG: {e}", exc_info=True)
            msg = f"🚨 *L-Bot Kritischer Fehler*\n\nTrade für {symbol} konnte nicht ausgeführt werden:\n_{e}_"
            send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)
    else:
        logger.info("Kein Einstiegssignal oder Longs deaktiviert.")


def babysit_open_position(exchange, params, get_state, set_state, telegram_config, logger):
    """
    Überwacht offene Positionen und stellt sicher, dass der Zustand zwischen DB und Börse konsistent ist.
    """
    account_name = exchange.account.get('name', 'Standard')
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    
    position_status = get_state(account_name, symbol, timeframe, 'position_status', 'closed')

    if position_status == 'open':
        try:
            open_positions = exchange.fetch_open_positions(symbol)
            
            if not open_positions:
                logger.info(f"Position für {symbol} an der Börse geschlossen (vermutlich SL/TP). Setze DB-Status zurück.")
                set_state(account_name, symbol, timeframe, 'position_status', 'closed')
                set_state(account_name, symbol, timeframe, 'sl_order_id', '0')
                set_state(account_name, symbol, timeframe, 'tp_order_id', '0')
                
                msg = f"ℹ️ *L-Bot Info*\n\nPosition für *{symbol}* wurde geschlossen (wahrscheinlich durch SL/TP)."
                send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)
            else:
                logger.info(f"Offene Position für {symbol} wird weiterhin überwacht.")
                # Hier könnte man in Zukunft die Trailing-Stop-Logik oder die Überprüfung von SL/TP-Orders einbauen
                
        except Exception as e:
            logger.error(f"Fehler beim Babysitting für {symbol}: {e}")
