# src/lbot/utils/trade_manager.py
import numpy as np
from .lstm_model import create_ann_features
from .telegram import send_message

def full_trade_cycle(exchange, model, scaler, params, settings, current_balance, db_get_state, db_set_state, telegram_config, logger):
    """
    Führt den kompletten Zyklus für eine mögliche Trade-Eröffnung im Live-Modus durch.
    """
    account_name = exchange.account.get('name', 'Standard')
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    
    # 1. Prüfen, ob bereits eine Position laut Datenbank offen ist
    position_status = db_get_state(account_name, symbol, timeframe, 'position_status', 'closed')
    if position_status == 'open':
        logger.info("Position ist bereits offen. Überspringe Trade-Eröffnung.")
        return

    # 2. Aktuelle Marktdaten und Ticker holen
    try:
        # Lade genug Daten für Features und die Sequenzlänge
        history_limit = settings['model_settings']['sequence_length'] + settings['strategy_filters']['ema_period'] + 5
        ohlcv = exchange.fetch_recent_ohlcv(symbol, timeframe, limit=history_limit)
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Marktdaten: {e}")
        return

    # 3. Features erstellen (inkl. EMA-Filter)
    ema_period = settings['strategy_filters'].get('ema_period', 200)
    data_with_features = create_ann_features(ohlcv, ema_period=ema_period)
    
    # 4. Trend-Filter anwenden
    if settings['strategy_filters'].get('use_trend_filter', False):
        latest_ema = data_with_features[f'ema_{ema_period}'].iloc[-1]
        if current_price < latest_ema:
            logger.info(f"TRADE VERHINDERT: Preis ({current_price}) liegt unter EMA-{ema_period} ({latest_ema:.2f}).")
            return

    # 5. Vorhersage treffen
    sequence_length = settings['model_settings']['sequence_length']
    latest_sequence_unscaled = data_with_features.iloc[-sequence_length:]
    
    # Skaliere nur die Feature-Spalten, nicht 'close' oder den EMA direkt
    feature_columns_to_scale = data_with_features.columns.drop(['close', f'ema_{ema_period}'], errors='ignore')
    scaled_values = scaler.transform(latest_sequence_unscaled[feature_columns_to_scale])
    
    # Ersetze die unskalierten Werte durch die skalierten im DataFrame für die richtige Struktur
    latest_sequence_scaled = latest_sequence_unscaled.copy()
    latest_sequence_scaled.loc[:, feature_columns_to_scale] = scaled_values

    # Erstelle den finalen Input für das Modell
    input_data = np.expand_dims(latest_sequence_scaled.values, axis=0)
    prediction = model.predict(input_data, verbose=0)[0][0]
    logger.info(f"Modell-Vorhersage für {symbol}: {prediction:.4f}")

    # 6. Trade-Entscheidung und Ausführung
    pred_threshold = params['strategy']['prediction_threshold']
    if prediction >= pred_threshold and params['behavior']['use_longs']:
        logger.info(f"Einstiegssignal! Vorhersage ({prediction:.4f}) > Schwellenwert ({pred_threshold}).")
        
        # Risikoparameter laden
        risk = params['risk']
        leverage = risk['leverage']
        risk_per_trade_pct = risk['risk_per_trade_pct'] / 100
        rr_ratio = risk['risk_reward_ratio']
        
        # Positionsgröße, SL und TP berechnen
        position_size_usd = current_balance * risk_per_trade_pct * leverage
        amount = position_size_usd / current_price
        sl_pct = risk_per_trade_pct / leverage
        
        stop_loss_price = current_price * (1 - sl_pct)
        take_profit_price = current_price * (1 + (sl_pct * rr_ratio))

        # Trade ausführen
        try:
            exchange.set_leverage(symbol, leverage)
            exchange.set_margin_mode(symbol, risk.get('margin_mode', 'isolated'))
            
            logger.info(f"Öffne LONG-Position: {amount:.4f} {symbol.split('/')[0]} im Wert von {position_size_usd:.2f} USD.")
            order = exchange.create_market_order(symbol, 'buy', amount)
            
            # SL und TP platzieren
            logger.info(f"Platziere Stop-Loss bei {stop_loss_price:.2f} und Take-Profit bei {take_profit_price:.2f}.")
            sl_order = exchange.place_trigger_market_order(symbol, 'sell', amount, stop_loss_price, {'reduceOnly': True})
            tp_order = exchange.place_trigger_market_order(symbol, 'sell', amount, take_profit_price, {'reduceOnly': True})
            
            # Zustand in DB speichern
            db_set_state(account_name, symbol, timeframe, 'position_status', 'open')
            db_set_state(account_name, symbol, timeframe, 'sl_order_id', sl_order['id'])
            db_set_state(account_name, symbol, timeframe, 'tp_order_id', tp_order['id'])

            # Telegram-Nachricht
            msg = (f"✅ *L-Bot Trade Eröffnet*\n\n"
                   f"*{symbol} ({timeframe})*\n"
                   f"Seite: LONG\n"
                   f"Größe: {position_size_usd:.2f} USDT\n"
                   f"Einstiegspreis: ~{current_price:.2f}\n"
                   f"Take Profit: {take_profit_price:.2f}\n"
                   f"Stop Loss: {stop_loss_price:.2f}")
            send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)

        except Exception as e:
            logger.critical(f"FEHLER BEI TRADE-AUSFÜHRUNG: {e}")
            msg = f"🚨 *L-Bot Kritischer Fehler*\n\nTrade für {symbol} konnte nicht ausgeführt werden:\n_{e}_"
            send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)
    else:
        logger.info("Kein Einstiegssignal oder Longs deaktiviert.")


def babysit_open_position(exchange, params, db_get_state, db_set_state, telegram_config, logger):
    """
    Überwacht offene Positionen und stellt sicher, dass der Zustand zwischen DB und Börse konsistent ist.
    """
    account_name = exchange.account.get('name', 'Standard')
    symbol = params['market']['symbol']
    timeframe = params['market']['timeframe']
    
    position_status = db_get_state(account_name, symbol, timeframe, 'position_status', 'closed')

    if position_status == 'open':
        try:
            open_positions = exchange.fetch_open_positions(symbol)
            
            if not open_positions:
                # Position ist an der Börse geschlossen, aber in DB noch offen -> SL/TP wurde ausgelöst
                logger.info(f"Position für {symbol} an der Börse geschlossen (vermutlich SL/TP). Setze DB-Status zurück.")
                db_set_state(account_name, symbol, timeframe, 'position_status', 'closed')
                
                msg = f"ℹ️ *L-Bot Info*\n\nPosition für *{symbol}* wurde geschlossen (wahrscheinlich durch SL/TP)."
                send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), msg)
            else:
                logger.info(f"Offene Position für {symbol} wird weiterhin überwacht.")
                # Hier könnte man in Zukunft die Trailing-Stop-Logik einbauen
                
        except Exception as e:
            logger.error(f"Fehler beim Babysitting für {symbol}: {e}")
