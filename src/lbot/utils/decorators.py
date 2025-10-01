# src/lbot/utils/decorators.py
from functools import wraps
from .guardian import Guardian, PreFlightCheckError
from .telegram import send_message
from .exchange import Exchange

def run_with_guardian_checks(func):
    """
    Ein Decorator, der sicherstellt, dass die Guardian Pre-Flight-Checks
    bestanden werden, bevor die eigentliche Bot-Logik ausgeführt wird.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        account = args[0]
        telegram_config = args[1]
        params = args[2]
        logger = args[5]
        model_path = args[6]
        scaler_path = args[7]
        
        account_name = account.get('name', 'Standard-Account')
        symbol = params['market']['symbol']
        
        try:
            exchange = Exchange(account)
            guardian = Guardian(exchange, params, model_path, scaler_path, logger)
            guardian.run_pre_flight_checks()
            return func(*args, **kwargs)

        except PreFlightCheckError as e:
            logger.critical(f"Guardian hat den Start für {account_name} ({symbol}) verhindert.")
            message = f"🚨 *L-Bot Gestoppt* ({symbol})\n\nGrund: Pre-Flight-Check fehlgeschlagen!\n\n_{e}_"
            send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
        
        except Exception as e:
            logger.critical(f"Ein kritischer Fehler ist im Guardian-Decorator aufgetreten: {e}", exc_info=True)
            message = f"🚨 *Kritischer Systemfehler* im Guardian-Decorator für {symbol}."
            send_message(telegram_config.get('bot_token'), telegram_config.get('chat_id'), message)
            
    return wrapper
