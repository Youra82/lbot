# src/jaegerbot/utils/exchange.py
import ccxt
import pandas as pd
from datetime import datetime, timezone

class Exchange:
    def __init__(self, account_config):
        self.account = account_config
        self.exchange = getattr(ccxt, 'bitget')({
            'apiKey': self.account.get('apiKey'),
            'secret': self.account.get('secret'),
            'password': self.account.get('password'),
            'options': {
                'defaultType': 'swap',
            },
        })
        self.markets = self.exchange.load_markets()

    def fetch_recent_ohlcv(self, symbol, timeframe, limit=100):
        """
        Ruft die letzten OHLCV-Daten ab und gibt sie als DataFrame
        mit einem korrekten DatetimeIndex zurück.
        """
        since = None
        data = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # KORREKTUR: Zeitstempel als Index setzen, um den AttributeError zu beheben.
        df.set_index('timestamp', inplace=True)
        
        # Index sortieren, um sicherzustellen, dass die Daten chronologisch sind.
        df.sort_index(inplace=True)
        return df

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def set_margin_mode(self, symbol, mode='isolated'):
        try:
            self.exchange.set_margin_mode(mode, symbol)
        except Exception as e:
            print(f"Warnung: Margin-Modus konnte nicht gesetzt werden: {e}")

    def set_leverage(self, symbol, level=10):
        self.exchange.set_leverage(level, symbol)

    def create_market_order(self, symbol, side, amount, params={}):
        return self.exchange.create_order(symbol, 'market', side, amount, params=params)
    
    def place_trigger_market_order(self, symbol, side, amount, trigger_price, params={}):
        return self.exchange.create_order(symbol, 'market', side, amount, params={'triggerPrice': trigger_price, 'reduceOnly': params.get('reduceOnly', False)})

    def fetch_open_positions(self, symbol):
        positions = self.exchange.fetch_positions([symbol])
        open_positions = [p for p in positions if p.get('contracts', 0.0) > 0.0]
        return open_positions

    def fetch_open_trigger_orders(self, symbol):
        return self.exchange.fetch_open_orders(symbol, params={'type': 'market', 'stop': True})
    
    def cancel_trigger_order(self, order_id, symbol):
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_open_limit_orders(self, symbol):
        return self.exchange.fetch_open_orders(symbol, params={'type': 'limit'})

    def cancel_limit_order(self, order_id, symbol):
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_balance_usdt(self):
        """
        Ruft den gesamten verfügbaren USDT-Saldo vom Konto ab.
        Wird für das dynamische Risikomanagement benötigt.
        """
        try:
            balance = self.exchange.fetch_balance()
            # Wir suchen nach USDT in den Futures/Margin Konten
            if 'USDT' in balance:
                return balance['USDT']['free']
            # Fallback für Spot-Konten, falls nötig
            elif 'total' in balance and 'USDT' in balance['total']:
                 return balance['total']['USDT']
            else:
                return 0
        except Exception as e:
            print(f"Fehler beim Abrufen des Kontostandes: {e}")
            return 0
