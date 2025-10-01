# src/lbot/utils/exchange.py
import ccxt
import pandas as pd
from time import sleep

class Exchange:
    def __init__(self, account_config):
        self.account = account_config
        self.exchange = getattr(ccxt, 'bitget')({
            'apiKey': self.account.get('apiKey'),
            'secret': self.account.get('secret'),
            'password': self.account.get('password'),
            'options': { 'defaultType': 'swap' },
        })
        self.markets = self.exchange.load_markets()

    def _format_dataframe(self, data):
        """ Konvertiert die rohen OHLCV-Daten in einen formatierten DataFrame. """
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    def fetch_ohlcv_since(self, symbol, timeframe, since, limit=1000):
        """ Holt alle OHLCV-Daten seit einem bestimmten Timestamp. """
        all_data = []
        while True:
            try:
                data = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not data:
                    break
                all_data.extend(data)
                since = data[-1][0] + 1 # Zeitstempel der nächsten Kerze
                sleep(self.exchange.rateLimit / 1000) # Respektiere Rate-Limits
            except Exception as e:
                print(f"Fehler beim Abrufen von OHLCV-Daten: {e}. Versuche erneut...")
                sleep(5)
                continue
        return self._format_dataframe(all_data)

    def fetch_recent_ohlcv(self, symbol, timeframe, limit=100):
        """ Holt die letzten N Kerzen (für Live-Trading benötigt). """
        data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return self._format_dataframe(data)

    def fetch_ticker(self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def set_margin_mode(self, symbol, mode='isolated'):
        try: self.exchange.set_margin_mode(mode, symbol)
        except Exception: pass

    def set_leverage(self, symbol, level=10):
        self.exchange.set_leverage(level, symbol)

    def create_market_order(self, symbol, side, amount, params={}):
        return self.exchange.create_order(symbol, 'market', side, amount, params=params)
    
    def place_trigger_market_order(self, symbol, side, amount, trigger_price, params={}):
        order_params = {'triggerPrice': trigger_price, 'reduceOnly': params.get('reduceOnly', False)}
        return self.exchange.create_order(symbol, 'market', side, amount, params=order_params)

    def fetch_open_positions(self, symbol):
        positions = self.exchange.fetch_positions([symbol])
        return [p for p in positions if p.get('contracts', 0.0) > 0.0]

    def fetch_open_trigger_orders(self, symbol):
        return self.exchange.fetch_open_orders(symbol, params={'type': 'market', 'stop': True})
    
    def cancel_trigger_order(self, order_id, symbol):
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_balance_usdt(self):
        try:
            balance = self.exchange.fetch_balance()
            if 'USDT' in balance: return balance['USDT']['free']
            elif 'total' in balance and 'USDT' in balance['total']: return balance['total']['USDT']
            else: return 0
        except Exception: return 0
