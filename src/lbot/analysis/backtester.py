# src/lbot/analysis/backtester.py
import pandas as pd
import numpy as np
import logging

# Logging-Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Backtester:
    def __init__(self, data, model, scaler, params, start_capital=1000, sequence_length=24):
        self.data = data
        self.model = model
        self.scaler = scaler
        self.params = params
        self.start_capital = start_capital
        self.sequence_length = sequence_length
        self.trades = []
        self.equity_curve = [start_capital]

    def run(self):
        """ Führt den kompletten Backtest durch. """
        
        # Bereite die Features vor und skaliere sie
        feature_columns_to_scale = self.data.columns.drop('close', errors='ignore')
        scaled_features = self.scaler.transform(self.data[feature_columns_to_scale])
        
        position = None  # None, 'long', 'short'
        entry_price = 0
        
        # Iteriere durch die Daten, beginnend nach der ersten vollen Sequenz
        for i in range(self.sequence_length, len(scaled_features)):
            current_price = self.data['close'].iloc[i]
            
            # Position schließen, falls SL/TP getroffen
            if position:
                pnl_pct = (current_price - entry_price) / entry_price
                if position == 'long':
                    if pnl_pct <= -self.sl_pct:
                        self._close_position(i, 'SL')
                        position = None
                    elif pnl_pct >= self.tp_pct:
                        self._close_position(i, 'TP')
                        position = None
                elif position == 'short':
                    if pnl_pct >= self.sl_pct:
                        self._close_position(i, 'SL')
                        position = None
                    elif pnl_pct <= -self.tp_pct:
                        self._close_position(i, 'TP')
                        position = None
            
            # Wenn keine Position offen ist, nach neuem Signal suchen
            if not position:
                # Erstelle die Sequenz für die Vorhersage
                sequence = scaled_features[i - self.sequence_length : i]
                input_data = np.expand_dims(sequence, axis=0)
                
                prediction = self.model.predict(input_data, verbose=0)[0][0]
                
                # Einstiegslogik
                pred_threshold = self.params['strategy']['prediction_threshold']
                if prediction >= pred_threshold and self.params['behavior']['use_longs']:
                    position = 'long'
                    entry_price = current_price
                    leverage = self.params['risk']['leverage']
                    risk_per_trade = self.params['risk']['risk_per_trade_pct'] / 100
                    rr_ratio = self.params['risk']['risk_reward_ratio']
                    
                    self.sl_pct = risk_per_trade / leverage
                    self.tp_pct = self.sl_pct * rr_ratio
                    
                    self._open_position(i, 'long')

        # Metriken berechnen und zurückgeben
        return self._calculate_metrics()

    def _open_position(self, index, side):
        self.trades.append({
            'entry_index': index,
            'entry_date': self.data.index[index],
            'entry_price': self.data['close'].iloc[index],
            'side': side,
            'status': 'open'
        })

    def _close_position(self, index, reason):
        trade = self.trades[-1]
        trade['status'] = 'closed'
        trade['exit_index'] = index
        trade['exit_date'] = self.data.index[index]
        trade['exit_price'] = self.data['close'].iloc[index]
        trade['reason'] = reason
        
        pnl_pct = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
        if trade['side'] == 'short':
            pnl_pct *= -1
            
        # Simples PnL-Update der Equity-Kurve
        leverage = self.params['risk']['leverage']
        capital = self.equity_curve[-1]
        pnl_amount = capital * pnl_pct * leverage
        self.equity_curve.append(capital + pnl_amount)

    def _calculate_metrics(self):
        if not self.trades:
            return {'total_pnl_pct': 0, 'win_rate': 0, 'max_drawdown_pct': 0, 'num_trades': 0}

        df_trades = pd.DataFrame(self.trades)
        closed_trades = df_trades[df_trades['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return {'total_pnl_pct': 0, 'win_rate': 0, 'max_drawdown_pct': 0, 'num_trades': 0}

        # PnL & Win Rate
        pnl = (closed_trades['exit_price'] - closed_trades['entry_price']) / closed_trades['entry_price']
        pnl[closed_trades['side'] == 'short'] *= -1
        
        final_equity = self.equity_curve[-1]
        total_pnl_pct = (final_equity / self.start_capital - 1) * 100
        win_rate = (pnl > 0).mean() * 100

        # Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = abs(drawdown.min() * 100)

        return {
            'total_pnl_pct': total_pnl_pct,
            'win_rate': win_rate,
            'max_drawdown_pct': max_drawdown_pct,
            'num_trades': len(closed_trades)
        }
