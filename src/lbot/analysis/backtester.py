# src/lbot/analysis/backtester.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Backtester:
    def __init__(self, data, model, scaler, params, start_capital=1000, sequence_length=24, fee_rate_pct=0.06, slippage_pct=0.02):
        self.data = data
        self.model = model
        self.scaler = scaler
        self.params = params
        self.start_capital = start_capital
        self.sequence_length = sequence_length
        self.fee_rate = fee_rate_pct / 100
        self.slippage = slippage_pct / 100
        self.trades = []
        self.equity_curve = [start_capital]

    def _apply_slippage(self, price, side):
        """ Simuliert den Preisrutsch bei einer Market Order. """
        if side == 'long': # Kauf-Preis ist leicht höher
            return price * (1 + self.slippage)
        elif side == 'short': # Verkauf-Preis ist leicht niedriger
            return price * (1 - self.slippage)
        return price

    def run(self):
        """ Führt den kompletten Backtest durch. """
        feature_columns_to_scale = self.data.columns.drop('close', errors='ignore')
        scaled_features = self.scaler.transform(self.data[feature_columns_to_scale])
        
        position = None  # None, 'long'
        entry_price = 0
        
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
            
            # Wenn keine Position offen ist, nach neuem Signal suchen
            if not position:
                sequence = scaled_features[i - self.sequence_length : i]
                input_data = np.expand_dims(sequence, axis=0)
                prediction = self.model.predict(input_data, verbose=0)[0][0]
                
                pred_threshold = self.params['strategy']['prediction_threshold']
                if prediction >= pred_threshold and self.params['behavior']['use_longs']:
                    position = 'long'
                    leverage = self.params['risk']['leverage']
                    risk_per_trade = self.params['risk']['risk_per_trade_pct'] / 100
                    rr_ratio = self.params['risk']['risk_reward_ratio']
                    
                    self.sl_pct = risk_per_trade / leverage
                    self.tp_pct = self.sl_pct * rr_ratio
                    
                    self._open_position(i, 'long')
                    # Der entry_price wird in _open_position gesetzt, inkl. Slippage
                    entry_price = self.trades[-1]['entry_price']

        return self._calculate_metrics()

    def _open_position(self, index, side):
        raw_price = self.data['close'].iloc[index]
        entry_price_with_slippage = self._apply_slippage(raw_price, side)
        
        self.trades.append({
            'entry_index': index,
            'entry_date': self.data.index[index],
            'entry_price': entry_price_with_slippage,
            'side': side,
            'status': 'open'
        })

    def _close_position(self, index, reason):
        trade = self.trades[-1]
        if trade['status'] != 'open': return

        raw_price = self.data['close'].iloc[index]
        # Slippage beim Schließen einer Long-Position (Verkauf)
        exit_price_with_slippage = self._apply_slippage(raw_price, 'short')

        trade['status'] = 'closed'
        trade['exit_index'] = index
        trade['exit_date'] = self.data.index[index]
        trade['exit_price'] = exit_price_with_slippage
        trade['reason'] = reason
        
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        
        pnl_pct = (exit_price - entry_price) / entry_price
        
        # PnL-Update der Equity-Kurve
        leverage = self.params['risk']['leverage']
        capital = self.equity_curve[-1]
        
        # Kosten kalkulieren
        entry_cost = capital * leverage * self.fee_rate
        exit_cost = capital * leverage * (1 + pnl_pct) * self.fee_rate
        total_fees = entry_cost + exit_cost
        
        pnl_amount = (capital * pnl_pct * leverage) - total_fees
        self.equity_curve.append(capital + pnl_amount)

    def _calculate_metrics(self):
        df_trades = pd.DataFrame(self.trades)
        closed_trades = df_trades[df_trades['status'] == 'closed'].copy()
        
        if closed_trades.empty:
            return {'total_pnl_pct': 0, 'win_rate': 0, 'max_drawdown_pct': 0, 'num_trades': 0}

        pnl = (closed_trades['exit_price'] - closed_trades['entry_price']) / closed_trades['entry_price']
        
        final_equity = self.equity_curve[-1]
        total_pnl_pct = (final_equity / self.start_capital - 1) * 100
        win_rate = (pnl > 0).mean() * 100

        # Max Drawdown
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = abs(drawdown.min() * 100) if not drawdown.empty else 0

        return {
            'total_pnl_pct': total_pnl_pct,
            'win_rate': win_rate,
            'max_drawdown_pct': max_drawdown_pct,
            'num_trades': len(closed_trades)
        }
