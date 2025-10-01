# src/lbot/analysis/backtester.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Backtester:
    def __init__(self, data, model, scaler, params, settings, start_capital=1000):
        self.data = data
        self.model = model
        self.scaler = scaler
        self.params = params
        self.settings = settings
        self.start_capital = start_capital
        
        model_conf = self.settings['model_settings']
        backtest_conf = self.settings['backtest_settings']
        self.filter_conf = self.settings.get('strategy_filters', {})
        
        self.sequence_length = model_conf['sequence_length']
        self.fee_rate = backtest_conf.get('fee_rate_pct', 0.06) / 100
        self.slippage = backtest_conf.get('slippage_pct', 0.02) / 100
        
        self.trades = []
        self.equity_curve = [start_capital]

    def _apply_slippage(self, price, side):
        if side == 'long': return price * (1 + self.slippage)
        elif side == 'short': return price * (1 - self.slippage)
        return price
        
    def _is_trend_filter_ok(self, index, side):
        if not self.filter_conf.get('use_trend_filter', False):
            return True

        ema_period = self.filter_conf.get('ema_period', 200)
        ema_col = f'ema_{ema_period}'
        
        if ema_col not in self.data.columns:
            return True

        current_price = self.data['close'].iloc[index]
        ema_value = self.data[ema_col].iloc[index]
        
        if side == 'long':
            return current_price > ema_value
        elif side == 'short':
            return current_price < ema_value
        return False

    def run(self):
        ema_period = self.filter_conf.get('ema_period', 200)
        
        # KORRIGIERTE ZEILE: Schließe jetzt auch die EMA-Spalte von der Skalierung aus
        feature_columns_to_scale = self.data.columns.drop(['close', f'ema_{ema_period}'], errors='ignore')
        
        scaled_features_df = self.data.copy()
        scaled_features_df[feature_columns_to_scale] = self.scaler.transform(self.data[feature_columns_to_scale])
        
        position = None
        entry_price = 0
        
        # Wir iterieren jetzt über den DataFrame mit den skalierten Features
        for i in range(self.sequence_length, len(scaled_features_df)):
            current_price = self.data['close'].iloc[i] # Originalpreis für PnL-Berechnung
            
            if position:
                pnl_pct = (current_price - entry_price) / entry_price
                if position == 'long':
                    if pnl_pct <= -self.sl_pct:
                        self._close_position(i, 'SL')
                        position = None
                    elif pnl_pct >= self.tp_pct:
                        self._close_position(i, 'TP')
                        position = None
            
            if not position:
                # Nimm die Sequenz aus dem skalierten DataFrame
                sequence_df = scaled_features_df.iloc[i - self.sequence_length : i]
                
                # Stelle sicher, dass 'close' und der EMA nicht in den finalen Numpy-Array für das Modell kommen
                model_input_features = sequence_df.drop(['close', f'ema_{ema_period}'], errors='ignore').values
                input_data = np.expand_dims(model_input_features, axis=0)
                
                prediction = self.model.predict(input_data, verbose=0)[0][0]
                
                pred_threshold = self.params['strategy']['prediction_threshold']
                
                if (prediction >= pred_threshold and 
                    self.params['behavior']['use_longs'] and 
                    self._is_trend_filter_ok(i, 'long')):
                    
                    position = 'long'
                    leverage = self.params['risk']['leverage']
                    risk_per_trade = self.params['risk']['risk_per_trade_pct'] / 100
                    rr_ratio = self.params['risk']['risk_reward_ratio']
                    
                    self.sl_pct = risk_per_trade / leverage
                    self.tp_pct = self.sl_pct * rr_ratio
                    
                    self._open_position(i, 'long')
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
        exit_price_with_slippage = self._apply_slippage(raw_price, 'short')
        trade['status'] = 'closed'
        trade['exit_index'] = index
        trade['exit_date'] = self.data.index[index]
        trade['exit_price'] = exit_price_with_slippage
        trade['reason'] = reason
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        pnl_pct = (exit_price - entry_price) / entry_price
        leverage = self.params['risk']['leverage']
        capital = self.equity_curve[-1]
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
        win_rate = (pnl > 0).mean() * 100 if not pnl.empty else 0
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
