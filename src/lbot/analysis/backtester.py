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
        
        model_conf = self.settings.get('model_settings', {})
        backtest_conf = self.settings.get('backtest_settings', {})
        self.filter_conf = self.settings.get('strategy_filters', {})
        
        self.sequence_length = model_conf.get('sequence_length', 24)
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
        if ema_col not in self.data.columns: return True
        current_price = self.data['close'].iloc[index]
        ema_value = self.data[ema_col].iloc[index]
        if side == 'long': return current_price > ema_value
        return False

    # NEU: Funktion für den Volatilitäts-Filter
    def _is_volatility_filter_ok(self, index):
        if not self.filter_conf.get('use_volatility_filter', False):
            return True
        
        atr_period = self.filter_conf.get('atr_period', 14)
        natr_col = f'natr_{atr_period}'
        
        if natr_col not in self.data.columns: return True
            
        min_natr = self.params['strategy'].get('min_natr', 0)
        max_natr = self.params['strategy'].get('max_natr', 999)
        
        current_natr = self.data[natr_col].iloc[index]
        
        return min_natr <= current_natr <= max_natr


    def run(self):
        ema_period = self.filter_conf.get('ema_period', 200)
        atr_period = self.filter_conf.get('atr_period', 14)
        
        # Features, die nicht skaliert werden sollen
        cols_to_drop_for_scaling = ['close', f'ema_{ema_period}', f'atr_{atr_period}', f'natr_{atr_period}']
        feature_columns_to_scale = self.data.columns.drop(cols_to_drop_for_scaling, errors='ignore')
        
        scaled_features_df = self.data.copy()
        try:
            scaled_features_df[feature_columns_to_scale] = self.scaler.transform(self.data[feature_columns_to_scale])
        except Exception as e:
            logging.error(f"Fehler bei der Skalierung im Backtester: {e}")
            return self._calculate_metrics()
        
        position = None
        entry_price = 0
        
        model_input_feature_names = self.scaler.get_feature_names_out()

        for i in range(self.sequence_length, len(scaled_features_df)):
            if position:
                # ... (Logik zum Schließen der Position bleibt unverändert) ...
                current_price = self.data['close'].iloc[i]
                pnl_pct = (current_price - entry_price) / entry_price
                if position == 'long':
                    if pnl_pct <= -self.sl_pct:
                        self._close_position(i, 'SL')
                        position = None
                    elif pnl_pct >= self.tp_pct:
                        self._close_position(i, 'TP')
                        position = None
            
            if not position:
                sequence_df = scaled_features_df.iloc[i - self.sequence_length : i]
                model_input_values = sequence_df[model_input_feature_names].values
                input_data = np.expand_dims(model_input_values, axis=0)
                
                prediction = self.model.predict(input_data, verbose=0)[0][0]
                pred_threshold = self.params['strategy']['prediction_threshold']
                
                # MODIFIZIERT: Alle Filter werden jetzt hier angewendet
                if (prediction >= pred_threshold and 
                    self.params['behavior'].get('use_longs', False) and 
                    self._is_trend_filter_ok(i, 'long') and
                    self._is_volatility_filter_ok(i)):
                    
                    position = 'long'
                    # ... (Rest der Logik zur Trade-Eröffnung bleibt unverändert) ...
                    leverage = self.params['risk']['leverage']
                    risk_per_trade = self.params['risk']['risk_per_trade_pct'] / 100
                    rr_ratio = self.params['risk']['risk_reward_ratio']
                    self.sl_pct = risk_per_trade / leverage
                    self.tp_pct = self.sl_pct * rr_ratio
                    self._open_position(i, 'long')
                    entry_price = self.trades[-1]['entry_price']

        return self._calculate_metrics()
    
    # ... (Restliche Funktionen _open_position, _close_position, _calculate_metrics bleiben unverändert) ...
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
        trade_to_close = next((t for t in reversed(self.trades) if t['status'] == 'open'), None)
        if not trade_to_close: return
        
        raw_price = self.data['close'].iloc[index]
        exit_price_with_slippage = self._apply_slippage(raw_price, 'short')
        
        trade_to_close['status'] = 'closed'
        trade_to_close['exit_index'] = index
        trade_to_close['exit_date'] = self.data.index[index]
        trade_to_close['exit_price'] = exit_price_with_slippage
        trade_to_close['reason'] = reason
        
        entry_price = trade_to_close['entry_price']
        exit_price = trade_to_close['exit_price']
        pnl_pct = (exit_price - entry_price) / entry_price
        
        leverage = self.params['risk']['leverage']
        capital = self.equity_curve[-1]
        
        entry_cost = capital * leverage * self.fee_rate
        exit_cost = capital * leverage * (1 + pnl_pct) * self.fee_rate
        total_fees = entry_cost + exit_cost
        
        pnl_amount = (capital * pnl_pct * leverage) - total_fees
        self.equity_curve.append(capital + pnl_amount)

    def _calculate_metrics(self):
        if not self.trades:
            return {'total_pnl_pct': 0, 'win_rate': 0, 'max_drawdown_pct': 0, 'num_trades': 0}
            
        df_trades = pd.DataFrame(self.trades)
        
        if 'status' not in df_trades.columns or df_trades[df_trades['status'] == 'closed'].empty:
             return {'total_pnl_pct': 0, 'win_rate': 0, 'max_drawdown_pct': 0, 'num_trades': 0}

        closed_trades = df_trades[df_trades['status'] == 'closed'].copy()
        
        pnl = (closed_trades['exit_price'] - closed_trades['entry_price']) / closed_trades['entry_price']
        final_equity = self.equity_curve[-1]
        total_pnl_pct = (final_equity / self.start_capital - 1) * 100
        win_rate = (pnl > 0).mean() * 100 if not pnl.empty else 0
        
        equity_series = pd.Series(self.equity_curve)
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series - peak) / peak
        max_drawdown_pct = abs(drawdown.min() * 100) if not drawdown.empty and not drawdown.isnull().all() else 0
        
        return {
            'total_pnl_pct': total_pnl_pct,
            'win_rate': win_rate,
            'max_drawdown_pct': max_drawdown_pct,
            'num_trades': len(closed_trades)
        }
