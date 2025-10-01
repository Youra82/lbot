# src/jaegerbot/utils/guardian.py
import os
import hashlib

class PreFlightCheckError(Exception):
    pass

class Guardian:
    def __init__(self, exchange, params, model_path, scaler_path, logger):
        self.exchange = exchange
        self.params = params
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.logger = logger
        self.checklist = [
            self._check_config_sanity,
            self._check_risk_parameters,
            self._check_artifacts_exist,
            self._check_exchange_connection
        ]

    def run_pre_flight_checks(self):
        self.logger.info("Guardian: Starte Pre-Flight-Checks...")
        for check_function in self.checklist:
            check_name = check_function.__name__
            try:
                check_function()
                self.logger.info(f"  ✅ Check bestanden: {check_name}")
            except Exception as e:
                self.logger.critical(f"  ❌ CHECK FEHLGESCHLAGEN: {check_name} -> {e}")
                raise PreFlightCheckError(f"Guardian-Check '{check_name}' fehlgeschlagen: {e}")
        self.logger.info("Guardian: Alle Checks erfolgreich bestanden.")
        return True

    def _check_config_sanity(self):
        required_keys = ['market', 'strategy', 'risk', 'behavior']
        for key in required_keys:
            if key not in self.params:
                raise ValueError(f"Obligatorische Sektion '{key}' fehlt in der Konfiguration.")

    def _check_risk_parameters(self):
        risk_pct = self.params['risk'].get('risk_per_trade_pct', 0)
        leverage = self.params['risk'].get('leverage', 0)
        if not 0 < risk_pct <= 10:
            raise ValueError(f"risk_per_trade_pct ({risk_pct}%) liegt außerhalb des sicheren Bereichs (0-10%).")
        if not 0 < leverage <= 25:
            raise ValueError(f"leverage ({leverage}x) liegt außerhalb des sicheren Bereichs (1-25x).")
        if self.params['risk'].get('risk_reward_ratio', 0) <= 0:
            raise ValueError("risk_reward_ratio muss größer als 0 sein.")

    def _check_artifacts_exist(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelldatei nicht gefunden unter: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler-Datei nicht gefunden unter: {self.scaler_path}")

    def _check_exchange_connection(self):
        try:
            server_time = self.exchange.exchange.fetch_time()
            if not server_time:
                raise ConnectionError("Server-Zeit konnte nicht abgerufen werden.")
        except Exception as e:
            raise ConnectionError(f"Verbindung zur Börse fehlgeschlagen: {e}")
