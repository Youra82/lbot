# src/lbot/analysis/result_selector.py
import os
import sys
import json
import pandas as pd

# --- Pfad-Konfiguration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'optimization_results.json')
SETTINGS_FILE = os.path.join(PROJECT_ROOT, 'settings.json')

def main():
    print("--- L-Bot Interaktiver Strategie-Aktivator ---")

    # 1. Lade Ergebnisse und Einstellungen
    if not os.path.exists(RESULTS_FILE):
        print(f"\nFehler: Ergebnisdatei nicht gefunden unter {RESULTS_FILE}")
        print("Bitte führe zuerst die Optimierungs-Pipeline aus (`run_pipeline.sh`).")
        return

    with open(RESULTS_FILE, 'r') as f:
        all_results = json.load(f)
    with open(SETTINGS_FILE, 'r') as f:
        settings = json.load(f)
        
    constraints = settings['optimization_settings'].get('constraints', {})
    print(f"\nFiltere Strategien anhand der Regeln in settings.json: {constraints}")

    # 2. Filtere und sortiere die Ergebnisse
    qualified_strategies = []
    for result in all_results:
        metrics = result['metrics']
        if (metrics['max_drawdown_pct'] <= constraints.get('max_drawdown_pct', 99) and
            metrics['win_rate'] >= constraints.get('min_win_rate_pct', 0) and
            metrics['total_pnl_pct'] >= constraints.get('min_pnl_pct', 0)):
            qualified_strategies.append(result)
            
    if not qualified_strategies:
        print("\nKeine Strategien haben die Kriterien erfüllt. Es gibt nichts zu aktivieren.")
        return
        
    qualified_strategies.sort(key=lambda x: x['score'], reverse=True)

    # 3. Zeige die qualifizierten Strategien in einer Tabelle an
    display_data = []
    for i, strat in enumerate(qualified_strategies):
        display_data.append({
            "#": i + 1,
            "Strategie": f"{strat['symbol']} ({strat['timeframe']})",
            "Score": strat['score'],
            "PnL (%)": strat['metrics']['total_pnl_pct'],
            "Max DD (%)": strat['metrics']['max_drawdown_pct'],
            "Win-Rate (%)": strat['metrics']['win_rate'],
            "Trades": strat['metrics']['num_trades'],
        })
        
    df = pd.DataFrame(display_data)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("\nFolgende Strategien haben sich qualifiziert (sortiert nach bestem Score):")
    print(df.to_string(index=False))

    # 4. Frage den Benutzer nach seiner Auswahl
    while True:
        try:
            choice = input("\nBitte wähle die zu aktivierenden Strategien (z.B. '1 3 4'), 'a' für alle, oder 'q' zum Abbrechen: ").lower()
            if choice == 'q':
                print("Abgebrochen. Es wurden keine Änderungen vorgenommen.")
                return

            selected_indices = []
            if choice == 'a':
                selected_indices = list(range(len(qualified_strategies)))
            else:
                selected_indices = [int(i) - 1 for i in choice.split()]

            # Validiere die Auswahl
            if any(i < 0 or i >= len(qualified_strategies) for i in selected_indices):
                raise ValueError("Ungültige Auswahl. Bitte nur Nummern aus der Liste verwenden.")
            
            break
        except ValueError as e:
            print(f"Fehler: {e}. Bitte versuche es erneut.")
            
    # 5. Aktualisiere die settings.json
    new_active_strategies = []
    print("\nFolgende Strategien werden aktiviert:")
    for index in selected_indices:
        strat = qualified_strategies[index]
        print(f" -> {strat['symbol']} ({strat['timeframe']})")
        new_active_strategies.append({
            "symbol": strat['symbol'],
            "timeframe": strat['timeframe']
        })
        
    settings['live_trading_settings']['active_strategies'] = new_active_strategies
    
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)
        
    print(f"\n✅ settings.json wurde erfolgreich mit {len(new_active_strategies)} Strategie(n) aktualisiert!")


if __name__ == "__main__":
    main()
