-----

# L-Bot

Ein vollautomatischer Trading-Bot für Krypto-Futures auf der Bitget-Börse, basierend auf einem **Long Short-Term Memory (LSTM)** neuronalen Netzwerk.

Dieses System ist die Weiterentwicklung des JaegerBots und nutzt ein Modell mit "Gedächtnis", um Handelsentscheidungen nicht nur auf Basis einer Momentaufnahme, sondern auf Basis von **Sequenzen** vergangener Marktdaten zu treffen. Es ist für den autonomen Betrieb auf einem Ubuntu-Server konzipiert.

## Kernstrategie (Prädiktives LSTM-Netzwerk)

Die Handelsthese bleibt, dass Märkte wiederkehrende Muster aufweisen. Der L-Bot hebt dies auf die nächste Stufe, indem er diese Muster im zeitlichen Kontext analysiert.

  * **Sequenz-Analyse:** Im Gegensatz zu einem einfachen ANN, das nur die letzte Kerze betrachtet, analysiert das LSTM-Modell des L-Bots eine ganze Sequenz (z.B. die letzten 24 Kerzen) als zusammenhängendes "Video" des Marktgeschehens.
  * **Gedächtnis:** Das LSTM-Modell besitzt interne Gedächtnis-Zellen, die es ihm ermöglichen, relevante Informationen aus früheren Kerzen der Sequenz zu behalten und für die aktuelle Vorhersage zu nutzen.
  * **Vorhersage-Ziel:** Das Modell ist darauf trainiert, basierend auf der Sequenz eine signifikante Preisbewegung über einen zukünftigen Zeitraum vorherzusagen, was kurzfristiges Rauschen herausfiltert.
  * **Einstieg:** Ein Trade wird nur initiiert, wenn die vom LSTM-Modell berechnete Wahrscheinlichkeit für eine bevorstehende Bewegung einen festgelegten Schwellenwert überschreitet.
  * **Risikomanagement:**
      * Die **Positionsgröße** wird dynamisch vor jedem Trade berechnet. Sie basiert auf einem festen Prozentsatz des **aktuellen, live von der Börse abgerufenen Kontostandes**.
      * Nach der Trade-Eröffnung werden sofort ein fester **Stop Loss** und ein fester **Take Profit** platziert und die Preise auf die von der Börse geforderte Genauigkeit gerundet.

## Architektur & Sicherheitsfeatures

Der L-Bot ist auf maximale Robustheit und Autonomie ausgelegt.

1.  **Der Cronjob (Der Wecker):** Ein einziger Cronjob läuft alle 15 Minuten und startet den `master_runner`.
2.  **Der Master-Runner (Der Dirigent):** Das `master_runner.py`-Skript ist das Gehirn der Automatisierung. Bei jedem Lauf:
      * **Babysitter-Check:** Zuerst prüft er für alle aktiven Strategien, ob eine offene Position ungeschützt ("nackt") ist und sichert sie bei Bedarf sofort ab.
      * **Präzisions-Scheduling:** Danach prüft er, ob für eine Strategie ein neuer, exakter Zeit-Block (z.B. eine neue 4-Stunden-Kerze) begonnen hat und startet nur bei Bedarf den eigentlichen Handelsprozess.
      * **Log-Sammler:** Er fängt die gesamte Ausgabe des Handelsprozesses auf und schreibt sie in die zentrale `cron.log`.
3.  **Der Handelsprozess (Der Agent):**
      * Der **Guardian-Decorator** führt vor jedem Trade automatisierte Sicherheits-Checks durch.
      * Die **Trade-Manager-Logik** führt den Handel aus, inklusive der robusten "Hausmeister-Routine".

-----

## Installation 🚀

Führe die folgenden Schritte auf einem frischen Ubuntu-Server aus.

#### 1\. Projekt klonen

```bash
# Ersetze dies mit dem Link zu deinem neuen L-Bot Git-Repository
git clone https://github.com/Youra82/lbot.git
```

#### 2\. Installations-Skript ausführen

```bash
cd lbot
chmod +x install.sh
bash ./install.sh
```

#### 3\. API-Schlüssel eintragen

Erstelle eine Kopie der Vorlage und trage deine Schlüssel ein. Der Bot-Name in der Datei wurde zu "lbot" geändert.

```bash
cp secret.json.example secret.json
nano secret.json
```

Speichere mit `Strg + X`, dann `Y`, dann `Enter`.

-----

## Konfiguration & Automatisierung

#### 1\. Neue LSTM-Modelle trainieren (WICHTIG)

Nach der Installation musst du zuerst die neuen, auf LSTM basierenden Modelle trainieren.

```bash
bash ./run_pipeline.sh
```

Das Skript fragt dich nach den zu trainierenden Symbolen und Zeitfenstern. Dieser Prozess kann je nach Datenmenge mehrere Stunden dauern. Die `config_...json`-Dateien werden in `src/lbot/strategy/configs/` gespeichert.

#### 2\. Strategien für den Handel aktivieren

Bearbeite die zentrale Steuerungsdatei `settings.json`, um die Strategien zu definieren, die der `master_runner` überwachen soll.

```bash
nano settings.json
```

**Beispiel `settings.json`:**

```json
{
    "security_settings": {
        "trade_manager_hash": null
    },
    "live_trading_settings": {
        "use_auto_optimizer_results": false,
        "active_strategies": [
            {
                "symbol": "AAVE/USDT:USDT",
                "timeframe": "1d"
            },
            {
                "symbol": "BIO/USDT:USDT",
                "timeframe": "4h"
            }
        ]
    },
    "optimization_settings": {
        "enabled": false
    }
}
```

#### 3\. Automatisierung per Cronjob einrichten

Richte den automatischen Prozess für den Live-Handel ein.

```bash
crontab -e
```

Füge die folgende **eine Zeile** am Ende der Datei ein. Passe den Pfad an, falls dein Bot nicht unter `/home/ubuntu/lbot` liegt.

```
# Starte den L-Bot Master-Runner alle 15 Minuten
*/15 * * * * /home/ubuntu/lbot/.venv/bin/python3 /home/ubuntu/lbot/master_runner.py >> /home/ubuntu/lbot/logs/cron.log 2>&1
```

-----

## Tägliche Verwaltung & Wichtige Befehle ⚙️

#### Logs ansehen

Die zentrale `cron.log`-Datei enthält **alle** wichtigen Informationen.

  * **Logs live mitverfolgen (der wichtigste Befehl):**

    ```bash
    tail -f logs/cron.log
    ```

    *(Mit `Strg + C` beenden)*

  * **Die letzten 200 Zeilen der zentralen Log-Datei anzeigen:**

    ```bash
    tail -n 200 logs/cron.log
    ```

  * **Zentrale Log-Datei nach Fehlern durchsuchen:**

    ```bash
    grep -i "ERROR" logs/cron.log
    ```

#### Cronjob manuell testen

Um den `master_runner` sofort auszuführen:

```bash
/home/ubuntu/lbot/.venv/bin/python3 /home/ubuntu/lbot/master_runner.py >> /home/ubuntu/lbot/logs/cron.log 2>&1
```

#### Bot aktualisieren

Um die neueste Version des Codes von deinem Git-Repository zu holen:

```bash
bash ./update.sh
```

-----

### ⚠️ Disclaimer

Dieses Material dient ausschließlich zu Bildungs- und Unterhaltungszwecken. Es handelt sich nicht um eine Finanzberatung. Der Nutzer trägt die alleinige Verantwortung für alle Handlungen. Der Autor haftet nicht für etwaige Verluste.
