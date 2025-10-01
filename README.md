# L-Bot

Ein vollautomatischer Trading-Bot für Krypto-Futures auf der Bitget-Börse, basierend auf einem **Long Short-Term Memory (LSTM)** neuronalen Netzwerk.

Dieses System ist für den autonomen Betrieb auf einem Ubuntu-Server konzipiert und nutzt ein Modell mit "Gedächtnis", um Handelsentscheidungen nicht nur auf Basis einer Momentaufnahme, sondern auf Basis von **Sequenzen** vergangener Marktdaten zu treffen.

## Kernstrategie (Prädiktives LSTM-Netzwerk)

Die Handelsthese ist, dass Märkte wiederkehrende Muster im zeitlichen Kontext aufweisen, die ein LSTM-Modell effektiv lernen kann.

* **Sequenz-Analyse:** Das LSTM-Modell analysiert eine ganze Sequenz (z.B. die letzten 24 Kerzen) als zusammenhängendes "Video" des Marktgeschehens, anstatt nur einen einzelnen Zeitpunkt zu betrachten.
* **Gedächtnis:** Das Modell besitzt interne Gedächtnis-Zellen, die es ihm ermöglichen, relevante Informationen aus früheren Kerzen der Sequenz zu behalten und für die aktuelle Vorhersage zu nutzen.
* **Vorhersage-Ziel:** Das Modell ist darauf trainiert, eine signifikante Preisbewegung über einen zukünftigen Zeitraum vorherzusagen, was kurzfristiges Marktrauschen herausfiltert.
* **Einstieg:** Ein Trade wird nur initiiert, wenn die vom LSTM-Modell berechnete Wahrscheinlichkeit für eine bevorstehende Bewegung einen optimierten Schwellenwert überschreitet.
* **Dynamisches Risikomanagement:**
    * Die **Positionsgröße** wird dynamisch vor jedem Trade berechnet. Sie basiert auf einem festen Prozentsatz des **aktuellen, live von der Börse abgerufenen Kontostandes**.
    * Nach der Trade-Eröffnung werden sofort ein fester **Stop Loss** und ein fester **Take Profit** platziert.

## Architektur & Sicherheitsfeatures

Der L-Bot ist auf maximale Robustheit und Autonomie ausgelegt.

1.  **Der Cronjob (Der Wecker):** Ein einziger Cronjob läuft alle 15 Minuten und startet den `master_runner`. Er ist die einzige externe Abhängigkeit.
2.  **Der Master-Runner (Der Dirigent):** Das `master_runner.py`-Skript ist das Gehirn der Automatisierung. Es stellt sicher, dass jede Strategie genau einmal pro Kerze ausgeführt wird und verwaltet offene Positionen.
    * **Babysitter-Check:** Zuerst prüft er für alle aktiven Strategien, ob eine offene Position ungeschützt ist und sichert sie bei Bedarf sofort ab.
    * **Präzisions-Scheduling:** Danach prüft er, ob für eine Strategie ein neuer, exakter Zeit-Block (z.B. eine neue 4-Stunden-Kerze) begonnen hat und startet nur dann den Handelsprozess.
3.  **Der Handelsprozess (Der Agent):**
    * Der **Guardian-Decorator** führt vor jedem Trade automatisierte Sicherheits-Checks durch (API-Verbindung, Konfiguration, Risikoparameter).
    * Die **Trade-Manager-Logik** führt den Handel aus, inklusive robuster Routinen zur Absicherung der Position.

---

## Installation 🚀

Führe die folgenden Schritte auf einem frischen Ubuntu-Server aus.

#### 1. Projekt klonen

```bash
# Ersetze dies mit dem Link zu deinem L-Bot Git-Repository
git clone [https://github.com/Youra82/lbot.git](https://github.com/Youra82/lbot.git)
---

## Installation 🚀

Führe die folgenden Schritte auf einem frischen Ubuntu-Server aus.

#### 1. Projekt klonen

```bash
# Ersetze dies mit dem Link zu deinem L-Bot Git-Repository
git clone [https://github.com/Youra82/lbot.git](https://github.com/Youra82/lbot.git)
````

#### 2\. Installations-Skript ausführen

Das Skript installiert alle System- und Python-Abhängigkeiten.

```bash
cd lbot
chmod +x install.sh
bash ./install.sh
```

#### 3\. API-Schlüssel eintragen

Erstelle die `secret.json` Datei und trage deine API-Schlüssel und Telegram-Daten ein.

```bash
# Öffne die Datei im Nano-Editor
nano secret.json
```

Füge den folgenden Inhalt ein und ersetze die Platzhalter durch deine echten Daten:

```json
{
    "lbot": [
        {
            "name": "MeinBitgetKonto",
            "apiKey": "DEIN_API_KEY_HIER_EINFUEGEN",
            "secret": "DEIN_API_SECRET_HIER_EINFUEGEN",
            "password": "DEIN_API_PASSWORT_HIER_EINFUEGEN"
        }
    ],
    "telegram": {
        "bot_token": "DEIN_TELEGRAM_BOT_TOKEN_HIER_EINFUEGEN",
        "chat_id": "DEINE_TELEGRAM_CHAT_ID_HIER_EINFUEGEN"
    }
}
```

Speichere die Datei mit `Strg + X`, dann `Y`, dann `Enter`.

-----

## Konfiguration & Automatisierung

#### 1\. Modelle trainieren & Parameter optimieren (WICHTIG)

Nach der Installation musst du zuerst die Modelle trainieren und die Handelsparameter optimieren.

```bash
# Startet die interaktive Pipeline für Training & Optimierung
bash ./run_pipeline.sh
```

Das Skript fragt dich nach den zu trainierenden Symbolen (z.B. `BTC ETH SOL`) und Zeitfenstern (z.B. `4h 1d`). Dieser Prozess kann je nach Datenmenge mehrere Stunden dauern. Die resultierenden `config_...json`-Dateien werden in `src/lbot/strategy/configs/` gespeichert.

#### 2\. Strategien für den Live-Handel aktivieren

Bearbeite die zentrale Steuerungsdatei `settings.json`, um die Strategien zu definieren, die der `master_runner` überwachen soll.

```bash
nano settings.json
```

Passe den Abschnitt `active_strategies` an. Nur die hier eingetragenen Paare werden live gehandelt. **Stelle sicher, dass für jede aktive Strategie eine `config_...json`-Datei aus Schritt 1 existiert\!**

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

Hier sind die wichtigsten Befehle, um deinen Bot im Auge zu behalten.

#### 🖥️ Logs überwachen

Die Logs sind deine wichtigste Informationsquelle.

```bash
# Logs live mitverfolgen (der wichtigste Befehl)
tail -f logs/cron.log

# Die letzten 200 Zeilen der zentralen Log-Datei anzeigen
tail -n 200 logs/cron.log

# Log-Datei einer spezifischen Strategie ansehen (Beispiel für BTC 4h)
tail -f logs/lbot_BTCUSDTUSDT_4h.log

# Alle Log-Dateien nach kritischen Fehlern durchsuchen
grep -ri "CRITICAL" logs/

# Alle Log-Dateien nach dem Wort "Position" durchsuchen
grep -ri "Position" logs/
```

*(Live-Logs mit `Strg + C` beenden)*

#### 📈 Analyse & Ergebnisse

```bash
# Backtest-Ergebnisse für alle trainierten Strategien anzeigen
bash ./show_results.sh
```

#### 💡 Prozess-Management

```bash
# Überprüfen, ob der Master-Runner-Prozess gerade aktiv ist
ps aux | grep master_runner.py

# Falls ein Prozess hängt, kann er damit beendet werden (normalerweise nicht nötig)
pkill -f master_runner.py
```

#### 🛠️ System-Wartung

```bash
# Speicherplatz des Daten-Caches anzeigen
du -sh data/cache/

# Den gesamten Daten-Cache manuell leeren (wird beim nächsten Lauf neu aufgebaut)
rm -rf data/cache/*

# Alle Log-Dateien leeren, um Platz zu schaffen
find logs/ -type f -delete
```

#### 🔄 Bot auf den neuesten Stand bringen

Um die neueste Version des Codes von deinem Git-Repository zu holen und notwendige Abhängigkeiten zu installieren.

```bash
# Schritt 1: Code aktualisieren (sichert deine secret.json)
bash ./update.sh

# Schritt 2: Python-Pakete auf den neuesten Stand bringen
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```

-----

### ⚠️ Disclaimer

Dieses Material dient ausschließlich zu Bildungs- und Unterhaltungszwecken. Es handelt sich nicht um eine Finanzberatung. Der Nutzer trägt die alleinige Verantwortung für alle Handlungen. Der Autor haftet nicht für etwaige Verluste.

```
```
