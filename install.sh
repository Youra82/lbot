#!/bin/bash
# Bricht das Skript bei jedem Fehler sofort ab
set -e

# --- Farbcodes für eine schönere Ausgabe ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}        L-Bot Vollautomatische Installation           ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# --- Pfade definieren ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# --- System-Pakete aktualisieren und installieren ---
echo -e "\n${YELLOW}Schritt 1/3: System-Pakete werden aktualisiert (apt)...${NC}"
sudo apt-get update -y &> /dev/null
sudo apt-get install python3-pip python3-venv -y &> /dev/null
echo -e "${GREEN}✔ System-Pakete sind aktuell.${NC}"

# --- Virtuelle Umgebung erstellen ---
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n${YELLOW}Schritt 2/3: Virtuelle Python-Umgebung wird erstellt...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✔ Virtuelle Umgebung in '$VENV_DIR' wurde erstellt.${NC}"
else
    echo -e "\n${YELLOW}Schritt 2/3: Virtuelle Python-Umgebung existiert bereits.${NC}"
fi

# --- Python-Abhängigkeiten installieren ---
echo -e "\n${YELLOW}Schritt 3/3: Python-Pakete werden installiert (pip)...${NC}"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip &> /dev/null
pip install -r "$REQUIREMENTS_FILE"
deactivate
echo -e "${GREEN}✔ Alle Python-Pakete aus requirements.txt sind installiert.${NC}"

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${GREEN}✅ Installation erfolgreich abgeschlossen!${NC}"
echo -e "${BLUE}=======================================================${NC}"
