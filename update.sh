#!/bin/bash
set -e

echo "--- Sicheres L-Bot Update wird ausgeführt ---"
cp secret.json secret.json.bak
git fetch origin
git reset --hard origin/main
cp secret.json.bak secret.json
rm secret.json.bak
echo "✅ L-Bot Update erfolgreich abgeschlossen."
