# src/lbot/utils/telegram.py
import requests

def send_message(bot_token, chat_id, text):
    """ Sendet eine Nachricht über einen Telegram-Bot. """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown'
    }
    if not bot_token or not chat_id:
        print("Telegram-Token oder Chat-ID nicht konfiguriert. Nachricht wird nicht gesendet.")
        print(f"Nachricht: {text}")
        return

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Senden der Telegram-Nachricht: {e}")
