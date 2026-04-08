import requests

TOKEN = "8600372885:AAH9MaxJx1xYhcZfRWDKFsbHKRUkOAfJaM8"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

print(requests.get(url).json())