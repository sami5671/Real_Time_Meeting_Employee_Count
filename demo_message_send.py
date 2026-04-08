import requests

TOKEN = "8600372885:AAH9MaxJx1xYhcZfRWDKFsbHKRUkOAfJaM8"
chat_id = "6354552421"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
res = requests.post(url, data={
    "chat_id": chat_id,
    "text": "Test message hi saima"
})

print(res.json())