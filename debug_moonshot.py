import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("MOONSHOT_API_KEY")
print("has_key:", bool(api_key))
print("prefix:", api_key[:6] if api_key else None)

url = "https://api.moonshot.cn/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.2,
    "max_tokens": 32,
}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
print(resp.status_code)
print(resp.text)