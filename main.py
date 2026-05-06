import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI(title="Avito Bot")

CLIENT_ID = os.getenv("AVITO_CLIENT_ID")
CLIENT_SECRET = os.getenv("AVITO_CLIENT_SECRET")
REDIRECT_URI = "https://arkadiybiggi-danisavitobot.hf.space/callback"

AVITO_AUTH_URL = "https://www.avito.ru/oauth"
AVITO_TOKEN_URL = "https://api.avito.ru/token"
AVITO_API_BASE = "https://api.avito.ru"

# Хранилище токена в памяти (для теста)
token_storage = {}


@app.get("/")
async def home():
    if "access_token" in token_storage:
        return HTMLResponse(
            "<h2>✅ Бот авторизован</h2>"
            "<p><a href='/messages'>Посмотреть сообщения</a></p>"
        )
    return HTMLResponse(
        "<h2>Avito Bot</h2>"
        "<p><a href='/login'>Авторизоваться через Авито</a></p>"
    )


@app.get("/login")
async def login():
    auth_url = (
        f"{AVITO_AUTH_URL}?"
        f"client_id={CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={REDIRECT_URI}&"
        f"scope=messenger:read messenger:write user:read"
    )
    return RedirectResponse(auth_url)


@app.get("/callback")
async def callback(code: str = None, error: str = None):
    if error:
        return HTMLResponse(f"<h2>❌ Ошибка авторизации: {error}</h2>")

    if not code:
        return HTMLResponse("<h2>❌ Код авторизации не получен</h2>")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            AVITO_TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
        )

    if response.status_code != 200:
        return HTMLResponse(f"<h2>❌ Ошибка получения токена:</h2><pre>{response.text}</pre>")

    token_data = response.json()
    token_storage["access_token"] = token_data.get("access_token")
    token_storage["user_id"] = token_data.get("user_id")

    return HTMLResponse(
        "<h2>✅ Авторизация успешна!</h2>"
        "<p><a href='/messages'>Посмотреть входящие сообщения</a></p>"
    )


@app.get("/messages")
async def get_messages():
    if "access_token" not in token_storage:
        return HTMLResponse("<h2>❌ Не авторизован. <a href='/login'>Войти</a></h2>")

    access_token = token_storage["access_token"]
    user_id = token_storage.get("user_id")
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        chats_resp = await client.get(
            f"{AVITO_API_BASE}/messenger/v3/accounts/{user_id}/chats",
            headers=headers,
            params={"limit": 20},
        )

    if chats_resp.status_code != 200:
        return {"error": chats_resp.text}

    chats_data = chats_resp.json()
    chats = chats_data.get("chats", [])

    if not chats:
        return HTMLResponse("<h2>📭 Нет активных чатов</h2>")

    html = "<h2>📬 Входящие чаты</h2><ul>"
    for chat in chats:
        chat_id = chat.get("id")
        last_message = chat.get("last_message", {})
        text = last_message.get("content", {}).get("text", "—")
        author = last_message.get("author_id", "неизвестно")
        html += f"<li><b>Чат {chat_id}</b><br>Автор: {author}<br>Сообщение: {text}</li><br>"
    html += "</ul>"

    return HTMLResponse(html)
