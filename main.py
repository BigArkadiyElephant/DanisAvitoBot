import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("avitobot")

CLIENT_ID = os.getenv("AVITO_CLIENT_ID")
CLIENT_SECRET = os.getenv("AVITO_CLIENT_SECRET")
REDIRECT_URI = "https://arkadiybiggi-danisavitobot.hf.space/callback"

AVITO_AUTH_URL = "https://www.avito.ru/oauth"
AVITO_TOKEN_URL = "https://api.avito.ru/token"
AVITO_API_BASE = "https://api.avito.ru"

POLL_INTERVAL = 10  # секунд между опросами

token_storage: dict = {}
# Хранит последние сообщения по chat_id для отображения в интерфейсе
messages_cache: list[dict] = []


async def fetch_chats() -> list[dict]:
    access_token = token_storage.get("access_token")
    user_id = token_storage.get("user_id")
    if not access_token or not user_id:
        return []

    headers = {"Authorization": f"Bearer {access_token}"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{AVITO_API_BASE}/messenger/v3/accounts/{user_id}/chats",
            headers=headers,
            params={"limit": 20},
        )

    if resp.status_code == 401:
        logger.warning("Токен истёк, нужна повторная авторизация")
        token_storage.clear()
        return []

    if resp.status_code != 200:
        logger.error("Ошибка получения чатов: %s", resp.text)
        return []

    return resp.json().get("chats", [])


async def poll_avito():
    """Фоновый polling: каждые POLL_INTERVAL секунд читает новые сообщения."""
    logger.info("Polling запущен (интервал %ds)", POLL_INTERVAL)
    while True:
        try:
            if token_storage.get("access_token"):
                chats = await fetch_chats()
                if chats:
                    messages_cache.clear()
                    for chat in chats:
                        last = chat.get("last_message", {})
                        messages_cache.append({
                            "chat_id": chat.get("id"),
                            "author": last.get("author_id", "—"),
                            "text": last.get("content", {}).get("text", "—"),
                            "created": last.get("created", ""),
                        })
                    logger.info("Получено %d чатов", len(chats))
        except Exception:
            logger.exception("Ошибка в poll_avito")

        await asyncio.sleep(POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(poll_avito())
    yield
    task.cancel()


app = FastAPI(title="Avito Bot", lifespan=lifespan)


@app.get("/")
async def home():
    if token_storage.get("access_token"):
        return HTMLResponse(
            "<h2>✅ Бот авторизован и работает</h2>"
            f"<p>User ID: {token_storage.get('user_id')}</p>"
            "<p><a href='/messages'>📬 Посмотреть сообщения</a></p>"
        )
    return HTMLResponse(
        "<h2>Avito Bot</h2>"
        "<p><a href='/login'>🔑 Авторизоваться через Авито</a></p>"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "authorized": bool(token_storage.get("access_token"))}


@app.get("/login")
async def login():
    from urllib.parse import urlencode
    params = urlencode({
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": "user:read",
    })
    return RedirectResponse(f"{AVITO_AUTH_URL}?{params}")


@app.get("/callback")
async def callback(request: Request, code: str = None, error: str = None):
    all_params = dict(request.query_params)
    logger.info("Callback params: %s", all_params)
    if error:
        return HTMLResponse(f"<h2>❌ Ошибка: {error}</h2><pre>{all_params}</pre>")
    if not code:
        return HTMLResponse(
            f"<h2>❌ Код авторизации не получен</h2>"
            f"<pre>Параметры от Авито: {all_params}</pre>"
            f"<p><a href='/login'>Попробовать снова</a></p>"
        )

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            AVITO_TOKEN_URL,
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": REDIRECT_URI,
            },
        )

    if resp.status_code != 200:
        return HTMLResponse(
            f"<h2>❌ Ошибка получения токена:</h2><pre>{resp.text}</pre>"
        )

    data = resp.json()
    token_storage["access_token"] = data.get("access_token")
    token_storage["user_id"] = data.get("user_id")
    logger.info("Авторизация успешна, user_id=%s", token_storage["user_id"])

    return HTMLResponse(
        "<h2>✅ Авторизация успешна! Polling запущен.</h2>"
        "<p><a href='/messages'>📬 Посмотреть сообщения</a></p>"
    )


@app.get("/messages")
async def get_messages():
    if not token_storage.get("access_token"):
        return HTMLResponse(
            "<h2>❌ Не авторизован. <a href='/login'>Войти</a></h2>"
        )

    if not messages_cache:
        return HTMLResponse(
            "<h2>📭 Нет сообщений (или polling ещё не получил данные)</h2>"
            "<p><a href='/messages'>🔄 Обновить</a></p>"
        )

    html = "<h2>📬 Последние сообщения</h2><table border='1' cellpadding='8'>"
    html += "<tr><th>Chat ID</th><th>Автор ID</th><th>Сообщение</th><th>Время</th></tr>"
    for m in messages_cache:
        html += (
            f"<tr>"
            f"<td>{m['chat_id']}</td>"
            f"<td>{m['author']}</td>"
            f"<td>{m['text']}</td>"
            f"<td>{m['created']}</td>"
            f"</tr>"
        )
    html += "</table><p><a href='/messages'>🔄 Обновить</a></p>"
    return HTMLResponse(html)
