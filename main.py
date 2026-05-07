import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("avitobot")

CLIENT_ID = os.getenv("AVITO_CLIENT_ID")
CLIENT_SECRET = os.getenv("AVITO_CLIENT_SECRET")

AVITO_TOKEN_URL = "https://api.avito.ru/token"
AVITO_API_BASE = "https://api.avito.ru"

POLL_INTERVAL = 10

token_storage: dict = {}
messages_cache: list[dict] = []


async def get_token() -> str | None:
    """Получает access_token через client_credentials."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            AVITO_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(CLIENT_ID, CLIENT_SECRET),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        logger.error("Ошибка получения токена: %s", resp.text)
        return None
    data = resp.json()
    logger.info("Ответ Авито: %s", data)
    token = data.get("access_token")
    if token:
        logger.info("Токен получен успешно")
    return token


async def get_user_id(token: str) -> str | None:
    """Получает user_id текущего аккаунта."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{AVITO_API_BASE}/core/v1/accounts/self",
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 200:
        logger.error("Ошибка получения user_id: %s", resp.text)
        return None
    return str(resp.json().get("id"))


async def fetch_chats(token: str, user_id: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{AVITO_API_BASE}/messenger/v3/accounts/{user_id}/chats",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": 20},
        )
    if resp.status_code == 401:
        logger.warning("Токен истёк")
        token_storage.clear()
        return []
    if resp.status_code != 200:
        logger.error("Ошибка получения чатов: %s", resp.text)
        return []
    return resp.json().get("chats", [])


async def poll_avito():
    """Фоновый polling: авторизуется и каждые POLL_INTERVAL секунд читает чаты."""
    logger.info("Polling запущен (интервал %ds)", POLL_INTERVAL)

    # Получаем токен при старте
    token = await get_token()
    if not token:
        logger.error("Не удалось получить токен при старте")
        return

    user_id = await get_user_id(token)
    if not user_id:
        logger.error("Не удалось получить user_id")
        return

    token_storage["access_token"] = token
    token_storage["user_id"] = user_id
    logger.info("Авторизован, user_id=%s", user_id)

    while True:
        try:
            # Обновляем токен если истёк
            if not token_storage.get("access_token"):
                token = await get_token()
                if token:
                    token_storage["access_token"] = token

            if token_storage.get("access_token"):
                chats = await fetch_chats(
                    token_storage["access_token"],
                    token_storage["user_id"],
                )
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
            "<h2>✅ Бот работает</h2>"
            f"<p>User ID: {token_storage.get('user_id')}</p>"
            "<p><a href='/messages'>📬 Посмотреть сообщения</a></p>"
        )
    return HTMLResponse("<h2>⏳ Бот запускается...</h2>")


@app.get("/health")
async def health():
    return {"status": "ok", "authorized": bool(token_storage.get("access_token"))}


@app.get("/messages")
async def get_messages():
    if not token_storage.get("access_token"):
        return HTMLResponse("<h2>⏳ Бот ещё не авторизован</h2>")

    if not messages_cache:
        return HTMLResponse(
            "<h2>📭 Нет сообщений</h2>"
            "<p><a href='/messages'>🔄 Обновить</a></p>"
        )

    html = "<h2>📬 Последние сообщения</h2>"
    html += "<table border='1' cellpadding='8'>"
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
