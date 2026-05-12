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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты — вежливый и опытный продавец-консультант на Авито. "
    "Отвечай покупателям коротко (2-3 предложения), по делу, дружелюбно. "
    "Цель — заинтересовать покупателя и довести до сделки. "
    "Не выдумывай характеристики товара — если не знаешь, скажи что уточнишь.",
)

ENABLE_AUTO_REPLY = os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true"

AVITO_TOKEN_URL = "https://api.avito.ru/token"
AVITO_API_BASE = "https://api.avito.ru"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

POLL_INTERVAL = 10

token_storage: dict = {}
messages_cache: list[dict] = []
# Хранит id последнего обработанного сообщения по каждому чату
replied_messages: dict[str, str] = {}
# Хранит последний сгенерированный ответ Gemini по каждому чату
generated_replies: dict[str, dict] = {}


async def get_token() -> str | None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                AVITO_TOKEN_URL,
                data={
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "grant_type": "client_credentials",
                },
            )
        if resp.status_code != 200:
            logger.error("Ошибка токена %s: %s", resp.status_code, resp.text)
            return None
        return resp.json().get("access_token")
    except Exception:
        logger.exception("Исключение в get_token")
        return None


async def get_user_id(token: str) -> str | None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{AVITO_API_BASE}/core/v1/accounts/self",
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code != 200:
        logger.error("Ошибка user_id: %s", resp.text)
        return None
    return str(resp.json().get("id"))


async def fetch_chats(token: str, user_id: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{AVITO_API_BASE}/messenger/v2/accounts/{user_id}/chats/",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": 20},
        )
    if resp.status_code == 401:
        token_storage.clear()
        return []
    if resp.status_code != 200:
        logger.error("Ошибка чатов: %s", resp.text)
        return []
    return resp.json().get("chats", [])


async def send_message(token: str, user_id: str, chat_id: str, text: str) -> bool:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{AVITO_API_BASE}/messenger/v1/accounts/{user_id}/chats/{chat_id}/messages",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"message": {"text": text}, "type": "text"},
        )
    if resp.status_code not in (200, 201):
        logger.error("Ошибка отправки в чат %s: %s %s", chat_id, resp.status_code, resp.text)
        return False
    logger.info("Сообщение отправлено в чат %s", chat_id)
    return True


async def generate_reply(buyer_message: str, item_title: str = "") -> str | None:
    """Генерирует ответ через Gemini."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY не задан")
        return None

    context = f"\n\nТовар: {item_title}" if item_title else ""
    prompt = (
        f"{SYSTEM_PROMPT}{context}\n\n"
        f"Сообщение покупателя: {buyer_message}\n\n"
        f"Твой ответ:"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
        if resp.status_code != 200:
            logger.error("Ошибка Gemini: %s %s", resp.status_code, resp.text)
            return None
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        logger.exception("Ошибка генерации ответа")
        return None


def _is_system_message(text: str, author_id: str) -> bool:
    if author_id in ("1", "0"):
        return True
    if text.startswith("[Системное сообщение]"):
        return True
    return False


async def process_chat(token: str, user_id: str, chat: dict) -> None:
    """Проверяет последнее сообщение в чате и при необходимости отвечает."""
    chat_id = chat.get("id")
    last_message = chat.get("last_message", {})
    msg_id = last_message.get("id")
    author_id = str(last_message.get("author_id", ""))
    text = last_message.get("content", {}).get("text", "")

    if not msg_id:
        return

    # Уже обрабатывали это сообщение — пропускаем (даже если генерация падала)
    if replied_messages.get(chat_id) == msg_id:
        return

    # Помечаем как обработанное сразу — чтобы не было ретрая на каждый poll
    replied_messages[chat_id] = msg_id

    # Сообщение от нас — пропускаем
    if author_id == user_id:
        return

    # Системные сообщения от Авито — пропускаем
    if _is_system_message(text, author_id):
        return

    if not text:
        return

    logger.info("Новое сообщение в %s: %s", chat_id, text[:80])

    # Заголовок объявления для контекста
    context = chat.get("context", {}).get("value", {})
    item_title = context.get("title", "") if isinstance(context, dict) else ""

    reply = await generate_reply(text, item_title)
    if not reply:
        logger.warning("Не удалось сгенерировать ответ для %s", chat_id)
        return

    logger.info("Ответ Gemini для %s: %s", chat_id, reply[:80])

    # Сохраняем сгенерированный ответ для отображения на /replies
    generated_replies[chat_id] = {
        "buyer_text": text,
        "reply": reply,
        "item_title": item_title,
        "sent": False,
    }

    if ENABLE_AUTO_REPLY:
        if await send_message(token, user_id, chat_id, reply):
            generated_replies[chat_id]["sent"] = True
    else:
        logger.info("Auto-reply выключен, ответ не отправлен")


async def poll_avito():
    logger.info("Polling запущен (интервал %ds, auto_reply=%s)", POLL_INTERVAL, ENABLE_AUTO_REPLY)

    token = await get_token()
    if not token:
        return
    user_id = await get_user_id(token)
    if not user_id:
        return

    token_storage["access_token"] = token
    token_storage["user_id"] = user_id
    logger.info("Авторизован, user_id=%s", user_id)

    while True:
        try:
            if not token_storage.get("access_token"):
                new_token = await get_token()
                if new_token:
                    token_storage["access_token"] = new_token

            current_token = token_storage.get("access_token")
            current_user_id = token_storage.get("user_id")
            if current_token and current_user_id:
                chats = await fetch_chats(current_token, current_user_id)
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
                        await process_chat(current_token, current_user_id, chat)
                    logger.info("Обработано %d чатов", len(chats))
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
            f"<p>Auto-reply: {'ВКЛ' if ENABLE_AUTO_REPLY else 'ВЫКЛ (просмотр)'}</p>"
            f"<p>AI: Gemini 1.5 Flash</p>"
            "<ul>"
            "<li><a href='/messages'>📬 Входящие сообщения</a></li>"
            "<li><a href='/replies'>🤖 Сгенерированные ответы</a></li>"
            "</ul>"
        )
    return HTMLResponse("<h2>⏳ Бот запускается...</h2>")


@app.get("/health")
async def health():
    return {"status": "ok", "authorized": bool(token_storage.get("access_token"))}


@app.get("/models")
async def list_models():
    """Список доступных Gemini моделей для текущего ключа."""
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY не задан"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        )
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    models = []
    for m in data.get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            models.append(m.get("name", "").replace("models/", ""))
    return {"current_model": GEMINI_MODEL, "available_models": models}


@app.get("/replies")
async def get_replies():
    if not token_storage.get("access_token"):
        return HTMLResponse("<h2>⏳ Бот ещё не авторизован</h2>")
    if not generated_replies:
        return HTMLResponse(
            "<h2>🤖 Сгенерированные ответы</h2>"
            "<p>Пока ничего не сгенерировано. Жди новых сообщений от покупателей.</p>"
            "<p><a href='/replies'>🔄 Обновить</a></p>"
        )

    mode = "ВКЛ — отправлено в Авито" if ENABLE_AUTO_REPLY else "ВЫКЛ — только просмотр"
    html = f"<h2>🤖 Сгенерированные ответы Gemini</h2><p>Auto-reply: <b>{mode}</b></p>"
    html += "<table border='1' cellpadding='8' style='border-collapse:collapse'>"
    html += (
        "<tr style='background:#eee'>"
        "<th>Chat ID</th><th>Объявление</th><th>Покупатель написал</th>"
        "<th>Ответ бота</th><th>Отправлено?</th></tr>"
    )
    for chat_id, data in generated_replies.items():
        sent_mark = "✅" if data.get("sent") else "—"
        html += (
            f"<tr>"
            f"<td>{chat_id}</td>"
            f"<td>{data.get('item_title', '')}</td>"
            f"<td>{data.get('buyer_text', '')}</td>"
            f"<td style='background:#f0fff0'>{data.get('reply', '')}</td>"
            f"<td style='text-align:center'>{sent_mark}</td>"
            f"</tr>"
        )
    html += "</table><p><a href='/replies'>🔄 Обновить</a> | <a href='/'>🏠 Главная</a></p>"
    return HTMLResponse(html)


@app.get("/messages")
async def get_messages():
    if not token_storage.get("access_token"):
        return HTMLResponse("<h2>⏳ Бот ещё не авторизован</h2>")
    if not messages_cache:
        return HTMLResponse("<h2>📭 Нет сообщений</h2>")

    html = "<h2>📬 Последние сообщения</h2>"
    html += "<table border='1' cellpadding='8'>"
    html += "<tr><th>Chat ID</th><th>Автор ID</th><th>Сообщение</th><th>Время</th></tr>"
    for m in messages_cache:
        html += (
            f"<tr><td>{m['chat_id']}</td><td>{m['author']}</td>"
            f"<td>{m['text']}</td><td>{m['created']}</td></tr>"
        )
    html += "</table><p><a href='/messages'>🔄 Обновить</a></p>"
    return HTMLResponse(html)
