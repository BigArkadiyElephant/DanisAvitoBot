import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("avitobot")

CLIENT_ID = os.getenv("AVITO_CLIENT_ID")
CLIENT_SECRET = os.getenv("AVITO_CLIENT_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_PROMPT = (
    "Ты — менеджер по продажам услуг разработки. Мы делаем под ключ: "
    "чат-боты с ИИ, нейросетевые решения для бизнеса, сайты. "
    "Пишешь от первого лица потенциальным клиентам на Авито.\n\n"
    "ПРАВИЛА:\n"
    "1. Отвечай коротко (1-3 предложения), без воды.\n"
    "2. Внимательно читай ИСТОРИЮ диалога — не задавай повторно вопросы, "
    "на которые уже получил ответ.\n"
    "3. Если клиент сказал 'не интересно' / 'нет' — поблагодари и не дави.\n"
    "4. Если клиент задаёт уточняющий вопрос — отвечай конкретно.\n"
    "5. Цель: понять задачу клиента и договориться о созвоне или ТЗ.\n"
    "6. Не выдумывай цены и сроки — если спросят, скажи что зависит от ТЗ "
    "и предложи обсудить детали."
)

PROMPT_FILE = "prompt.json"


def _load_prompt() -> str:
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("prompt", os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT))
    except FileNotFoundError:
        return os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT)
    except Exception:
        logger.exception("Ошибка чтения файла промпта")
        return os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT)


def _save_prompt(prompt: str) -> None:
    try:
        with open(PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt}, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Не удалось сохранить промпт")


# Состояние в памяти (можно менять через /admin)
runtime_state = {
    "system_prompt": _load_prompt(),
    "enable_auto_reply": os.getenv("ENABLE_AUTO_REPLY", "true").lower() == "true",
}


def get_prompt() -> str:
    return runtime_state["system_prompt"]


def is_auto_reply_enabled() -> bool:
    return runtime_state["enable_auto_reply"]

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


async def fetch_chat_messages(token: str, user_id: str, chat_id: str, limit: int = 30) -> list[dict]:
    """Получает историю сообщений конкретного чата."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{AVITO_API_BASE}/messenger/v3/accounts/{user_id}/chats/{chat_id}/messages/",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": limit},
        )
    if resp.status_code != 200:
        logger.error("Ошибка истории чата %s: %s %s", chat_id, resp.status_code, resp.text)
        return []
    data = resp.json()
    # API может возвращать как {messages: [...]} так и список
    return data.get("messages", data) if isinstance(data, dict) else data


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


def _build_gemini_contents(history: list[dict], our_user_id: str) -> list[dict]:
    """Преобразует историю Авито в формат Gemini contents (multi-turn)."""
    contents = []
    # История из Авито приходит в обратном порядке (новые сверху) — переворачиваем
    for msg in reversed(history):
        text = msg.get("content", {}).get("text", "")
        if not text:
            continue
        author_id = str(msg.get("author_id", ""))
        if author_id in ("1", "0") or text.startswith("[Системное сообщение]"):
            continue
        role = "model" if author_id == our_user_id else "user"
        # Gemini не любит подряд идущие сообщения одной роли — склеиваем
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + text
        else:
            contents.append({"role": role, "parts": [{"text": text}]})
    return contents


async def generate_reply(
    history: list[dict],
    our_user_id: str,
    item_title: str = "",
) -> str | None:
    """Генерирует ответ через Gemini с учётом истории диалога."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY не задан")
        return None

    contents = _build_gemini_contents(history, our_user_id)
    if not contents or contents[-1]["role"] != "user":
        logger.info("Нет нового сообщения от клиента для ответа")
        return None

    system_text = get_prompt()
    if item_title:
        system_text += f"\n\nКонтекст: клиент пишет по объявлению «{item_title}»."

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_text}]},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                json=payload,
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

    # Тянем историю чата чтобы Gemini понимал контекст
    history = await fetch_chat_messages(token, user_id, chat_id)
    if not history:
        logger.warning("Пустая история для чата %s", chat_id)
        return

    reply = await generate_reply(history, user_id, item_title)
    if not reply:
        logger.warning("Не удалось сгенерировать ответ для %s", chat_id)
        return

    logger.info("Ответ Gemini для %s: %s", chat_id, reply[:80])

    # Сохраняем сгенерированный ответ для отображения на /replies
    generated_replies[chat_id] = {
        "buyer_text": text,
        "reply": reply,
        "item_title": item_title,
        "history_len": len(history),
        "sent": False,
    }

    if is_auto_reply_enabled():
        if await send_message(token, user_id, chat_id, reply):
            generated_replies[chat_id]["sent"] = True
    else:
        logger.info("Auto-reply выключен, ответ не отправлен")


async def poll_avito():
    logger.info("Polling запущен (интервал %ds, auto_reply=%s)", POLL_INTERVAL, is_auto_reply_enabled())

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
        auto = "ВКЛ" if is_auto_reply_enabled() else "ВЫКЛ (просмотр)"
        return HTMLResponse(
            "<h2>✅ Бот работает</h2>"
            f"<p>User ID: {token_storage.get('user_id')}</p>"
            f"<p>Auto-reply: <b>{auto}</b></p>"
            f"<p>Модель: {GEMINI_MODEL}</p>"
            "<ul>"
            "<li><a href='/admin'>⚙️ Админ-панель (промпт + тест)</a></li>"
            "<li><a href='/replies'>🤖 Сгенерированные ответы</a></li>"
            "<li><a href='/messages'>📬 Входящие сообщения</a></li>"
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


ADMIN_PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Admin</title>
<style>
body{font-family:system-ui,Arial;max-width:900px;margin:20px auto;padding:0 16px}
h2{margin-top:24px}
textarea{width:100%;box-sizing:border-box;font-family:monospace;font-size:14px;padding:8px}
input[type=text]{width:100%;box-sizing:border-box;padding:8px;font-size:14px}
button{padding:10px 18px;font-size:14px;cursor:pointer;background:#2563eb;color:white;border:0;border-radius:6px}
button:hover{background:#1d4ed8}
.row{margin:12px 0}
.box{background:#f5f5f5;padding:12px;border-radius:8px;margin:8px 0;white-space:pre-wrap}
.reply{background:#e6ffe6}
nav a{margin-right:12px}
label{display:block;margin-bottom:6px;font-weight:600}
</style></head>
<body>
<nav><a href="/">🏠 Главная</a> <a href="/replies">🤖 Ответы</a> <a href="/messages">📬 Сообщения</a></nav>

<h2>⚙️ Системный промпт</h2>
<form method="post" action="/admin/prompt">
<div class="row"><label>Промпт (как бот должен себя вести)</label>
<textarea name="prompt" rows="14">__PROMPT__</textarea></div>
<button type="submit">💾 Сохранить</button>
</form>

<h2>🔘 Авто-ответы в Авито</h2>
<form method="post" action="/admin/toggle">
<p>Сейчас: <b>__AUTO__</b></p>
<button type="submit">Переключить</button>
</form>

<h2>🧪 Тестовый чат</h2>
<p>Напиши сообщение от лица клиента — бот сгенерирует ответ как в Авито.</p>
<form method="post" action="/admin/test">
<div class="row"><label>Сообщение клиента</label>
<input type="text" name="message" value="__TEST_MSG__" placeholder="Здравствуйте, сколько стоит чат-бот?"></div>
<button type="submit">🚀 Сгенерировать ответ</button>
</form>
__TEST_RESULT__

</body></html>
"""

# Память тестового чата для удобства
last_test = {"message": "", "reply": ""}


def render_admin() -> str:
    test_html = ""
    if last_test["reply"]:
        test_html = (
            f"<h3>Результат теста</h3>"
            f"<div class='box'>👤 {last_test['message']}</div>"
            f"<div class='box reply'>🤖 {last_test['reply']}</div>"
        )
    return (
        ADMIN_PAGE
        .replace("__PROMPT__", get_prompt())
        .replace("__AUTO__", "ВКЛ" if is_auto_reply_enabled() else "ВЫКЛ")
        .replace("__TEST_MSG__", last_test["message"])
        .replace("__TEST_RESULT__", test_html)
    )


@app.get("/admin")
async def admin_page():
    return HTMLResponse(render_admin())


@app.post("/admin/prompt")
async def admin_save_prompt(prompt: str = Form(...)):
    runtime_state["system_prompt"] = prompt.strip()
    _save_prompt(prompt.strip())
    logger.info("Промпт обновлён (%d символов) и сохранён в файл", len(prompt))
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/toggle")
async def admin_toggle():
    runtime_state["enable_auto_reply"] = not runtime_state["enable_auto_reply"]
    logger.info("Auto-reply -> %s", runtime_state["enable_auto_reply"])
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/test")
async def admin_test(message: str = Form(...)):
    fake_history = [{"author_id": "test_user", "content": {"text": message}}]
    reply = await generate_reply(fake_history, "self", item_title="Услуги разработки")
    last_test["message"] = message
    last_test["reply"] = reply or "(не удалось сгенерировать)"
    return RedirectResponse("/admin", status_code=303)


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

    mode = "ВКЛ — отправлено в Авито" if is_auto_reply_enabled() else "ВЫКЛ — только просмотр"
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
