import asyncio
import html as html_module
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
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def _load_prompt_from_supabase() -> str | None:
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/bot_settings",
            params={"key": "eq.system_prompt", "select": "value"},
            headers=_supabase_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                logger.info("Промпт загружен из Supabase (%d символов)", len(rows[0]["value"]))
                return rows[0]["value"]
            logger.info("В Supabase ещё нет сохранённого промпта")
            return None
        logger.error("Supabase load %s: %s", resp.status_code, resp.text)
    except Exception:
        logger.exception("Ошибка загрузки промпта из Supabase")
    return None


def _save_prompt_to_supabase(prompt: str) -> bool:
    if not (SUPABASE_URL and SUPABASE_KEY):
        return False
    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/bot_settings",
            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates"},
            json={"key": "system_prompt", "value": prompt},
            timeout=10,
        )
        if resp.status_code in (200, 201, 204):
            logger.info("Промпт сохранён в Supabase")
            return True
        logger.error("Supabase save %s: %s", resp.status_code, resp.text)
    except Exception:
        logger.exception("Ошибка сохранения промпта в Supabase")
    return False


def _load_prompt() -> str:
    """Приоритет: Supabase → локальный файл → SYSTEM_PROMPT env → DEFAULT_PROMPT."""
    from_db = _load_prompt_from_supabase()
    if from_db:
        return from_db
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            fallback = json.load(f).get("prompt", os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT))
    except FileNotFoundError:
        fallback = os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT)
    except Exception:
        logger.exception("Ошибка чтения файла промпта")
        fallback = os.getenv("SYSTEM_PROMPT", DEFAULT_PROMPT)
    # Supabase был пуст — сразу фиксируем туда, чтобы следующий старт загрузил оттуда
    _save_prompt_to_supabase(fallback)
    return fallback


def _save_prompt(prompt: str) -> None:
    """Сохраняем в Supabase + дублируем в файл (для отладки и fallback)."""
    _save_prompt_to_supabase(prompt)
    try:
        with open(PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt}, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Не удалось сохранить промпт в файл")


def _load_disabled_chats() -> set[str]:
    if not (SUPABASE_URL and SUPABASE_KEY):
        return set()
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/bot_settings",
            params={"key": "eq.disabled_chats", "select": "value"},
            headers=_supabase_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            rows = resp.json()
            if rows:
                return set(json.loads(rows[0]["value"]))
    except Exception:
        logger.exception("Ошибка загрузки disabled_chats из Supabase")
    return set()


def _save_disabled_chats() -> None:
    if not (SUPABASE_URL and SUPABASE_KEY):
        return
    try:
        httpx.post(
            f"{SUPABASE_URL}/rest/v1/bot_settings",
            headers={**_supabase_headers(), "Prefer": "resolution=merge-duplicates"},
            json={"key": "disabled_chats", "value": json.dumps(list(disabled_chats))},
            timeout=10,
        )
    except Exception:
        logger.exception("Ошибка сохранения disabled_chats в Supabase")


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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

POLL_INTERVAL = 10

token_storage: dict = {}
messages_cache: list[dict] = []
replied_messages: dict[str, str] = {}
generated_replies: dict[str, dict] = {}
disabled_chats: set[str] = _load_disabled_chats()


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


def _build_messages(history: list[dict], our_user_id: str, system_text: str) -> list[dict]:
    """Преобразует историю Авито в OpenAI-формат messages (DeepSeek-совместимый)."""
    messages = [{"role": "system", "content": system_text}]
    # История из Авито приходит newest-first — переворачиваем в хронологический порядок
    for msg in reversed(history):
        text = msg.get("content", {}).get("text", "")
        if not text:
            continue
        author_id = str(msg.get("author_id", ""))
        if author_id in ("1", "0") or text.startswith("[Системное сообщение]"):
            continue
        role = "assistant" if author_id == our_user_id else "user"
        # Склеиваем подряд идущие сообщения одной роли
        if len(messages) > 1 and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + text
        else:
            messages.append({"role": role, "content": text})
    return messages


async def generate_reply(
    history: list[dict],
    our_user_id: str,
    item_title: str = "",
) -> str | None:
    """Генерирует ответ через GPT-4o-mini с учётом истории диалога."""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не задан")
        return None

    system_text = get_prompt()
    if item_title:
        system_text += f"\n\nКонтекст: клиент пишет по объявлению «{item_title}»."

    messages = _build_messages(history, our_user_id, system_text)
    non_system = [m for m in messages if m["role"] != "system"]
    if not non_system or non_system[-1]["role"] != "user":
        logger.info("Нет нового сообщения от клиента для ответа")
        return None

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": OPENAI_MODEL, "messages": messages, "max_tokens": 500},
            )
        if resp.status_code != 200:
            logger.error("Ошибка OpenAI: %s %s", resp.status_code, resp.text)
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
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

    if chat_id in disabled_chats:
        return

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

    # На старте помечаем все существующие сообщения как обработанные
    # чтобы не сжечь квоту Gemini при перезапуске бота
    init_chats = await fetch_chats(token, user_id)
    for chat in init_chats:
        cid = chat.get("id")
        mid = chat.get("last_message", {}).get("id")
        if cid and mid:
            replied_messages[cid] = mid
    logger.info("Инициализировано %d чатов — Gemini вызывается только для новых сообщений", len(init_chats))

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
        storage = "Supabase ✅" if (SUPABASE_URL and SUPABASE_KEY) else "локальный файл ⚠️"
        return HTMLResponse(
            "<h2>✅ Бот работает</h2>"
            f"<p>User ID: {token_storage.get('user_id')}</p>"
            f"<p>Auto-reply: <b>{auto}</b></p>"
            f"<p>Модель: {OPENAI_MODEL}</p>"
            f"<p>Хранилище промпта: <b>{storage}</b></p>"
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
    return {
        "current_model": OPENAI_MODEL,
        "available_models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "api_key_set": bool(OPENAI_API_KEY),
    }


ADMIN_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AvitoBot · Console</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#1E1F2E;color:#E4E6F1;font-family:'Inter',sans-serif;min-height:100vh;padding:24px 20px;display:flex;flex-direction:column}
h1,h2,.brand{font-family:'Space Grotesk',sans-serif}
.wrap{max-width:1200px;margin:0 auto;flex:1;display:flex;flex-direction:column;gap:0}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:12px;font-size:20px;font-weight:700;letter-spacing:.5px}
.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#00D9FF,#0096B3);display:grid;place-items:center;color:#0B0C16;font-weight:700;font-size:18px;box-shadow:0 0 18px rgba(0,217,255,.45)}
.accent{color:#00D9FF}
.online{display:flex;align-items:center;gap:8px;font-size:13px;color:#8A8DA8;font-family:'Inter';font-weight:500}
.dot-live{width:8px;height:8px;border-radius:50%;background:#00D9FF;box-shadow:0 0 10px #00D9FF;flex-shrink:0}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:22px}
.stat-card{background:#2A2B3D;border:1px solid #34354A;border-radius:16px;padding:18px 22px;display:flex;justify-content:space-between;align-items:center}
.stat-label{font-size:11px;color:#8A8DA8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;font-weight:600}
.stat-val{font-family:'Space Grotesk';font-size:26px;font-weight:700;color:#fff}
.stat-val.cyan{color:#00D9FF;text-shadow:0 0 14px rgba(0,217,255,.6)}
.pill{background:rgba(0,217,255,.12);color:#00D9FF;border:1px solid rgba(0,217,255,.35);padding:4px 10px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}
.pill.off{background:rgba(255,160,0,.1);color:#FFA500;border-color:rgba(255,160,0,.35)}
.grid{display:grid;grid-template-columns:1fr 1.05fr;gap:20px;flex:1}
.card{background:#2A2B3D;border:1px solid #34354A;border-radius:16px;padding:22px}
.card-title{display:flex;align-items:center;gap:10px;margin-bottom:18px}
.card-dot{width:8px;height:8px;border-radius:50%;background:#00D9FF;box-shadow:0 0 10px #00D9FF}
.card-title h2{font-size:15px;font-weight:700;letter-spacing:.4px}
.field{margin-bottom:16px}
.field label{display:block;font-size:11px;color:#8A8DA8;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:8px;font-weight:600}
textarea{width:100%;background:#1E1F2E;border:1px solid #34354A;border-radius:12px;padding:12px 14px;color:#E4E6F1;font-family:'Inter';font-size:13px;line-height:1.6;resize:vertical;outline:none;min-height:220px}
textarea:focus{border-color:#00D9FF;box-shadow:0 0 0 3px rgba(0,217,255,.1)}
.toggle-row{display:flex;align-items:center;justify-content:space-between;background:#1E1F2E;border:1px solid #34354A;border-radius:12px;padding:14px 16px;margin-top:14px}
.toggle-label{font-size:14px;font-weight:500}
.toggle-sub{font-size:12px;color:#8A8DA8;margin-top:3px}
.toggle-btn{padding:7px 16px;border-radius:8px;font-family:'Inter';font-size:12px;font-weight:600;cursor:pointer;border:1px solid rgba(0,217,255,.4);background:rgba(0,217,255,.1);color:#00D9FF;transition:all .2s}
.toggle-btn:hover{background:rgba(0,217,255,.2)}
.toggle-btn.active{background:#00D9FF;color:#0B0C16;border-color:#00D9FF;box-shadow:0 0 12px rgba(0,217,255,.4)}
.actions{display:flex;gap:10px;margin-top:16px}
.btn{flex:1;padding:12px;border-radius:12px;font-family:'Inter';font-size:13px;font-weight:600;cursor:pointer;border:1px solid #34354A;background:#1E1F2E;color:#E4E6F1;transition:all .2s}
.btn:hover{border-color:#555}
.btn.primary{background:#00D9FF;color:#0B0C16;border-color:#00D9FF;box-shadow:0 0 18px rgba(0,217,255,.4)}
.btn.primary:hover{background:#33E1FF;box-shadow:0 0 24px rgba(0,217,255,.55)}
.chat-body{display:flex;flex-direction:column;gap:12px;max-height:360px;overflow-y:auto;padding-right:4px;margin-bottom:16px}
.chat-body::-webkit-scrollbar{width:4px}
.chat-body::-webkit-scrollbar-track{background:#1E1F2E}
.chat-body::-webkit-scrollbar-thumb{background:#34354A;border-radius:999px}
.chat-empty{text-align:center;color:#8A8DA8;font-size:13px;padding:40px 0}
.msg{display:flex;gap:12px;align-items:flex-start}
.msg.bot{flex-direction:row-reverse}
.avatar{width:36px;height:36px;border-radius:50%;flex-shrink:0;display:grid;place-items:center;font-family:'Space Grotesk';font-weight:700;font-size:13px}
.av-user{background:linear-gradient(135deg,#6B5BFF,#3B2EAA);color:#fff}
.av-bot{background:linear-gradient(135deg,#00D9FF,#006D85);color:#0B0C16;box-shadow:0 0 10px rgba(0,217,255,.35)}
.bubble{max-width:78%;padding:11px 14px;border-radius:14px;font-size:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.bubble.user{background:#1E1F2E;border:1px solid #34354A;border-top-left-radius:4px}
.bubble.bot{background:rgba(0,217,255,.07);border:1px solid rgba(0,217,255,.22);border-top-right-radius:4px}
.chat-input-row{display:flex;gap:10px}
.chat-input-row input{flex:1;background:#1E1F2E;border:1px solid #34354A;border-radius:12px;padding:12px 14px;color:#E4E6F1;font-family:'Inter';font-size:14px;outline:none;transition:border-color .2s}
.chat-input-row input:focus{border-color:#00D9FF}
.chat-input-row button{padding:0 20px;border-radius:12px;background:#00D9FF;color:#0B0C16;border:none;font-weight:700;font-size:14px;cursor:pointer;box-shadow:0 0 14px rgba(0,217,255,.4);transition:all .2s;white-space:nowrap}
.chat-input-row button:hover{background:#33E1FF;box-shadow:0 0 20px rgba(0,217,255,.6)}
.clear-row{display:flex;justify-content:flex-end;margin-top:10px}
.btn-clear{background:rgba(231,76,60,.1);color:#e74c3c;border:1px solid rgba(231,76,60,.3);padding:7px 14px;border-radius:8px;font-size:12px;font-weight:600;cursor:pointer;font-family:'Inter';transition:all .2s}
.btn-clear:hover{background:#e74c3c;color:#fff}
footer{background:#161724;border-top:1px solid #2A2B3D;text-align:center;padding:22px 24px;margin-top:28px}
footer .fname{font-family:'Space Grotesk';font-weight:700;font-size:15px;color:#E4E6F1;margin-bottom:8px}
footer .contacts{display:flex;gap:24px;justify-content:center;flex-wrap:wrap;font-size:13px;color:#8A8DA8}
footer a{color:#8A8DA8;text-decoration:none;transition:color .2s}
footer a:hover{color:#00D9FF}
@media(max-width:760px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">

<div class="topbar">
  <div class="brand">
    <div class="logo">A</div>
    <div>Avito<span class="accent">Bot</span> <span style="color:#8A8DA8;font-weight:500;font-size:14px;font-family:'Inter'">&nbsp;/ prompt console</span></div>
  </div>
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div class="online"><span class="dot-live"></span>online &middot; auto-reply: __STATUS__</div>
    <a href="/admin/chats" style="color:#00D9FF;text-decoration:none;font-size:13px;border:1px solid rgba(0,217,255,.3);padding:5px 12px;border-radius:8px;background:rgba(0,217,255,.07)">⚙️ Управление чатами</a>
    <a href="/replies" style="color:#8A8DA8;text-decoration:none;font-size:13px">💬 Ответы</a>
    <a href="/messages" style="color:#8A8DA8;text-decoration:none;font-size:13px">📬 Сообщения</a>
  </div>
</div>

<div class="stats">
  <div class="stat-card">
    <div><div class="stat-label">Чатов загружено</div><div class="stat-val cyan">__STAT_CHATS__</div></div>
    <div class="pill">live</div>
  </div>
  <div class="stat-card">
    <div><div class="stat-label">Ответов сгенерировано</div><div class="stat-val">__STAT_REPLIES__</div></div>
    <div class="pill">gemini</div>
  </div>
  <div class="stat-card">
    <div><div class="stat-label">Режим бота</div><div class="stat-val cyan">__STAT_MODE__</div></div>
    <div class="pill __PILL_CLS__">__STAT_PILL__</div>
  </div>
</div>

<div class="grid">

  <div class="card">
    <div class="card-title"><span class="card-dot"></span><h2>BOT CONFIG</h2></div>
    <form method="post" action="/admin/prompt">
      <div class="field">
        <label>Системный промпт — сохраняется в Supabase</label>
        <textarea name="prompt">__PROMPT__</textarea>
      </div>
      <div class="actions">
        <button type="button" class="btn" onclick="location.href='/admin'">Сбросить</button>
        <button type="submit" class="btn primary">💾 Сохранить и задеплоить</button>
      </div>
    </form>
    <div class="toggle-row">
      <div>
        <div class="toggle-label">Авто-ответы в Авито</div>
        <div class="toggle-sub">__AUTO_SUB__</div>
      </div>
      <form method="post" action="/admin/toggle">
        <button type="submit" class="toggle-btn __TOGGLE_CLS__">__TOGGLE_LBL__</button>
      </form>
    </div>
  </div>

  <div class="card">
    <div class="card-title"><span class="card-dot"></span><h2>LIVE PREVIEW</h2></div>
    <div class="chat-body" id="chatbox">__CHAT_MESSAGES__</div>
    <form method="post" action="/admin/test">
      <div class="chat-input-row">
        <input type="text" name="message" placeholder="Тестовое сообщение от покупателя…" autofocus>
        <button type="submit">Send</button>
      </div>
    </form>
    __CLEAR_BTN__
  </div>

</div>
</div>

<footer>
  <div class="fname">Аркадий | Нейросети | Чат-боты</div>
  <div class="contacts">
    <span>📞 <a href="tel:89990027781">8 999 002 77 81</a></span>
    <span>✉️ <a href="mailto:arkadiynovichkov@mail.ru">arkadiynovichkov@mail.ru</a></span>
  </div>
</footer>
<script>var cb=document.getElementById('chatbox');if(cb)cb.scrollTop=cb.scrollHeight;</script>
</body></html>
"""

# История тестового чата: [{"role": "user"/"bot", "text": "..."}]
test_chat_history: list[dict] = []


def _render_chat_messages() -> str:
    if not test_chat_history:
        return "<div class='chat-empty'>Начни диалог — напиши сообщение от клиента ниже</div>"
    out = ""
    for msg in test_chat_history:
        text = html_module.escape(msg["text"])
        if msg["role"] == "user":
            out += f"<div class='msg'><div class='avatar av-user'>КЛ</div><div class='bubble user'>{text}</div></div>"
        else:
            out += f"<div class='msg bot'><div class='avatar av-bot'>АЛ</div><div class='bubble bot'>{text}</div></div>"
    return out


def _test_history_to_avito_format() -> list[dict]:
    # Возвращаем newest-first — так ожидает _build_gemini_contents (он делает reversed внутри)
    return [
        {"author_id": "self" if m["role"] == "bot" else "test_user", "content": {"text": m["text"]}}
        for m in reversed(test_chat_history)
    ]


def render_admin() -> str:
    auto_on = is_auto_reply_enabled()
    clear_btn = ""
    if test_chat_history:
        clear_btn = "<div class='clear-row'><form method='post' action='/admin/test/clear'><button type='submit' class='btn-clear'>🗑 Очистить чат</button></form></div>"
    return (
        ADMIN_PAGE
        .replace("__PROMPT__", html_module.escape(get_prompt()))
        .replace("__STATUS__", "ON" if auto_on else "OFF")
        .replace("__STAT_CHATS__", str(len(messages_cache)))
        .replace("__STAT_REPLIES__", str(len(generated_replies)))
        .replace("__STAT_MODE__", "Авто" if auto_on else "Просмотр")
        .replace("__PILL_CLS__", "" if auto_on else "off")
        .replace("__STAT_PILL__", "отправляет" if auto_on else "только просмотр")
        .replace("__AUTO_SUB__", "Бот сам пишет в Авито" if auto_on else "Ответы только в панели")
        .replace("__TOGGLE_CLS__", "active" if auto_on else "")
        .replace("__TOGGLE_LBL__", "ВКЛ" if auto_on else "ВЫКЛ")
        .replace("__CHAT_MESSAGES__", _render_chat_messages())
        .replace("__CLEAR_BTN__", clear_btn)
    )


CHATS_PAGE_HEADER = """<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AvitoBot · Чаты</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#1E1F2E;color:#E4E6F1;font-family:'Inter',sans-serif;min-height:100vh;padding:24px 20px}
h1,h2,.brand{font-family:'Space Grotesk',sans-serif}
.wrap{max-width:900px;margin:0 auto}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;flex-wrap:wrap;gap:12px}
.brand{display:flex;align-items:center;gap:12px;font-size:20px;font-weight:700}
.logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#00D9FF,#0096B3);display:grid;place-items:center;color:#0B0C16;font-weight:700;font-size:18px;box-shadow:0 0 18px rgba(0,217,255,.45)}
.accent{color:#00D9FF}
nav a{color:#8A8DA8;text-decoration:none;font-size:13px;margin-left:16px;transition:color .2s}
nav a:hover{color:#00D9FF}
.page-title{font-size:17px;font-weight:700;margin-bottom:18px;color:#E4E6F1}
.chat-card{background:#2A2B3D;border:1px solid #34354A;border-radius:14px;padding:16px 20px;margin-bottom:12px;display:flex;align-items:center;justify-content:space-between;gap:16px;transition:border-color .2s}
.chat-card:hover{border-color:#555}
.chat-card.disabled{opacity:.55;border-color:#2A2B3D}
.chat-info{flex:1;min-width:0}
.chat-id{font-family:'Space Grotesk';font-size:13px;color:#8A8DA8;margin-bottom:4px;word-break:break-all}
.chat-msg{font-size:14px;color:#E4E6F1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.chat-label{font-size:11px;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px;font-weight:600}
.label-on{color:#27ae60}
.label-off{color:#e74c3c}
.btn-on{padding:8px 18px;border-radius:9px;font-family:'Inter';font-size:13px;font-weight:600;cursor:pointer;border:1px solid rgba(0,217,255,.4);background:rgba(0,217,255,.1);color:#00D9FF;transition:all .2s;white-space:nowrap}
.btn-on:hover{background:rgba(0,217,255,.2)}
.btn-off{padding:8px 18px;border-radius:9px;font-family:'Inter';font-size:13px;font-weight:600;cursor:pointer;border:1px solid rgba(231,76,60,.4);background:rgba(231,76,60,.1);color:#e74c3c;transition:all .2s;white-space:nowrap}
.btn-off:hover{background:#e74c3c;color:#fff}
.empty{text-align:center;color:#8A8DA8;padding:60px 0;font-size:15px}
.refresh{display:inline-block;margin-bottom:20px;color:#00D9FF;text-decoration:none;font-size:13px;border:1px solid rgba(0,217,255,.3);padding:6px 14px;border-radius:8px;background:rgba(0,217,255,.07);transition:all .2s}
.refresh:hover{background:rgba(0,217,255,.15)}
</style></head><body><div class="wrap">
<div class="topbar">
  <div class="brand"><div class="logo">A</div><div>Avito<span class="accent">Bot</span></div></div>
  <nav>
    <a href="/admin">⚙️ Панель</a>
    <a href="/replies">💬 Ответы</a>
    <a href="/messages">📬 Сообщения</a>
  </nav>
</div>
"""

CHATS_PAGE_FOOTER = """
</div></body></html>"""


def render_chats() -> str:
    html = CHATS_PAGE_HEADER
    html += "<a class='refresh' href='/admin/chats'>🔄 Обновить список</a>"
    html += f"<div class='page-title'>⚙️ Управление чатами <span style='color:#8A8DA8;font-size:14px;font-weight:400'>({len(messages_cache)} чатов)</span></div>"
    if not messages_cache:
        html += "<div class='empty'>Чаты ещё не загружены. Подожди несколько секунд и обнови страницу.</div>"
    else:
        for m in messages_cache:
            cid = m["chat_id"]
            is_disabled = cid in disabled_chats
            card_cls = "chat-card disabled" if is_disabled else "chat-card"
            label_cls = "label-off" if is_disabled else "label-on"
            label_txt = "БОТ ВЫКЛЮЧЕН" if is_disabled else "БОТ АКТИВЕН"
            btn_cls = "btn-on" if is_disabled else "btn-off"
            btn_txt = "Включить бота" if is_disabled else "Выключить бота"
            msg_preview = html_module.escape(str(m.get("text", ""))[:80])
            html += (
                f"<div class='{card_cls}'>"
                f"<div class='chat-info'>"
                f"<div class='chat-label {label_cls}'>{label_txt}</div>"
                f"<div class='chat-id'>{html_module.escape(cid)}</div>"
                f"<div class='chat-msg'>{msg_preview}</div>"
                f"</div>"
                f"<form method='post' action='/admin/chats/toggle'>"
                f"<input type='hidden' name='chat_id' value='{html_module.escape(cid)}'>"
                f"<button type='submit' class='{btn_cls}'>{btn_txt}</button>"
                f"</form>"
                f"</div>"
            )
    html += CHATS_PAGE_FOOTER
    return html


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


@app.get("/admin/chats")
async def admin_chats_page():
    return HTMLResponse(render_chats())


@app.post("/admin/chats/toggle")
async def admin_chats_toggle(chat_id: str = Form(...)):
    if chat_id in disabled_chats:
        disabled_chats.discard(chat_id)
        logger.info("Бот включён для чата %s", chat_id)
    else:
        disabled_chats.add(chat_id)
        logger.info("Бот выключен для чата %s", chat_id)
    _save_disabled_chats()
    return RedirectResponse("/admin/chats", status_code=303)


@app.post("/admin/test")
async def admin_test(message: str = Form(...)):
    test_chat_history.append({"role": "user", "text": message.strip()})
    history = _test_history_to_avito_format()
    try:
        reply = await generate_reply(history, "self", item_title="Услуги разработки")
    except Exception as e:
        reply = f"❌ Исключение: {e}"
    if reply is None:
        reply = "⚠️ ИИ не ответил. Проверь OPENAI_API_KEY в HF Secrets и логи Space (кнопка Logs)."
    test_chat_history.append({"role": "bot", "text": reply})
    return RedirectResponse("/admin", status_code=303)


@app.post("/admin/test/clear")
async def admin_test_clear():
    test_chat_history.clear()
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
