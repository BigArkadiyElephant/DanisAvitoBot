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
        storage = "Supabase ✅" if (SUPABASE_URL and SUPABASE_KEY) else "локальный файл ⚠️"
        return HTMLResponse(
            "<h2>✅ Бот работает</h2>"
            f"<p>User ID: {token_storage.get('user_id')}</p>"
            f"<p>Auto-reply: <b>{auto}</b></p>"
            f"<p>Модель: {GEMINI_MODEL}</p>"
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


ADMIN_PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Панель управления ботом</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,Arial,sans-serif;background:#f0f2f5;min-height:100vh;display:flex;flex-direction:column}
header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);color:white;padding:20px 32px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
header h1{font-size:20px;font-weight:700;letter-spacing:.3px}
header p{font-size:13px;opacity:.65;margin-top:3px}
nav a{color:rgba(255,255,255,.8);text-decoration:none;margin-left:16px;font-size:14px}
nav a:hover{color:white}
main{max-width:860px;width:100%;margin:28px auto;padding:0 20px;flex:1}
.card{background:white;border-radius:16px;padding:26px;margin-bottom:22px;box-shadow:0 2px 12px rgba(0,0,0,.07)}
.card h2{font-size:17px;color:#1a1a2e;margin-bottom:16px;display:flex;align-items:center;gap:8px}
label{display:block;font-size:13px;color:#666;margin-bottom:8px;font-weight:500}
textarea{width:100%;font-family:'Courier New',monospace;font-size:13px;padding:14px;border:1.5px solid #e0e0e0;border-radius:10px;resize:vertical;line-height:1.6;color:#333;transition:border-color .2s}
textarea:focus{outline:none;border-color:#0f3460}
input[type=text]{width:100%;padding:11px 16px;font-size:14px;border:1.5px solid #e0e0e0;border-radius:10px;color:#333;transition:border-color .2s}
input[type=text]:focus{outline:none;border-color:#0f3460}
.btn{padding:11px 22px;font-size:14px;font-weight:600;cursor:pointer;border:0;border-radius:10px;transition:all .2s}
.btn-primary{background:#0f3460;color:white}
.btn-primary:hover{background:#16213e;transform:translateY(-1px)}
.btn-toggle{background:#e8f4fd;color:#0f3460;border:1.5px solid #0f3460}
.btn-toggle:hover{background:#0f3460;color:white}
.btn-send{background:#0f3460;color:white;white-space:nowrap}
.btn-send:hover{background:#16213e}
.btn-clear{background:#fff0f0;color:#c0392b;border:1.5px solid #e74c3c;font-size:13px;padding:8px 16px;cursor:pointer;border-radius:8px;font-weight:500;transition:all .2s}
.btn-clear:hover{background:#e74c3c;color:white}
.row{margin-top:14px}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600}
.badge-on{background:#e6f9f0;color:#27ae60}
.badge-off{background:#fef9e7;color:#e67e22}
.chat-box{margin:14px 0;max-height:380px;overflow-y:auto;background:#f8f9fb;border-radius:12px;padding:16px;display:flex;flex-direction:column;gap:10px}
.chat-empty{color:#aaa;font-size:14px;text-align:center;padding:20px}
.msg{display:flex;gap:10px;align-items:flex-start}
.msg-user{flex-direction:row-reverse}
.bubble{padding:10px 14px;border-radius:14px;max-width:78%;font-size:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.msg-user .bubble{background:#0f3460;color:white;border-radius:14px 4px 14px 14px}
.msg-bot .bubble{background:white;border:1px solid #e0e0e0;color:#333;border-radius:4px 14px 14px 14px}
.avatar{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;background:#e8edf5}
.input-row{display:flex;gap:10px;margin-top:12px}
.input-row input{flex:1}
.chat-footer{display:flex;justify-content:flex-end;margin-top:8px}
footer{background:#1a1a2e;color:rgba(255,255,255,.75);text-align:center;padding:20px 32px;margin-top:auto}
footer .fname{font-weight:700;font-size:16px;color:white;margin-bottom:8px}
footer .contacts{font-size:14px;display:flex;gap:24px;justify-content:center;flex-wrap:wrap}
footer a{color:rgba(255,255,255,.7);text-decoration:none}
footer a:hover{color:white}
</style>
</head>
<body>
<header>
  <div>
    <h1>🤖 Панель управления ботом</h1>
    <p>Авито Бот · Gemini AI · Управление ответами</p>
  </div>
  <nav>
    <a href="/">🏠 Главная</a>
    <a href="/replies">💬 Ответы</a>
    <a href="/messages">📬 Сообщения</a>
  </nav>
</header>
<main>

<div class="card">
  <h2>⚙️ Системный промпт</h2>
  <form method="post" action="/admin/prompt">
    <label>Характер и цели бота — сохраняется в Supabase навсегда</label>
    <textarea name="prompt" rows="14">__PROMPT__</textarea>
    <div class="row"><button type="submit" class="btn btn-primary">💾 Сохранить промпт</button></div>
  </form>
</div>

<div class="card">
  <h2>🔘 Авто-ответы в Авито</h2>
  <p>Статус: <span class="badge __AUTO_CLS__">__AUTO__</span></p>
  <div class="row">
    <form method="post" action="/admin/toggle">
      <button type="submit" class="btn btn-toggle">Переключить</button>
    </form>
  </div>
</div>

<div class="card">
  <h2>🧪 Тестовый чат с ботом</h2>
  <p style="color:#888;font-size:13px;margin-bottom:4px">Пиши от лица клиента — бот отвечает с учётом всей истории диалога</p>
  <div class="chat-box" id="chatbox">__CHAT_MESSAGES__</div>
  <form method="post" action="/admin/test">
    <div class="input-row">
      <input type="text" name="message" placeholder="Сообщение клиента..." autofocus>
      <button type="submit" class="btn btn-send">➤ Отправить</button>
    </div>
  </form>
  __CLEAR_BTN__
</div>

</main>
<footer>
  <div class="fname">Аркадий | Нейросети | Чат-боты</div>
  <div class="contacts">
    <span>📞 <a href="tel:89990027781">8 999 002 77 81</a></span>
    <span>✉️ <a href="mailto:arkadiynovichkov@mail.ru">arkadiynovichkov@mail.ru</a></span>
  </div>
</footer>
<script>
  var cb = document.getElementById('chatbox');
  if(cb) cb.scrollTop = cb.scrollHeight;
</script>
</body></html>
"""

# История тестового чата: [{"role": "user"/"bot", "text": "..."}]
test_chat_history: list[dict] = []


def _render_chat_messages() -> str:
    if not test_chat_history:
        return "<div class='chat-empty'>Начни диалог — напиши сообщение ниже</div>"
    html = ""
    for msg in test_chat_history:
        if msg["role"] == "user":
            html += f"<div class='msg msg-user'><div class='avatar'>👤</div><div class='bubble'>{msg['text']}</div></div>"
        else:
            html += f"<div class='msg msg-bot'><div class='avatar'>🤖</div><div class='bubble'>{msg['text']}</div></div>"
    return html


def _test_history_to_avito_format() -> list[dict]:
    return [
        {"author_id": "self" if m["role"] == "bot" else "test_user", "content": {"text": m["text"]}}
        for m in test_chat_history
    ]


def render_admin() -> str:
    auto_on = is_auto_reply_enabled()
    clear_btn = ""
    if test_chat_history:
        clear_btn = "<div class='chat-footer'><form method='post' action='/admin/test/clear'><button type='submit' class='btn-clear'>🗑 Очистить чат</button></form></div>"
    return (
        ADMIN_PAGE
        .replace("__PROMPT__", get_prompt())
        .replace("__AUTO_CLS__", "badge-on" if auto_on else "badge-off")
        .replace("__AUTO__", "ВКЛ — бот отвечает в Авито" if auto_on else "ВЫКЛ — только просмотр")
        .replace("__CHAT_MESSAGES__", _render_chat_messages())
        .replace("__CLEAR_BTN__", clear_btn)
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
    test_chat_history.append({"role": "user", "text": message.strip()})
    history = _test_history_to_avito_format()
    reply = await generate_reply(history, "self", item_title="Услуги разработки")
    test_chat_history.append({"role": "bot", "text": reply or "(не удалось сгенерировать)"})
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
