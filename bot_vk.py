"""
bot_vk.py — ВКонтакте бот «АгроПомощник»
Три режима:
  РЕЖИМ 1 (score >= 0.12) — строго по документам
  РЕЖИМ 2 (score >= 0.015) — по теме АПК, из общих знаний с предупреждением
  РЕЖИМ 3 (score < 0.015) — не по теме, кратко
"""
import os, logging, re, math, pickle, requests, json
import json as _json, time as _time, threading
from pathlib import Path as _Path

# ── Загрузка .env ─────────────────────────────────────────────────────
env_path = _Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

VK_TOKEN   = os.environ.get("VK_TOKEN", "")
VK_CONFIRM = os.environ.get("VK_CONFIRM", "")
POLZA_KEY  = os.environ.get("POLZA_API_KEY", "")
SECRET     = os.environ.get("VK_SECRET", "")

POLZA_URL   = "https://polza.ai/api/v1/chat/completions"
POLZA_MODEL = "deepseek/deepseek-chat"
VK_API_URL  = "https://api.vk.com/method/"
VK_API_VER  = "5.199"

# ── Защита от дублей ──────────────────────────────────────────────────
_IDS_FILE = _Path("/tmp/processed_ids.json")

def _load_ids():
    try:
        if _IDS_FILE.exists():
            data = _json.loads(_IDS_FILE.read_text())
            now = _time.time()
            return {k: v for k, v in data.items() if now - v < 7200}
    except:
        pass
    return {}

def _save_ids(d):
    try: _IDS_FILE.write_text(_json.dumps(d))
    except: pass

processed_ids = _load_ids()

# ── TF-IDF индекс ─────────────────────────────────────────────────────
INDEX = {"chunks": [], "vectors": [], "doc_freq": {}, "N": 0}

def load_index():
    global INDEX
    p = _Path("knowledge_index.pkl")
    if p.exists():
        with open(p, "rb") as f:
            INDEX = pickle.load(f)
        log.info(f"Индекс загружен: {INDEX['N']} чанков")
    else:
        log.warning("knowledge_index.pkl не найден")

STOPWORDS = {
    'кто','что','как','это','все','для','при','или','они','она','оно','нет','был',
    'быть','есть','вот','так','уже','еще','ещё','там','тут','где','без','над',
    'под','про','вам','вас','нас','мне','том','эти','эта','этот','эту','ним',
    'них','той','тот','три','два','раз','лет','год','кого','чем','тем','тех',
    'тому','того','всё','всех','всем','наш','ваш','can','the','and','for',
    'not','you','are','but','was','ней','ооо','зао','оао','пао',
}

def tokenize(text):
    # Нормализуем опечатки
    text = text.lower()
    text = re.sub(r'(\w)а[йи]т', r'\1ает', text)  # "работаит" -> "работает"
    text = re.sub(r'^што\b', 'что', text)
    return [w for w in re.findall(r'[а-яёa-z]{3,}', text) if w not in STOPWORDS]

THRESHOLD_DOCS    = 0.12
THRESHOLD_RELATED = 0.015

def search_kb(query, n=8):  # было 6 → стало 8
    if not INDEX["chunks"]: return [], 0.0
    words = tokenize(query)
    if not words: return [], 0.0
    N = INDEX["N"] or 1
    df = INDEX["doc_freq"]
    tf = {}
    for w in words: tf[w] = tf.get(w, 0) + 1
    qvec = {w: (c/len(words)) * math.log(N/df[w])
            for w, c in tf.items() if w in df and df[w] > 0}
    if not qvec: return [], 0.0
    scored = []
    for i, cvec in enumerate(INDEX["vectors"]):
        common = set(qvec) & set(cvec)
        if not common: continue
        dot = sum(qvec[w]*cvec[w] for w in common)
        n1 = math.sqrt(sum(x**2 for x in qvec.values()))
        n2 = math.sqrt(sum(x**2 for x in cvec.values()))
        s = dot/(n1*n2) if n1 and n2 else 0
        if s > 0: scored.append((s, i))
    scored.sort(reverse=True)
    if not scored: return [], 0.0
    return [INDEX["chunks"][i] for _, i in scored[:n]], scored[0][0]

# ── Промпты ───────────────────────────────────────────────────────────
SYSTEM_BASE = """ВАЖНО: Всегда отвечай ТОЛЬКО на русском языке. Никогда не используй украинский, английский или другие языки.
Не используй markdown: никаких **, ##, __, [текст](ссылка).
Каждый блок отделяй пустой строкой.
Не повторяй один и тот же ответ дважды.
"""

SYSTEM_MODE1 = SYSTEM_BASE + """
Ты — АгроПомощник, эксперт по АПК России.
Тебе дан контекст из официальных документов. Отвечай подробно на его основе.

Правила:
— Используй ВСЕ факты из контекста: названия систем, функции, интеграции, цифры
— Раскрывай каждый пункт максимально подробно — пользователь хочет узнать всё
— НЕ выдумывай детали которых нет в контексте
— Если в контексте нет точного ответа — честно скажи об этом и ответь из общих знаний об АПК
— Структура: 📌 главное, ✅ пункты с подробным описанием, 💡 совет, 📎 источник в конце
— Отвечай один раз без повторений"""

SYSTEM_MODE2 = SYSTEM_BASE + """
Ты — АгроПомощник, эксперт по АПК России.
В базе документов по этому вопросу точной информации нет.

Правила:
— ПЕРВОЕ ПРЕДЛОЖЕНИЕ всегда: "📚 В загруженных документах информация по этому вопросу отсутствует. Ниже приведена информация из общих знаний."
— После — подробный полезный ответ
— Структура: 📌 главное, ✅ пункты, 💡 совет
— В конце: "💡 Для точных данных: mcx.gov.ru или региональный Минсельхоз"
— Отвечай один раз без повторений"""

SYSTEM_MODE3 = SYSTEM_BASE + """
Ты — АгроПомощник. Специализируешься на вопросах АПК.
Пользователь задал вопрос не по теме.

Правила:
— Первая строка: "🤖 Этот вопрос не по моей теме."
— Вторая строка: "В загруженных документах информации нет. Вопрос не относится к тематике АгроПомощника."
— Третья строка: куда обратиться (одно предложение)
— ВСЁ. Больше ничего."""

SYSTEM_SECURITY = SYSTEM_BASE + """
Ты — АгроПомощник. Специализируешься на вопросах АПК.
Пользователь задаёт вопрос связанный с обходом законодательства или безопасности данных.

Правила:
— Вежливо откажи
— Сошлись на 152-ФЗ "О персональных данных" и требования Минцифры
— Не давай никаких инструкций по обходу защиты"""

SYSTEM_CLARIFY = SYSTEM_BASE + """
Ты — АгроПомощник, эксперт по АПК России.
Пользователь задал широкий или неконкретный вопрос об АПК.

Правила:
— Коротко признай вопрос
— Задай 2-3 уточняющих вопроса чтобы дать точный ответ:
  например: размер хозяйства, направление деятельности, текущие системы, бюджет
— НЕ давай развёрнутый ответ сразу — сначала уточни"""

# Вопросы требующие уточнения
CLARIFY_TRIGGERS = [
    'проблем', 'помогите', 'что делать', 'с чего начать',
    'посоветуй', 'не знаю', 'помощь нужна', 'хотим внедрить',
    'хочу внедрить', 'как улучшить', 'с чего начинать'
]

# Вопросы безопасности
SECURITY_TRIGGERS = [
    'обойти', 'отключить защиту', 'персональных данных обход',
    'без авторизации', 'взломать', 'отключить логирование',
    'удалить логи', 'скрыть действия'
]

# ── История диалогов ──────────────────────────────────────────────────
user_history = {}

def get_history(uid): return user_history.setdefault(uid, [])

def trim_history(uid):
    h = get_history(uid)
    if len(h) > 12: user_history[uid] = h[-12:]

# ── Очистка текста ────────────────────────────────────────────────────
def clean_md(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [текст](url) -> текст
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── Polza.ai ──────────────────────────────────────────────────────────
def call_polza(system, messages, max_tokens=1000):
    headers = {"Authorization": f"Bearer {POLZA_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": POLZA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    resp = requests.post(POLZA_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        log.error(f"Polza {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise Exception(data["error"].get("message", str(data["error"])))
    return data["choices"][0]["message"]["content"]

# ── Определение типа запроса ──────────────────────────────────────────
# Слова-уточнения — пользователь просит раскрыть предыдущий ответ
EXPAND_TRIGGERS = [
    'подробнее', 'расскажи подробнее', 'подробней', 'расширь',
    'подробно', 'ещё', 'еще', 'продолжи', 'дальше', 'больше',
    'раскрой', 'объясни', 'поподробнее', 'детальнее',
]

def detect_intent(question):
    q_lower = question.strip().lower()
    if any(t in q_lower for t in SECURITY_TRIGGERS):
        return "security"
    # Короткий вопрос-уточнение (типа "подробнее", "расскажи ещё")
    if len(question) < 40 and any(q_lower == t or q_lower.startswith(t) for t in EXPAND_TRIGGERS):
        return "expand"
    if len(question) < 60 and any(t in q_lower for t in CLARIFY_TRIGGERS):
        return "clarify"
    return "normal"

# ── Обработка составного вопроса ─────────────────────────────────────
def split_compound_question(question):
    """Разбивает составной вопрос на части если есть явные разделители"""
    separators = [' и ', '? ', '. А ', '. И ', '\n']
    parts = [question]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = [p.strip() for p in new_parts if len(p.strip()) > 15]
    if len(parts) > 1:
        return parts[:3]
    return None

# ── RAG: основная функция ─────────────────────────────────────────────
def ask_with_rag(uid, question):
    history = get_history(uid)
    recent = history[-2:] if len(history) > 2 else history[:]  # было 6 → стало 2

    intent = detect_intent(question)

    # Уточнение предыдущего ответа — "подробнее", "расскажи ещё"
    if intent == "expand":
        if history:
            # Берём предыдущий вопрос из истории и просим раскрыть подробнее
            prev_question = None
            for msg in reversed(history):
                if msg["role"] == "user":
                    content = msg["content"]
                    # Убираем системный контекст если есть
                    if "ВОПРОС:" in content:
                        prev_question = content.split("ВОПРОС:")[-1].strip()
                    else:
                        prev_question = content
                    break
            if prev_question:
                expanded = f"{prev_question} — раскрой подробнее, дай больше деталей и примеров"
                log.info(f"[{uid}] → EXPAND: '{expanded[:50]}'")
                return _single_rag_full(uid, expanded, recent, history)
        # Если истории нет — просим уточнить
        vk_send(uid, "Уточните пожалуйста — о чём именно рассказать подробнее?")
        return ""

    # Безопасность
    if intent == "security":
        system = SYSTEM_SECURITY
        user_msg = question
        max_tok = 300
        log.info(f"[{uid}] → БЕЗОПАСНОСТЬ")

    # Уточняющие вопросы
    elif intent == "clarify":
        system = SYSTEM_CLARIFY
        user_msg = question
        max_tok = 400
        log.info(f"[{uid}] → УТОЧНЕНИЕ")

    else:
        # Пробуем разбить составной вопрос
        parts = split_compound_question(question)
        if parts and len(parts) > 1:
            log.info(f"[{uid}] Составной вопрос: {len(parts)} частей")
            answers = []
            for i, part in enumerate(parts, 1):
                ans = _single_rag(uid, part, recent)
                answers.append(f"— Вопрос {i}: {part}\n{ans}")
            answer = "\n\n".join(answers)
            answer = clean_md(answer)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            trim_history(uid)
            return answer

        return _single_rag_full(uid, question, recent, history)

    history.append({"role": "user", "content": user_msg})
    trim_history(uid)
    try:
        answer = call_polza(system, recent + [{"role": "user", "content": user_msg}], max_tok)
        answer = clean_md(answer)
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        return _handle_error(e)

def _single_rag(uid, question, recent):
    """RAG для одной части вопроса без истории"""
    chunks, score = search_kb(question)
    if score >= THRESHOLD_DOCS:
        parts = [f"[{c['source']}]:\n{c['text'][:500]}" for c in chunks]  # было 700 → 500
        ctx = "\n\n".join(parts)
        system = SYSTEM_MODE1
        user_msg = f"КОНТЕКСТ:\n{ctx}\n\nВОПРОС: {question}"
        max_tok = 600
    elif score >= THRESHOLD_RELATED:
        system = SYSTEM_MODE2
        user_msg = question
        max_tok = 600
    else:
        system = SYSTEM_MODE3
        user_msg = question
        max_tok = 150
    try:
        return clean_md(call_polza(system, recent + [{"role": "user", "content": user_msg}], max_tok))
    except Exception as e:
        return _handle_error(e)

def _single_rag_full(uid, question, recent, history):
    """RAG для одного вопроса с сохранением истории"""
    chunks, max_score = search_kb(question)
    log.info(f"[{uid}] score={max_score:.4f} | '{question[:50]}'")

    if max_score >= THRESHOLD_DOCS:
        parts = [f"[{c['source']}]:\n{c['text'][:500]}" for c in chunks]  # было 700 → 500
        context = "\n\n".join(parts)
        system = SYSTEM_MODE1
        user_msg = f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ:\n{context}\n\nВОПРОС: {question}"
        max_tok = 1300  # было 1000 → 1300
        log.info(f"[{uid}] → РЕЖИМ 1 (документы)")

    elif max_score >= THRESHOLD_RELATED:
        system = SYSTEM_MODE2
        user_msg = question
        max_tok = 900
        log.info(f"[{uid}] → РЕЖИМ 2 (общие знания АПК)")

    else:
        system = SYSTEM_MODE3
        user_msg = question
        max_tok = 200
        log.info(f"[{uid}] → РЕЖИМ 3 (не по теме)")

    history.append({"role": "user", "content": user_msg})
    trim_history(uid)

    try:
        answer = call_polza(system, recent + [{"role": "user", "content": user_msg}], max_tok)
        answer = clean_md(answer)
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        return _handle_error(e)

def _handle_error(e):
    err = str(e)
    log.error(f"Polza error: {err}")
    if "402" in err: return "⚠️ Ошибка: недостаточно средств на Polza.ai."
    if "401" in err: return "⚠️ Ошибка: неверный API ключ."
    if "429" in err: return "⚠️ Превышен лимит запросов. Подождите 30 секунд."
    return "⚠️ Произошла ошибка, попробуйте снова через несколько секунд."

# ── VK API ────────────────────────────────────────────────────────────
def vk_typing(user_id):
    """Отправляет один сигнал 'печатает...' (держится ~10 сек)"""
    params = {
        "user_id": user_id,
        "type": "typing",
        "access_token": VK_TOKEN,
        "v": VK_API_VER,
    }
    try:
        requests.post(VK_API_URL + "messages.setActivity", data=params, timeout=5)
    except:
        pass

def vk_typing_loop(user_id, stop_event):
    """Повторяет 'печатает...' каждые 8 сек пока бот думает"""
    while not stop_event.is_set():
        vk_typing(user_id)
        stop_event.wait(8)

def vk_send(user_id, text):
    import random
    for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
        params = {
            "user_id": user_id, "message": chunk,
            "random_id": random.randint(1, 2**31),
            "access_token": VK_TOKEN, "v": VK_API_VER,
        }
        resp = requests.post(VK_API_URL + "messages.send", data=params, timeout=10)
        log.info(f"VK send {user_id}: {resp.status_code}")

def vk_send_keyboard(user_id, text):
    import random
    keyboard = {
        "one_time": False,
        "buttons": [
            [{"action": {"type": "text", "label": "🌾 Зерновод"}, "color": "positive"},
             {"action": {"type": "text", "label": "🐄 Животноводство"}, "color": "positive"}],
            [{"action": {"type": "text", "label": "🎣 Рыболовство"}, "color": "positive"},
             {"action": {"type": "text", "label": "🌱 Овощеводство"}, "color": "positive"}],
            [{"action": {"type": "text", "label": "💰 Гранты и субсидии"}, "color": "primary"},
             {"action": {"type": "text", "label": "📱 Цифровизация АПК"}, "color": "primary"}],
            [{"action": {"type": "text", "label": "🏦 Льготные кредиты"}, "color": "primary"},
             {"action": {"type": "text", "label": "📊 Стратегия до 2030"}, "color": "primary"}],
            [{"action": {"type": "text", "label": "🔄 Начать заново"}, "color": "secondary"}],
        ]
    }
    params = {
        "user_id": user_id, "message": text,
        "keyboard": json.dumps(keyboard, ensure_ascii=False),
        "random_id": random.randint(1, 2**31),
        "access_token": VK_TOKEN, "v": VK_API_VER,
    }
    requests.post(VK_API_URL + "messages.send", data=params, timeout=10)

QUICK_Q = {
    "🌾 Зерновод":          "Какие меры государственной поддержки и цифровые инструменты предусмотрены для зернового хозяйства в документах АПК?",
    "🐄 Животноводство":    "Какая господдержка и цифровые системы доступны для животноводства согласно документам?",
    "🎣 Рыболовство":       "Какие цифровые инструменты и меры поддержки предусмотрены для рыбохозяйственного комплекса в документах?",
    "🌱 Овощеводство":      "Какие субсидии, гранты и программы доступны для овощеводства?",
    "💰 Гранты и субсидии": "Расскажи про подсистему Гранты МФХ и другие субсидии в АИС Субсидии АПК подробно.",
    "📱 Цифровизация АПК":  "Какие цифровые технологии внедряются в АПК согласно стратегии 3309-р? Что такое единая цифровая платформа?",
    "🏦 Льготные кредиты":  "Расскажи подробно про подсистему льготного кредитования в АИС Субсидии АПК.",
    "📊 Стратегия до 2030":  "Расскажи подробно об основных задачах, индикаторах и дорожной карте Стратегии ЦТ АПК до 2030.",
}

WELCOME = (
    "👋 Здравствуйте! Я АгроПомощник — система управления знаниями для агропредприятий России.\n\n"
    "📚 База знаний:\n"
    "— Стратегия ЦТ АПК до 2030 (Пост. Правительства № 3309-р)\n"
    "— Монография «Цифровая трансформация АПК» (Худякова Е.В., 2022)\n"
    "— НД Минсельхоза РФ (АИС «Субсидии АПК»)\n\n"
    "🤖 Как работаю:\n"
    "— Вопрос есть в документах → подробный ответ с источником\n"
    "— Вопрос по АПК, но нет в файлах → отвечаю из общих знаний, предупреждаю\n"
    "— Вопрос не по теме → кратко объясняю и направляю\n\n"
    "Выберите тему или напишите вопрос 👇"
)

def handle_message(user_id, text):
    text = text.strip()
    if text.lower() in ["начать","старт","start","/start","привет","hello","🔄 начать заново"]:
        user_history.pop(user_id, None)
        vk_send_keyboard(user_id, WELCOME)
        return
    if text in QUICK_Q:
        stop = threading.Event()
        t = threading.Thread(target=vk_typing_loop, args=(user_id, stop), daemon=True)
        t.start()
        try:
            answer = ask_with_rag(user_id, QUICK_Q[text])
        finally:
            stop.set()
        vk_send(user_id, answer)
        return
    stop = threading.Event()
    t = threading.Thread(target=vk_typing_loop, args=(user_id, stop), daemon=True)
    t.start()
    try:
        answer = ask_with_rag(user_id, text)
    finally:
        stop.set()
    vk_send(user_id, answer)

# ── Flask Webhook ─────────────────────────────────────────────────────
from flask import Flask, request as flask_req

app = Flask(__name__)

@app.route("/vk_webhook", methods=["POST"])
def vk_webhook():
    data = flask_req.get_json(silent=True) or {}
    event_type = data.get("type")

    if event_type == "confirmation":
        return VK_CONFIRM, 200

    if data.get("secret") != SECRET:
        log.warning("Wrong secret key!")
        return "ok", 200

    if event_type == "message_new":
        msg = data.get("object", {}).get("message", {})
        user_id = msg.get("from_id")
        text = msg.get("text", "").strip()
        msg_id = msg.get("id", 0)
        msg_date = msg.get("date", 0)

        # Фильтр старых сообщений (60 сек)
        now = int(_time.time())
        if msg_date and (now - msg_date) > 60:
            log.info(f"Старое сообщение (возраст {now-msg_date}с) — пропускаем")
            return "ok", 200

        # Дедупликация
        msg_id_str = str(msg_id)
        if msg_id and msg_id_str in processed_ids:
            log.info(f"Дубль {msg_id} — пропускаем")
            return "ok", 200
        if msg_id:
            processed_ids[msg_id_str] = _time.time()
            cleaned = {k: v for k, v in processed_ids.items() if _time.time() - v < 7200}
            processed_ids.clear()
            processed_ids.update(cleaned)
            _save_ids(processed_ids)

        if user_id and text:
            log.info(f"[{user_id}] '{text[:60]}'")
            handle_message(user_id, text)

    return "ok", 200

@app.route("/", methods=["GET"])
def index():
    return "АгроПомощник VK ✅", 200

if __name__ == "__main__":
    load_index()
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Запущен на порту {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
