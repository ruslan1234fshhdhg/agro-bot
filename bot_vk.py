"""
bot_vk.py — ВКонтакте бот «АгроПомощник»
Три режима без check_relevance:
  РЕЖИМ 1 (score >= 0.12) — ответ строго по документам
  РЕЖИМ 2 (score >= 0.015) — по теме АПК, из общих знаний с предупреждением
  РЕЖИМ 3 (score < 0.015) — не по теме, кратко
"""
import os, logging, re, math, pickle, requests, json
import json as _json, time as _time
from pathlib import Path as _Path

# ── Загрузка .env ─────────────────────────────────────────────────────
env_path = Path(_Path(__file__).parent / ".env") if False else _Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

VK_TOKEN    = os.environ.get("VK_TOKEN", "")
VK_CONFIRM  = os.environ.get("VK_CONFIRM", "")
POLZA_KEY   = os.environ.get("POLZA_API_KEY", "")
SECRET      = os.environ.get("VK_SECRET", "")

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

# Стоп-слова — убираем слова дающие ложные совпадения
STOPWORDS = {
    'кто','что','как','это','все','для','при','или','они','она','оно','нет','был',
    'быть','есть','вот','так','уже','еще','ещё','там','тут','где','без','над',
    'под','про','вам','вас','нас','мне','том','эти','эта','этот','эту','ним',
    'них','той','тот','три','два','раз','лет','год','кого','чем','тем','тех',
    'тому','того','всё','всех','всем','наш','ваш','can','the','and','for',
    'not','you','are','but','was','ней','ооо','зао','оао','пао',
}

def tokenize(text):
    return [w for w in re.findall(r'[а-яёa-z]{3,}', text.lower()) if w not in STOPWORDS]

# ── Пороги трёх режимов ───────────────────────────────────────────────
THRESHOLD_DOCS    = 0.12   # >= 0.12 → в документах
THRESHOLD_RELATED = 0.015  # >= 0.015 → по теме АПК

def search_kb(query, n=6):
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
SYSTEM_MODE1 = """Ты — АгроПомощник, эксперт по АПК России.

Тебе дан контекст из официальных документов. Отвечай подробно на его основе.

Правила:
— Используй конкретику из контекста: цифры, названия программ, подсистем, ставки
— Структура: 📌 главное, ✅ пункты, 💡 совет, 📎 источник
— В конце ОБЯЗАТЕЛЬНО укажи источник через 📎
— Не используй ** ## __ (никакого markdown)
— Каждый блок отделяй пустой строкой
— Отвечай только один раз, без повторений"""

SYSTEM_MODE2 = """Ты — АгроПомощник, эксперт по АПК России.

В базе документов по этому вопросу нет точной информации.

Правила:
— Начни СТРОГО с фразы: "📚 В моей базе документов по этому вопросу информации нет, отвечаю из общих знаний об АПК:"
— После — подробный полезный ответ из общих знаний
— Структура: 📌 главное, ✅ пункты, 💡 совет
— В конце: "💡 Для точных данных: mcx.gov.ru или региональный Минсельхоз"
— Не используй ** ## __
— Отвечай только один раз, без повторений"""

SYSTEM_MODE3 = """Ты — АгроПомощник. Специализируешься на вопросах АПК и цифровой трансформации сельского хозяйства России.

Пользователь задал вопрос не по теме.

Правила:
— Начни с: "🤖 Этот вопрос не по моей теме."
— Одним предложением: чем ты занимаешься
— Одним предложением: куда обратиться по этому вопросу
— ВСЁ. Больше ничего не добавляй.
— Не используй ** ## __"""

# ── История диалогов ──────────────────────────────────────────────────
user_history = {}

def get_history(uid): return user_history.setdefault(uid, [])

def trim_history(uid):
    h = get_history(uid)
    if len(h) > 12: user_history[uid] = h[-12:]

# ── Очистка markdown ──────────────────────────────────────────────────
def clean_md(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── Polza.ai ──────────────────────────────────────────────────────────
def call_polza(system, messages, max_tokens=1000):
    headers = {"Authorization": f"Bearer {POLZA_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": POLZA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    resp = requests.post(POLZA_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        log.error(f"Polza {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise Exception(data["error"].get("message", str(data["error"])))
    return data["choices"][0]["message"]["content"]

# ── RAG: три режима без check_relevance ───────────────────────────────
def ask_with_rag(uid, question):
    chunks, max_score = search_kb(question)
    history = get_history(uid)
    recent = history[-6:] if len(history) > 6 else history[:]

    log.info(f"[{uid}] score={max_score:.4f} | вопрос: {question[:50]}")

    if max_score >= THRESHOLD_DOCS:
        # РЕЖИМ 1: есть в документах
        parts = []
        for c in chunks:
            parts.append(f"[{c['source']}]:\n{c['text'][:700]}")
        context = "\n\n".join(parts)
        system = SYSTEM_MODE1
        user_msg = f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ:\n{context}\n\nВОПРОС: {question}"
        max_tok = 1200
        log.info(f"[{uid}] → РЕЖИМ 1 (документы)")

    elif max_score >= THRESHOLD_RELATED:
        # РЕЖИМ 2: по теме АПК, нет в документах
        system = SYSTEM_MODE2
        user_msg = question
        max_tok = 1000
        log.info(f"[{uid}] → РЕЖИМ 2 (общие знания АПК)")

    else:
        # РЕЖИМ 3: не по теме
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
        err = str(e)
        log.error(f"Polza error: {err}")
        if "402" in err: return "⚠️ Ошибка: недостаточно средств на Polza.ai."
        if "401" in err: return "⚠️ Ошибка: неверный API ключ."
        return "⚠️ Произошла ошибка, попробуйте снова."

# ── VK API ────────────────────────────────────────────────────────────
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

# ── Быстрые кнопки ────────────────────────────────────────────────────
QUICK_Q = {
    "🌾 Зерновод":          "Я зерновой фермер. Какие меры государственной поддержки и цифровые инструменты для меня предусмотрены в документах АПК?",
    "🐄 Животноводство":    "Занимаюсь животноводством. Какая господдержка и цифровые системы для меня доступны?",
    "🎣 Рыболовство":       "Занимаюсь рыболовством. Какие цифровые инструменты и меры поддержки предусмотрены в нормативных документах?",
    "🌱 Овощеводство":      "Занимаюсь овощеводством. Какие субсидии, гранты и программы мне доступны?",
    "💰 Гранты и субсидии": "Какие виды грантов и субсидий АПК существуют? Расскажи про подсистему Гранты МФХ в АИС Субсидии АПК.",
    "📱 Цифровизация АПК":  "Какие цифровые технологии внедряются в АПК согласно стратегии 3309-р? Что такое единая цифровая платформа АПК?",
    "🏦 Льготные кредиты":  "Расскажи про подсистему льготного кредитования в АИС Субсидии АПК. Какие виды льготных кредитов предусмотрены?",
    "📊 Стратегия до 2030":  "Расскажи подробно об основных задачах, индикаторах и дорожной карте Стратегии цифровой трансформации АПК до 2030.",
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

# ── Обработка сообщений ───────────────────────────────────────────────
def handle_message(user_id, text):
    text = text.strip()
    if text.lower() in ["начать","старт","start","/start","привет","hello","🔄 начать заново"]:
        user_history.pop(user_id, None)
        vk_send_keyboard(user_id, WELCOME)
        return
    if text in QUICK_Q:
        answer = ask_with_rag(user_id, QUICK_Q[text])
        vk_send(user_id, answer)
        return
    answer = ask_with_rag(user_id, text)
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
