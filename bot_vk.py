"""
bot_vk.py — ВКонтакте бот «АгроПомощник»
"""
import os, logging, re, math, pickle, requests, json
from pathlib import Path

# ── Загрузка .env ─────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

VK_TOKEN    = os.environ.get("VK_TOKEN", "")
VK_GROUP_ID = os.environ.get("VK_GROUP_ID", "")
VK_CONFIRM  = os.environ.get("VK_CONFIRM", "")
POLZA_KEY   = os.environ.get("POLZA_API_KEY", "")
SECRET      = os.environ.get("VK_SECRET", "")

POLZA_URL   = "https://polza.ai/api/v1/chat/completions"
POLZA_MODEL = "deepseek/deepseek-chat"
VK_API_URL  = "https://api.vk.com/method/"
VK_API_VER  = "5.199"

# ── Защита от дублей ──────────────────────────────────────────────────
processed_ids = set()

# ── TF-IDF индекс ─────────────────────────────────────────────────────
INDEX = {"chunks": [], "vectors": [], "doc_freq": {}, "N": 0}

def load_index():
    global INDEX
    p = Path("knowledge_index.pkl")
    if p.exists():
        with open(p, "rb") as f:
            INDEX = pickle.load(f)
        log.info(f"Индекс загружен: {INDEX['N']} чанков")
    else:
        log.warning("knowledge_index.pkl не найден")

def tokenize(text):
    return re.findall(r'[а-яёa-z]+', text.lower())

RELEVANCE_THRESHOLD = 0.05

def search_kb(query, n=4):
    if not INDEX["chunks"]: return [], 0.0
    words = tokenize(query)
    if not words: return [], 0.0
    N = INDEX["N"] or 1
    df = INDEX["doc_freq"]
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
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
SYSTEM_WITH_DOCS = """Ты — АгроПомощник, умный помощник для сельскохозяйственных предпринимателей России.

Тебе предоставлен контекст из официальных документов АПК. Отвечай строго по нему.

Правила оформления ответа:
— Пиши живым, простым языком
— Используй эмодзи для структуры: 📌 для главного, ✅ для пунктов, 💡 для совета, 📎 для источника
— НЕ используй markdown: никаких **, ##, __, ---
— Каждый смысловой блок отделяй пустой строкой
— В конце всегда пиши источник через 📎
— Один чёткий ответ, без повторений
— Максимум 6-8 предложений"""

SYSTEM_NO_DOCS = """Ты — АгроПомощник, умный помощник для сельскохозяйственных предпринимателей России.

По этому вопросу в базе документов ничего не найдено. Действуй по одному из двух сценариев:

СЦЕНАРИЙ А — вопрос близок к АПК (фермерство, урожай, животные, рыбалка как бизнес, субсидии, техника и т.д.):
Начни с: "📚 В моей базе документов по этому вопросу ничего нет, но из общих знаний могу сказать:"
Затем дай краткий полезный ответ (3-5 предложений) с эмодзи для структуры.
В конце: "💡 Для точной информации советую: mcx.gov.ru"

СЦЕНАРИЙ Б — вопрос совсем не по теме (покупки, кино, погода, удочки для рыбалки и т.д.):
Начни с: "😊 Это немного не по моей части —"
Одним предложением объясни что ты про АПК и цифровизацию сельского хозяйства.
Дай один короткий дружелюбный совет куда обратиться (2 предложения максимум).
НЕ давай развёрнутых советов по теме вне АПК.

Правила оформления:
— Живой простой язык
— Эмодзи для структуры
— Никаких **, ##, __, markdown
— Один ответ, без повторений"""

# ── История диалогов ──────────────────────────────────────────────────
user_history = {}

def get_history(uid):
    return user_history.setdefault(uid, [])

def trim_history(uid):
    h = get_history(uid)
    if len(h) > 10:
        user_history[uid] = h[-10:]

# ── Очистка markdown ──────────────────────────────────────────────────
def clean_markdown(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ── Polza.ai ──────────────────────────────────────────────────────────
def call_polza(system, messages):
    headers = {"Authorization": f"Bearer {POLZA_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": POLZA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 700,
        "temperature": 0.4,
    }
    resp = requests.post(POLZA_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        log.error(f"Polza {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise Exception(data["error"].get("message", str(data["error"])))
    return data["choices"][0]["message"]["content"]

def ask_with_rag(uid, question):
    chunks, max_score = search_kb(question)
    docs_found = bool(chunks) and max_score >= RELEVANCE_THRESHOLD

    if docs_found:
        parts = []
        for c in chunks:
            label = {"strategy": "Документ 1 (Стратегия ЦТ АПК)",
                     "monograph": "Документ 2 (Монография Худяковой)",
                     "normative": "Документ 3 (НД Минсельхоза)"
                     }.get(c["doc_id"], c["doc_id"])
            parts.append(f"[{label}]:\n{c['text'][:400]}")
        context = "\n\n".join(parts)
        system = SYSTEM_WITH_DOCS
        user_msg = f"КОНТЕКСТ ИЗ ДОКУМЕНТОВ:\n{context}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}"
        log.info(f"[{uid}] score={max_score:.3f} -> ДОКУМЕНТЫ")
    else:
        system = SYSTEM_NO_DOCS
        user_msg = question
        log.info(f"[{uid}] score={max_score:.3f} -> ОБЩИЕ ЗНАНИЯ / ОФФТОП")

    history = get_history(uid)
    recent = history[-6:] if len(history) > 6 else history[:]
    history.append({"role": "user", "content": user_msg})
    trim_history(uid)

    try:
        answer = call_polza(system, recent + [{"role": "user", "content": user_msg}])
        answer = clean_markdown(answer)
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
            "user_id": user_id,
            "message": chunk,
            "random_id": random.randint(1, 2**31),
            "access_token": VK_TOKEN,
            "v": VK_API_VER,
        }
        requests.post(VK_API_URL + "messages.send", data=params, timeout=10)

def vk_send_keyboard(user_id, text):
    import random
    keyboard = {
        "one_time": False,
        "buttons": [
            [
                {"action": {"type": "text", "label": "🌾 Зерновод"}, "color": "positive"},
                {"action": {"type": "text", "label": "🐄 Животноводство"}, "color": "positive"},
            ],
            [
                {"action": {"type": "text", "label": "🎣 Рыболовство"}, "color": "positive"},
                {"action": {"type": "text", "label": "🌱 Овощеводство"}, "color": "positive"},
            ],
            [
                {"action": {"type": "text", "label": "💰 Гранты и субсидии"}, "color": "primary"},
                {"action": {"type": "text", "label": "📱 Цифровизация АПК"}, "color": "primary"},
            ],
            [
                {"action": {"type": "text", "label": "🏦 Льготные кредиты"}, "color": "primary"},
                {"action": {"type": "text", "label": "📊 Стратегия до 2030"}, "color": "primary"},
            ],
            [
                {"action": {"type": "text", "label": "🔄 Начать заново"}, "color": "secondary"},
            ],
        ]
    }
    params = {
        "user_id": user_id,
        "message": text,
        "keyboard": json.dumps(keyboard, ensure_ascii=False),
        "random_id": random.randint(1, 2**31),
        "access_token": VK_TOKEN,
        "v": VK_API_VER,
    }
    requests.post(VK_API_URL + "messages.send", data=params, timeout=10)

# ── Быстрые кнопки ────────────────────────────────────────────────────
QUICK_Q = {
    "🌾 Зерновод":         "Я зерновой фермер. Какие меры государственной поддержки и цифровые инструменты для меня предусмотрены?",
    "🐄 Животноводство":   "Занимаюсь животноводством. Какая поддержка и цифровые системы для меня доступны?",
    "🎣 Рыболовство":      "Занимаюсь рыболовством как бизнесом. Какие цифровые инструменты и меры поддержки актуальны?",
    "🌱 Овощеводство":     "Занимаюсь овощеводством. Какие субсидии, гранты и программы мне доступны?",
    "💰 Гранты и субсидии":"Хочу получить грант для агробизнеса. Какие виды грантов существуют и как работает подсистема Гранты МФХ?",
    "📱 Цифровизация АПК": "Какие цифровые технологии внедряются в АПК согласно стратегии 3309-р? Что такое единая цифровая платформа?",
    "🏦 Льготные кредиты": "Как работает льготное кредитование сельхозтоваропроизводителей? Какие ставки и условия?",
    "📊 Стратегия до 2030": "Расскажи об основных задачах и индикаторах Стратегии цифровой трансформации АПК до 2030.",
}

WELCOME = (
    "👋 Здравствуйте! Я АгроПомощник — система управления знаниями для агропредприятий России.\n\n"
    "📚 Моя база знаний:\n"
    "— Стратегия цифровой трансформации АПК до 2030 (Пост. Правительства № 3309-р)\n"
    "— Монография «Цифровая трансформация АПК» (Худякова Е.В., 2022)\n"
    "— Нормативные документы Минсельхоза РФ (АИС «Субсидии АПК»)\n\n"
    "🤖 Как работаю: ищу ответ в документах → если нашёл, отвечаю по ним → если нет, предупреждаю и отвечаю из общих знаний.\n\n"
    "Выберите тему или напишите вопрос 👇"
)

# ── Обработка сообщений ───────────────────────────────────────────────
def handle_message(user_id, text):
    text = text.strip()

    if text.lower() in ["начать", "старт", "start", "/start", "привет", "hello", "🔄 начать заново"]:
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
from flask import Flask, request as flask_request

app = Flask(__name__)

@app.route("/vk_webhook", methods=["POST"])
def vk_webhook():
    data = flask_request.get_json(silent=True) or {}
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

        # Защита от дублей
        if msg_id and msg_id in processed_ids:
            log.info(f"Дубль {msg_id} — пропускаем")
            return "ok", 200

        if msg_id:
            processed_ids.add(msg_id)
            if len(processed_ids) > 1000:
                old = list(processed_ids)[:500]
                for m in old:
                    processed_ids.discard(m)

        if user_id and text:
            log.info(f"Сообщение от {user_id}: {text[:50]}")
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
