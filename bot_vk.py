"""
bot_vk.py — ВКонтакте бот «АгроПомощник»
Платформа: VK (vk.com) — отечественный сервис
Поиск: TF-IDF по базе знаний из 3 PDF (RAG)
ИИ:    Polza.ai — DeepSeek V3
Регистрация бота: vk.com/dev → Создать приложение → Сообщество
Бесплатно, без VPN, работает в России
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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# ── Настройки ─────────────────────────────────────────────────────────
VK_TOKEN       = os.environ.get("VK_TOKEN", "")         # токен сообщества VK
VK_GROUP_ID    = os.environ.get("VK_GROUP_ID", "")      # ID сообщества (без минуса)
VK_CONFIRM     = os.environ.get("VK_CONFIRM", "")       # строка подтверждения сервера
POLZA_KEY      = os.environ.get("POLZA_API_KEY", "")    # Polza.ai ключ
SECRET         = os.environ.get("VK_SECRET", "secret")  # секретный ключ webhook

POLZA_URL      = "https://polza.ai/api/v1/chat/completions"
POLZA_MODEL    = "deepseek/deepseek-chat"

VK_API_URL     = "https://api.vk.com/method/"
VK_API_VERSION = "5.199"

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
        log.warning("knowledge_index.pkl не найден — запустите indexer.py")

def tokenize(text: str) -> list:
    return re.findall(r'[а-яёa-z]+', text.lower())

# ── TF-IDF поиск ─────────────────────────────────────────────────────
RELEVANCE_THRESHOLD = 0.05

def search_kb(query: str, n: int = 4):
    if not INDEX["chunks"]: return [], 0.0
    words = tokenize(query)
    if not words: return [], 0.0
    N = INDEX["N"] or 1
    df = INDEX["doc_freq"]
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    qvec = {w: (c / len(words)) * math.log(N / df[w])
            for w, c in tf.items() if w in df and df[w] > 0}
    if not qvec: return [], 0.0
    scored = []
    for i, cvec in enumerate(INDEX["vectors"]):
        common = set(qvec) & set(cvec)
        if not common: continue
        dot = sum(qvec[w] * cvec[w] for w in common)
        n1 = math.sqrt(sum(x**2 for x in qvec.values()))
        n2 = math.sqrt(sum(x**2 for x in cvec.values()))
        s = dot / (n1 * n2) if n1 and n2 else 0
        if s > 0: scored.append((s, i))
    scored.sort(reverse=True)
    if not scored: return [], 0.0
    return [INDEX["chunks"][i] for _, i in scored[:n]], scored[0][0]

# ── Промпты ───────────────────────────────────────────────────────────
SYSTEM_WITH_DOCS = """Ты - АгроПомощник, система управления знаниями для сельскохозяйственных предпринимателей России.

База знаний - три официальных документа:
1. Распоряжение Правительства РФ N 3309-р - Стратегия цифровой трансформации АПК до 2030
2. Монография Цифровая трансформация АПК (Худякова Е.В., РГАУ-МСХА, 2022)
3. Нормативные документы Минсельхоза РФ (АИС Субсидии АПК, приказ N503-2024)

Правила:
- Отвечай на основе предоставленного КОНТЕКСТА из документов
- Указывай источник: [Документ 1], [Документ 2] или [Документ 3]
- Конкретные цифры, суммы, названия подсистем
- Структура: ответ - детали - следующий шаг
- Кратко и по делу, на русском языке"""

SYSTEM_NO_DOCS = """Ты - АгроПомощник, система управления знаниями для сельскохозяйственных предпринимателей России.

В базе знаний точного ответа не найдено.

Правила:
- Начни с: "📚 В моей базе знаний точной информации не нашлось, но вот что я знаю об АПК:"
- Дай полезный ответ из общих знаний об АПК России
- Заверши: "💡 Для актуальной информации: mcx.gov.ru или региональное минсельхоз."
- На русском языке, дружелюбно"""

# ── История диалогов ──────────────────────────────────────────────────
user_history: dict = {}

def get_history(uid: int) -> list:
    return user_history.setdefault(uid, [])

def trim_history(uid: int):
    h = get_history(uid)
    if len(h) > 10:
        user_history[uid] = h[-10:]

# ── Polza.ai запрос ───────────────────────────────────────────────────
def call_polza(system: str, messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {POLZA_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": POLZA_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 1000,
        "temperature": 0.3,
    }
    resp = requests.post(POLZA_URL, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        log.error(f"Polza {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise Exception(data["error"].get("message", str(data["error"])))
    return data["choices"][0]["message"]["content"]

# ── RAG ───────────────────────────────────────────────────────────────
def ask_with_rag(uid: int, question: str) -> str:
    chunks, max_score = search_kb(question)
    docs_found = bool(chunks) and max_score >= RELEVANCE_THRESHOLD

    if docs_found:
        parts = []
        for c in chunks:
            label = {"strategy": "Документ 1", "monograph": "Документ 2",
                     "normative": "Документ 3"}.get(c["doc_id"], c["doc_id"])
            parts.append(f"[{label}]: {c['text'][:400]}")
        context = "\n\n".join(parts)
        system = SYSTEM_WITH_DOCS
        user_msg = f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}"
        log.info(f"[{uid}] score={max_score:.3f} -> ДОКУМЕНТЫ")
    else:
        system = SYSTEM_NO_DOCS
        user_msg = question
        log.info(f"[{uid}] score={max_score:.3f} -> ОБЩИЕ ЗНАНИЯ")

    history = get_history(uid)
    recent = history[-6:] if len(history) > 6 else history[:]
    history.append({"role": "user", "content": user_msg})
    trim_history(uid)

    try:
        answer = call_polza(system, recent + [{"role": "user", "content": user_msg}])
        history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        err = str(e)
        log.error(f"Polza error: {err}")
        if "402" in err: return "⚠️ Недостаточно средств на Polza.ai."
        if "401" in err: return "⚠️ Неверный POLZA_API_KEY."
        return f"⚠️ Ошибка: {err[:200]}"

# ── VK API: отправка сообщения ────────────────────────────────────────
def vk_send(user_id: int, text: str):
    """Отправить сообщение пользователю VK."""
    import random
    # VK ограничивает сообщения до 4096 символов
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        params = {
            "user_id": user_id,
            "message": chunk,
            "random_id": random.randint(1, 2**31),
            "access_token": VK_TOKEN,
            "v": VK_API_VERSION,
        }
        resp = requests.post(VK_API_URL + "messages.send", data=params, timeout=10)
        log.info(f"VK send to {user_id}: {resp.status_code}")

def vk_send_keyboard(user_id: int, text: str):
    """Отправить сообщение с кнопками-меню."""
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
                {"action": {"type": "text", "label": "🔄 Сбросить историю"}, "color": "secondary"},
            ],
        ]
    }
    params = {
        "user_id": user_id,
        "message": text,
        "keyboard": json.dumps(keyboard, ensure_ascii=False),
        "random_id": random.randint(1, 2**31),
        "access_token": VK_TOKEN,
        "v": VK_API_VERSION,
    }
    resp = requests.post(VK_API_URL + "messages.send", data=params, timeout=10)
    log.info(f"VK send keyboard to {user_id}: {resp.status_code}")

# ── Быстрые кнопки ────────────────────────────────────────────────────
QUICK_Q = {
    "🌾 Зерновод":        "Я зерновой фермер. Какие меры государственной поддержки и цифровые инструменты для меня предусмотрены?",
    "🐄 Животноводство":  "Занимаюсь животноводством. Какая поддержка и цифровые системы для меня доступны?",
    "🎣 Рыболовство":     "Занимаюсь рыболовством. Какие цифровые инструменты и меры поддержки актуальны?",
    "🌱 Овощеводство":    "Занимаюсь овощеводством. Какие субсидии, гранты и программы мне доступны?",
    "💰 Гранты и субсидии":"Хочу получить грант. Какие виды грантов существуют, как работает подсистема Гранты МФХ?",
    "📱 Цифровизация АПК": "Какие цифровые технологии внедряются в АПК согласно стратегии N3309-р?",
    "🏦 Льготные кредиты": "Как работает льготное кредитование сельхозтоваропроизводителей?",
    "📊 Стратегия до 2030":"Расскажи об основных задачах и индикаторах Стратегического направления ЦТ АПК до 2030.",
}

# ── Обработка входящего сообщения ─────────────────────────────────────
def handle_message(user_id: int, text: str):
    text = text.strip()

    # Старт / приветствие
    if text.lower() in ["начать", "старт", "start", "/start", "привет"]:
        user_history.pop(user_id, None)
        vk_send_keyboard(user_id,
            "👋 Здравствуйте! Я АгроПомощник — система управления знаниями для агропредприятий.\n\n"
            "📚 База знаний (3 документа):\n"
            "• Распоряжение Правительства РФ № 3309-р\n"
            "• Монография Худяковой Е.В. (РГАУ-МСХА, 2022)\n"
            "• НД Минсельхоза РФ (АИС «Субсидии АПК»)\n\n"
            "🤖 Поиск: TF-IDF + DeepSeek V3 (Polza.ai)\n\n"
            "Выберите тему или напишите вопрос 👇"
        )
        return

    # Сброс истории
    if text in ["🔄 Сбросить историю", "сброс", "reset"]:
        user_history.pop(user_id, None)
        vk_send_keyboard(user_id, "✅ История очищена! Задайте новый вопрос.")
        return

    # Быстрые кнопки
    if text in QUICK_Q:
        question = QUICK_Q[text]
        answer = ask_with_rag(user_id, question)
        vk_send(user_id, answer)
        return

    # Обычный вопрос
    answer = ask_with_rag(user_id, text)
    vk_send(user_id, answer)

# ── Webhook сервер (Flask) ────────────────────────────────────────────
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/vk_webhook", methods=["POST"])
def vk_webhook():
    data = request.get_json(silent=True) or {}
    log.info(f"VK event: {data.get('type')} from group {data.get('group_id')}")

    # Подтверждение сервера
    if data.get("type") == "confirmation":
        return VK_CONFIRM, 200

    # Проверка секретного ключа
    if data.get("secret") != SECRET:
        log.warning("Wrong secret key!")
        return "ok", 200

    # Новое сообщение
    if data.get("type") == "message_new":
        msg = data.get("object", {}).get("message", {})
        user_id = msg.get("from_id")
        text = msg.get("text", "").strip()
        if user_id and text:
            handle_message(user_id, text)

    return "ok", 200

@app.route("/", methods=["GET"])
def index():
    return "АгроПомощник VK Bot работает! ✅"

# ── Запуск ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_index()
    port = int(os.environ.get("PORT", 5000))
    log.info(f"АгроПомощник VK запущен на порту {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
