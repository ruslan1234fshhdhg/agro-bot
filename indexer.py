"""
indexer.py — Индексирует PDF-файлы в TF-IDF базу знаний.
Запустить один раз перед запуском бота: python indexer.py
"""
import json, math, re, pickle, sys
from pathlib import Path
import pdfplumber

PDF_FILES = {
    "strategy": {
        "path": "docs/strategy_3309.pdf",
        "source": "Распоряжение Правительства РФ № 3309-р от 23.11.2023 — Стратегия ЦТ АПК до 2030"
    },
    "monograph": {
        "path": "docs/monograph_khudyakova.pdf",
        "source": "Монография «Цифровая трансформация АПК» (Худякова Е.В., РГАУ-МСХА, 2022)"
    },
    "normative": {
        "path": "docs/normative_minselhoz.pdf",
        "source": "Нормативные документы Минсельхоза РФ (АИС Субсидии АПК, приказ №503-2024)"
    },
}

def extract_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def chunk_text(text: str, doc_id: str, source: str, size=120, overlap=15) -> list[dict]:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i:i+size]
        chunk_text = " ".join(chunk_words).strip()
        if len(chunk_text) > 80:
            chunks.append({
                "chunk_id": f"{doc_id}_{idx:04d}",
                "doc_id": doc_id,
                "source": source,
                "text": chunk_text,
            })
            idx += 1
        i += size - overlap
    return chunks

def build_tfidf(chunks: list[dict]) -> dict:
    from collections import defaultdict
    N = len(chunks)
    doc_freq = defaultdict(int)
    for c in chunks:
        for w in set(re.findall(r'[а-яёa-z]+', c["text"].lower())):
            doc_freq[w] += 1
    vectors = []
    for c in chunks:
        words = re.findall(r'[а-яёa-z]+', c["text"].lower())
        tf = defaultdict(int)
        for w in words:
            tf[w] += 1
        vec = {}
        for w, cnt in tf.items():
            if doc_freq[w] > 0:
                idf = math.log(N / doc_freq[w])
                vec[w] = (cnt / len(words)) * idf
        vectors.append(vec)
    return {"chunks": chunks, "vectors": vectors, "doc_freq": dict(doc_freq), "N": N}

def main():
    all_chunks = []
    docs_dir = Path("docs")
    if not docs_dir.exists():
        print("Создайте папку docs/ и поместите туда PDF-файлы")
        print("Имена файлов: strategy_3309.pdf, monograph_khudyakova.pdf, normative_minselhoz.pdf")
        sys.exit(1)

    for doc_id, info in PDF_FILES.items():
        p = info["path"]
        if not Path(p).exists():
            print(f"Файл не найден: {p}, пропускаю...")
            continue
        print(f"Индексирую {doc_id}...")
        text = extract_text(p)
        chunks = chunk_text(text, doc_id, info["source"])
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} чанков")

    if not all_chunks:
        print("Нет PDF-файлов для индексирования!")
        sys.exit(1)

    index = build_tfidf(all_chunks)
    with open("knowledge_index.pkl", "wb") as f:
        pickle.dump(index, f)
    print(f"\nИндекс готов: {len(all_chunks)} чанков, сохранён в knowledge_index.pkl")

if __name__ == "__main__":
    main()
