"""เชฟน้อย — Thai cooking assistant chatbot with RAG.

Pipeline: Load docs (data/*.md + uploads) → Clean → Chunk → Embed (Gemini sentence embedding)
→ FAISS index → Retrieve top-k per question → Grounded prompt → Gemini LLM answers with citations.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import requests
import streamlit as st

try:
    import faiss
except ImportError:  # numpy fallback gives identical results for a flat inner-product index
    faiss = None

NOT_FOUND = "ไม่พบข้อมูล"

SYSTEM_PROMPT = f"""คุณคือ "เชฟน้อย" ผู้ช่วยตอบคำถามเรื่องการทำอาหารไทย โดยตอบจากเอกสารอ้างอิง (Context) ที่ได้รับเท่านั้น
กติกา:
1. ใช้เฉพาะข้อมูลใน Context ห้ามใช้ความรู้ภายนอก ห้ามเดา และห้ามแต่งปริมาณหรือขั้นตอนเพิ่ม
2. ทุกประโยคที่ใช้ข้อมูลจาก Context ให้ใส่เลขอ้างอิง เช่น [1] หรือ [1][3] ท้ายประโยค
3. ถ้า Context ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบว่า "{NOT_FOUND}" ตามด้วยประโยคสั้นๆ ว่าฐานข้อมูลยังไม่มีเรื่องนี้ และอาจแนะนำหัวข้อที่มีในฐานข้อมูล
4. ถ้าคำถามไม่เกี่ยวกับอาหาร ให้ตอบว่า "{NOT_FOUND}" และชวนกลับมาถามเรื่องอาหาร
5. ตอบภาษาเดียวกับคำถาม (ไทย/อังกฤษ)
6. เมื่อให้สูตรอาหาร ใช้รูปแบบ: ชื่อเมนู + เวลา + จำนวนเสิร์ฟ → **วัตถุดิบ** (รายการ) → **วิธีทำ** (ขั้นตอนมีหมายเลข) → **เคล็ดลับ**
7. ถ้าผู้ใช้บอกวัตถุดิบที่มี ให้เลือกเมนูใน Context ที่ใช้วัตถุดิบเหล่านั้นได้"""

RAG_TEMPLATE = """Context (เอกสารที่ค้นได้จากฐานข้อมูล):
{context}

คำถาม: {question}

ตอบโดยอิงจาก Context ข้างบนเท่านั้น พร้อมเลขอ้างอิง [n] ถ้าไม่มีข้อมูลให้ตอบว่า "ไม่พบข้อมูล\""""

SUGGESTIONS = [
    "สอนทำกะเพราหมูสับไข่ดาว",
    "มีไข่ มะเขือเทศ หัวหอม ทำอะไรได้บ้าง?",
    "ไก่ต้องสุกที่อุณหภูมิเท่าไหร่?",
    "ใช้อะไรแทนน้ำปลาได้บ้าง?",
]

WELCOME = (
    "สวัสดีครับ! ผม **เชฟน้อย** 👨‍🍳 ผู้ช่วยทำอาหารไทยที่ตอบจากคลังสูตรอาหาร "
    "ถามสูตร วิธีทำ ของทดแทน หรือความปลอดภัยอาหารได้เลย — ทุกคำตอบจะมีแหล่งอ้างอิงให้ดูครับ"
)

DATA_DIR = Path(__file__).parent / "data"
BASE = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768


def secret(name, default):
    # st.secrets raises when no secrets file exists, so fall back to the default
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


API_KEY = secret("GEMINI_API_KEY", "")
MODEL = secret("GEMINI_MODEL", "gemini-flash-latest")
HEADERS = {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}


# ---------- 1. Document loading & cleaning ----------

def clean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[​-‍﻿]", "", text)  # zero-width chars common in copied Thai text
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[  ]{2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(uploads):
    docs = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(DATA_DIR.glob("*.md"))]
    docs += [(p.name, p.read_text(encoding="utf-8")) for p in sorted(DATA_DIR.glob("*.txt"))]
    docs += [(f.name, f.getvalue().decode("utf-8", errors="ignore")) for f in uploads or []]
    return [(name, clean_text(text)) for name, text in docs if text.strip()]


# ---------- 2. Chunking ----------

def chunk_document(source, text, max_chars=1000, overlap=150):
    """Split on markdown headings so each recipe/topic stays one chunk;
    long sections fall back to a sliding window with overlap so no step is cut off without context."""
    sections = [s.strip() for s in re.split(r"\n(?=#{1,3} )", text) if s.strip()]
    chunks = []
    for sec in sections:
        title = sec.splitlines()[0].lstrip("# ").strip()[:80]
        if sec.startswith("#") and "\n" not in sec:
            continue  # heading with no body
        if len(sec) <= max_chars:
            chunks.append({"source": source, "title": title, "text": sec})
            continue
        start = 0
        while start < len(sec):
            piece = sec[start:start + max_chars]
            if start:
                piece = f"{title} (ต่อ)\n{piece}"
            chunks.append({"source": source, "title": title, "text": piece})
            start += max_chars - overlap
    return chunks


# ---------- 3. Embedding & vector search ----------

def embed(texts, task_type):
    vectors = []
    for i in range(0, len(texts), 100):  # API limit: 100 per batch
        batch = texts[i:i + 100]
        res = requests.post(
            f"{BASE}/{EMBED_MODEL}:batchEmbedContents",
            headers=HEADERS,
            json={
                "requests": [
                    {
                        "model": f"models/{EMBED_MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                        "outputDimensionality": EMBED_DIM,
                    }
                    for t in batch
                ]
            },
            timeout=120,
        )
        if res.status_code != 200:
            raise RuntimeError(f"Embedding error {res.status_code}: {res.text[:300]}")
        vectors.extend(e["values"] for e in res.json()["embeddings"])
    vecs = np.array(vectors, dtype="float32")
    # Normalize so inner product == cosine similarity (needed for dims < 3072)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@st.cache_resource(show_spinner="กำลังสร้าง vector index...")
def build_index(docs):
    """docs: tuple of (source, text). Cached, so embeddings are computed once per document set."""
    chunks = [c for source, text in docs for c in chunk_document(source, text)]
    vecs = embed([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    if faiss is not None:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
    else:
        index = vecs
    return chunks, index


def retrieve(question, chunks, index, k, min_score):
    q = embed([question], "RETRIEVAL_QUERY")
    k = min(k, len(chunks))
    if faiss is not None:
        scores, ids = index.search(q, k)
        scores, ids = scores[0], ids[0]
    else:
        sims = index @ q[0]
        ids = np.argsort(-sims)[:k]
        scores = sims[ids]
    return [{**chunks[i], "score": float(s)} for i, s in zip(ids, scores) if s >= min_score]


# ---------- 4. Prompt engineering ----------

def build_prompt(question, hits):
    if not hits:
        context = "(ไม่พบเอกสารที่เกี่ยวข้อง)"
    else:
        context = "\n\n".join(f"[{n}] ({h['source']} — {h['title']})\n{h['text']}" for n, h in enumerate(hits, 1))
    return RAG_TEMPLATE.format(context=context, question=question)


# ---------- 5. LLM ----------

def stream_chat(contents):
    """Yield text chunks from Gemini's SSE stream."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
    }
    with requests.post(
        f"{BASE}/{MODEL}:streamGenerateContent?alt=sse",
        headers=HEADERS,
        json=body,
        stream=True,
        timeout=120,
    ) as res:
        if res.status_code != 200:
            raise RuntimeError(f"API error {res.status_code}: {res.text[:300]}")
        for line in res.iter_lines(decode_unicode=False):
            if not line or not line.startswith(b"data:"):
                continue
            data = json.loads(line[5:].decode("utf-8"))
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if text:
                yield text


# ---------- 6. Chatbot interface ----------

def show_sources(hits):
    if not hits:
        st.caption("📚 แหล่งอ้างอิง: ไม่พบเอกสารที่เกี่ยวข้องเกินเกณฑ์ความคล้าย")
        return
    with st.expander(f"📚 แหล่งอ้างอิงที่ใช้ตอบ ({len(hits)} เอกสาร)"):
        for n, h in enumerate(hits, 1):
            st.markdown(f"**[{n}] {h['title']}** · `{h['source']}` · similarity = `{h['score']:.3f}`")
            st.caption(h["text"][:500] + ("…" if len(h["text"]) > 500 else ""))


st.set_page_config(page_title="เชฟน้อย — Cooking RAG Chatbot", page_icon="🍳")

if not API_KEY:
    st.error('ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Secrets (Manage app → Settings → Secrets)')
    st.stop()

if "messages" not in st.session_state:
    # user: {"role", "text"}; model: {"role", "text", "sources"}
    st.session_state.messages = []

with st.sidebar:
    st.title("🍳 เชฟน้อย")
    st.caption("ผู้ช่วยทำอาหารไทย · RAG Chatbot")
    top_k = st.slider("จำนวนเอกสารที่ค้น (top-k)", 1, 8, 4)
    min_score = st.slider("เกณฑ์ความคล้ายขั้นต่ำ", 0.0, 1.0, 0.50, 0.05,
                          help="chunk ที่คล้ายคำถามน้อยกว่าค่านี้จะไม่ถูกส่งให้ LLM")
    uploads = st.file_uploader("เพิ่มเอกสาร (.txt / .md)", type=["txt", "md"], accept_multiple_files=True)
    if st.button("🔄 เริ่มแชทใหม่", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

docs = load_documents(uploads)
try:
    chunks, index = build_index(tuple(docs))
except Exception as e:
    st.error(f"สร้าง index ไม่สำเร็จ: {e}")
    st.stop()

with st.sidebar:
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("เอกสาร", len(docs))
    c2.metric("Chunks", len(chunks))
    st.caption(
        f"LLM: `{MODEL}`  \nEmbedding: `{EMBED_MODEL}` ({EMBED_DIM}-d)  \n"
        "Vector DB: " + ("FAISS IndexFlatIP (cosine)" if faiss else "NumPy cosine")
    )
    with st.expander("📂 เอกสารในคลังความรู้"):
        for name, _ in docs:
            st.markdown(f"- `{name}`")

st.title("🍳 เชฟน้อย")
st.caption("แชทบอทผู้ช่วยทำอาหารไทย — ตอบจากคลังสูตรอาหารด้วย RAG พร้อมแสดงแหล่งอ้างอิงทุกคำตอบ")

with st.chat_message("assistant", avatar="👨‍🍳"):
    st.markdown(WELCOME)

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role, avatar="🧑" if role == "user" else "👨‍🍳"):
        st.markdown(m["text"])
        if role == "assistant":
            show_sources(m.get("sources", []))

prompt = st.chat_input("ถามเรื่องอาหาร เช่น 'ต้มยำกุ้งน้ำข้นทำยังไง'")

if not st.session_state.messages and not prompt:
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, use_container_width=True):
            prompt = s

if prompt:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="👨‍🍳"):
        try:
            hits = retrieve(prompt, chunks, index, top_k, min_score)
            # Earlier turns stay as plain text for conversation memory; only the latest turn carries Context
            contents = [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in st.session_state.messages]
            contents.append({"role": "user", "parts": [{"text": build_prompt(prompt, hits)}]})
            reply = st.write_stream(stream_chat(contents))
            st.session_state.messages.append({"role": "user", "text": prompt})
            st.session_state.messages.append({"role": "model", "text": reply, "sources": hits})
        except Exception as e:
            st.error(f"⚠️ {e}")
            st.stop()
    st.rerun()
