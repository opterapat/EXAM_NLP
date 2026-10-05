"""เชฟน้อย — Cooking chatbot with RAG.

Pipeline: Load docs → Chunk → Embed (Gemini embedding) → FAISS index
→ Retrieve top-k for each question → Augment prompt → Gemini generates answer with sources.
"""

import json
import re
from pathlib import Path

import numpy as np
import requests
import streamlit as st

try:
    import faiss
except ImportError:  # numpy fallback gives identical results for a flat inner-product index
    faiss = None

SYSTEM_PROMPT = """คุณคือ "เชฟน้อย" ผู้ช่วยทำอาหารที่เป็นมิตร ตอบเป็นภาษาไทย (ถ้าผู้ใช้พิมพ์ภาษาอื่น ให้ตอบภาษานั้น)
หน้าที่: แนะนำเมนู สูตรอาหาร วิธีทำทีละขั้น เทคนิคการปรุง การเลือก/ทดแทนวัตถุดิบ การจัดเก็บอาหาร และความปลอดภัยในครัว
รูปแบบคำตอบเมื่อให้สูตร:
- ชื่อเมนู + เวลาที่ใช้ + จำนวนที่ได้ (เสิร์ฟกี่คน)
- **วัตถุดิบ** เป็นรายการพร้อมปริมาณ
- **วิธีทำ** เป็นขั้นตอนมีหมายเลข
- **เคล็ดลับ** สั้นๆ 1-3 ข้อ
ถ้าผู้ใช้บอกวัตถุดิบที่มี ให้เสนอเมนูที่ทำได้จากของเหล่านั้น
ถ้าคำถามไม่เกี่ยวกับอาหารหรือการทำอาหาร ให้ปฏิเสธอย่างสุภาพและชวนกลับมาคุยเรื่องอาหาร
ระวังเรื่องอาหารแพ้ง่ายและความปลอดภัยของอาหาร (เช่น อุณหภูมิที่สุก) เมื่อเกี่ยวข้อง"""

RAG_TEMPLATE = """ข้อมูลอ้างอิงจากฐานข้อมูลสูตรอาหาร:
{context}

กติกา:
- ใช้ข้อมูลอ้างอิงด้านบนเป็นหลัก และอ้างอิงแหล่งด้วยเลข [1], [2] ท้ายประโยคที่ใช้ข้อมูลนั้น
- ถ้าข้อมูลอ้างอิงไม่มีคำตอบ ให้บอกว่า "ไม่พบในฐานข้อมูล" แล้วจึงตอบจากความรู้ทั่วไปได้
- ห้ามแต่งปริมาณหรือขั้นตอนขัดกับข้อมูลอ้างอิง

คำถาม: {question}"""

SUGGESTIONS = [
    "สอนทำกะเพราหมูสับไข่ดาว",
    "มีไข่ มะเขือเทศ หัวหอม ทำอะไรได้บ้าง?",
    "เมนูคลีนลดน้ำหนัก ทำง่ายใน 15 นาที",
    "ใช้อะไรแทนน้ำปลาได้บ้าง?",
]

WELCOME = "สวัสดีครับ! ผม **เชฟน้อย** 👨‍🍳 ถามเรื่องสูตรอาหาร วิธีทำ หรือบอกวัตถุดิบที่มีในตู้เย็นมาได้เลยครับ"

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


# ---------- RAG: chunk → embed → index → retrieve ----------

def chunk_document(source, text, max_chars=1200, overlap=150):
    """Split by '## ' headings (one recipe per chunk); fall back to sliding windows for long/plain text."""
    sections = [s.strip() for s in re.split(r"\n(?=## )", text) if s.strip()]
    chunks = []
    for sec in sections:
        title = sec.splitlines()[0].lstrip("# ").strip()[:80]
        if sec.startswith("# ") and "\n" not in sec:
            continue  # bare document title
        if len(sec) <= max_chars:
            chunks.append({"source": source, "title": title, "text": sec})
            continue
        start = 0
        while start < len(sec):
            chunks.append({"source": source, "title": title, "text": sec[start:start + max_chars]})
            start += max_chars - overlap
    return chunks


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
    vecs = embed([f"{c['title']}\n{c['text']}" for c in chunks], "RETRIEVAL_DOCUMENT")
    if faiss is not None:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
    else:
        index = vecs
    return chunks, index


def retrieve(question, chunks, index, k):
    q = embed([question], "RETRIEVAL_QUERY")
    k = min(k, len(chunks))
    if faiss is not None:
        scores, ids = index.search(q, k)
        scores, ids = scores[0], ids[0]
    else:
        sims = index @ q[0]
        ids = np.argsort(-sims)[:k]
        scores = sims[ids]
    return [{**chunks[i], "score": float(s)} for i, s in zip(ids, scores)]


def build_prompt(question, hits):
    context = "\n\n".join(f"[{n}] ({h['source']} — {h['title']})\n{h['text']}" for n, h in enumerate(hits, 1))
    return RAG_TEMPLATE.format(context=context, question=question)


# ---------- LLM ----------

def stream_chat(contents):
    """Yield text chunks from Gemini's SSE stream."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.4},
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


def show_sources(hits):
    with st.expander(f"📚 แหล่งอ้างอิง (RAG) — {len(hits)} chunks"):
        for n, h in enumerate(hits, 1):
            st.markdown(f"**[{n}] {h['title']}** · `{h['source']}` · similarity = `{h['score']:.3f}`")
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))


# ---------- UI ----------

st.set_page_config(page_title="เชฟน้อย — Cooking Chatbot", page_icon="🍳")

if not API_KEY:
    st.error("ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Secrets (Settings → Secrets)")
    st.stop()

if "messages" not in st.session_state:
    # user: {"role", "text"}; model: {"role", "text", "sources"}
    st.session_state.messages = []

with st.sidebar:
    st.title("🍳 เชฟน้อย")
    st.caption(f"Cooking Chatbot + RAG · LLM: `{MODEL}` · Embedding: `{EMBED_MODEL}`")
    use_rag = st.toggle("ใช้ RAG (ค้นจากฐานข้อมูล)", value=True)
    top_k = st.slider("จำนวน chunk ที่ค้น (top-k)", 1, 8, 4)
    uploads = st.file_uploader("เพิ่มเอกสารสูตรอาหาร (.txt / .md)", type=["txt", "md"], accept_multiple_files=True)
    if st.button("🔄 เริ่มแชทใหม่", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

docs = [(p.name, p.read_text(encoding="utf-8")) for p in sorted(DATA_DIR.glob("*.md"))]
docs += [(f.name, f.getvalue().decode("utf-8", errors="ignore")) for f in uploads or []]

try:
    chunks, index = build_index(tuple(docs))
except Exception as e:
    st.error(f"สร้าง index ไม่สำเร็จ: {e}")
    st.stop()

with st.sidebar:
    st.divider()
    st.metric("เอกสาร / chunks ในฐานข้อมูล", f"{len(docs)} / {len(chunks)}")
    st.caption("Vector store: " + ("FAISS IndexFlatIP (cosine)" if faiss else "NumPy cosine similarity"))
    with st.expander("ดูหัวข้อในฐานข้อมูล"):
        for c in chunks:
            st.markdown(f"- {c['title']} · `{c['source']}`")

st.title("🍳 เชฟน้อย")
st.caption("แชทบอทผู้ช่วยทำอาหาร + RAG — ตอบจากฐานข้อมูลสูตรอาหารพร้อมอ้างอิงแหล่งที่มา")

with st.chat_message("assistant", avatar="👨‍🍳"):
    st.markdown(WELCOME)

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role, avatar="🧑" if role == "user" else "👨‍🍳"):
        st.markdown(m["text"])
        if m.get("sources"):
            show_sources(m["sources"])

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
            hits = retrieve(prompt, chunks, index, top_k) if use_rag else []
            # Only the latest turn carries retrieved context; earlier turns stay as plain text
            contents = [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in st.session_state.messages]
            question = build_prompt(prompt, hits) if hits else prompt
            contents.append({"role": "user", "parts": [{"text": question}]})
            reply = st.write_stream(stream_chat(contents))
            st.session_state.messages.append({"role": "user", "text": prompt})
            st.session_state.messages.append({"role": "model", "text": reply, "sources": hits})
        except Exception as e:
            st.error(f"⚠️ {e}")
            st.stop()
    st.rerun()
