"""Gordon Ramsay (AI) — Thai cooking assistant chatbot with RAG.

Pipeline: Load docs (data/*.md + uploads) → Clean → Chunk → Embed (Gemini sentence embedding)
→ FAISS index → Retrieve top-k per question → Grounded prompt → Gemini LLM answers with citations.
"""

import base64
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

SYSTEM_PROMPT = f"""คุณคือ "Gordon Ramsay (AI)" ตัวละคร AI ผู้ช่วยตอบคำถามเรื่องการทำอาหารไทย ที่ได้แรงบันดาลใจจากสไตล์ของ Gordon Ramsay (ไม่ใช่ตัวจริง)
บุคลิก: พูดตรง กระชับ มีพลัง ใส่ใจรายละเอียด ใช้คำติดปากอย่าง "Beautiful!", "Yes, chef!", "Stunning!" ได้บ้าง แต่สุภาพและให้กำลังใจผู้ใช้เสมอ ห้ามดูถูกหรือด่าผู้ใช้
ตอบจากเอกสารอ้างอิง (Context) ที่ได้รับเท่านั้น
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

SUGGESTIONS = {
    "🍲 สูตรอาหาร": ["สอนทำกะเพราหมูสับไข่ดาว", "ต้มยำกุ้งน้ำข้นทำยังไง?"],
    "🥚 มีอะไรในตู้เย็น": ["มีไข่ มะเขือเทศ หัวหอม ทำอะไรได้บ้าง?", "มีอกไก่ อยากกินคลีนๆ ทำอะไรดี?"],
    "🔁 ของทดแทน": ["ใช้อะไรแทนน้ำปลาได้บ้าง?", "ไม่มีกะทิ ใช้อะไรแทนได้?"],
    "🛡️ ความปลอดภัย": ["ไก่ต้องสุกที่อุณหภูมิเท่าไหร่?", "อาหารสุกเก็บในตู้เย็นได้กี่วัน?"],
}

DATA_DIR = Path(__file__).parent / "data"
CHEF_IMG = Path(__file__).parent / "assets" / "chef.jpg"
BOT_AVATAR = str(CHEF_IMG) if CHEF_IMG.exists() else "👨‍🍳"
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

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
.stApp { font-family: 'Sarabun', sans-serif; }
.block-container { max-width: 860px; padding-top: 2rem; }
.hero { display: flex; gap: 16px; align-items: center; padding: 18px 20px; border-radius: 16px;
        background: linear-gradient(135deg, #FDE6DA, #FFF4EC); border: 1px solid #F3D3C1; margin-bottom: 14px; }
.hero .emoji { font-size: 46px; line-height: 1; }
.hero img { width: 84px; height: 84px; border-radius: 50%; object-fit: cover; border: 3px solid #fff; box-shadow: 0 2px 8px rgba(0,0,0,.15); flex-shrink: 0; }
.hero h2 { margin: 0; padding: 0; font-size: 1.45rem; color: #C2410C; }
.hero p { margin: 4px 0 0; color: #5B4A40; font-size: 0.98rem; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 6px 0 18px; }
.step { background: #fff; border: 1px solid #F0E2D6; border-radius: 12px; padding: 10px 12px; font-size: 0.9rem; color: #5B4A40; }
.step b { color: #C2410C; display: block; margin-bottom: 2px; }
.cat { font-weight: 600; color: #5B4A40; margin: 4px 0 2px; font-size: 0.95rem; }
.src { font-size: 0.85rem; color: #8A7D72; margin-top: 6px; }
.src span { display: inline-block; background: #FBEFE6; border-radius: 999px; padding: 1px 10px; margin: 2px 4px 2px 0; color: #7C4A2D; }
@media (max-width: 640px) { .steps { grid-template-columns: 1fr; } .hero .emoji { font-size: 36px; } .hero img { width: 64px; height: 64px; } }
</style>
"""


def doc_meta(text):
    title = text.splitlines()[0].lstrip("# ").strip()
    m = re.search(r"หมวด:\s*([^|\n]+)", text)
    return title, (m.group(1).strip() if m else "อื่นๆ")


def friendly_error(e):
    msg = str(e)
    if "429" in msg:
        return "⏳ ตอนนี้มีการใช้งาน API ถี่เกินไป รอประมาณ 1 นาทีแล้วลองถามใหม่อีกครั้งนะครับ"
    if any(code in msg for code in ("400", "401", "403")) and "key" in msg.lower():
        return "🔑 API key ไม่ถูกต้องหรือหมดอายุ — ผู้ดูแลต้องตั้งค่า GEMINI_API_KEY ใหม่ใน Secrets"
    if isinstance(e, requests.exceptions.RequestException):
        return "📡 เชื่อมต่อเซิร์ฟเวอร์ไม่สำเร็จ ลองกดถามใหม่อีกครั้งนะครับ"
    return f"⚠️ เกิดข้อผิดพลาด: {msg[:200]}"


def show_sources(hits):
    if not hits:
        st.markdown('<div class="src">📚 ไม่มีเอกสารในคลังที่เกี่ยวข้องกับคำถามนี้</div>', unsafe_allow_html=True)
        return
    tags = "".join(f"<span>[{n}] {h['title']}</span>" for n, h in enumerate(hits, 1))
    st.markdown(f'<div class="src">📚 อ้างอิงจาก: {tags}</div>', unsafe_allow_html=True)
    with st.expander("ดูข้อความในเอกสารอ้างอิง"):
        for n, h in enumerate(hits, 1):
            st.markdown(f"**[{n}] {h['title']}** · `{h['source']}` · ความคล้าย `{h['score']:.2f}`")
            st.caption(h["text"][:500] + ("…" if len(h["text"]) > 500 else ""))


st.set_page_config(page_title="Gordon Ramsay (AI) — ผู้ช่วยทำอาหารไทย", page_icon="🍳", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

if not API_KEY:
    st.error("🔑 ยังไม่ได้ตั้งค่า GEMINI_API_KEY — ไปที่ Manage app → Settings → Secrets")
    st.stop()

if "messages" not in st.session_state:
    # user: {"role", "text"}; model: {"role", "text", "sources"}
    st.session_state.messages = []

ask = None  # question picked from a button instead of typed

with st.sidebar:
    if CHEF_IMG.exists():
        st.image(str(CHEF_IMG), width=110)
    st.markdown("## 🍳 Gordon Ramsay (AI)")
    st.caption("ผู้ช่วยทำอาหารไทย · ตอบจากคลังสูตรอาหารด้วย RAG")
    st.caption("ℹ️ ตัวละคร AI ที่ได้แรงบันดาลใจจาก Gordon Ramsay — ไม่ใช่ตัวจริงและไม่มีส่วนเกี่ยวข้อง")
    if st.button("➕ เริ่มแชทใหม่", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

with st.sidebar:
    with st.expander("⚙️ ตั้งค่าขั้นสูง"):
        top_k = st.slider("จำนวนเอกสารที่ค้น (top-k)", 1, 8, 4)
        min_score = st.slider("เกณฑ์ความคล้ายขั้นต่ำ", 0.0, 1.0, 0.50, 0.05,
                              help="เอกสารที่คล้ายคำถามน้อยกว่าค่านี้จะไม่ถูกส่งให้ LLM")
        uploads = st.file_uploader("เพิ่มเอกสารเข้าคลัง (.txt / .md)", type=["txt", "md"],
                                   accept_multiple_files=True)

docs = load_documents(uploads)
try:
    chunks, index = build_index(tuple(docs))
except Exception as e:
    st.error(friendly_error(e))
    st.stop()

with st.sidebar:
    st.markdown("#### 📖 เมนูในคลัง — กดเพื่อดูสูตร")
    groups = {}
    for name, text in docs:
        title, cat = doc_meta(text)
        groups.setdefault(cat, []).append(title)
    for cat, titles in groups.items():
        if cat == "ความรู้ทั่วไป":
            continue
        with st.expander(f"{cat} ({len(titles)})"):
            for t in titles:
                if st.button(t, key=f"menu-{t}", use_container_width=True):
                    ask = f"สอนทำ{t}"
    if "ความรู้ทั่วไป" in groups:
        with st.expander(f"💡 ความรู้ทั่วไป ({len(groups['ความรู้ทั่วไป'])})"):
            for t in groups["ความรู้ทั่วไป"]:
                if st.button(t, key=f"menu-{t}", use_container_width=True):
                    ask = f"สรุปเรื่อง{t}"
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("เอกสาร", len(docs))
    c2.metric("Chunks", len(chunks))
    st.caption(
        f"LLM `{MODEL}` · Embedding `{EMBED_MODEL}` · "
        + ("FAISS" if faiss else "NumPy") + " vector search"
    )

if not st.session_state.messages:
    if CHEF_IMG.exists():
        b64 = base64.b64encode(CHEF_IMG.read_bytes()).decode()
        hero_img = f'<img src="data:image/jpeg;base64,{b64}" alt="Gordon Ramsay (AI)">'
    else:
        hero_img = '<div class="emoji">👨‍🍳</div>'
    st.markdown(
        f"""
<div class="hero">{hero_img}<div>
<h2>Yes, chef! I'm Gordon Ramsay (AI) 🔥</h2>
<p>ถามเรื่องสูตรอาหารไทย วิธีทำ ของทดแทน หรือความปลอดภัยอาหาร — ผมตอบจากคลังสูตรที่คัดไว้ พร้อมบอกแหล่งอ้างอิงทุกครั้ง</p>
</div></div>
<div class="steps">
<div class="step"><b>1. ถาม</b>พิมพ์คำถาม หรือกดคำถามตัวอย่างด้านล่าง</div>
<div class="step"><b>2. ค้นหา</b>ระบบค้นสูตรที่เกี่ยวข้องจากคลังเอกสาร</div>
<div class="step"><b>3. ตอบ + อ้างอิง</b>ดูแหล่งที่มาได้ใต้คำตอบ [1] [2]</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("**ลองถามแบบนี้ดูครับ 👇**")
    cols = st.columns(2)
    for i, (cat, qs) in enumerate(SUGGESTIONS.items()):
        with cols[i % 2]:
            st.markdown(f'<div class="cat">{cat}</div>', unsafe_allow_html=True)
            for q in qs:
                if st.button(q, key=f"sug-{q}", use_container_width=True):
                    ask = q

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role, avatar="🧑" if role == "user" else BOT_AVATAR):
        st.markdown(m["text"])
        if role == "assistant":
            show_sources(m.get("sources", []))

prompt = st.chat_input("พิมพ์คำถามเรื่องอาหาร เช่น “ต้มยำกุ้งใส่มะนาวตอนไหน”") or ask

if prompt:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            with st.spinner("🔎 กำลังค้นหาสูตรในคลัง..."):
                hits = retrieve(prompt, chunks, index, top_k, min_score)
            # Earlier turns stay as plain text for conversation memory; only the latest turn carries Context
            contents = [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in st.session_state.messages]
            contents.append({"role": "user", "parts": [{"text": build_prompt(prompt, hits)}]})
            reply = st.write_stream(stream_chat(contents))
            st.session_state.messages.append({"role": "user", "text": prompt})
            st.session_state.messages.append({"role": "model", "text": reply, "sources": hits})
        except Exception as e:
            st.error(friendly_error(e))
            st.stop()
    st.rerun()
