import json

import requests
import streamlit as st

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

SUGGESTIONS = [
    "สอนทำกะเพราหมูสับไข่ดาว",
    "มีไข่ มะเขือเทศ หัวหอม ทำอะไรได้บ้าง?",
    "เมนูคลีนลดน้ำหนัก ทำง่ายใน 15 นาที",
    "ใช้อะไรแทนน้ำปลาได้บ้าง?",
]

WELCOME = "สวัสดีครับ! ผม **เชฟน้อย** 👨‍🍳 ถามเรื่องสูตรอาหาร วิธีทำ หรือบอกวัตถุดิบที่มีในตู้เย็นมาได้เลยครับ"

def secret(name, default):
    # st.secrets raises when no secrets file exists, so fall back to the default
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


API_KEY = secret("GEMINI_API_KEY", "")
MODEL = secret("GEMINI_MODEL", "gemini-flash-latest")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent?alt=sse"


def stream_chat(history):
    """Yield text chunks from Gemini's SSE stream."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in history],
        "generationConfig": {"temperature": 0.7},
    }
    with requests.post(
        URL,
        headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
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


st.set_page_config(page_title="เชฟน้อย — Cooking Chatbot", page_icon="🍳")

with st.sidebar:
    st.title("🍳 เชฟน้อย")
    st.caption("แชทบอทผู้ช่วยทำอาหาร (Gemini)")
    if st.button("🔄 เริ่มใหม่", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🍳 เชฟน้อย")
st.caption("แชทบอทผู้ช่วยทำอาหาร — ถามสูตร วิธีทำ หรือบอกวัตถุดิบที่มี")

if not API_KEY:
    st.error("ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Secrets (Settings → Secrets)")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"model", "text": str}]

with st.chat_message("assistant", avatar="👨‍🍳"):
    st.markdown(WELCOME)

for m in st.session_state.messages:
    role = "user" if m["role"] == "user" else "assistant"
    with st.chat_message(role, avatar="🧑" if role == "user" else "👨‍🍳"):
        st.markdown(m["text"])

prompt = st.chat_input("ถามเรื่องอาหาร เช่น 'ต้มยำกุ้งน้ำข้นทำยังไง'")

if not st.session_state.messages and not prompt:
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, use_container_width=True):
            prompt = s

if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="👨‍🍳"):
        try:
            reply = st.write_stream(stream_chat(st.session_state.messages))
            st.session_state.messages.append({"role": "model", "text": reply})
        except Exception as e:
            st.session_state.messages.pop()
            st.error(f"⚠️ {e}")
    st.rerun()
