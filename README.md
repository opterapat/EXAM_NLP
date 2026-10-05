# 🍳 เชฟน้อย — RAG Chatbot ผู้ช่วยทำอาหารไทย

Web Application แชตบอตที่ตอบคำถามเรื่องการทำอาหารไทยจากคลังเอกสารสูตรอาหาร ด้วยเทคนิค **RAG (Retrieval-Augmented Generation)**
พัฒนาด้วย Streamlit + Gemini API และ Deploy บน Streamlit Community Cloud

## แนวคิดของ Domain

มือใหม่หัดทำอาหารมักค้นสูตรจากอินเทอร์เน็ตแล้วเจอข้อมูลขัดกัน หรือได้คำตอบจาก AI ที่แต่งปริมาณขึ้นเอง
"เชฟน้อย" จึงตอบ **จากคลังสูตรที่คัดไว้เท่านั้น** พร้อมแหล่งอ้างอิงทุกคำตอบ ครอบคลุม:

- สูตรอาหารไทย 12 เมนู (วัตถุดิบ ปริมาณ ขั้นตอน เคล็ดลับ)
- การใช้วัตถุดิบทดแทน
- ความปลอดภัยอาหาร อุณหภูมิที่สุก การเก็บรักษา และสารก่อภูมิแพ้
- เทคนิคพื้นฐานในครัว
- ถ้าถามเรื่องที่ไม่มีในเอกสาร → ตอบว่า **"ไม่พบข้อมูล"**

## โครงสร้างไฟล์

```
├── app.py                 # Streamlit app หลัก (RAG pipeline + Chat UI)
├── streamlit_app.py       # entry point สำหรับ deployment เดิม (เรียก app.py)
├── requirements.txt
├── test_questions.csv     # คำถามทดสอบ 15 ข้อ (ไม่มีคำตอบในเอกสาร 3 ข้อ)
├── data/                  # คลังความรู้ 15 ไฟล์ (1 เมนู/หัวข้อ ต่อไฟล์)
└── .streamlit/secrets.toml  # API key (อยู่ใน .gitignore — ไม่ขึ้น GitHub)
```

## RAG Pipeline

| ขั้นตอน | รายละเอียด |
|---|---|
| 1. Document Loading | โหลด `data/*.md`, `*.txt` + ไฟล์ที่ผู้ใช้อัปโหลดจาก sidebar |
| 2. Cleaning | Unicode NFC, ลบ zero-width chars (พบบ่อยในข้อความไทยที่ copy มา), ยุบช่องว่าง/บรรทัดว่างซ้ำ |
| 3. Chunking | แบ่งตามหัวข้อ Markdown → 1 เมนู = 1 chunk (สูตรไม่ถูกตัดกลางขั้นตอน); ถ้ายาวเกิน 1,000 ตัวอักษรใช้ sliding window overlap 150 และใส่ชื่อหัวข้อนำทุก chunk |
| 4. Embedding | `gemini-embedding-001` (sentence embedding รองรับไทย/อังกฤษ) 768 มิติ, ใช้ taskType `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`, normalize เวกเตอร์ |
| 5. Vector Search | **FAISS** `IndexFlatIP` (inner product บนเวกเตอร์ normalize = cosine similarity), top-k + เกณฑ์ความคล้ายขั้นต่ำ; index สร้างครั้งเดียวด้วย `st.cache_resource` |
| 6. Prompt Engineering | System prompt บังคับตอบจาก Context เท่านั้น, ใส่เลขอ้างอิง `[n]`, ตอบ "ไม่พบข้อมูล" เมื่อไม่มีคำตอบ, กำหนดรูปแบบสูตรอาหาร; temperature 0.2 |
| 7. LLM | Gemini (`gemini-flash-latest`) ผ่าน REST API แบบ streaming |
| 8. Chat Interface | คุยต่อเนื่อง (ส่งประวัติแชทให้ LLM), แสดงเอกสารอ้างอิง + similarity ใต้ทุกคำตอบ, ปุ่มคำถามตัวอย่าง, ปรับ top-k / เกณฑ์ / อัปโหลดเอกสารได้ |

> ใช้ Embedding ผ่าน API แทนการโหลด sentence-transformers ในเครื่อง เพื่อลดหน่วยความจำบน Streamlit Community Cloud

## วิธีใช้งาน

**บนเว็บ:** เปิด URL ของแอป → พิมพ์คำถามหรือกดปุ่มคำถามตัวอย่าง → กด "📚 แหล่งอ้างอิงที่ใช้ตอบ" เพื่อดูเอกสารที่ค้นได้

**รันในเครื่อง:**
```bash
pip install -r requirements.txt
mkdir .streamlit
echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml
streamlit run app.py
```

**Deploy:** Streamlit Community Cloud → เลือก repo นี้ → Advanced settings → Secrets ใส่ `GEMINI_API_KEY = "..."`

## แหล่งที่มาของเอกสาร

เอกสารใน `data/` เรียบเรียงโดยใช้ AI ช่วยสร้างเนื้อหา (อนุญาตตามโจทย์) โดยอ้างอิงสูตรอาหารไทยมาตรฐานที่แพร่หลาย
ส่วนอุณหภูมิความปลอดภัยอาหารและระยะเวลาเก็บรักษาอ้างอิงแนวทางของ USDA Food Safety and Inspection Service (FSIS)
และหลัก "Danger Zone" 5–60°C

## ตัวอย่าง Prompt ที่ใช้สั่ง AI ในการพัฒนา

1. *"เราจะทำแชทบอทเกี่ยวกับการทำอาหาร"* → สร้างแชทบอทพื้นฐาน + system prompt บทบาท "เชฟน้อย"
2. *"พาเอาขึ้น Streamlit"* → แปลงเป็น Streamlit app, ตั้งค่า git, ย้าย API key ไป Secrets
3. *"ต้องทำระบบ RAG data ด้วย"* → สร้างคลังสูตรอาหาร, Chunk → Embed → FAISS → Retrieve → Prompt พร้อมอ้างอิง
4. *"ดูใน checklist (NLP-SubTest2.ipynb)"* → ปรับให้ครบข้อกำหนด: `app.py`, แยก `data/` เป็น 15 ไฟล์, ทำความสะอาดข้อความ, prompt ตอบเฉพาะ Context + "ไม่พบข้อมูล", `test_questions.csv`, README

**Prompt ที่ใช้ในแอป (System Prompt):**
```
คุณคือ "เชฟน้อย" ผู้ช่วยตอบคำถามเรื่องการทำอาหารไทย โดยตอบจากเอกสารอ้างอิง (Context) ที่ได้รับเท่านั้น
1. ใช้เฉพาะข้อมูลใน Context ห้ามใช้ความรู้ภายนอก ห้ามเดา และห้ามแต่งปริมาณหรือขั้นตอนเพิ่ม
2. ทุกประโยคที่ใช้ข้อมูลจาก Context ให้ใส่เลขอ้างอิง เช่น [1] หรือ [1][3]
3. ถ้า Context ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูล" ...
```

## คำถามทดสอบ

ดู `test_questions.csv` — 15 ข้อ: ตอบได้จากเอกสาร 12 ข้อ, ไม่มีคำตอบในเอกสาร 3 ข้อ (มัสมั่น, เค้กช็อกโกแลต, ราคาทอง) ซึ่งแชตบอตต้องตอบ "ไม่พบข้อมูล"
