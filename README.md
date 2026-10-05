# เชฟน้อย — Cooking Chatbot + RAG

Streamlit + Gemini cooking assistant with Retrieval-Augmented Generation.

**Pipeline:** `data/*.md` (+ uploaded .txt/.md) → chunk by recipe (`## ` heading, sliding window for long text)
→ embed with `gemini-embedding-001` (768-d, normalized) → FAISS `IndexFlatIP` (cosine)
→ retrieve top-k per question → inject as numbered context → Gemini answers with `[n]` citations.

Sidebar: toggle RAG on/off (compare answers), top-k slider, upload extra documents, view indexed chunks.

Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.
Put `GEMINI_API_KEY` in `.streamlit/secrets.toml` (local) or in the app Secrets on Streamlit Cloud.
