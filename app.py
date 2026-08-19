"""Groove & Gear RAG Chatbot (S4 Pivot).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space
"""

import os
from pathlib import Path

import faiss
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_index():
    """โหลด music_gear_kb.md, แบ่งเป็น chunks และสร้าง embeddings."""

    # เปลี่ยนชื่อไฟล์ KB เป็นของร้านดนตรี
    kb_path = Path("music_gear_kb.md")

    if not kb_path.exists():
        raise FileNotFoundError("ไม่พบไฟล์ music_gear_kb.md")

    text = kb_path.read_text(encoding="utf-8")

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    return model, index, chunks


def retrieve_top_k(
    query: str,
    model,
    index,
    chunks: list[str],
    k: int = 3,
) -> list[str]:
    """Encode query แล้วค้นหา top-k chunks จาก FAISS."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for chunk_index in indices[0]:
        if chunk_index != -1:
            results.append(chunks[chunk_index])

    return results


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """ส่งคำถามและ context ที่ retrieve ได้ไปให้ Gemini สร้างคำตอบ."""

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("ไม่พบ GOOGLE_API_KEY ใน environment")

    if not context_chunks:
        return "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องในระบบของ Groove & Gear"

    context = "\n\n---\n\n".join(context_chunks)

    # ปรับ Prompt ให้เป็นร้านเครื่องดนตรี
    prompt = f"""
    คุณคือผู้ช่วยตอบคำถามของร้านเครื่องดนตรี "Groove & Gear"

    กฎการตอบ:
    - ใช้ข้อมูลจาก CONTEXT เท่านั้น
    - ห้ามแต่งข้อมูลหรือเดา
    - หากผู้ใช้ถามหลายคำถาม ให้ตอบทุกข้อเป็นรายการ (Bullet List)
    - หากคำถามข้อใดไม่มีข้อมูลใน CONTEXT ให้ตอบว่า
    "ขออภัยครับ ไม่พบข้อมูลนี้ในระบบของ Groove & Gear"
    - ตอบเป็นภาษาไทย
    - ตอบด้วยโทนเป็นมิตร เข้าใจหัวอกคนเล่นดนตรี
    - ตอบสั้น กระชับ และเข้าใจง่าย

    CONTEXT:
    {context}

    คำถาม:
    {query}

    คำตอบ:
    """.strip()

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = (response.text or "").strip()

    if not answer:
        return "ไม่สามารถสร้างคำตอบได้ในขณะนี้ครับ"

    return answer


def main():
    st.set_page_config(
        page_title="Groove & Gear",
        page_icon="🎸",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #f4f1e8;
            --muted: #a8aaa0;
            --panel: #171a19;
            --panel-soft: #202422;
            --line: rgba(244, 241, 232, 0.12);
            --lime: #d7f36a;
            --orange: #ff8a5b;
        }

        .stApp {
            background: radial-gradient(circle at 75% 0%, #30372a 0, #111312 36%, #0c0e0d 80%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: #101211;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] .block-container { padding: 2rem 1.25rem; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
        h1 { font-size: clamp(2.2rem, 5vw, 4.6rem) !important; line-height: 0.98 !important; }
        .brand-kicker {
            color: var(--lime); font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.16em; text-transform: uppercase; margin-bottom: 0.9rem;
        }
        .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 42rem; margin-bottom: 2rem; }
        .hero-rule { border-top: 1px solid var(--line); margin: 1.5rem 0 2rem; }
        .side-label { color: var(--muted); font-size: 0.72rem; letter-spacing: 0.13em; text-transform: uppercase; }
        .side-title { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.4rem; font-weight: 700; margin: 0.3rem 0 2rem; }
        .status-dot { color: var(--lime); font-size: 0.8rem; }
        .tip-box { border: 1px solid var(--line); border-radius: 8px; padding: 0.9rem; color: var(--muted); font-size: 0.86rem; line-height: 1.5; }
        [data-testid="stChatMessage"] {
            background: rgba(23, 26, 25, 0.74);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin: 0.7rem 0;
        }
        [data-testid="stChatInput"] {
            border-color: rgba(215, 243, 106, 0.45);
        }
        .stButton > button {
            border: 1px solid var(--line); border-radius: 6px; background: var(--panel);
            color: var(--ink); text-align: left; transition: border-color 150ms ease, transform 150ms ease;
        }
        .stButton > button:hover { border-color: var(--lime); color: var(--lime); transform: translateY(-1px); }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: rgba(23, 26, 25, 0.5); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="side-label">Groove & Gear</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="side-title">Your tone desk</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="status-dot">● Knowledge base online</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown(
            '<div class="side-label">ลองเริ่มจากคำถามนี้</div>', unsafe_allow_html=True)
        starter_prompts = [
            "ช่วยเลือกกีตาร์สำหรับมือใหม่",
            "แอมป์ซ้อมที่เหมาะกับห้องนอน",
            "เปรียบเทียบอุปกรณ์ในร้าน",
        ]
        for starter in starter_prompts:
            if st.button(starter, use_container_width=True):
                st.session_state.pending_prompt = starter
        st.divider()
        if st.button("ล้างบทสนทนา", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown(
            '<div class="tip-box">ถามเรื่องสเปค การใช้งาน หรือช่วยจับคู่ gear ให้เข้ากับสไตล์ของคุณ</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="brand-kicker">Music gear intelligence / 24.7</div>',
                unsafe_allow_html=True)
    st.title("Find your sound.")
    st.markdown(
        '<div class="hero-copy">ผู้ช่วยเลือกเครื่องดนตรีที่คุยกับคุณรู้เรื่อง ค้นจากคลังข้อมูลของ Groove & Gear แล้วสรุปให้แบบสั้น กระชับ และเป็นภาษาไทย</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)

    try:
        model, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "ยินดีต้อนรับสู่ **Groove & Gear**!\n\nถามสเปคเครื่องดนตรี ปรึกษาเรื่องแอมป์ หรือให้ช่วยจับคู่ gear กับสไตล์ของคุณได้เลยครับ 🎸"}
        ]

    for msg in st.session_state.messages:
        avatar = "GG" if msg["role"] == "assistant" else "YOU"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    prompt = st.chat_input("มีอะไรให้ Groove & Gear ช่วยแนะนำไหมครับ?")
    if not prompt and st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="YOU"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="GG"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(prompt, model, index, chunks, k=3)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
