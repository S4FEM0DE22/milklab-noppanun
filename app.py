"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space

นักศึกษาต้องเติม TODO 5 จุด ใน Session 3 Lab 2.2
"""

import os
from pathlib import Path

import faiss
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_index():
    """โหลด menu_kb.md, แบ่งเป็น chunks และสร้าง embeddings."""

    kb_path = Path("menu_kb.md")

    if not kb_path.exists():
        raise FileNotFoundError("ไม่พบไฟล์ menu_kb.md")

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
        return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้ของ MilkLab°"

    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
    คุณคือผู้ช่วยตอบคำถามเกี่ยวกับร้าน MilkLab°

    กฎการตอบ:
    - ใช้ข้อมูลจาก CONTEXT เท่านั้น
    - ห้ามแต่งข้อมูลหรือเดา
    - หากผู้ใช้ถามหลายคำถาม ให้ตอบทุกข้อเป็นรายการ (Bullet List)
    - หากคำถามข้อใดไม่มีข้อมูลใน CONTEXT ให้ตอบว่า
    "ไม่พบข้อมูลนี้ในฐานความรู้ของ MilkLab°"
    - ตอบเป็นภาษาไทย
    - ตอบสั้น กระชับ และเข้าใจง่าย

    CONTEXT:
    {context}

    คำถาม:
    {query}

    คำตอบ:
    """.strip()

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
    )

    answer = (response.text or "").strip()

    if not answer:
        return "ไม่สามารถสร้างคำตอบได้ในขณะนี้"

    return answer


def main():
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
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
