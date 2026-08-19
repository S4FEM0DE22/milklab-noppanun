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
        model="gemini-flash-latest",
        contents=prompt,
    )

    answer = (response.text or "").strip()

    if not answer:
        return "ไม่สามารถสร้างคำตอบได้ในขณะนี้ครับ"

    return answer


def main():
    # ปรับ UI และ Branding ให้เป็นร้านดนตรี
    st.set_page_config(page_title="Groove & Gear", page_icon="🎸")
    st.title("🎸 Groove & Gear Assistant")
    st.caption("แชตบอตผู้ช่วยร้านเครื่องดนตรี ถามสเปค ปรึกษาเรื่องแอมป์ ได้เลย!")

    try:
        model, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # กำหนด Welcome Message
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "ยินดีต้อนรับสู่ **Groove & Gear**! 🤘\nถามสเปคเครื่องดนตรี ปรึกษาเรื่องแอมป์ หรือให้เราป้ายยาอุปกรณ์เด็ดๆ พิมพ์มาได้เลยครับ!"}
        ]

    # ใส่ Avatar ให้ข้อความแชต
    for msg in st.session_state.messages:
        avatar = "🎧" if msg["role"] == "assistant" else "🧑‍🎤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    # เปลี่ยนข้อความในช่องพิมพ์
    if prompt := st.chat_input("มีอะไรให้ Groove & Gear ช่วยแนะนำไหมครับ?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎤"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🎧"):
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
