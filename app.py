"""Groove & Gear RAG Chatbot (S4 Pivot).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space
"""

import base64
import html
import json
import os
from pathlib import Path

import faiss
import streamlit as st
import streamlit.components.v1 as components
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
    min_score: float = 0.30,
) -> list[str]:
    """ค้นหา chunks ที่เกี่ยวข้องและตัดผลลัพธ์ที่คะแนนต่ำออก."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index != -1 and float(score) >= min_score:
            results.append(chunks[chunk_index])

    return results


@st.cache_data(ttl=3600, max_entries=100, show_spinner=False)
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
        - ข้อมูลร้าน สินค้า ราคา และทำเลใน CONTEXT เป็นข้อมูลสมมติสำหรับการสาธิตเท่านั้น
            ห้ามนำเสนอว่าเป็นข้อมูลร้านจริงหรือข้อมูลล่าสุด
    - ใช้ข้อมูลจาก CONTEXT เท่านั้น ห้ามใช้ความรู้ทั่วไปจากภายนอก
    - ห้ามแต่งข้อมูล ห้ามเดา และห้ามเติมรายละเอียดที่ไม่ได้เขียนไว้ใน CONTEXT
    - ข้อมูลสินค้า ราคา รหัสสินค้า สเปค โปรโมชั่น สต็อก การรับประกัน และการจัดส่ง
      ต้องตอบตาม CONTEXT เท่านั้น หากไม่มีข้อมูลตรงกันให้บอกว่าไม่พบข้อมูล
        - หากลูกค้าซื้อหลายรายการ ให้แสดงรายการสินค้า ราคาต่อชิ้น จำนวน และคำนวณยอดรวม
            จากราคาที่อยู่ใน CONTEXT เท่านั้น ตรวจเลขให้ถูกต้อง และแยกค่าจัดส่งออกจากราคาสินค้า
        - ใช้ส่วนลดหรือโปรโมชั่นได้เฉพาะที่ระบุไว้ใน CONTEXT ห้ามสร้างส่วนลดหรือยอดสุทธิขึ้นเอง
            หากไม่มีโปรโมชั่นตรงกับรายการ ให้แจ้งว่าไม่พบโปรโมชั่นที่ระบุไว้
    - หาก CONTEXT กล่าวถึงสินค้าคนละรุ่นกับคำถาม ห้ามนำมาแทนกันหรือสรุปว่าเหมือนกัน
    - ถ้าคำถามต้องการข้อมูลล่าสุด เช่น สต็อกหรือโปรโมชั่น ให้แนะนำให้ติดต่อแอดมิน
        - หากคำถามไม่เกี่ยวกับสินค้า บริการ หรือข้อมูลของร้าน ให้แจ้งอย่างสุภาพว่า
            "ขออภัยครับ ผมช่วยตอบได้เฉพาะเรื่องสินค้า บริการ และข้อมูลของ Groove & Gear ครับ"
        - หากคำถามเกี่ยวกับร้านแต่ไม่มีข้อมูลเพียงพอ ให้แยกแจ้งว่าไม่พบข้อมูล
            และไม่ควรตอบเหมือนเป็นคำถามนอกหัวข้อ
    - หากผู้ใช้ถามหลายคำถาม ให้ตอบทุกข้อเป็นรายการ (Bullet List)
    - หากไม่มีข้อมูลเพียงพอ ให้ตอบอย่างสุภาพว่า
      "ขออภัยครับ ตอนนี้ยังไม่พบข้อมูลนี้ในฐานความรู้ของ Groove & Gear
      หากแจ้งรุ่นหรือรายละเอียดเพิ่ม แอดมินจะช่วยตรวจสอบให้ได้ครับ"
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

        @keyframes rise-in {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes bar-pulse {
            0%, 100% { transform: scaleY(0.45); }
            50% { transform: scaleY(1); }
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
        .brand-name {
            color: var(--ink); font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.8rem, 7vw, 6.8rem); font-weight: 700;
            letter-spacing: 0; line-height: 0.9; margin: 0;
            animation: rise-in 700ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        .brand-name span { color: var(--lime); }
        .brand-tagline {
            color: var(--orange); font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.25rem, 2.5vw, 2.1rem); font-weight: 600;
            line-height: 1; margin: 1rem 0 0.7rem;
            animation: rise-in 700ms 100ms cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        .soundwave { align-items: center; display: flex; gap: 3px; height: 18px; margin-bottom: 1.4rem; }
        .soundwave i { animation: bar-pulse 900ms ease-in-out infinite; background: var(--lime); border-radius: 2px; display: block; height: 18px; transform-origin: center; width: 3px; }
        .soundwave i:nth-child(2) { animation-delay: 120ms; height: 12px; }
        .soundwave i:nth-child(3) { animation-delay: 240ms; height: 18px; }
        .soundwave i:nth-child(4) { animation-delay: 360ms; height: 9px; }
        .soundwave i:nth-child(5) { animation-delay: 480ms; height: 15px; }
        .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 42rem; margin-bottom: 2rem; }
        .hero-rule { border-top: 1px solid var(--line); margin: 1.5rem 0 2rem; }
        .category-rail { align-items: center; display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.25rem 0 1.5rem; animation: rise-in 650ms 250ms ease both; }
        .category-title { color: var(--orange); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; margin-right: 0.25rem; }
        .category-chip { border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 0.72rem; padding: 0.32rem 0.6rem; transition: border-color 180ms ease, color 180ms ease, transform 180ms ease; }
        .category-chip:hover { border-color: var(--lime); color: var(--lime); transform: translateY(-2px); }
        .service-strip { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); display: grid; grid-template-columns: repeat(3, 1fr); margin: 1.75rem 0 2rem; animation: rise-in 650ms 350ms ease both; }
        .service-item { border-right: 1px solid var(--line); padding: 0.85rem 1rem; transition: background 180ms ease; }
        .service-item:last-child { border-right: 0; }
        .service-item:hover { background: rgba(215, 243, 106, 0.05); }
        .service-item strong { color: var(--lime); display: block; font-family: 'Space Grotesk'; font-size: 1.1rem; }
        .service-item span { color: var(--muted); font-size: 0.72rem; }
        .signal-grid { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 0.5rem 0 2rem; }
        .signal { border: 1px solid var(--line); border-radius: 6px; padding: 0.65rem 0.85rem; background: rgba(32, 36, 34, 0.65); }
        .signal strong { display: block; color: var(--lime); font-family: 'Space Grotesk'; font-size: 1rem; }
        .signal span { color: var(--muted); font-size: 0.72rem; }
        .side-label { color: var(--muted); font-size: 0.72rem; letter-spacing: 0.13em; text-transform: uppercase; }
        .side-title { color: var(--ink); font-family: 'Space Grotesk'; font-size: 1.4rem; font-weight: 700; margin: 0.3rem 0 2rem; }
        .status-dot { color: var(--lime); font-size: 0.8rem; }
        .tip-box { border: 1px solid var(--line); border-radius: 8px; padding: 0.9rem; color: var(--muted); font-size: 0.86rem; line-height: 1.5; }
        .location-box { border-left: 2px solid var(--orange); padding-left: 0.75rem; color: var(--ink); font-size: 0.82rem; line-height: 1.45; }
        .location-box span { display: block; color: var(--muted); font-size: 0.7rem; letter-spacing: 0.08em; margin-bottom: 0.25rem; text-transform: uppercase; }
        .radio-card { background: linear-gradient(135deg, rgba(215, 243, 106, 0.1), rgba(255, 138, 91, 0.06)); border: 1px solid rgba(215, 243, 106, 0.28); border-radius: 8px; padding: 0.9rem; }
        .radio-head { align-items: center; display: flex; gap: 0.6rem; margin-bottom: 0.65rem; }
        .radio-mark { align-items: center; background: var(--lime); border-radius: 50%; color: #111312; display: flex; font-size: 0.7rem; height: 1.55rem; justify-content: center; width: 1.55rem; }
        .radio-name { color: var(--ink); font-family: 'Space Grotesk'; font-size: 0.95rem; font-weight: 700; }
        .radio-status { color: var(--orange); font-size: 0.62rem; letter-spacing: 0.1em; margin-left: auto; text-transform: uppercase; }
        .radio-bars { align-items: center; display: flex; gap: 3px; height: 1rem; margin-bottom: 0.55rem; }
        .radio-bars i { animation: bar-pulse 820ms ease-in-out infinite; background: var(--lime); border-radius: 2px; display: block; height: 0.8rem; width: 3px; }
        .radio-bars i:nth-child(2) { animation-delay: 120ms; height: 0.55rem; }
        .radio-bars i:nth-child(3) { animation-delay: 240ms; height: 1rem; }
        .radio-bars i:nth-child(4) { animation-delay: 360ms; height: 0.4rem; }
        .radio-bars i:nth-child(5) { animation-delay: 480ms; height: 0.7rem; }
        .radio-source { color: var(--muted); font-size: 0.68rem; letter-spacing: 0.08em; margin-bottom: 0.35rem; text-transform: uppercase; }
        .radio-player { background: rgba(12, 14, 13, 0.72); border: 1px solid var(--line); border-radius: 6px; display: block; height: 2.3rem; margin-top: 0.55rem; width: 100%; }
        .highlight-list { color: var(--muted); font-size: 0.82rem; line-height: 1.65; }
        .highlight-list strong { color: var(--ink); }
        .section-label { color: var(--orange); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; margin: 0.5rem 0 0.6rem; }
        [data-testid="stChatMessage"] {
            background: rgba(23, 26, 25, 0.74);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin: 0.7rem 0;
            animation: rise-in 500ms ease both;
        }
        [data-testid="stChatInput"] {
            border-color: rgba(215, 243, 106, 0.45);
            border-radius: 8px;
            box-shadow: 0 0 0 1px rgba(215, 243, 106, 0.05);
        }
        .stButton > button {
            border: 1px solid var(--line); border-radius: 6px; background: var(--panel);
            color: var(--ink); text-align: left; transition: border-color 150ms ease, transform 150ms ease;
        }
        .stButton > button:hover { border-color: var(--lime); color: var(--lime); transform: translateY(-1px); }
        .stButton > button:focus-visible { border-color: var(--lime); box-shadow: 0 0 0 2px rgba(215, 243, 106, 0.2); }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 8px; background: rgba(23, 26, 25, 0.5); }
        @media (max-width: 640px) {
            .service-strip { grid-template-columns: 1fr; }
            .service-item { border-bottom: 1px solid var(--line); border-right: 0; }
            .service-item:last-child { border-bottom: 0; }
            .hero-copy { font-size: 0.95rem; }
            [data-testid="stChatMessage"] { padding: 0.9rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
        }
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
        st.markdown(
            '<div class="location-box"><span>ที่อยู่ร้าน</span>เลขที่ 88/9 ซอยเสียงดนตรี 9 ถนนมิตรภาพ ตำบลในเมือง<br>อำเภอเมืองขอนแก่น จังหวัดขอนแก่น 40000<br><small>ด้านหลัง มทร.อีสาน วิทยาเขตขอนแก่น</small></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            '<div class="side-label">ลองเริ่มจากคำถามนี้</div>', unsafe_allow_html=True)
        starter_prompts = [
            "ช่วยเลือกกีตาร์สำหรับมือใหม่",
            "แอมป์ซ้อมที่เหมาะกับห้องนอน",
            "เปรียบเทียบอุปกรณ์ในร้าน",
            "อยากอัดเสียงร้อง ต้องใช้อะไรบ้าง",
            "งบ 10,000 บาท ซื้อชุดเริ่มต้นอะไรดี",
        ]
        for starter in starter_prompts:
            if st.button(starter, use_container_width=True):
                st.session_state.pending_prompt = starter
        st.markdown(
            '<div class="side-label">ถามเกี่ยวกับร้าน</div>', unsafe_allow_html=True)
        store_prompts = [
            "ร้านอยู่ที่ไหน",
            "ร้านเปิดกี่โมง",
            "มีบริการอะไรบ้าง",
            "จัดส่งสินค้าอย่างไร",
            "สินค้ารับประกันไหม",
        ]
        for store_prompt in store_prompts:
            if st.button(store_prompt, key=f"store_{store_prompt}", use_container_width=True):
                st.session_state.pending_prompt = store_prompt
        st.divider()
        if st.button("ล้างบทสนทนา", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown(
            '<div class="tip-box">ถามเรื่องสเปค การใช้งาน หรือช่วยจับคู่ gear ให้เข้ากับสไตล์ของคุณ</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="highlight-list"><strong>ทำไมต้อง Groove &amp; Gear</strong><br>• สินค้าหลากหลายสำหรับผู้เริ่มต้น<br>• ช่วยเลือกตามงบและสถานที่ใช้งาน<br>• มีบริการเซ็ตอัปกีตาร์และเบส</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown('<div class="side-label">Radio</div>',
                    unsafe_allow_html=True)
        radio_stream_url = os.getenv("RADIO_STREAM_URL", "").strip()
        local_radio_path = Path("assets/groove_radio_original.mp3")
        with st.container():
            source_label = "Live stream" if radio_stream_url else "Original Session 01"
            if radio_stream_url:
                audio_source = radio_stream_url
            elif local_radio_path.exists():
                audio_bytes = base64.b64encode(
                    local_radio_path.read_bytes()).decode("ascii")
                audio_source = f"data:audio/mpeg;base64,{audio_bytes}"
            else:
                st.markdown(
                    '<div class="tip-box">เติม RADIO_STREAM_URL เพื่อเปิดสถานีเพลงในร้าน</div>',
                    unsafe_allow_html=True,
                )
                audio_source = ""

            if audio_source:
                components.html(
                    f"""
                    <style>
                        * {{ box-sizing: border-box; }}
                        body {{ margin: 0; background: transparent; font-family: 'DM Sans', 'Space Grotesk', sans-serif; }}
                        .player {{
                            background: linear-gradient(135deg, rgba(23,26,25,0.95) 0%, rgba(12,14,13,0.98) 100%);
                            border: 1px solid rgba(215,243,106,0.35);
                            border-radius: 16px;
                            box-shadow: 0 20px 48px rgba(0,0,0,0.5), inset 0 1px 0 rgba(215,243,106,0.1);
                            color: #f4f1e8;
                            overflow: hidden;
                            padding: 11px;
                            position: relative;
                            transition: box-shadow 300ms ease;
                        }}
                        .player::before {{
                            content: '';
                            position: absolute;
                            top: 0; left: 0; right: 0;
                            height: 1px;
                            background: linear-gradient(90deg, transparent, rgba(215,243,106,0.3), transparent);
                            pointer-events: none;
                        }}
                        .top {{ align-items: center; display: flex; gap: 10px; margin-bottom: 2px; min-width: 0; }}
                        .cover {{
                            align-items: center;
                            background: conic-gradient(from 35deg, #d7f36a 0deg, #ff8a5b 50deg, #202522 120deg, #d7f36a 360deg);
                            border: 3px solid #0d0f0e;
                            border-radius: 14px;
                            box-shadow: 0 10px 28px rgba(215,243,106,0.15), 0 4px 12px rgba(0,0,0,0.4);
                            display: flex;
                            flex: 0 0 56px;
                            height: 56px;
                            justify-content: center;
                            position: relative;
                            width: 56px;
                            transition: transform 300ms ease, box-shadow 300ms ease;
                        }}
                        .player:hover .cover {{ transform: scale(1.05) translateY(-2px); box-shadow: 0 12px 32px rgba(215,243,106,0.2), 0 6px 16px rgba(0,0,0,0.5); }}
                        .cover::before {{
                            background: repeating-radial-gradient(circle, transparent 0 4px, rgba(13,15,14,0.3) 5px 6px);
                            border-radius: 50%;
                            content: '';
                            inset: 5px;
                            position: absolute;
                        }}
                        .cover::after {{
                            background: #171a19;
                            border: 5px solid #101211;
                            border-radius: 50%;
                            content: '';
                            height: 14px;
                            position: relative;
                            width: 14px;
                            box-shadow: 0 0 6px rgba(215,243,106,0.3);
                        }}
                        .info {{ flex: 1; min-width: 0; }}
                        .eyebrow {{
                            color: #d7f36a;
                            font-size: 10px;
                            font-weight: 700;
                            letter-spacing: .16em;
                            text-transform: uppercase;
                            opacity: 0.9;
                        }}
                        .title {{
                            font-size: 15px;
                            font-weight: 700;
                            letter-spacing: 0;
                            margin-top: 5px;
                            background: linear-gradient(135deg, #f4f1e8 0%, #d7f36a 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                        }}
                        .subtitle {{
                            color: #a8aaa0;
                            font-size: 10px;
                            margin-top: 5px;
                            letter-spacing: 0.5px;
                        }}
                        .live {{
                            background: linear-gradient(135deg, rgba(255,138,91,0.2), rgba(215,243,106,0.1));
                            border: 1px solid rgba(255,138,91,0.45);
                            border-radius: 99px;
                            color: #ff8a5b;
                            font-size: 9px;
                            letter-spacing: .12em;
                            margin-left: auto;
                            padding: 4px 7px;
                            font-weight: 600;
                            animation: liveGlow 2s ease-in-out infinite;
                        }}
                        @keyframes liveGlow {{ 0%, 100% {{ opacity: 0.7; }} 50% {{ opacity: 1; }} }}
                        .bars {{
                            align-items: center;
                            display: flex;
                            gap: 3px;
                            height: 20px;
                            margin: 8px 0 9px;
                            justify-content: center;
                        }}
                        .bars i {{
                            animation: soundPulse 0.75s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                            background: linear-gradient(180deg, #d7f36a 0%, #ff8a5b 100%);
                            border-radius: 2px;
                            height: 14px;
                            width: 3px;
                            opacity: 0.85;
                            box-shadow: 0 0 4px rgba(215,243,106,0.4);
                        }}
                        .bars i:nth-child(1) {{ animation-delay: 0s; height: 10px; }}
                        .bars i:nth-child(2) {{ animation-delay: 0.1s; height: 8px; }}
                        .bars i:nth-child(3) {{ animation-delay: 0.2s; height: 20px; }}
                        .bars i:nth-child(4) {{ animation-delay: 0.3s; height: 7px; }}
                        .bars i:nth-child(5) {{ animation-delay: 0.4s; height: 16px; }}
                        .bars i:nth-child(6) {{ animation-delay: 0.5s; height: 9px; }}
                        .bars i:nth-child(7) {{ animation-delay: 0.6s; height: 18px; }}
                        .bars i:nth-child(8) {{ animation-delay: 0.7s; height: 11px; }}
                        @keyframes soundPulse {{ 0%, 100% {{ transform: scaleY(0.5); opacity: 0.6; }} 50% {{ transform: scaleY(1); opacity: 1; }} }}
                        .timeline {{
                            align-items: center;
                            display: flex;
                            gap: 10px;
                            margin-bottom: 6px;
                        }}
                        .timeline span {{
                            color: #a8aaa0;
                            font-size: 10px;
                            min-width: 32px;
                            font-weight: 600;
                            font-family: monospace;
                        }}
                        .timeline span:last-child {{ text-align: right; }}
                        .controls {{
                            align-items: center;
                            display: flex;
                            gap: 9px;
                            justify-content: center;
                        }}
                        button {{
                            align-items: center;
                            background: transparent;
                            border: 1px solid rgba(215,243,106,0.2);
                            border-radius: 8px;
                            color: #a8aaa0;
                            cursor: pointer;
                            display: flex;
                            font-size: 13px;
                            height: 32px;
                            justify-content: center;
                            transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);
                            width: 32px;
                        }}
                        button:hover {{
                            border-color: rgba(215,243,106,0.5);
                            color: #d7f36a;
                            transform: translateY(-2px);
                            background: rgba(215,243,106,0.05);
                            box-shadow: 0 4px 12px rgba(215,243,106,0.1);
                        }}
                        button:active {{ transform: translateY(0); }}
                        #toggle {{
                            background: linear-gradient(135deg, #d7f36a 0%, #f4f1e8 100%);
                            border: none;
                            border-radius: 50%;
                            color: #101211;
                            font-size: 15px;
                            height: 44px;
                            width: 44px;
                            font-weight: 700;
                            box-shadow: 0 8px 20px rgba(215,243,106,0.3), 0 2px 8px rgba(0,0,0,0.3);
                            transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);
                        }}
                        #toggle:hover {{
                            background: linear-gradient(135deg, #f4f1e8 0%, #d7f36a 100%);
                            transform: scale(1.1) translateY(-2px);
                            box-shadow: 0 12px 28px rgba(215,243,106,0.4), 0 4px 12px rgba(0,0,0,0.4);
                        }}
                        input[type=range] {{
                            accent-color: #d7f36a;
                            cursor: pointer;
                            height: 6px;
                            border-radius: 3px;
                            background: linear-gradient(90deg, rgba(215,243,106,0.2), rgba(215,243,106,0.1));
                            border: none;
                            transition: background 200ms ease;
                        }}
                        input[type=range]:hover {{ background: linear-gradient(90deg, rgba(215,243,106,0.3), rgba(215,243,106,0.15)); }}
                        input[type=range]::-webkit-slider-thumb {{
                            appearance: none;
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: linear-gradient(135deg, #d7f36a, #ff8a5b);
                            cursor: pointer;
                            box-shadow: 0 2px 6px rgba(215,243,106,0.4);
                            transition: all 150ms ease;
                        }}
                        input[type=range]::-webkit-slider-thumb:hover {{ transform: scale(1.2); }}
                        input[type=range]::-moz-range-thumb {{
                            width: 14px;
                            height: 14px;
                            border-radius: 50%;
                            background: linear-gradient(135deg, #d7f36a, #ff8a5b);
                            cursor: pointer;
                            border: none;
                            box-shadow: 0 2px 6px rgba(215,243,106,0.4);
                            transition: all 150ms ease;
                        }}
                        input[type=range]::-moz-range-thumb:hover {{ transform: scale(1.2); }}
                        #progress {{ flex: 1; }}
                        #volume {{ width: 68px; }}
                        .volume-wrap {{
                            align-items: center;
                            display: flex;
                            gap: 6px;
                            margin-left: auto;
                            color: #a8aaa0;
                            font-size: 9px;
                            font-weight: 600;
                            letter-spacing: 0.1em;
                        }}
                        @media (max-width: 420px) {{ .player {{ padding: 10px; }} .cover {{ flex-basis: 48px; height: 48px; width: 48px; }} .cover::after {{ height: 10px; width: 10px; }} .live {{ font-size: 8px; padding: 3px 5px; }} .subtitle {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }} .controls {{ flex-wrap: wrap; }} .volume-wrap {{ flex: 1 0 100%; margin-left: 0; }} #volume {{ flex: 1; width: auto; }} }}
                    </style>
                    <div class="player">
                        <div class="top">
                            <div class="cover"></div>
                            <div class="info">
                                <div class="eyebrow">Groove &amp; Gear Radio</div>
                                <div class="title">{html.escape(source_label)}</div>
                                <div class="subtitle">Original session • Music for your tone hunt</div>
                            </div>
                            <div class="live">ON AIR</div>
                        </div>
                        <div class="bars"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
                        <audio id="audio" loop preload="auto" src="{audio_source}"></audio>
                        <div class="timeline"><span id="current">00:00</span><input id="progress" type="range" min="0" max="100" value="0" aria-label="ความคืบหน้าเพลง"><span id="duration">00:00</span></div>
                        <div class="controls">
                            <button id="back" aria-label="ย้อนกลับ 10 วินาที">↶</button>
                            <button id="toggle" aria-label="เล่นหรือหยุดเพลง">▶</button>
                            <button id="forward" aria-label="ข้ามไปข้างหน้า 10 วินาที">↷</button>
                            <div class="volume-wrap"><span>VOL</span><input id="volume" type="range" min="0" max="1" step="0.05" value="0.7" aria-label="ระดับเสียง"></div>
                        </div>
                    </div>
                    <script>
                        const audio = document.getElementById('audio');
                        const toggle = document.getElementById('toggle');
                        const progress = document.getElementById('progress');
                        const volume = document.getElementById('volume');
                        const current = document.getElementById('current');
                        const duration = document.getElementById('duration');
                        const formatTime = (value) => {{
                            if (!Number.isFinite(value)) return '00:00';
                            const minutes = Math.floor(value / 60).toString().padStart(2, '0');
                            const seconds = Math.floor(value % 60).toString().padStart(2, '0');
                            return `${{minutes}}:${{seconds}}`;
                        }};
                        const syncButton = () => {{ toggle.textContent = audio.paused ? '▶' : 'Ⅱ'; }};
                        const attemptAutoplay = () => {{
                            const playPromise = audio.play();
                            if (playPromise !== undefined) {{
                                playPromise.catch((err) => {{
                                    console.log('Autoplay blocked (browser policy):', err.message);
                                    syncButton();
                                }});
                            }} else {{
                                syncButton();
                            }}
                        }};
                        toggle.addEventListener('click', () => {{
                            if (audio.paused) {{
                                const playPromise = audio.play();
                                if (playPromise !== undefined) {{
                                    playPromise.catch((err) => console.error('Play failed:', err));
                                }}
                            }} else {{
                                audio.pause();
                            }}
                        }});
                        document.getElementById('back').addEventListener('click', () => {{ audio.currentTime = Math.max(0, audio.currentTime - 10); }});
                        document.getElementById('forward').addEventListener('click', () => {{ audio.currentTime = Math.min(audio.duration || audio.currentTime, audio.currentTime + 10); }});
                        progress.addEventListener('input', () => {{ if (audio.duration) audio.currentTime = progress.value / 100 * audio.duration; }});
                        volume.addEventListener('input', () => {{ audio.volume = parseFloat(volume.value); }});
                        audio.addEventListener('loadedmetadata', () => {{ duration.textContent = formatTime(audio.duration); }});
                        audio.addEventListener('timeupdate', () => {{ progress.value = audio.duration ? audio.currentTime / audio.duration * 100 : 0; current.textContent = formatTime(audio.currentTime); }});
                        audio.addEventListener('play', syncButton);
                        audio.addEventListener('pause', syncButton);
                        audio.addEventListener('ended', syncButton);
                        audio.addEventListener('error', (e) => {{ console.error('Audio error:', e.target.error); }});
                        audio.volume = 0.7;
                        volume.value = 0.7;
                        setTimeout(attemptAutoplay, 100);
                    </script>
                    """,
                    height=220,
                    scrolling=False,
                )

    st.markdown('<div class="brand-kicker">Music gear intelligence / 24.7</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<h1 class="brand-name">Groove <span>&amp;</span> Gear</h1>'
        '<div class="brand-tagline">Find your sound.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="soundwave" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">ผู้ช่วยเลือกเครื่องดนตรีที่คุยกับคุณรู้เรื่อง ค้นจากคลังข้อมูลของร้าน แล้วสรุปให้แบบสั้น กระชับ และเป็นภาษาไทย</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="location-box"><span>Visit Groove &amp; Gear</span>เลขที่ 88/9 ซอยเสียงดนตรี 9 ถนนมิตรภาพ ตำบลในเมือง อำเภอเมืองขอนแก่น จังหวัดขอนแก่น 40000<br>ด้านหลัง มทร.อีสาน วิทยาเขตขอนแก่น</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="signal-grid"><div class="signal"><strong>19</strong><span>หมวดความรู้</span></div><div class="signal"><strong>29</strong><span>รายการสินค้า</span></div><div class="signal"><strong>RAG</strong><span>อ้างอิงจากคลังร้าน</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="category-rail"><span class="category-title">IN THE SHOP</span><span class="category-chip">GUITAR</span><span class="category-chip">BASS</span><span class="category-chip">RECORDING</span><span class="category-chip">KEYBOARD</span><span class="category-chip">DRUMS</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="service-strip"><div class="service-item"><strong>80 บาท</strong><span>ค่าส่งปกติต่อคำสั่งซื้อ</span></div><div class="service-item"><strong>500 บาท</strong><span>เซ็ตอัปกีตาร์และเบส</span></div><div class="service-item"><strong>2,000 บาท</strong><span>ยอดรับส่งด่วนฟรี</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hero-rule"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Start with a direction</div>',
                unsafe_allow_html=True)
    prompt_columns = st.columns(4)
    featured_prompts = [
        ("🎸", "กีตาร์", "กีตาร์โปร่งหรือไฟฟ้าดี"),
        ("🎙️", "อัดเสียง", "อยากเริ่มอัดเสียงที่บ้าน"),
        ("🎹", "คีย์บอร์ด", "เลือกคีย์บอร์ดสำหรับเรียน"),
        ("💸", "ตามงบ", "ช่วยจัดชุดเริ่มต้นตามงบ"),
    ]
    for column, (icon, label, prompt_text) in zip(prompt_columns, featured_prompts):
        with column:
            if st.button(f"{icon}  {label}", key=f"featured_{label}", use_container_width=True):
                st.session_state.pending_prompt = prompt_text

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
        avatar = "🎧" if msg["role"] == "assistant" else "🧑‍🎤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

    prompt = st.chat_input("มีอะไรให้ Groove & Gear ช่วยแนะนำไหมครับ?")
    if not prompt and st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🎤"):
            st.write(prompt)

        with st.chat_message("assistant", avatar="🎧"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(prompt, model, index, chunks, k=3)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                st.caption(
                    f"พบข้อมูลอ้างอิง {len(context)} ส่วนจาก Knowledge Base")
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
