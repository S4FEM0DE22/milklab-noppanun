"""Groove & Gear Agent Harness (S3 Pivot).

จัดการระบบ Agent และ Function Calling (Tools) สำหรับร้านเครื่องดนตรี
"""

import json
import os
import sys
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. นิยามฟังก์ชันการทำงานของ Tools (Mock functions)
# ---------------------------------------------------------------------------


def check_amp_compatibility(instrument_type: str, amp_type: str) -> str:
    """เช็กความเข้ากันได้ของเครื่องดนตรีและแอมป์"""
    ins = instrument_type.lower()
    amp = amp_type.lower()

    if "เบส" in ins or "bass" in ins:
        if "กีตาร์" in amp or "guitar" in amp:
            return "อันตราย: การใช้เบสกับแอมป์กีตาร์อาจทำให้ดอกลำโพงแตกได้ เนื่องจากแอมป์กีตาร์ไม่ได้ออกแบบมาให้รับความถี่ต่ำและแรงกระแทกจากเบส แนะนำให้ใช้แอมป์เบสโดยเฉพาะครับ"

    if "กีตาร์" in ins or "guitar" in ins:
        if "เบส" in amp or "bass" in amp:
            return "ปลอดภัย: สามารถใช้กีตาร์เล่นกับแอมป์เบสได้ ลำโพงไม่พัง แต่โทนเสียงที่ได้อาจจะทุ้มและขาดความใส (High-end) ไปบ้างครับ"

    return f"โดยทั่วไปสามารถใช้งานร่วมกันได้ แต่แนะนำให้ใช้อุปกรณ์ที่ออกแบบมาเฉพาะทางจะได้เสียงที่ดีที่สุดครับ"


def recommend_starter_gear(instrument_category: str, budget: int) -> str:
    """แนะนำสินค้าเบื้องต้นตามประเภทและงบประมาณ"""
    cat = instrument_category.lower()
    if "คีย์บอร์ด" in cat or "keyboard" in cat:
        if budget < 5000:
            return "ในงบนี้ แนะนำเป็น 'คีย์บอร์ดไฟฟ้า 61 คีย์' (ราคา 4,900 บาท) พกพาง่าย มีจังหวะในตัวครับ"
        else:
            return "ถ้างบถึง แนะนำ 'เปียโนไฟฟ้า 88 คีย์' (ราคา 12,500 บาท) ทัชชิ่งเหมือนจริง ซ้อมได้ยาวๆ ครับ"

    elif "กีตาร์" in cat or "guitar" in cat:
        return "แนะนำ 'กีตาร์โปร่ง ทรง Dreadnought' (ราคา 3,500 บาท) เสียงกังวาน เล่นง่าย เหมาะกับมือใหม่สุดๆ ครับ"

    return f"สำหรับ {instrument_category} ในงบ {budget} บาท ทักแอดมินมาเพื่อรับคำแนะนำเชิงลึกได้เลยครับ!"


def log_order(order_message: str) -> str:
    """เรียกใช้ระบบบันทึกออเดอร์ (เชื่อมกับ sales_logger)"""
    # ในการใช้งานจริง คุณสามารถ import extract_order_info และ log_to_sheet จาก sales_logger.py มาใช้ได้เลย
    return f"ระบบได้รับออเดอร์แล้ว: ระบบเตรียมแพ็คสินค้าและจะสรุปยอดให้ในแชตครับ (บันทึกออเดอร์สำเร็จ)"


# ---------------------------------------------------------------------------
# 2. กำหนด TOOL SCHEMA สำหรับ Gemini
# ---------------------------------------------------------------------------

music_tools = [
    types.FunctionDeclaration(
        name="check_amp_compatibility",
        description="ตรวจสอบความเข้ากันได้และความปลอดภัย เมื่อลูกค้าถามว่าเอาเครื่องดนตรีประเภทหนึ่งไปเสียบเล่นกับแอมป์อีกประเภทหนึ่งได้หรือไม่",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "instrument_type": types.Schema(type=types.Type.STRING, description="ประเภทเครื่องดนตรีที่ลูกค้าจะเล่น เช่น กีตาร์, เบส, คีย์บอร์ด"),
                "amp_type": types.Schema(type=types.Type.STRING, description="ประเภทแอมป์ที่ลูกค้าจะนำไปเสียบ เช่น แอมป์กีตาร์, แอมป์เบส"),
            },
            required=["instrument_type", "amp_type"]
        )
    ),
    types.FunctionDeclaration(
        name="recommend_starter_gear",
        description="แนะนำสินค้ารุ่นเริ่มต้น เมื่อลูกค้าถามหาเครื่องดนตรีและบอกงบประมาณ",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "instrument_category": types.Schema(type=types.Type.STRING, description="หมวดหมู่เครื่องดนตรี เช่น กีตาร์, คีย์บอร์ด, กลอง"),
                "budget": types.Schema(type=types.Type.INTEGER, description="งบประมาณที่ลูกค้ามี (ใส่ตัวเลขเท่านั้น หากไม่ระบุให้ใส่ 0)"),
            },
            required=["instrument_category", "budget"]
        )
    ),
    types.FunctionDeclaration(
        name="log_order",
        description="ใช้บันทึกคำสั่งซื้อ เมื่อลูกค้าตกลงซื้อสินค้าหรือสั่งซื้อสินค้า",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "order_message": types.Schema(type=types.Type.STRING, description="ข้อความแชตทั้งหมดของลูกค้าที่บอกว่าสั่งอะไร ส่งที่ไหน"),
            },
            required=["order_message"]
        )
    )
]

# ดึงฟังก์ชันมาไว้ใน Dictionary เพื่อให้เรียกใช้ง่ายๆ เมื่อ LLM คืนค่าชื่อฟังก์ชันมา
TOOL_FUNCTIONS = {
    "check_amp_compatibility": check_amp_compatibility,
    "recommend_starter_gear": recommend_starter_gear,
    "log_order": log_order
}

# ---------------------------------------------------------------------------
# 3. Agent Harness Logic
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
คุณคือแอดมินร้าน "Groove & Gear" (ร้านขายเครื่องดนตรีและอุปกรณ์ครบวงจร)
บุคลิก: เป็นกันเอง คุยสนุก เป็นนักดนตรีตัวจริงที่พร้อมป้ายยาอุปกรณ์ให้ลูกค้า
หน้าที่:
1. ตอบคำถามเรื่องเครื่องดนตรี
2. หากลูกค้าถามเรื่องความปลอดภัย/การเสียบแอมป์ข้ามประเภท ให้เรียกใช้เครื่องมือ `check_amp_compatibility`
3. หากลูกค้าให้งบประมาณมาและให้แนะนำสินค้า ให้เรียกใช้เครื่องมือ `recommend_starter_gear`
4. หากลูกค้าสั่งซื้อสินค้า ให้เรียกใช้เครื่องมือ `log_order`
"""


def chat_with_agent(user_message: str, api_key: str | None = None):
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=key)

    # สร้าง Tools object
    agent_tools = types.Tool(function_declarations=music_tools)

    print(f"User: {user_message}")

    # ส่งข้อความไปหา Gemini พร้อม Tools
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[agent_tools]
        )
    )

    # ตรวจสอบว่า Gemini ต้องการเรียกใช้ Tool (Function Calling) หรือไม่
    if response.function_calls:
        for function_call in response.function_calls:
            func_name = function_call.name
            args = {k: v for k, v in function_call.args.items()}

            print(
                f"\n[Agent คิด...] กำลังเรียกใช้ Tool: {func_name} ด้วยข้อมูล {args}")

            if func_name in TOOL_FUNCTIONS:
                # เรียกใช้ฟังก์ชัน Python ที่เราเตรียมไว้
                tool_result = TOOL_FUNCTIONS[func_name](**args)
                print(f"[ผลลัพธ์จาก Tool]: {tool_result}")

                # ส่งผลลัพธ์กลับไปให้ Gemini เพื่อสร้างข้อความตอบกลับลูกค้า
                final_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        user_message,
                        response.candidates[0].content,
                        types.Part.from_function_response(
                            name=func_name,
                            response={"result": tool_result}
                        )
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION)
                )
                print(f"\nGroove & Gear Admin: {final_response.text}")
    else:
        # ตอบกลับปกติโดยไม่ใช้ Tool
        print(f"\nGroove & Gear Admin: {response.text}")


if __name__ == "__main__":
    load_dotenv()
    # ลองเทสด้วยคำถามที่บอตต้องใช้ Tool
    test_msg = "พี่ครับ ผมเอาเบสไปเสียบเล่นกับแอมป์กีตาร์ของที่บ้านได้มั้ยครับ?"
    chat_with_agent(test_msg)
