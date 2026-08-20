"""Groove & Gear Sales Logger (S2 Pivot).

Usage:
    python sales_logger.py --message "พี่ครับ เอาแอมป์เบส 15W ตัวนึงครับ ส่งที่คอนโด XYZ"

Reads GOOGLE_API_KEY from env. Extracts order details and logs to Google Sheets (or console).
"""

import argparse
import datetime
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google import genai

# ถ้ามีการใช้ Google Sheets API จริงๆ ให้ import google_auth_oauthlib ฯลฯ ตรงนี้
# แต่ในตัวอย่างนี้เราจะจำลอง (Mock) การบันทึกลง Sheet เหมือนใน MilkLab นะครับ

PROMPT_TEMPLATE = """\
คุณคือผู้ช่วยจัดการออเดอร์ของร้าน "Groove & Gear" (ร้านขายเครื่องดนตรีและอุปกรณ์เสริม)

จงสกัดข้อมูลคำสั่งซื้อจากข้อความของลูกค้า แล้วตอบกลับเป็นรูปแบบ JSON เท่านั้น โดยมีโครงสร้างดังนี้:
{{
    "customer_name": "ชื่อลูกค้า (ถ้าไม่ระบุให้ใช้ 'ไม่ระบุ')",
    "category": "หมวดหมู่สินค้า เช่น กีตาร์, เบส, คีย์บอร์ด, กลอง, ไมโครโฟน, อุปกรณ์เสริม",
    "item_details": "ชื่อสินค้า รุ่น หรือรายละเอียดที่ลูกค้าต้องการ",
    "price": "ราคาสินค้า (ถ้าลูกค้าไม่บอกราคาให้ใส่ 0)",
    "notes": "หมายเหตุอื่นๆ เช่น ที่อยู่จัดส่ง, เบอร์โทร, คำขอพิเศษ (ถ้ามี)"
}}

ข้อความลูกค้า: "{user_message}"

ตอบกลับเฉพาะ JSON เท่านั้น ห้ามมีข้อความอื่นปน
"""


def extract_order_info(message: str, api_key: str | None = None) -> dict[str, Any]:
    """Extract order information from user message using Gemini."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    client = genai.Client(api_key=key)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEMPLATE.format(user_message=message),
        )

        # คลีนข้อความเผื่อ LLM ใส่ Markdown ```json ... ``` มาให้
        raw_text = (getattr(response, "text", None) or "").strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"Error extracting order info: {e}", file=sys.stderr)
        return {
            "customer_name": "ไม่ระบุ",
            "category": "ไม่ระบุ",
            "item_details": message,
            "price": 0,
            "notes": "ดึงข้อมูลล้มเหลว บันทึกข้อความดิบแทน"
        }


def log_to_sheet(order_data: dict[str, Any]) -> str:
    """Mock function to log data to Google Sheets."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # จำลองว่าเขียนลงคอลัมน์ [Timestamp, Customer_Name, Category, Item_Details, Price, Notes]
    row_data = [
        timestamp,
        order_data.get("customer_name", "ไม่ระบุ"),
        order_data.get("category", "ไม่ระบุ"),
        order_data.get("item_details", "ไม่ระบุ"),
        order_data.get("price", 0),
        order_data.get("notes", "")
    ]

    # TODO: นำ row_data ไปเชื่อมต่อ Google Sheets API ของจริงตรงนี้

    # จำลองว่าสำเร็จ และคืนค่าข้อความยืนยันสำหรับแชตบอต
    print(f"\n[Mock Google Sheet Log] Appended row: {row_data}")

    return f"🎵 จัดออเดอร์ {order_data.get('item_details')} ให้แล้วนะครับ เตรียมแพ็คลงกล่องกันกระแทกอย่างดีครับผม! 📦"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract music gear order and log to database/sheet")
    parser.add_argument(
        "--message", help="User's chat message containing the order")
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    message = args.message
    if not message:
        message = input("ข้อความจากลูกค้า: ").strip()

    if not message:
        print("กรุณาใส่ข้อความ")
        return 1

    print("กำลังประมวลผลออเดอร์...")
    order_info = extract_order_info(message)
    reply_msg = log_to_sheet(order_info)

    print(f"\nข้อความตอบกลับจากระบบ:\n{reply_msg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
