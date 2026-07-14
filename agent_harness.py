"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema parse response เป็น tool call
เรียก tool จริง print trace log
"""

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:  # pragma: no cover - graceful fallback for environments without dependency
    genai = None

import sales_logger


TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """Send the Thai command to Gemini and request a strict JSON tool call."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")
    if genai is None or not hasattr(genai, "Client"):
        raise RuntimeError("google.genai is not available")

    client = genai.Client(api_key=key)
    tool_schema_json = json.dumps(TOOL_SCHEMA, ensure_ascii=False)
    prompt = f"""คุณคือตัวช่วยจัดการขายของ MilkLab

ผู้ใช้จะส่งคำสั่งภาษาไทย ให้แปลงคำสั่งนั้นเป็นคำสั่ง tool เดียวต่อไปนี้
Schema:
{tool_schema_json}

กฎ:
- ตอบเฉพาะ JSON เท่านั้น ไม่มีคำอธิบาย
- รูปแบบต้องเป็น {{"tool": "ชื่อ tool", "args": {{...}}}}
- ถ้าเป็นคำสั่งบันทึกขาย ให้ใช้ log_sale
- ถ้าสอบถามยอดขายของวันที่ ให้ใช้ query_sales
- ถ้าเป็นคำสั่งส่งแจ้งเตือน ให้ใช้ send_alert

คำสั่งของผู้ใช้: {cmd}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    text = (getattr(response, "text", None) or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini did not return valid JSON: {text}") from exc

    if not isinstance(parsed, dict) or "tool" not in parsed or "args" not in parsed:
        raise RuntimeError("Gemini response was not a tool call")

    tool_name = parsed["tool"]
    args = parsed.get("args", {})
    if not isinstance(args, dict):
        raise RuntimeError("Tool args must be an object")
    return {"tool": tool_name, "args": args}


def dispatch_tool(tool_call: dict) -> str:
    """Execute the requested tool and return a readable summary."""
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    if tool_name == "log_sale":
        menu = str(args.get("menu", ""))
        qty = int(args.get("qty", 0))
        price = float(args.get("price", 0))
        row = sales_logger.append_to_sheet(menu, qty, price)
        provider = sales_logger.send_notification(
            f"บันทึก {menu} x{qty} = {row['total']} บาท")
        return f"log_sale OK: row appended at {row['timestamp']} via {provider}"

    if tool_name == "query_sales":
        date = str(args.get("date", ""))
        total = sales_logger.query_sales(date)
        return f"ยอดขายวันที่ {date} = {total} บาท"

    if tool_name == "send_alert":
        message = str(args.get("message", ""))
        provider = sales_logger.send_notification(message)
        return f"alert sent via {provider}"

    raise RuntimeError(f"Unsupported tool: {tool_name}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    print(f"[USER] {args.cmd}")

    tool_call = parse_command(args.cmd)
    print(f"[LLM]  tool={tool_call['tool']} args={tool_call['args']}")

    result = dispatch_tool(tool_call)
    print(f"[TOOL] {tool_call['tool']} {result}")
    print(f"[USER] ← {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
