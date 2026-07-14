"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"

รับคำสั่งภาษาไทย ส่งให้ Gemini พร้อม tool schema
parse response เป็น tool call เรียก tool จริง
และบันทึก trace ลง agent_trace.log
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

import sales_logger


TRACE_FILE = "agent_trace.log"

TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {
                    "type": "string",
                    "description": "ชื่อเมนูหรือสินค้า",
                },
                "qty": {
                    "type": "integer",
                    "description": "จำนวนที่ขาย ต้องมากกว่า 0",
                },
                "price": {
                    "type": "number",
                    "description": "ราคาต่อหน่วย ต้องไม่ติดลบ",
                },
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายรวมของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "วันที่รูปแบบ YYYY-MM-DD",
                },
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่งข้อความแจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "ข้อความที่ต้องการส่ง",
                },
            },
            "required": ["message"],
        },
    },
]


def write_trace(event_type: str, content: str) -> None:
    """Append one trace record to agent_trace.log."""
    timestamp = datetime.now(
        ZoneInfo("Asia/Bangkok")
    ).strftime("%Y-%m-%d %H:%M:%S")

    with open(TRACE_FILE, "a", encoding="utf-8") as file:
        file.write(f"{timestamp} | {event_type} | {content}\n")


def parse_command(cmd: str, api_key: str | None = None) -> dict[str, Any]:
    """Send a Thai command to Gemini and request a strict JSON tool call."""
    key = api_key or os.getenv("GOOGLE_API_KEY", "").strip()

    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    if genai is None or not hasattr(genai, "Client"):
        raise RuntimeError(
            "google.genai is not available; install package google-genai"
        )

    client = genai.Client(api_key=key)
    tool_schema_json = json.dumps(TOOL_SCHEMA, ensure_ascii=False)

    prompt = f"""
คุณคือตัวช่วยจัดการยอดขายของ MilkLab

ผู้ใช้จะส่งคำสั่งภาษาไทย ให้เลือกใช้ tool เพียงหนึ่งตัวจาก schema นี้:

{tool_schema_json}

กฎ:
- ตอบเฉพาะ JSON เท่านั้น
- ห้ามใส่ Markdown หรือคำอธิบายเพิ่มเติม
- รูปแบบคำตอบต้องเป็น:
  {{"tool": "ชื่อ tool", "args": {{...}}}}
- คำสั่งบันทึกยอดขายให้ใช้ log_sale
- คำสั่งดูยอดขายให้ใช้ query_sales
- คำสั่งแจ้งเตือนให้ใช้ send_alert
- ห้ามแก้จำนวนติดลบให้เป็นจำนวนบวก
- ห้ามแต่งข้อมูลที่ผู้ใช้ไม่ได้ระบุ
- วันที่ต้องอยู่ในรูปแบบ YYYY-MM-DD

คำสั่งของผู้ใช้:
{cmd}
""".strip()

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )

    text = (getattr(response, "text", None) or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini did not return valid JSON: {text}"
        ) from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini response must be a JSON object")

    if "tool" not in parsed or "args" not in parsed:
        raise RuntimeError("Gemini response was not a valid tool call")

    tool_name = parsed["tool"]
    args = parsed["args"]

    if not isinstance(tool_name, str):
        raise RuntimeError("Tool name must be a string")

    if not isinstance(args, dict):
        raise RuntimeError("Tool args must be an object")

    return {
        "tool": tool_name,
        "args": args,
    }


def validate_tool_call(tool_call: dict[str, Any]) -> None:
    """Validate all arguments before executing a tool."""
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    allowed_tools = {
        "log_sale",
        "query_sales",
        "send_alert",
    }

    if tool_name not in allowed_tools:
        raise ValueError(f"unsupported tool: {tool_name}")

    if not isinstance(args, dict):
        raise ValueError("tool args must be an object")

    if tool_name == "log_sale":
        menu = str(args.get("menu", "")).strip()

        try:
            qty = int(args.get("qty"))
        except (TypeError, ValueError) as exc:
            raise ValueError("quantity must be an integer") from exc

        try:
            price = float(args.get("price"))
        except (TypeError, ValueError) as exc:
            raise ValueError("price must be a number") from exc

        if not menu:
            raise ValueError("menu must not be empty")

        if qty <= 0:
            raise ValueError("quantity must be positive")

        if price < 0:
            raise ValueError("price must not be negative")

        # เก็บค่าที่แปลงชนิดข้อมูลแล้วกลับเข้า args
        args["menu"] = menu
        args["qty"] = qty
        args["price"] = price

    elif tool_name == "query_sales":
        date = str(args.get("date", "")).strip()

        if not date:
            raise ValueError("date must not be empty")

        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("date must use YYYY-MM-DD format") from exc

        args["date"] = date

    elif tool_name == "send_alert":
        message = str(args.get("message", "")).strip()

        if not message:
            raise ValueError("message must not be empty")

        if len(message) > 1000:
            raise ValueError("message is too long")

        args["message"] = message


def dispatch_tool(tool_call: dict[str, Any]) -> str:
    """Execute a validated tool call and return a readable summary."""
    tool_name = tool_call["tool"]
    args = tool_call["args"]

    if tool_name == "log_sale":
        menu = args["menu"]
        qty = args["qty"]
        price = args["price"]

        row = sales_logger.append_to_sheet(
            menu=menu,
            qty=qty,
            price=price,
        )

        provider = sales_logger.send_notification(
            f"บันทึก {menu} x{qty} = {row['total']} บาท"
        )

        return (
            f"log_sale OK: row appended at {row['timestamp']} "
            f"via {provider}; total={row['total']} บาท"
        )

    if tool_name == "query_sales":
        date = args["date"]
        total = sales_logger.query_sales(date)

        return f"query_sales OK: ยอดขายวันที่ {date} = {total} บาท"

    if tool_name == "send_alert":
        message = args["message"]
        provider = sales_logger.send_notification(message)

        return f"send_alert OK: alert sent via {provider}"

    raise RuntimeError(f"Unsupported tool: {tool_name}")


def build_user_response(
    tool_call: dict[str, Any],
    tool_result: str,
) -> str:
    """Create a readable final response for the user."""
    tool_name = tool_call["tool"]
    args = tool_call["args"]

    if tool_name == "log_sale":
        total = args["qty"] * args["price"]
        return f"บันทึกแล้ว ยอดรวม {total:g} บาท"

    if tool_name == "query_sales":
        return tool_result.replace("query_sales OK: ", "", 1)

    if tool_name == "send_alert":
        return "ส่งข้อความแจ้งเตือนเรียบร้อยแล้ว"

    return tool_result


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="MilkLab Agent Harness"
    )
    parser.add_argument(
        "--cmd",
        required=True,
        help="คำสั่งภาษาไทย",
    )
    args = parser.parse_args()

    command = args.cmd.strip()

    print(f"[USER] {command}")
    write_trace("user_input", command)

    try:
        tool_call = parse_command(command)

        tool_call_json = json.dumps(
            tool_call,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        print(
            f"[LLM] tool={tool_call['tool']} "
            f"args={tool_call['args']}"
        )
        write_trace("llm_response", tool_call_json)

        # Guardrail ต้องทำก่อน dispatch_tool เสมอ
        validate_tool_call(tool_call)

        result = dispatch_tool(tool_call)

        print(f"[TOOL] {result}")
        write_trace("tool_result", result)

        user_response = build_user_response(tool_call, result)
        print(f"[USER] ← {user_response}")

        return 0

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

        print(f"[ERROR] {error_message}", file=sys.stderr)
        write_trace("tool_result", error_message)

        return 1


if __name__ == "__main__":
    sys.exit(main())