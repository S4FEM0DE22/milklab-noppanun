"""Groove & Gear Caption Generator (S1 Pivot).

Usage:
    python caption_generator.py --item "แอมป์เบสฝึกซ้อม 15W" --n 3

Reads GOOGLE_API_KEY from env. Generates Thai captions for music gear items.
"""

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from google import genai


PROMPT_TEMPLATE = """\
คุณคือ social media manager ของร้าน "Groove & Gear" ร้านขายเครื่องดนตรีและอุปกรณ์ครบวงจร

ข้อมูลสินค้า:
{item_context}

จงเขียนแคปชั่นภาษาไทย 2 ถึง 3 ประโยคเพื่อโปรโมตสินค้านี้ โดยเป็นสไตล์ {style}

เงื่อนไข:
- โทนเป็นกันเอง เข้าใจหัวอกคนเล่นดนตรี ใช้คำง่าย ใส่ emoji เครื่องดนตรีได้
- ต้องมี call-to-action ปิดท้าย เช่น สั่งซื้อเลย, ทักแชท, หรือ เข้ามาลองเทสเสียง
- ห้ามใช้ em dash
- คำแคปชั่นต้องไม่เกิน 280 ตัวอักษร
- ถ้าเป็นสไตล์ cute ให้ดูเป็นมิตร เข้าถึงง่าย และต้อนรับมือใหม่
- ถ้าเป็นสไตล์ minimal ให้สั้น กระชับ โฟกัสที่สเปคและดูเป็นมืออาชีพ
- ถ้าเป็นสไตล์ gen-z ให้เท่ คูล กวนๆ แบบวัยรุ่นทำเพลง
"""


def _parse_item_input(item: str | dict[str, Any] | None) -> str | dict[str, Any]:
    """Parse CLI item input, allowing either plain text or JSON objects."""
    if item is None:
        return ""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return ""
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    return item


def _build_item_context(item: str | dict[str, Any] | None) -> str:
    """Create a compact item context block for the prompt."""
    parsed_item = _parse_item_input(item)
    if isinstance(parsed_item, dict):
        # รองรับทั้งคีย์เก่า (menu) และคีย์ใหม่ (item) เผื่อระบบอื่นส่งมา
        item_data = parsed_item.get(
            "item", parsed_item.get("menu", parsed_item))
        if not isinstance(item_data, dict):
            return f"ชื่อสินค้า: {parsed_item}"

        name = item_data.get("name") or item_data.get(
            "title") or item_data.get("item")
        price = item_data.get("price")
        # เปลี่ยนจาก ingredients เป็น features หรือ desc สำหรับเครื่องดนตรี
        features = item_data.get("features", item_data.get("desc", []))
        if isinstance(features, str):
            features = [features]

        parts: list[str] = []
        if name:
            parts.append(f"ชื่อสินค้า: {name}")
        if price is not None:
            parts.append(f"ราคา: {price} บาท")
        if features:
            parts.append("จุดเด่น: " + ", ".join(str(f) for f in features))
        return "\n".join(parts) if parts else "ชื่อสินค้า: ไม่ระบุ"

    return f"ชื่อสินค้า: {parsed_item}" if parsed_item else "ชื่อสินค้า: ไม่ระบุ"


def generate_caption(
    item: str | dict[str, Any] | None,
    api_key: str | None = None,
    max_attempts: int = 3,
    style: str = "cute",
) -> str:
    """Generate a Thai caption for the given music gear item."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    item_context = _build_item_context(item)
    client = genai.Client(api_key=key)

    for _ in range(max_attempts):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEMPLATE.format(
                item_context=item_context, style=style),
        )
        caption = (getattr(response, "text", None) or "").strip()
        if len(caption) <= 280:
            return caption

    return caption


def generate_captions(
    item: str | dict[str, Any] | None,
    n: int = 1,
    api_key: str | None = None,
    styles: list[str] | None = None,
) -> list[str]:
    """Generate multiple captions for the same item."""
    if n <= 0:
        return []

    count = max(n, len(styles or [])) if styles is not None else n
    style_list = styles or ["cute"] * count
    if len(style_list) < count:
        style_list = style_list + ["cute"] * (count - len(style_list))

    return [generate_caption(item, api_key=api_key, style=style_list[index]) for index in range(count)]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate Thai captions for Groove & Gear items")
    # เปลี่ยนจาก --menu เป็น --item เพื่อให้เข้ากับโดเมน (แต่ยังรองรับ --menu เป็น alias)
    parser.add_argument(
        "--item", "--menu", dest="item", help="Item name or JSON object with name, price and features")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of captions to generate")
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    item = args.item
    if not item:
        item = input("สินค้าที่จะโปรโมต: ").strip()

    if not item:
        print("กรุณาใส่ชื่อสินค้า")
        return 1

    captions = generate_captions(item, n=args.n, styles=[
                                 "cute", "minimal", "gen-z"][: args.n])
    print()
    for index, caption in enumerate(captions, start=1):
        if args.n > 1:
            print(f"[{index}] {caption}")
        else:
            print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
