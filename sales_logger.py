"""MilkLab Sales Logger (S2).

Usage:
    python sales_logger.py --menu "นมหมีฮอกไกโด" --qty 2 --price 65

Reads GOOGLE_SHEETS_CREDENTIALS and TELEGRAM_BOT_TOKEN (or LINE_CHANNEL_TOKEN) from env.
Appends row [timestamp, menu, qty, price, total] to a Google Sheet,
then sends a notification via Telegram or LINE bot.
"""

import argparse
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

try:
    import gspread
except ImportError:  # pragma: no cover - graceful fallback for environments without dependency
    gspread = None


def _parse_credentials() -> dict:
    raw = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "").strip()
    if not raw:
        raise RuntimeError("GOOGLE_SHEETS_CREDENTIALS not set")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SHEETS_CREDENTIALS must be valid JSON") from exc


def _get_worksheet():
    if gspread is None:
        raise RuntimeError("gspread is not installed")

    credentials = _parse_credentials()
    try:
        client = gspread.service_account_from_dict(credentials)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to authenticate Google Sheets: {exc}") from exc

    spreadsheet_name = os.getenv(
        "GOOGLE_SHEETS_SPREADSHEET_NAME", "S2 Worksheet").strip() or "S2 Worksheet"
    spreadsheet_url = os.getenv("GOOGLE_SHEETS_SPREADSHEET_URL", "").strip()

    try:
        if spreadsheet_url:
            workbook = client.open_by_url(spreadsheet_url)
        else:
            workbook = client.open(spreadsheet_name)
    except Exception as exc:
        if spreadsheet_url:
            raise RuntimeError(
                f"Unable to open spreadsheet {spreadsheet_url}: {exc}") from exc
        raise RuntimeError(
            f"Unable to open spreadsheet '{spreadsheet_name}': {exc}") from exc

    worksheet_attr = getattr(workbook, "worksheet", None)
    if callable(worksheet_attr):
        try:
            return worksheet_attr(spreadsheet_name)
        except Exception:
            return workbook.sheet1

    if worksheet_attr is not None:
        return worksheet_attr

    return workbook.sheet1


def append_to_sheet(menu: str, qty: int, price: float) -> dict:
    """Append a sales row to Google Sheets and return the stored payload."""
    worksheet = _get_worksheet()
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    total = qty * price
    row = [timestamp, menu, qty, price, total]
    worksheet.append_row(row)
    return {"timestamp": timestamp, "menu": menu, "qty": qty, "price": price, "total": total}


def query_sales(date: str) -> float:
    """Sum sales totals for a given date using the appended sheet rows."""
    worksheet = _get_worksheet()
    rows = worksheet.get_all_values()
    total = 0.0
    prefix = date.strip()
    for row in rows:
        if len(row) < 5:
            continue
        timestamp = row[0]
        if timestamp.startswith(prefix):
            try:
                total += float(row[4])
            except (TypeError, ValueError):
                continue
    return total


def send_notification(message: str) -> str:
    """Send a notification through Telegram or LINE when credentials are available."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if bot_token and chat_id:
        import requests

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(
            url, json={"chat_id": chat_id, "text": message}, timeout=10)
        response.raise_for_status()
        return "telegram"

    line_token = os.getenv("LINE_CHANNEL_TOKEN", "").strip()
    line_user_id = os.getenv("LINE_USER_ID", "").strip()
    if line_token and line_user_id:
        import requests

        response = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {line_token}"},
            json={"to": line_user_id, "messages": [
                {"type": "text", "text": message}]},
            timeout=10,
        )
        response.raise_for_status()
        return "line"

    raise RuntimeError("No notification credentials available")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="MilkLab Sales Logger")
    parser.add_argument("--menu", required=True, help="ชื่อเมนู")
    parser.add_argument("--qty", type=int, required=True, help="จำนวนขวด")
    parser.add_argument("--price", type=float,
                        required=True, help="ราคาต่อขวด")
    args = parser.parse_args()

    try:
        row = append_to_sheet(args.menu, args.qty, args.price)
        total = row["total"]
    except Exception as exc:
        print(f"[ERROR] บันทึก Sheet ล้มเหลว: {exc}", file=sys.stderr)
        print("[HINT] ตรวจ GOOGLE_SHEETS_CREDENTIALS และ share Sheet กับ service account email", file=sys.stderr)
        return 1

    try:
        provider = send_notification(
            f"บันทึก {args.menu} x{args.qty} = {total} บาท")
    except Exception as exc:
        print(
            f"[WARN] บันทึก Sheet สำเร็จแต่ส่งแจ้งเตือนล้มเหลว: {exc}", file=sys.stderr)
        return 0

    print(f"[OK] บันทึกและแจ้งเตือนผ่าน {provider} เรียบร้อย ยอด {total} บาท")
    return 0


if __name__ == "__main__":
    sys.exit(main())
