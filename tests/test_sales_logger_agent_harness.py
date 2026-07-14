import json

import agent_harness
import sales_logger


class FakeWorksheet:
    def __init__(self):
        self.rows = []

    def append_row(self, row):
        self.rows.append(row)


class FakeSpreadsheet:
    def __init__(self, worksheet):
        self.worksheet = worksheet


class FakeClient:
    def __init__(self, worksheet):
        self.worksheet = worksheet

    def open(self, title):
        return FakeSpreadsheet(self.worksheet)

    def open_by_url(self, url):
        return FakeSpreadsheet(self.worksheet)


class FakeGspreadModule:
    def __init__(self, worksheet):
        self.worksheet = worksheet

    def service_account_from_dict(self, credentials):
        return FakeClient(self.worksheet)


def test_append_to_sheet_appends_timestamped_row(monkeypatch):
    worksheet = FakeWorksheet()
    monkeypatch.setenv("GOOGLE_SHEETS_CREDENTIALS",
                       json.dumps({"type": "service_account"}))
    monkeypatch.setattr(sales_logger, "gspread", FakeGspreadModule(worksheet))

    row = sales_logger.append_to_sheet("นมหมีฮอกไกโด", 2, 65)

    assert row["menu"] == "นมหมีฮอกไกโด"
    assert row["qty"] == 2
    assert row["price"] == 65
    assert row["total"] == 130
    assert len(worksheet.rows) == 1
    assert worksheet.rows[0][0] != ""
    assert worksheet.rows[0][1:] == ["นมหมีฮอกไกโด", 2, 65, 130]


def test_parse_command_returns_tool_call(monkeypatch):
    class FakeModels:
        def generate_content(self, **kwargs):
            return type("Response", (), {"text": '{"tool": "log_sale", "args": {"menu": "นมหมีฮอกไกโด", "qty": 2, "price": 65}}'})()

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = FakeModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(agent_harness.genai, "Client",
                        lambda api_key=None: FakeClient(api_key=api_key))

    tool_call = agent_harness.parse_command("บันทึกขายนมหมี 2 ขวด ขวดละ 65")

    assert tool_call["tool"] == "log_sale"
    assert tool_call["args"] == {"menu": "นมหมีฮอกไกโด", "qty": 2, "price": 65}
