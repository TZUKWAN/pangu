"""Trade-calendar fallback tests.

The web runtime should avoid degrading to weekday-only logic when the primary
akshare import fails but a real bundled or public calendar source is available.
All tests use monkeypatches; they do not depend on live network or a real
akshare installation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from engine.web import server


def test_dates_from_akshare_calendar_json_reads_file(tmp_path, monkeypatch):
    """find_spec locates akshare/file_fold/calendar.json without importing akshare."""
    pkg = tmp_path / "akshare"
    (pkg / "file_fold").mkdir(parents=True)
    (pkg / "file_fold" / "calendar.json").write_text(
        json.dumps(["20260101", "20260102", "20260105", "20260106"]),
        encoding="utf-8",
    )

    def fake_find_spec(name, *args, **kwargs):
        if name == "akshare":
            return SimpleNamespace(submodule_search_locations=[str(pkg)])
        return None

    monkeypatch.setattr(server.importlib.util, "find_spec", fake_find_spec)
    assert server._dates_from_akshare_calendar_json() == {
        "20260101",
        "20260102",
        "20260105",
        "20260106",
    }


def test_dates_from_akshare_calendar_json_missing_returns_empty(monkeypatch):
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda *a, **k: None)
    assert server._dates_from_akshare_calendar_json() == set()


def test_dates_from_akshare_calendar_json_strips_dash(tmp_path, monkeypatch):
    pkg = tmp_path / "akshare"
    (pkg / "file_fold").mkdir(parents=True)
    (pkg / "file_fold" / "calendar.json").write_text(
        json.dumps(["2026-01-02", "2026-01-05"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server.importlib.util,
        "find_spec",
        lambda *a, **k: SimpleNamespace(submodule_search_locations=[str(pkg)]),
    )
    assert server._dates_from_akshare_calendar_json() == {"20260102", "20260105"}


class _FakeResp:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_dates_from_http_calendar_parses_dates(monkeypatch):
    """The HTTP fallback trusts only responses with enough date-like entries."""
    import urllib.request

    payload = json.dumps(
        ["20260101", "20260102", "20260105"]
        + [f"2026{m:02d}{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    )
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResp(payload))
    dates = server._dates_from_http_calendar()
    assert "20260101" in dates
    assert len(dates) >= 200


def test_dates_from_http_calendar_fails_silently(monkeypatch):
    import urllib.request

    def boom(*a, **k):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert server._dates_from_http_calendar() == set()


def _block_akshare_import(monkeypatch):
    """Simulate the py_mini_racer import failure raised by importing akshare."""
    import builtins

    real_import = builtins.__import__

    def patched(name, *a, **k):
        if name == "akshare":
            raise ImportError("py_mini_racer circular import")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", patched)


def test_load_calendar_falls_back_to_calendar_json_when_akshare_dead(monkeypatch, tmp_path):
    pkg = tmp_path / "akshare"
    (pkg / "file_fold").mkdir(parents=True)
    (pkg / "file_fold" / "calendar.json").write_text(
        json.dumps(["20260701", "20260702", "20260703"]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server.importlib.util,
        "find_spec",
        lambda *a, **k: SimpleNamespace(submodule_search_locations=[str(pkg)]),
    )
    _block_akshare_import(monkeypatch)
    monkeypatch.setattr(server, "_trade_calendar_cache", None)

    dates, warnings, source = server._load_trade_calendar()
    assert source == "akshare.file_fold.calendar_json"
    assert dates == {"20260701", "20260702", "20260703"}
    assert any("py_mini_racer" in w or "akshare" in w for w in warnings)


def test_load_calendar_falls_back_to_http_when_calendar_json_missing(monkeypatch):
    _block_akshare_import(monkeypatch)
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda *a, **k: None)
    monkeypatch.setattr(server, "_dates_from_http_calendar", lambda: {"20260701", "20260702"})
    monkeypatch.setattr(server, "_trade_calendar_cache", None)

    dates, warnings, source = server._load_trade_calendar()
    assert dates == {"20260701", "20260702"}
    assert source == "http.public_calendar"
    assert any("akshare" in w for w in warnings)


def test_load_calendar_all_fail_returns_weekday_fallback(monkeypatch):
    _block_akshare_import(monkeypatch)
    monkeypatch.setattr(server.importlib.util, "find_spec", lambda *a, **k: None)
    monkeypatch.setattr(server, "_dates_from_http_calendar", lambda: set())
    monkeypatch.setattr(server, "_trade_calendar_cache", None)

    dates, warnings, source = server._load_trade_calendar()
    assert dates == set()
    assert source == "weekday_fallback"
    assert any("akshare" in w for w in warnings)


def test_runtime_context_reflects_calendar_json_source(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    dt = datetime(2026, 7, 1, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(server, "_beijing_now", lambda: dt)
    monkeypatch.setattr(
        server,
        "_load_trade_calendar",
        lambda: ({"20260701", "20260702"}, [], "akshare.file_fold.calendar_json"),
    )
    rt = server._runtime_context()
    assert rt["calendar_source"] == "akshare.file_fold.calendar_json"
    assert rt["calendar_coverage"] == "含 akshare 内置交易日历，覆盖到内置文件末端"
    assert rt["is_trade_day"] is True


def test_runtime_context_reflects_http_calendar_source(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    dt = datetime(2026, 7, 1, 14, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(server, "_beijing_now", lambda: dt)
    monkeypatch.setattr(
        server,
        "_load_trade_calendar",
        lambda: ({"20260701", "20260702"}, [], "http.public_calendar"),
    )
    rt = server._runtime_context()
    assert rt["calendar_source"] == "http.public_calendar"
    assert rt["calendar_coverage"] == "含公开 HTTP 交易日历兜底"
    assert rt["is_trade_day"] is True
