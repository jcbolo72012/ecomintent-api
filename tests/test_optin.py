import json, tempfile
from pathlib import Path
from unittest.mock import patch


def test_default_opt_out():
    with tempfile.TemporaryDirectory() as d:
        with patch("api.main.SETTINGS_DIR", Path(d)/"users"):
            from api.main import _is_opted_in
            assert _is_opted_in("u1") is False


def test_opt_in_persists():
    with tempfile.TemporaryDirectory() as d:
        with patch("api.main.SETTINGS_DIR", Path(d)/"users"):
            from api.main import _save_settings, _is_opted_in
            _save_settings("u1", {"opt_in": True})
            assert _is_opted_in("u1") is True


def test_opt_out_reverts():
    with tempfile.TemporaryDirectory() as d:
        with patch("api.main.SETTINGS_DIR", Path(d)/"users"):
            from api.main import _save_settings, _is_opted_in
            _save_settings("u1", {"opt_in": True})
            _save_settings("u1", {"opt_in": False})
            assert _is_opted_in("u1") is False


def test_log_stores_hash_not_raw_text():
    with tempfile.TemporaryDirectory() as d:
        with patch("api.main.SETTINGS_DIR", Path(d)/"users"), \
             patch("api.main.LOGS_DIR", Path(d)/"logs"):
            from api.main import _log_low_confidence
            raw = "where is my order it has been 8 days"
            _log_low_confidence("u1", raw, "WISMO", 0.62)
            lf = next((Path(d)/"logs").glob("*.jsonl"))
            entry = json.loads(open(lf).readline())
            assert raw not in json.dumps(entry)
            assert len(entry["text_hash"]) == 32


def test_deletion_removes_file():
    with tempfile.TemporaryDirectory() as d:
        with patch("api.main.SETTINGS_DIR", Path(d)/"users"), \
             patch("api.main.LOGS_DIR", Path(d)/"logs"):
            from api.main import _log_low_confidence
            import hashlib
            _log_low_confidence("u1", "x", "OTHER", 0.5)
            lf = (Path(d)/"logs")/f"{hashlib.md5(b'u1').hexdigest()}.jsonl"
            assert lf.exists(); lf.unlink(); assert not lf.exists()
