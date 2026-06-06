"""
Tests for HMAC-SHA256 webhook signature validation.
Run with: pytest gorgias/tests/test_webhook_signature.py -v
"""
import base64
import hmac
import hashlib
import pytest
from unittest.mock import patch, MagicMock

TEST_SECRET = "test_app_secret_12345"
TEST_BODY = b'{"event": {"type": "ticket-created", "object": {"id": 123}}}'


def make_signature(body: bytes, secret: str) -> str:
    return base64.b64encode(
        hmac.new(secret.encode(), body, hashlib.sha256).digest()
    ).decode()


# Patch the env var and volume paths before importing
@pytest.fixture(autouse=True)
def patch_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GORGIAS_APP_SECRET", TEST_SECRET)
    monkeypatch.setenv("GORGIAS_APP_ID", "test_app_id")
    monkeypatch.setenv("APP_BASE_URL", "https://test.modal.run")
    # Patch volume paths to use temp directory
    monkeypatch.setattr("gorgias.webhook.LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr("gorgias.oauth.TOKEN_DIR", tmp_path / "tokens")
    monkeypatch.setattr("gorgias.settings.SETTINGS_DIR", tmp_path / "settings")


def test_valid_signature_passes():
    from gorgias.webhook import verify_signature
    sig = make_signature(TEST_BODY, TEST_SECRET)
    assert verify_signature(TEST_BODY, sig) is True


def test_invalid_signature_fails():
    from gorgias.webhook import verify_signature
    assert verify_signature(TEST_BODY, "invalidsignature") is False


def test_missing_signature_fails():
    from gorgias.webhook import verify_signature
    assert verify_signature(TEST_BODY, "") is False


def test_empty_body_valid_signature_passes():
    from gorgias.webhook import verify_signature
    body = b""
    sig = make_signature(body, TEST_SECRET)
    assert verify_signature(body, sig) is True


def test_tampered_body_fails():
    from gorgias.webhook import verify_signature
    sig = make_signature(TEST_BODY, TEST_SECRET)
    tampered = TEST_BODY + b" extra"
    assert verify_signature(tampered, sig) is False


def test_wrong_secret_fails():
    from gorgias.webhook import verify_signature
    sig = make_signature(TEST_BODY, "wrong_secret")
    assert verify_signature(TEST_BODY, sig) is False


def test_signature_timing_safe():
    """Ensure comparison uses constant-time compare (hmac.compare_digest)."""
    import inspect
    from gorgias import webhook
    src = inspect.getsource(webhook.verify_signature)
    assert "compare_digest" in src, "Must use hmac.compare_digest for timing-safe comparison"
