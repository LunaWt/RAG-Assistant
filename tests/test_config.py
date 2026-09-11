import pytest
from pydantic import ValidationError

from app.config import Settings

CREDENTIALS = {"llm_api_key": "key", "hf_token": "token"}


def test_remote_http_base_url_is_rejected():
    """The key rides in the Authorization header of every request, so http leaks it."""
    with pytest.raises(ValidationError):
        Settings(**CREDENTIALS, llm_base_url="http://api.example.com/v1")


def test_loopback_http_base_url_is_allowed():
    url = "http://127.0.0.1:11434/v1"
    assert Settings(**CREDENTIALS, llm_base_url=url).llm_base_url == url
