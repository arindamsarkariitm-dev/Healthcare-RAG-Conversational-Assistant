import os
from pathlib import Path


def test_env_file_exists():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    assert env_path.exists(), ".env should exist in the project root"


def test_env_has_key_entry():
    env_path = Path(__file__).resolve().parents[1] / ".env"
    contents = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in contents
