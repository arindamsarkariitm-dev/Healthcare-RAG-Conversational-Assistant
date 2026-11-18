from pathlib import Path


def test_requirements_contains_adapter_packages():
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    lines = requirements.read_text(encoding="utf-8").splitlines()
    assert any("langchain-community" in line for line in lines)
    assert any("langchain-openai" in line for line in lines)
