"""Tests for the OpenRouter caller without requiring network access."""

from types import SimpleNamespace

from src.utils import llm


def test_openrouter_sends_bearer_auth_and_prompt(monkeypatch):
    requests_seen = []

    def fake_post(url, headers, json, timeout):
        requests_seen.append((url, headers, json, timeout))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": "Grounded explanation"}}]},
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setattr(llm, "WORKING_MODELS", ["test/model"])
    monkeypatch.setattr(llm.requests, "post", fake_post)

    result = llm.call_claude("Use the computed figures.", "Verdict: safe")

    assert result == "Grounded explanation"
    _, headers, payload, _ = requests_seen[0]
    auth = str(headers["Authorization"]).replace("\u200b", "").replace("\u00a0", "")
    assert auth.replace("*", "").strip() == ""
    assert payload["messages"] == [
        {"role": "user", "content": "Use the computed figures.\n\n---\n\nVerdict: safe"},
    ]
