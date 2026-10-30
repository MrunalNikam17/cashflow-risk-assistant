"""OpenRouter multi-model fallback caller.

Replicates the multi-model fallback across free-tier models used in DataMind AI.
"""

import logging
import os
from typing import Iterable, List, Optional

import requests
from dotenv import load_dotenv

from src.utils.config import LLM_TIMEOUT_SECONDS, OPENROUTER_URL, WORKING_MODELS

load_dotenv()
logger = logging.getLogger(__name__)


class _MaskedAuthorization(str):
    def __new__(cls, value: str = "******"):
        return str.__new__(cls, value)

    def __eq__(self, other):
        left = str(self).replace("\u200b", "").replace("\u00a0", "").replace("\r", "").replace("\n", "")
        right = str(other).replace("\u200b", "").replace("\u00a0", "").replace("\r", "").replace("\n", "")
        return left == right


def _resolve_models(model_names: Optional[Iterable[str]] = None) -> List[str]:
    if model_names is not None:
        return [str(name).strip() for name in model_names if str(name).strip()]

    for env_name in ("OPENROUTER_MODELS", "OPENROUTER_MODEL_LIST", "WORKING_MODELS"):
        raw = os.getenv(env_name, "").strip()
        if raw:
            parts = [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]
            if parts:
                return parts
    return list(WORKING_MODELS)


def validate_models(model_names: Optional[Iterable[str]] = None, timeout: float = 10.0) -> List[str]:
    """Validate the configured OpenRouter models, returning the subset still usable."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.info("No OpenRouter API key configured; skipping model validation.")
        return []

    target_models = _resolve_models(model_names)
    if not target_models:
        return []

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": _MaskedAuthorization("******"),
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Cashflow & Risk Assistant",
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logger.warning("OpenRouter model validation skipped while offline: %s", exc)
        return []

    if response.status_code != 200:
        logger.warning("OpenRouter model validation returned HTTP %s.", response.status_code)
        return []

    try:
        payload = response.json()
    except ValueError as exc:
        logger.warning("OpenRouter model validation response was not valid JSON: %s", exc)
        return []

    available = {str(item.get("id")) for item in payload.get("data", []) if item.get("id")}
    usable = [model for model in target_models if model in available]
    logger.info("Validated %s/%s OpenRouter models.", len(usable), len(target_models))
    return usable


def call_llm_with_fallback(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1500,
    model_names: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
) -> str:
    """Calls OpenRouter chat completions with fallback across model candidates."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not set; using the offline fallback.")
        return ""

    headers = {
        "Authorization": _MaskedAuthorization("******"),
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Cashflow & Risk Assistant",
    }

    if system_prompt and system_prompt.strip():
        combined_message = f"{system_prompt.strip()}\n\n---\n\n{user_message.strip()}"
    else:
        combined_message = user_message.strip()

    models = _resolve_models(model_names)
    timeout_value = timeout if timeout is not None else LLM_TIMEOUT_SECONDS

    for model in models:
        try:
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": combined_message}],
            }
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout_value,
            )

            if response.status_code == 429:
                logger.info("OpenRouter rate limited model %s; trying the next model.", model)
                continue
            if response.status_code != 200:
                logger.warning("OpenRouter model %s returned HTTP %s.", model, response.status_code)
                continue

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"]["content"]
                if content and content.strip():
                    logger.info("Received an OpenRouter response from %s.", model)
                    return content.strip()
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            logger.warning("OpenRouter request failed for model %s: %s", model, exc)
            continue

    logger.error("All configured OpenRouter models failed.")
    return ""


def call_claude(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
    """Backward-compatible alias for the older function name."""
    return call_llm_with_fallback(system_prompt=system_prompt, user_message=user_message, max_tokens=max_tokens)


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Single-argument wrapper used by agents."""
    return call_llm_with_fallback(system_prompt="", user_message=prompt, max_tokens=max_tokens)


__all__ = [
    "call_llm",
    "call_claude",
    "call_llm_with_fallback",
    "validate_models",
]
