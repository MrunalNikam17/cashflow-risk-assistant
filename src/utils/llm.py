"""OpenRouter multi-model fallback caller.

Replicates the multi-model fallback across free-tier models used in DataMind AI
(https://github.com/MrunalNikam17/datamind-ai/blob/main/utils/llm.py).
"""

import logging
import os
import requests
from dotenv import load_dotenv

from src.utils.config import WORKING_MODELS, OPENROUTER_URL, LLM_TIMEOUT_SECONDS

load_dotenv()
logger = logging.getLogger(__name__)


def call_claude(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
    """Calls OpenRouter chat completions with fallback across free-tier models.

    Args:
        system_prompt: Optional instructions / persona for the model.
        user_message: The prompt or input content.
        max_tokens: Maximum tokens in completion.

    Returns:
        The text response from the first model that successfully responds,
        or an empty string / error indicator if all models fail or no API key is set.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not set; using the offline fallback.")
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Cashflow & Risk Assistant",
    }

    # Keep instructions in one user message for compatibility with models that
    # do not support a separate system role.
    if system_prompt and system_prompt.strip():
        combined_message = f"{system_prompt.strip()}\n\n---\n\n{user_message.strip()}"
    else:
        combined_message = user_message.strip()

    for model in WORKING_MODELS:
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
                timeout=LLM_TIMEOUT_SECONDS,
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

        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as e:
            logger.warning("OpenRouter request failed for model %s: %s", model, e)
            continue

    logger.error("All configured OpenRouter models failed.")
    return ""


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Single-argument wrapper used by agents."""
    return call_claude(system_prompt="", user_message=prompt, max_tokens=max_tokens)
