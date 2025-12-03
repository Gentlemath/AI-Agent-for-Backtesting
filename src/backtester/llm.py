from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .inference_auth_token import get_access_token


@dataclass
class LLMConfig:
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0


# Defaults, overridable via env vars
DEFAULT_MODEL = os.environ.get("ARGONNE_INFERENCE_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct")
DEFAULT_BASE_URL = os.environ.get(
    "ARGONNE_INFERENCE_URL",
    "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
)


class LLMGenerationError(RuntimeError):
    """Raised when the LLM call fails or returns malformed output."""
    pass


class ArgonneLLM:
    """Client for the Argonne inference platform (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 120.0,
    ):
        resolved_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        if not resolved_url:
            raise ValueError("Argonne LLM base URL missing. Set ARGONNE_INFERENCE_URL.")

        self.config = LLMConfig(
            base_url=resolved_url,
            model=model or DEFAULT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )

        self.timeout = timeout

        # Initialize OpenAI-compatible client with Argonne token
        token = get_access_token()
        self.client = OpenAI(api_key=token, base_url=self.config.base_url)

    def call_reasoning_api(self, user: str, system: Optional[str] = None) -> dict:
        """
        Call the Argonne chat completion endpoint and return a plain dict
        shaped like the standard OpenAI response.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            resp = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise LLMGenerationError(f"Argonne inference call failed: {exc}") from exc

        # Normalize to a plain dict for logging / downstream use
        return {
            "id": getattr(resp, "id", None),
            "choices": [
                {
                    "message": {
                        "role": resp.choices[0].message.role,
                        "content": resp.choices[0].message.content,
                    },
                    "finish_reason": resp.choices[0].finish_reason,
                }
            ],
            "model": self.config.model,
        }
