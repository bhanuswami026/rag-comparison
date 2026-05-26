"""Small LLM provider wrapper for Gemini and OpenAI."""

from __future__ import annotations

import os
from typing import Any

from config import LLM_PROVIDERS


class LLMProviderError(Exception):
    """Raised when an LLM provider cannot complete a request."""


def get_provider_config(provider_name: str) -> dict[str, str]:
    try:
        return LLM_PROVIDERS[provider_name]
    except KeyError as exc:
        raise LLMProviderError(f"Unknown LLM provider: {provider_name}") from exc


def get_api_key(provider_name: str) -> str | None:
    config = get_provider_config(provider_name)
    return os.getenv(config["api_key_env"])


def describe_provider(provider_name: str) -> dict[str, Any]:
    config = get_provider_config(provider_name)
    api_key = get_api_key(provider_name)
    return {
        "name": provider_name,
        "provider": config["provider"],
        "model": config["model"],
        "api_key_env": config["api_key_env"],
        "has_api_key": bool(api_key),
    }


def generate_answer(
    provider_name: str,
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
) -> str:
    config = get_provider_config(provider_name)
    api_key = get_api_key(provider_name)

    if not api_key:
        raise LLMProviderError(
            f"{config['api_key_env']} is not set. Add the key before running {provider_name}."
        )

    if config["provider"] == "openai":
        return generate_openai_answer(
            api_key=api_key,
            model=config["model"],
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    if config["provider"] == "gemini":
        return generate_gemini_answer(
            api_key=api_key,
            model=config["model"],
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    raise LLMProviderError(f"Unsupported provider type: {config['provider']}")


def generate_openai_answer(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer clearly and only use the context provided by the app.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def generate_gemini_answer(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    return response.text.strip() if response.text else ""
