"""Shared construction of OpenAI-compatible clients."""

from openai import AsyncOpenAI


def build_openai_client(base_url: str, api_key: str, timeout: float) -> AsyncOpenAI:
    """Build an ``AsyncOpenAI`` client for an OpenAI-compatible endpoint.

    A placeholder key keeps construction valid for local providers that ignore
    auth; real requests still send the configured key.
    """
    return AsyncOpenAI(base_url=base_url, api_key=api_key or "not-required", timeout=timeout)
