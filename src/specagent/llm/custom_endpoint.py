"""
Custom LLM client for OpenAI-compatible inference endpoints.

Provides a simple wrapper for local/custom inference endpoints that implement
the OpenAI chat completions API format.
"""

import json
from typing import Optional

import requests


class CustomEndpointLLM:
    """LLM client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        endpoint_url: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout: int = 30,
    ):
        """
        Initialize custom endpoint client.

        Args:
            endpoint_url: Full URL to the /v1/chat/completions endpoint
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def invoke(self, prompt: str) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The input prompt text

        Returns:
            The generated response text

        Raises:
            requests.HTTPError: If the API request fails
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            self.endpoint_url, json=payload, timeout=self.timeout
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]


def create_custom_llm(
    endpoint_url: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> CustomEndpointLLM:
    """
    Create a custom LLM client.

    Args:
        endpoint_url: OpenAI-compatible endpoint URL. If None, uses settings.
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Configured CustomEndpointLLM instance
    """
    if endpoint_url is None:
        from specagent.config import settings

        # Default to configured endpoint or fallback
        endpoint_url = getattr(
            settings,
            "custom_endpoint_url",
            "http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions",
        )

    return CustomEndpointLLM(
        endpoint_url=endpoint_url, temperature=temperature, max_tokens=max_tokens
    )
