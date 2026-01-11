"""
LLM factory for creating LLM instances based on configuration.

Provides a unified interface for creating LLM clients regardless of backend
(HuggingFace, custom endpoint, local model, etc.).
"""

from typing import Protocol


class LLMProtocol(Protocol):
    """Protocol that all LLM clients must implement."""

    def invoke(self, prompt: str) -> str:
        """Call the LLM with a prompt and return the response."""
        ...


def create_llm(temperature: float | None = None) -> LLMProtocol:
    """
    Create an LLM client based on configuration settings.

    Args:
        temperature: Optional temperature override (0.0-1.0). If None, uses settings.llm_temperature

    Returns:
        LLM client that implements the LLMProtocol

    The function checks settings in this order:
        1. use_custom_endpoint → CustomEndpointLLM
        2. use_local_llm → Local GGUF model (TODO)
        3. default → HuggingFaceEndpoint
    """
    from specagent.config import settings

    # Use provided temperature or fall back to settings
    temp = temperature if temperature is not None else settings.llm_temperature

    if settings.use_custom_endpoint:
        # Use custom OpenAI-compatible endpoint with retry for serverless cold starts
        from specagent.llm.custom_endpoint import CustomEndpointLLM

        return CustomEndpointLLM(
            endpoint_url=settings.custom_endpoint_url,
            temperature=temp,
            max_tokens=settings.llm_max_tokens,
            timeout=120,  # 2 minute timeout for slow inference
            max_retries=5,  # Retry up to 5 times for serverless cold starts
            retry_delay=5.0,  # Start with 5s delay (exponential: 5s, 10s, 20s, 40s, 80s)
        )

    elif settings.use_local_llm:
        # Use local GGUF model (not yet implemented)
        raise NotImplementedError(
            "Local GGUF model support not yet implemented. "
            "Set use_custom_endpoint=True or use_local_llm=False"
        )

    else:
        # Use HuggingFace Inference API
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=settings.llm_model,
            huggingfacehub_api_token=settings.hf_api_key_value,
            temperature=temp,
            max_new_tokens=settings.llm_max_tokens,
        )


# Alias for backwards compatibility
get_llm = create_llm
