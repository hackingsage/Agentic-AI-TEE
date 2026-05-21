"""LLM client abstraction — pluggable providers with a unified interface.

Supports: Anthropic Claude, OpenAI, Google Gemini, Ollama (local), and Mock.
Never logs full prompts or responses. Only logs: task_id, step_number,
input_tokens, output_tokens, latency_ms.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from enclave.agent.models import LLMResponse, Message

logger = logging.getLogger(__name__)


# ─── Model Catalog ──────────────────────────────────────────────────────────── #


@dataclass
class ModelInfo:
    """Metadata about a specific model."""

    id: str
    name: str
    provider: str
    input_price_per_m: float = 0.0   # $ per 1M input tokens
    output_price_per_m: float = 0.0  # $ per 1M output tokens
    context_window: int = 128_000
    description: str = ""


# Catalog of popular models, grouped by provider
MODEL_CATALOG: dict[str, list[ModelInfo]] = {
    "anthropic": [
        ModelInfo(
            id="claude-sonnet-4-20250514",
            name="Claude Sonnet 4",
            provider="anthropic",
            input_price_per_m=3.0,
            output_price_per_m=15.0,
            context_window=200_000,
            description="Best balance of speed and intelligence",
        ),
        ModelInfo(
            id="claude-opus-4-20250514",
            name="Claude Opus 4",
            provider="anthropic",
            input_price_per_m=15.0,
            output_price_per_m=75.0,
            context_window=200_000,
            description="Most capable, highest quality",
        ),
        ModelInfo(
            id="claude-haiku-3-5-20241022",
            name="Claude 3.5 Haiku",
            provider="anthropic",
            input_price_per_m=0.80,
            output_price_per_m=4.0,
            context_window=200_000,
            description="Fastest, most cost-effective",
        ),
    ],
    "openai": [
        ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            input_price_per_m=2.50,
            output_price_per_m=10.0,
            context_window=128_000,
            description="Flagship multimodal model",
        ),
        ModelInfo(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            provider="openai",
            input_price_per_m=0.15,
            output_price_per_m=0.60,
            context_window=128_000,
            description="Fast and affordable",
        ),
        ModelInfo(
            id="o3-mini",
            name="o3-mini",
            provider="openai",
            input_price_per_m=1.10,
            output_price_per_m=4.40,
            context_window=200_000,
            description="Reasoning model, great for code",
        ),
    ],
    "gemini": [
        ModelInfo(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            provider="gemini",
            input_price_per_m=0.15,
            output_price_per_m=0.60,
            context_window=1_000_000,
            description="Fast, 1M context, thinking model",
        ),
        ModelInfo(
            id="gemini-2.5-pro",
            name="Gemini 2.5 Pro",
            provider="gemini",
            input_price_per_m=1.25,
            output_price_per_m=10.0,
            context_window=1_000_000,
            description="Most capable Gemini, 1M context",
        ),
    ],
    "ollama": [
        ModelInfo(
            id="llama3.1:8b",
            name="Llama 3.1 8B",
            provider="ollama",
            description="Fast local model, good for general tasks",
        ),
        ModelInfo(
            id="codellama:13b",
            name="Code Llama 13B",
            provider="ollama",
            description="Optimized for code generation",
        ),
        ModelInfo(
            id="mistral:7b",
            name="Mistral 7B",
            provider="ollama",
            description="Efficient, strong reasoning",
        ),
        ModelInfo(
            id="deepseek-coder-v2:16b",
            name="DeepSeek Coder V2 16B",
            provider="ollama",
            description="State-of-the-art open code model",
        ),
    ],
    "openrouter": [
        ModelInfo(
            id="anthropic/claude-sonnet-4",
            name="Claude Sonnet 4 (OR)",
            provider="openrouter",
            input_price_per_m=3.0,
            output_price_per_m=15.0,
            context_window=200_000,
            description="Anthropic Claude via OpenRouter",
        ),
        ModelInfo(
            id="openai/gpt-4o",
            name="GPT-4o (OR)",
            provider="openrouter",
            input_price_per_m=2.50,
            output_price_per_m=10.0,
            context_window=128_000,
            description="OpenAI GPT-4o via OpenRouter",
        ),
        ModelInfo(
            id="google/gemini-2.5-flash",
            name="Gemini 2.5 Flash (OR)",
            provider="openrouter",
            input_price_per_m=0.15,
            output_price_per_m=0.60,
            context_window=1_000_000,
            description="Google Gemini via OpenRouter",
        ),
        ModelInfo(
            id="meta-llama/llama-3.1-70b-instruct",
            name="Llama 3.1 70B (OR)",
            provider="openrouter",
            input_price_per_m=0.52,
            output_price_per_m=0.75,
            context_window=131_072,
            description="Meta Llama 3.1 via OpenRouter",
        ),
        ModelInfo(
            id="deepseek/deepseek-r1",
            name="DeepSeek R1 (OR)",
            provider="openrouter",
            input_price_per_m=0.55,
            output_price_per_m=2.19,
            context_window=163_840,
            description="DeepSeek reasoning model via OpenRouter",
        ),
    ],
    "groq": [
        ModelInfo(
            id="llama-3.3-70b-versatile",
            name="Llama 3.3 70B",
            provider="groq",
            input_price_per_m=0.59,
            output_price_per_m=0.79,
            context_window=128_000,
            description="Versatile high-speed 70B model",
        ),
        ModelInfo(
            id="llama-3.1-8b-instant",
            name="Llama 3.1 8B",
            provider="groq",
            input_price_per_m=0.05,
            output_price_per_m=0.08,
            context_window=128_000,
            description="Fast and cost-effective 8B model",
        ),
        ModelInfo(
            id="deepseek-r1-distill-llama-70b",
            name="DeepSeek R1 Distill 70B",
            provider="groq",
            input_price_per_m=0.59,
            output_price_per_m=0.79,
            context_window=128_000,
            description="DeepSeek R1 reasoning model distilled to Llama 70B",
        ),
    ],
    "mock": [
        ModelInfo(
            id="mock",
            name="Mock LLM",
            provider="mock",
            description="Testing only — returns predefined responses",
        ),
    ],
}

# Flat lookup: model_id → ModelInfo
ALL_MODELS: dict[str, ModelInfo] = {
    m.id: m
    for models in MODEL_CATALOG.values()
    for m in models
}

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
    "groq": "Groq",
    "ollama": "Ollama (Local)",
    "mock": "Mock (Testing)",
}

PROVIDER_ENV_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": "",  # no key needed
    "mock": "",
}

# ─── Abstract Base ──────────────────────────────────────────────────────────── #


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        """Call the LLM with a system prompt and message history.

        Args:
            system: System prompt.
            messages: Conversation history.
            max_tokens: Maximum output tokens.
            task_id: For structured logging only.
            step_number: For structured logging only.

        Returns:
            LLMResponse with text, token counts, and latency.
        """
        raise NotImplementedError


# ─── Anthropic ──────────────────────────────────────────────────────────────── #


class AnthropicClient(LLMClient):
    """Anthropic Claude API client.

    In production, API calls route through the privacy proxy via vsock.
    The API key is injected by the proxy — never stored in the enclave.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        import anthropic

        start = time.monotonic()

        client_kwargs: dict[str, Any] = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = anthropic.AsyncAnthropic(**client_kwargs)

        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=api_messages,
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        result = LLMResponse(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=elapsed_ms,
        )

        # Structured logging — never log content
        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "anthropic",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── OpenAI ─────────────────────────────────────────────────────────────────── #


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4o, o3-mini, etc.).

    Uses the official openai async client.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI SDK not installed. Run: pip install -e '.[openai]'"
            )

        start = time.monotonic()

        client_kwargs: dict[str, Any] = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        client = AsyncOpenAI(**client_kwargs)

        # Build messages with system prompt
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=api_messages,
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        result = LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "openai",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── OpenRouter ─────────────────────────────────────────────────────────────── #


class OpenRouterClient(LLMClient):
    """OpenRouter API client — access any model via a single API.

    OpenRouter provides an OpenAI-compatible API at https://openrouter.ai/api/v1
    that routes to 200+ models from all major providers. Uses httpx directly
    to avoid requiring the openai SDK as a dependency.
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url or self.OPENROUTER_BASE_URL

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        import httpx

        start = time.monotonic()

        # Build OpenAI-compatible messages
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/enclave-tee",
            "X-Title": "Enclave TEE Agent",
        }

        import asyncio

        max_retries = 6
        backoff = 2.0  # initial sleep duration in seconds
        data = {}

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": self._model,
                            "max_tokens": max_tokens,
                            "messages": api_messages,
                        },
                    )

                    if resp.status_code == 429:
                        # Check Retry-After header or use backoff
                        retry_after = resp.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else backoff
                        logger.warning(
                            f"Rate limit (429) from OpenRouter. Retrying in {wait_time:.1f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(wait_time)
                        backoff *= 2
                        continue

                    if resp.status_code >= 500:
                        logger.warning(
                            f"Transient OpenRouter server error ({resp.status_code}). Retrying in {backoff:.1f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    break
            except httpx.HTTPError as exc:
                if attempt == max_retries:
                    raise
                logger.warning(
                    f"HTTP error during OpenRouter call: {exc}. Retrying in {backoff:.1f}s... "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        elapsed_ms = (time.monotonic() - start) * 1000

        choices = data.get("choices", [{}])
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})

        result = LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "openrouter",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── Groq ───────────────────────────────────────────────────────────────────── #


class GroqClient(LLMClient):
    """Groq API client — access high-speed open-weight models.

    Groq provides an OpenAI-compatible API at https://api.groq.com/openai/v1
    that routes to high-speed Llama, Mixtral, Gemma, and DeepSeek models.
    Uses httpx directly to avoid requiring the groq SDK as a dependency.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url or self.GROQ_BASE_URL

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        import httpx
        import asyncio

        start = time.monotonic()

        # Build OpenAI-compatible messages
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 6
        backoff = 2.0  # initial sleep duration in seconds
        data = {}

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": self._model,
                            "max_tokens": max_tokens,
                            "messages": api_messages,
                        },
                    )

                    if resp.status_code == 429:
                        # Check Retry-After header or use backoff
                        retry_after = resp.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else backoff
                        logger.warning(
                            f"Rate limit (429) from Groq. Retrying in {wait_time:.1f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(wait_time)
                        backoff *= 2
                        continue

                    if resp.status_code >= 500:
                        logger.warning(
                            f"Transient Groq server error ({resp.status_code}). Retrying in {backoff:.1f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    break
            except httpx.HTTPError as exc:
                if attempt == max_retries:
                    raise
                logger.warning(
                    f"HTTP error during Groq call: {exc}. Retrying in {backoff:.1f}s... "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(backoff)
                backoff *= 2

        elapsed_ms = (time.monotonic() - start) * 1000

        choices = data.get("choices", [{}])
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})

        result = LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "groq",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── Google Gemini ──────────────────────────────────────────────────────────── #


class GeminiClient(LLMClient):
    """Google Gemini API client.

    Uses the official google-genai SDK.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "Google GenAI SDK not installed. Run: pip install -e '.[gemini]'"
            )

        start = time.monotonic()

        client = genai.Client(api_key=self._api_key)

        # Build contents: system instruction is separate in Gemini
        contents: list[types.Content] = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=m.content)],
            ))

        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        text = response.text or ""
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        result = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "gemini",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── Ollama (Local) ─────────────────────────────────────────────────────────── #


class OllamaClient(LLMClient):
    """Ollama local LLM client.

    Connects to a local Ollama instance via its REST API.
    No API key required. Uses httpx (already a project dependency).
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        import httpx

        start = time.monotonic()

        # Build Ollama chat messages
        api_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
        ]
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": api_messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = (time.monotonic() - start) * 1000

        text = data.get("message", {}).get("content", "")
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        result = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )

        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "ollama",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── Mock ───────────────────────────────────────────────────────────────────── #


class MockLLMClient(LLMClient):
    """Mock LLM client for testing.

    Returns predefined responses in sequence. When responses are exhausted,
    returns a default task_complete response.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_count = 0
        self._calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Access recorded call history for assertions."""
        return self._calls

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
    ) -> LLMResponse:
        self._calls.append({
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "task_id": task_id,
            "step_number": step_number,
        })

        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            # Default: return task_complete
            text = (
                "<thinking>All steps complete.</thinking>\n"
                "<task_complete>\n"
                "  <summary>Task completed successfully.</summary>\n"
                "</task_complete>"
            )

        self._call_count += 1

        return LLMResponse(
            text=text,
            input_tokens=100 * len(messages),
            output_tokens=len(text),
            latency_ms=50.0,
        )


# ─── Factory ────────────────────────────────────────────────────────────────── #


def create_llm_client(
    provider: str,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
) -> LLMClient:
    """Factory function to create an LLM client from configuration.

    Args:
        provider: One of "anthropic", "openai", "gemini", "openrouter", "ollama", "mock".
        model: Model identifier (e.g. "gpt-4o", "gemini-2.5-flash").
        api_key: Provider API key (not needed for ollama/mock).
        base_url: Optional override for the API endpoint.

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: If the provider is not recognized.
    """
    provider = provider.lower().strip()

    if provider == "anthropic":
        return AnthropicClient(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key or None,
            base_url=base_url or None,
        )
    elif provider == "openai":
        return OpenAIClient(
            model=model or "gpt-4o",
            api_key=api_key or None,
            base_url=base_url or None,
        )
    elif provider == "gemini":
        return GeminiClient(
            model=model or "gemini-2.5-flash",
            api_key=api_key or None,
        )
    elif provider == "openrouter":
        return OpenRouterClient(
            model=model or "anthropic/claude-sonnet-4",
            api_key=api_key or None,
            base_url=base_url or None,
        )
    elif provider == "groq":
        return GroqClient(
            model=model or "llama-3.3-70b-versatile",
            api_key=api_key or None,
            base_url=base_url or None,
        )
    elif provider == "ollama":
        return OllamaClient(
            model=model or "llama3.1:8b",
            base_url=base_url or "http://localhost:11434",
        )
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported: anthropic, openai, gemini, openrouter, groq, ollama, mock"
        )


def get_available_providers() -> dict[str, bool]:
    """Check which providers have API keys configured or are available.

    Returns:
        Dict of provider → is_available.
    """
    import os

    available: dict[str, bool] = {
        "mock": True,  # Always available
    }

    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        available["anthropic"] = True

    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        available["openai"] = True

    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        available["gemini"] = True

    # OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        available["openrouter"] = True

    # Groq
    if os.getenv("GROQ_API_KEY"):
        available["groq"] = True

    # Ollama — check if the SDK-less REST API is reachable
    # We just mark it as available and let the user try; connection errors
    # will surface at call time.
    available["ollama"] = True

    return available
