"""LLM client abstraction — pluggable providers with a unified interface.

Supports: Anthropic Claude, OpenAI, Google Gemini, Ollama (local), and Mock.
Never logs full prompts or responses. Only logs: task_id, step_number,
input_tokens, output_tokens, latency_ms.
"""

from __future__ import annotations

import logging
import time
import uuid
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from enclave.agent.models import (
    LLMResponse,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
)

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
            id="meta-llama/llama-4-scout-17b-16e-instruct",
            name="Llama 4 Scout 17B",
            provider="groq",
            input_price_per_m=0.11,
            output_price_per_m=0.34,
            context_window=128_000,
            description="Multimodal, tool use, fast — best for coding",
        ),
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
            id="openai/gpt-oss-120b",
            name="GPT-OSS 120B",
            provider="groq",
            input_price_per_m=1.50,
            output_price_per_m=3.00,
            context_window=131_072,
            description="OpenAI flagship open-weight 120B, browser search & code exec",
        ),
        ModelInfo(
            id="openai/gpt-oss-20b",
            name="GPT-OSS 20B",
            provider="groq",
            input_price_per_m=0.30,
            output_price_per_m=0.60,
            context_window=131_072,
            description="OpenAI lightweight open-weight 20B, fast and capable",
        ),
        ModelInfo(
            id="qwen/qwen3-32b",
            name="Qwen3 32B",
            provider="groq",
            input_price_per_m=0.29,
            output_price_per_m=0.39,
            context_window=131_072,
            description="Dual-mode reasoning, strong for agentic tasks (preview)",
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


# ─── Mapping Helpers ───────────────────────────────────────────────────────── #


def _block_to_dict(block: ContentBlock) -> dict[str, Any]:
    """Convert ContentBlock to Anthropic Messages API block structure."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    elif isinstance(block, ThinkingBlock):
        # ThinkingBlocks are typically filtered before sending to the API,
        # but if they make it here, serialize them properly
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    elif isinstance(block, ToolUseBlock):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    elif isinstance(block, ToolResultBlock):
        res: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content
        }
        if block.is_error:
            res["is_error"] = True
        return res
    raise ValueError(f"Unknown block type: {type(block)}")


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-formatted tools to OpenAI-formatted tool schemas."""
    if not tools:
        return []
    openai_tools = []
    for tool in tools:
        import copy
        # Deep copy input_schema to avoid modifying original registry schemas in place
        parameters = copy.deepcopy(tool["input_schema"])
        if "required" in parameters and not parameters["required"]:
            parameters.pop("required", None)

        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": parameters
            }
        })
    return openai_tools


def _convert_messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Messages (with content blocks) to OpenAI-compatible messages API list."""
    api_messages = []
    for m in messages:
        if isinstance(m.content, str):
            api_messages.append({"role": m.role, "content": m.content})
        else:
            if m.role == "assistant":
                text_content = ""
                tool_calls = []
                for b in m.content:
                    if isinstance(b, TextBlock):
                        text_content += b.text
                    elif isinstance(b, ToolUseBlock):
                        tool_calls.append({
                            "id": b.id,
                            "type": "function",
                            "function": {
                                "name": b.name,
                                "arguments": json.dumps(b.input)
                            }
                        })
                msg: dict[str, Any] = {"role": "assistant"}
                if text_content:
                    msg["content"] = text_content
                elif not tool_calls:
                    msg["content"] = ""
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                api_messages.append(msg)
            elif m.role == "user":
                is_tool_results = any(isinstance(b, ToolResultBlock) for b in m.content)
                if is_tool_results:
                    for b in m.content:
                        if isinstance(b, ToolResultBlock):
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": b.tool_use_id,
                                "content": b.content
                            })
                        elif isinstance(b, TextBlock):
                            api_messages.append({"role": "user", "content": b.text})
                else:
                    parts = []
                    for b in m.content:
                        if isinstance(b, TextBlock):
                            parts.append({"type": "text", "text": b.text})
                    api_messages.append({"role": "user", "content": parts})
    return api_messages


def _convert_tools_to_gemini(tools: list[dict[str, Any]]) -> list[Any]:
    """Convert tools to google-genai types.Tool declarations."""
    from google.genai import types
    gemini_tools = []
    
    def convert_schema(s: dict[str, Any]) -> types.Schema:
        t_type = s.get("type", "object").upper()
        properties = {}
        for k, v in s.get("properties", {}).items():
            properties[k] = convert_schema(v)
            
        enum_vals = s.get("enum")
        
        return types.Schema(
            type=t_type,
            description=s.get("description"),
            properties=properties or None,
            required=s.get("required") or None,
            enum=enum_vals or None,
        )
        
    decls = []
    for tool in tools:
        decls.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=convert_schema(tool["input_schema"])
            )
        )
    if decls:
        gemini_tools.append(types.Tool(function_declarations=decls))
    return gemini_tools


# ─── Abstract Base ──────────────────────────────────────────────────────────── #


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        """Call the LLM with a system prompt and message history.

        Args:
            system: System prompt.
            messages: Conversation history.
            tools: Registered tool definitions.
            max_tokens: Maximum output tokens.
            task_id: For structured logging only.
            step_number: For structured logging only.
            on_chunk: Callback for streaming text tokens.
            enable_thinking: Enable extended thinking (Anthropic only).
            on_thinking_chunk: Callback for streaming thinking tokens.

        Returns:
            LLMResponse with structured content blocks, token counts, and latency.
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
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic
            client_kwargs: dict[str, Any] = {}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = anthropic.AsyncAnthropic(**client_kwargs)
        return self._client

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        start = time.monotonic()

        api_messages = []
        for m in messages:
            if isinstance(m.content, str):
                api_messages.append({"role": m.role, "content": m.content})
            else:
                # Filter out ThinkingBlocks — they must not be sent back to the API
                # as regular content (Anthropic uses a separate mechanism)
                filtered = [_block_to_dict(b) for b in m.content if not isinstance(b, ThinkingBlock)]
                api_messages.append({
                    "role": m.role,
                    "content": filtered
                })

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = tools

        # Extended thinking support — requires higher max_tokens budget
        if enable_thinking:
            # Thinking uses output tokens, so we need a larger budget.
            # Anthropic requires max_tokens >= 1024 when thinking is enabled.
            thinking_budget = max(max_tokens, 16384)
            kwargs["max_tokens"] = thinking_budget
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget - 4096,  # Reserve 4096 for actual output
            }

        if on_chunk is not None or on_thinking_chunk is not None:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'type'):
                                if event.delta.type == "text_delta":
                                    if on_chunk:
                                        await on_chunk(event.delta.text)
                                elif event.delta.type == "thinking_delta":
                                    if on_thinking_chunk:
                                        await on_thinking_chunk(event.delta.thinking)
            response = await stream.get_final_message()
        else:
            response = await client.messages.create(**kwargs)

        elapsed_ms = (time.monotonic() - start) * 1000

        content_blocks: list[ContentBlock] = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(ThinkingBlock(
                    thinking=block.thinking,
                    signature=getattr(block, 'signature', ''),
                ))
            elif block.type == "tool_use":
                content_blocks.append(ToolUseBlock(
                    id=block.id,
                    name=block.name,
                    input=block.input
                ))

        result = LLMResponse(
            content=content_blocks,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=elapsed_ms,
            stop_reason=response.stop_reason or "",
        )

        logger.info(
            "llm_call",
            extra={
                "task_id": task_id,
                "step": step_number,
                "model": self._model,
                "provider": "anthropic",
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "has_thinking": bool(result.thinking_text),
                "latency_ms": round(elapsed_ms, 2),
            },
        )

        return result


# ─── OpenAI ─────────────────────────────────────────────────────────────────── #


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4o, o3-mini, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise RuntimeError(
                    "OpenAI SDK not installed. Run: pip install -e '.[openai]'"
                )
            client_kwargs: dict[str, Any] = {}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**client_kwargs)
        return self._client

    async def call(
        self,
        system: str,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        client = self._get_client()

        start = time.monotonic()

        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(_convert_messages_to_openai(messages))

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)

        response = await client.chat.completions.create(**kwargs)

        elapsed_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        content_blocks: list[ContentBlock] = []
        if choice.message.content:
            content_blocks.append(TextBlock(text=choice.message.content))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                content_blocks.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=args
                ))

        stop_reason = ""
        if choice.finish_reason == "stop":
            stop_reason = "end_turn"
        elif choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = choice.finish_reason or ""

        result = LLMResponse(
            content=content_blocks,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
            stop_reason=stop_reason,
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
    """OpenRouter API client."""

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
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        import httpx
        import asyncio

        start = time.monotonic()

        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(_convert_messages_to_openai(messages))

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/enclave-tee",
            "X-Title": "Enclave TEE Agent",
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)

        max_retries = 6
        backoff = 2.0
        data = {}

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    if resp.status_code == 429:
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
        choice = choices[0] if choices else {}
        message_data = choice.get("message", {})
        text = message_data.get("content", "")
        tool_calls = message_data.get("tool_calls", [])
        usage = data.get("usage", {})

        content_blocks: list[ContentBlock] = []
        if text:
            content_blocks.append(TextBlock(text=text))

        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                args = func.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                args = {}
            content_blocks.append(ToolUseBlock(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                name=func.get("name", ""),
                input=args
            ))

        finish_reason = choice.get("finish_reason", "")
        stop_reason = ""
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = finish_reason or ""

        result = LLMResponse(
            content=content_blocks,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
            stop_reason=stop_reason,
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


def _parse_groq_failed_generation(failed_gen: str) -> dict[str, Any] | None:
    """Parse Groq's failed_generation field to recover the intended tool call.

    Groq's Llama models sometimes generate tool calls in malformed formats like:
      <function=bash{"command": "uname -a"}</function>
      <function=bash [{"command": "uname -a"}](1)</function>

    This function attempts to extract the tool name and arguments from these
    malformed generations so we can still execute the intended tool call.

    Returns:
        dict with 'name' and 'args' keys, or None if parsing fails.
    """
    if not failed_gen or not failed_gen.strip():
        return None

    try:
        # Pattern 1: <function=name{json_args}</function>
        match = re.search(r'<function=(\w+)\s*(\{.*?\})\s*</function>', failed_gen, re.DOTALL)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = json.loads(args_str)
            return {"name": name, "args": args}

        # Pattern 2: <function=name [json_args](number)</function>
        match = re.search(r'<function=(\w+)\s*\[(.*?)\]\s*\(\d+\)\s*</function>', failed_gen, re.DOTALL)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = json.loads(args_str)
            return {"name": name, "args": args}

        # Pattern 3: <function=name>json_args</function>
        match = re.search(r'<function=(\w+)>\s*(\{.*?\})\s*</function>', failed_gen, re.DOTALL)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            args = json.loads(args_str)
            return {"name": name, "args": args}

        logger.debug(f"Could not parse Groq failed_generation: {failed_gen[:200]}")
        return None
    except (json.JSONDecodeError, AttributeError, IndexError) as exc:
        logger.debug(f"JSON parse error in failed_generation recovery: {exc}")
        return None


class GroqClient(LLMClient):
    """Groq API client."""

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
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
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        import httpx
        import asyncio

        start = time.monotonic()

        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(_convert_messages_to_openai(messages))

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)
            payload["parallel_tool_calls"] = False

        max_retries = 3
        backoff = 2.0
        data = {}

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else backoff
                        # Cap wait time to 30 seconds — don't honor absurd Retry-After values
                        wait_time = min(wait_time, 30.0)
                        logger.warning(
                            f"Rate limit (429) from Groq. Retrying in {wait_time:.1f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(wait_time)
                        backoff = min(backoff * 2, 30.0)
                        continue

                    if resp.status_code >= 500:
                        logger.warning(
                            f"Transient Groq server error ({resp.status_code}). Retrying in {backoff:.1f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt == max_retries:
                            resp.raise_for_status()
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 30.0)
                        continue

                    if resp.status_code == 400:
                        # Check for tool_use_failed — parse the failed_generation
                        error_data = resp.json() if resp.text else {}
                        error_info = error_data.get("error", {})
                        if error_info.get("code") == "tool_use_failed":
                            failed_gen = error_info.get("failed_generation", "")
                            parsed = _parse_groq_failed_generation(failed_gen)
                            if parsed:
                                elapsed_ms = (time.monotonic() - start) * 1000
                                logger.info(
                                    f"Recovered tool call from Groq failed_generation: "
                                    f"{parsed['name']}({list(parsed['args'].keys())})"
                                )
                                content_blocks: list[ContentBlock] = [
                                    ToolUseBlock(
                                        id=f"call_{uuid.uuid4().hex[:16]}",
                                        name=parsed["name"],
                                        input=parsed["args"],
                                    )
                                ]
                                result = LLMResponse(
                                    content=content_blocks,
                                    input_tokens=0,
                                    output_tokens=0,
                                    latency_ms=elapsed_ms,
                                    stop_reason="tool_use",
                                )
                                logger.info(
                                    "llm_call",
                                    extra={
                                        "task_id": task_id,
                                        "step": step_number,
                                        "model": self._model,
                                        "provider": "groq",
                                        "recovered": True,
                                        "input_tokens": 0,
                                        "output_tokens": 0,
                                        "latency_ms": round(elapsed_ms, 2),
                                    },
                                )
                                return result
                            else:
                                logger.warning(
                                    f"Groq tool_use_failed but could not parse: {failed_gen[:200]}"
                                )
                        # Non-recoverable 400 — raise immediately, don't retry
                        logger.error(f"Groq API error ({resp.status_code}): {resp.text[:500]}")
                        raise httpx.HTTPStatusError(
                            f"Groq API error ({resp.status_code})",
                            request=resp.request,
                            response=resp,
                        )

                    if resp.status_code >= 400:
                        logger.error(f"Groq API error ({resp.status_code}): {resp.text[:500]}")
                        raise httpx.HTTPStatusError(
                            f"Groq API error ({resp.status_code}): {resp.text[:500]}",
                            request=resp.request,
                            response=resp,
                        )

                    data = resp.json()
                    break
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                if attempt == max_retries:
                    raise
                logger.warning(
                    f"HTTP error during Groq call: {exc}. Retrying in {backoff:.1f}s... "
                    f"(Attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

        elapsed_ms = (time.monotonic() - start) * 1000

        choices = data.get("choices", [{}])
        choice = choices[0] if choices else {}
        message_data = choice.get("message", {})
        text = message_data.get("content", "")
        tool_calls = message_data.get("tool_calls", [])
        usage = data.get("usage", {})

        content_blocks: list[ContentBlock] = []
        if text:
            content_blocks.append(TextBlock(text=text))

        for tc in tool_calls:
            func = tc.get("function", {})
            try:
                args = func.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                args = {}
            content_blocks.append(ToolUseBlock(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                name=func.get("name", ""),
                input=args
            ))

        finish_reason = choice.get("finish_reason", "")
        stop_reason = ""
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = finish_reason or ""

        result = LLMResponse(
            content=content_blocks,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
            stop_reason=stop_reason,
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
    """Google Gemini API client."""

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
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
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

        # Build contents
        contents: list[types.Content] = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            parts = []
            if isinstance(m.content, str):
                parts.append(types.Part(text=m.content))
            else:
                for b in m.content:
                    if isinstance(b, TextBlock):
                        parts.append(types.Part(text=b.text))
                    elif isinstance(b, ToolUseBlock):
                        parts.append(types.Part(
                            function_call=types.FunctionCall(
                                name=b.name,
                                args=b.input
                            )
                        ))
                    elif isinstance(b, ToolResultBlock):
                        # Find the corresponding tool name by tracing preceding assistant messages
                        tool_name = ""
                        for prev_msg in messages:
                            if not isinstance(prev_msg.content, str):
                                for prev_block in prev_msg.content:
                                    if isinstance(prev_block, ToolUseBlock) and prev_block.id == b.tool_use_id:
                                        tool_name = prev_block.name
                                        break
                            if tool_name:
                                break
                        if not tool_name:
                            tool_name = "unknown_tool"
                        
                        parts.append(types.Part(
                            function_response=types.FunctionResponse(
                                name=tool_name,
                                response={"result": b.content}
                            )
                        ))
            contents.append(types.Content(role=role, parts=parts))

        config_kwargs: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        if tools:
            config_kwargs["tools"] = _convert_tools_to_gemini(tools)

        config = types.GenerateContentConfig(**config_kwargs)

        response = await client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        content_blocks: list[ContentBlock] = []
        if response.text:
            content_blocks.append(TextBlock(text=response.text))

        if response.function_calls:
            for fc in response.function_calls:
                content_blocks.append(ToolUseBlock(
                    id=f"call_{uuid.uuid4().hex[:16]}",
                    name=fc.name,
                    input=fc.args
                ))

        stop_reason = ""
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == "STOP" or finish_reason == "stop":
                if response.function_calls:
                    stop_reason = "tool_use"
                else:
                    stop_reason = "end_turn"
            elif finish_reason == "MAX_TOKENS" or finish_reason == "max_tokens":
                stop_reason = "max_tokens"
            else:
                stop_reason = str(finish_reason)

        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        result = LLMResponse(
            content=content_blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            stop_reason=stop_reason,
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
    """Ollama local LLM client."""

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
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        import httpx

        start = time.monotonic()

        api_messages = [{"role": "system", "content": system}]
        api_messages.extend(_convert_messages_to_openai(messages))

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = _convert_tools_to_openai(tools)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        elapsed_ms = (time.monotonic() - start) * 1000

        message_data = data.get("message", {})
        text = message_data.get("content", "")
        tool_calls = message_data.get("tool_calls", [])

        content_blocks: list[ContentBlock] = []
        if text:
            content_blocks.append(TextBlock(text=text))

        for tc in tool_calls:
            function_data = tc.get("function", {})
            try:
                args = function_data.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
            except Exception:
                args = {}
            content_blocks.append(ToolUseBlock(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                name=function_data.get("name", ""),
                input=args
            ))

        stop_reason = "end_turn"
        if tool_calls:
            stop_reason = "tool_use"

        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0

        result = LLMResponse(
            content=content_blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            stop_reason=stop_reason,
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

    Accepts predefined responses (XML string fallbacks or structured content lists/LLMResponse objects).
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
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
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        task_id: str = "",
        step_number: int = 0,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        enable_thinking: bool = False,
        on_thinking_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
    ) -> LLMResponse:
        self._calls.append({
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "task_id": task_id,
            "step_number": step_number,
            "tools": tools,
        })

        if self._call_count < len(self._responses):
            res_obj = self._responses[self._call_count]
        else:
            res_obj = (
                "<thinking>All steps complete.</thinking>\n"
                "<task_complete>\n"
                "  <summary>Task completed successfully.</summary>\n"
                "</task_complete>"
            )

        content_blocks: list[ContentBlock] = []
        stop_reason = "end_turn"

        if isinstance(res_obj, LLMResponse):
            self._call_count += 1
            return res_obj
        elif isinstance(res_obj, list):
            content_blocks = res_obj
            stop_reason = "tool_use" if any(isinstance(b, ToolUseBlock) for b in content_blocks) else "end_turn"
        elif isinstance(res_obj, str):
            # Parse XML fallback for backward compatibility with existing tests
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", res_obj, re.DOTALL)
            if thinking_match:
                content_blocks.append(TextBlock(text=thinking_match.group(1).strip()))

            tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", res_obj, re.DOTALL)
            for tc_str in tool_calls:
                stop_reason = "tool_use"
                name_match = re.search(r"<name>(.*?)</name>", tc_str, re.DOTALL)
                name = name_match.group(1).strip() if name_match else ""

                args: dict[str, Any] = {}
                args_match = re.search(r"<args>(.*?)</args>", tc_str, re.DOTALL)
                if args_match:
                    args_str = args_match.group(1).strip()
                    # extract tag/value parameters
                    params = re.findall(r"<(.*?)>(.*?)</\1>", args_str, re.DOTALL)
                    for k, v in params:
                        val = v.strip()
                        if val.lower() == "true":
                            args[k] = True
                        elif val.lower() == "false":
                            args[k] = False
                        elif val.isdigit():
                            args[k] = int(val)
                        else:
                            try:
                                args[k] = float(val)
                            except ValueError:
                                args[k] = val

                content_blocks.append(ToolUseBlock(
                    id=f"call_{uuid.uuid4().hex[:16]}",
                    name=name,
                    input=args
                ))

            complete_match = re.search(r"<task_complete>(.*?)</task_complete>", res_obj, re.DOTALL)
            if complete_match:
                summary_match = re.search(r"<summary>(.*?)</summary>", complete_match.group(1), re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else complete_match.group(1).strip()
                content_blocks.append(TextBlock(text=summary))

            if not content_blocks:
                content_blocks.append(TextBlock(text=res_obj))
        else:
            raise TypeError(f"Unsupported mock response type: {type(res_obj)}")

        self._call_count += 1

        return LLMResponse(
            content=content_blocks,
            input_tokens=100 * len(messages),
            output_tokens=len(res_obj) if isinstance(res_obj, str) else 10,
            latency_ms=50.0,
            stop_reason=stop_reason,
        )


# ─── Factory ────────────────────────────────────────────────────────────────── #


def create_llm_client(
    provider: str,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
) -> LLMClient:
    """Factory function to create an LLM client from configuration."""
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
            model=model or "meta-llama/llama-4-scout-17b-16e-instruct",
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
    """Check which providers have API keys configured or are available."""
    import os

    available: dict[str, bool] = {
        "mock": True,
    }

    if os.getenv("ANTHROPIC_API_KEY"):
        available["anthropic"] = True

    if os.getenv("OPENAI_API_KEY"):
        available["openai"] = True

    if os.getenv("GOOGLE_API_KEY"):
        available["gemini"] = True

    if os.getenv("OPENROUTER_API_KEY"):
        available["openrouter"] = True

    if os.getenv("GROQ_API_KEY"):
        available["groq"] = True

    available["ollama"] = True

    return available
