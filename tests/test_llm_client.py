import pytest
import httpx
from unittest.mock import AsyncMock, patch
from enclave.agent.llm_client import OpenRouterClient, GroqClient
from enclave.agent.models import Message


@pytest.mark.asyncio
async def test_openrouter_client_retries_on_rate_limit():
    client = OpenRouterClient(api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    mock_response_429 = httpx.Response(
        status_code=429,
        request=request,
        headers={"Retry-After": "0.1"},
        json={"error": "Rate limit exceeded"},
    )
    mock_response_200 = httpx.Response(
        status_code=200,
        request=request,
        json={
            "choices": [{"message": {"content": "Hello from model"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
        },
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [mock_response_429, mock_response_200]

        # Use a short backoff for testing
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            response = await client.call("system prompt", messages)

            assert response.text == "Hello from model"
            assert response.input_tokens == 10
            assert response.output_tokens == 15
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once_with(0.1)


@pytest.mark.asyncio
async def test_openrouter_client_retries_on_server_error():
    client = OpenRouterClient(api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    mock_response_503 = httpx.Response(
        status_code=503,
        request=request,
        json={"error": "Service Unavailable"},
    )
    mock_response_200 = httpx.Response(
        status_code=200,
        request=request,
        json={
            "choices": [{"message": {"content": "Hello from model"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        },
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [mock_response_503, mock_response_200]

        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            response = await client.call("system prompt", messages)

            assert response.text == "Hello from model"
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_groq_client_retries_on_rate_limit():
    client = GroqClient(api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_response_429 = httpx.Response(
        status_code=429,
        request=request,
        headers={"Retry-After": "0.1"},
        json={"error": "Rate limit exceeded"},
    )
    mock_response_200 = httpx.Response(
        status_code=200,
        request=request,
        json={
            "choices": [{"message": {"content": "Hello from Groq"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
        },
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [mock_response_429, mock_response_200]

        # Use a short backoff for testing
        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            response = await client.call("system prompt", messages)

            assert response.text == "Hello from Groq"
            assert response.input_tokens == 10
            assert response.output_tokens == 15
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once_with(0.1)


@pytest.mark.asyncio
async def test_groq_client_retries_on_server_error():
    client = GroqClient(api_key="test-key")
    messages = [Message(role="user", content="Hello")]

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    mock_response_503 = httpx.Response(
        status_code=503,
        request=request,
        json={"error": "Service Unavailable"},
    )
    mock_response_200 = httpx.Response(
        status_code=200,
        request=request,
        json={
            "choices": [{"message": {"content": "Hello from Groq"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
        },
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = [mock_response_503, mock_response_200]

        with patch("asyncio.sleep", AsyncMock()) as mock_sleep:
            response = await client.call("system prompt", messages)

            assert response.text == "Hello from Groq"
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once_with(2.0)
