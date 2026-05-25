"""API call tool — makes outbound HTTP requests through a domain allowlist.

All outbound requests are filtered against a configurable domain allowlist
to prevent data exfiltration.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 50 * 1024  # 50KB
DEFAULT_TIMEOUT = 30
VALID_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


class APICallTool(BaseTool):
    """Make outbound HTTP requests through a domain allowlist.

    Filters all requests against a configurable allowlist to prevent
    data exfiltration. Response bodies are truncated to 50KB.
    """

    name = "api_call"
    description = (
        "Make HTTP requests to external APIs. All requests are filtered through "
        "a domain allowlist. Supports GET, POST, PUT, PATCH, DELETE methods. "
        "Response bodies are truncated to 50KB."
    )

    def __init__(self, domain_allowlist: list[str] | None = None) -> None:
        self._domain_allowlist = set(domain_allowlist) if domain_allowlist else None

    def _is_domain_allowed(self, url: str) -> bool:
        """Check if the URL's domain is in the allowlist."""
        if self._domain_allowlist is None:
            return True  # No allowlist = all domains allowed

        parsed = urlparse(url)
        domain = parsed.hostname or ""

        # Check exact match and suffix match (e.g., *.example.com)
        for allowed in self._domain_allowlist:
            if domain == allowed or domain.endswith(f".{allowed}"):
                return True
        return False

    def validate_args(self, args: dict[str, Any]) -> str | None:
        method = args.get("method")
        if not method:
            return "Missing required argument: 'method'"
        if method.upper() not in VALID_METHODS:
            return f"Invalid method '{method}'. Must be one of: {VALID_METHODS}"

        url = args.get("url")
        if not url:
            return "Missing required argument: 'url'"
        if not isinstance(url, str):
            return "'url' must be a string"

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"URL scheme must be http or https, got '{parsed.scheme}'"

        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        method: str = kwargs["method"].upper()
        url: str = kwargs["url"]
        headers: dict[str, str] = kwargs.get("headers", {})
        body: str | dict | None = kwargs.get("body")
        timeout: int = min(kwargs.get("timeout", DEFAULT_TIMEOUT), 120)

        # Domain allowlist check
        if not self._is_domain_allowed(url):
            parsed = urlparse(url)
            return ToolOutput(
                success=False,
                error=(
                    f"Domain '{parsed.hostname}' is not in the allowlist. "
                    f"Allowed domains: {self._domain_allowlist}"
                ),
            )

        try:
            import httpx

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, dict) else None,
                    content=body if isinstance(body, str) else None,
                )

                response_text = response.text[:MAX_RESPONSE_BYTES]
                if len(response.text) > MAX_RESPONSE_BYTES:
                    response_text += "\n... (truncated)"

                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response_text,
                }

                return ToolOutput(
                    success=(200 <= response.status_code < 400),
                    result=result,
                    error=(
                        f"HTTP {response.status_code}" if response.status_code >= 400 else None
                    ),
                )

        except Exception as exc:
            logger.error(
                "api_call_error",
                extra={
                    "method": method,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return ToolOutput(
                success=False,
                error=f"HTTP request failed: {type(exc).__name__}: {exc}",
            )

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        "description": "HTTP method to use",
                    },
                    "url": {
                        "type": "string",
                        "description": "Full URL to request",
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers as key-value pairs",
                    },
                    "body": {
                        "type": "string",
                        "description": "Request body (string or JSON object)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 30, max: 120)",
                    },
                },
                "required": ["method", "url"],
            },
        }
