"""Browser automation tool — headless Chromium via Playwright.

Screenshots returned as base64 PNG. Falls back to a mock in test environments
where Playwright is not available.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

VALID_OPERATIONS = {"navigate", "screenshot", "click", "type_text", "get_text", "get_html"}


class BrowserTool(BaseTool):
    """Browser automation using headless Chromium.

    Supports navigation, screenshots, clicking, typing, and text extraction.
    Uses Playwright when available; falls back to mock for testing.
    """

    name = "browser"
    description = (
        "Automate a headless web browser. Can navigate to URLs, take screenshots, "
        "click elements, type text into fields, and extract page text. Screenshots "
        "are returned as base64-encoded PNG. Use CSS selectors for element targeting."
    )

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._browser: Any = None
        self._page: Any = None
        self._playwright: Any = None

    async def _ensure_browser(self) -> None:
        """Lazily initialize the browser on first use."""
        if self._page is not None:
            return

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self._headless)
            self._page = await self._browser.new_page()
            logger.info("browser_initialized", extra={"headless": self._headless})
        except ImportError:
            logger.warning("playwright_not_available", extra={"fallback": "mock"})
            # Create a minimal mock for testing
            self._page = _MockPage()

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._page = None
        self._playwright = None

    def validate_args(self, args: dict[str, Any]) -> str | None:
        operation = args.get("operation")
        if not operation:
            return "Missing required argument: 'operation'"
        if operation not in VALID_OPERATIONS:
            return f"Invalid operation '{operation}'. Must be one of: {VALID_OPERATIONS}"
        if operation == "navigate" and not args.get("url"):
            return "Missing required argument: 'url' for navigate operation"
        if operation in ("click", "type_text") and not args.get("selector"):
            return "Missing required argument: 'selector'"
        if operation == "type_text" and not args.get("text"):
            return "Missing required argument: 'text'"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        operation: str = kwargs["operation"]
        await self._ensure_browser()

        try:
            if operation == "navigate":
                return await self._navigate(kwargs["url"])
            elif operation == "screenshot":
                return await self._screenshot()
            elif operation == "click":
                return await self._click(kwargs["selector"])
            elif operation == "type_text":
                return await self._type_text(kwargs["selector"], kwargs["text"])
            elif operation == "get_text":
                selector = kwargs.get("selector")
                return await self._get_text(selector)
            elif operation == "get_html":
                return await self._get_html()
            else:
                return ToolOutput(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            logger.error(
                "browser_error",
                extra={"operation": operation, "error": str(exc)},
            )
            return ToolOutput(
                success=False,
                error=f"Browser operation '{operation}' failed: {exc}",
            )

    async def _navigate(self, url: str) -> ToolOutput:
        await self._page.goto(url, wait_until="domcontentloaded")
        title = await self._page.title()
        return ToolOutput(success=True, result=f"Navigated to '{url}'. Page title: '{title}'")

    async def _screenshot(self) -> ToolOutput:
        screenshot_bytes = await self._page.screenshot(type="png")
        b64 = base64.b64encode(screenshot_bytes).decode("ascii")
        return ToolOutput(success=True, result=f"data:image/png;base64,{b64}")

    async def _click(self, selector: str) -> ToolOutput:
        await self._page.click(selector)
        return ToolOutput(success=True, result=f"Clicked element: '{selector}'")

    async def _type_text(self, selector: str, text: str) -> ToolOutput:
        await self._page.fill(selector, text)
        return ToolOutput(success=True, result=f"Typed text into '{selector}'")

    async def _get_text(self, selector: str | None = None) -> ToolOutput:
        if selector:
            element = await self._page.query_selector(selector)
            if element is None:
                return ToolOutput(success=False, error=f"Element not found: '{selector}'")
            text = await element.text_content()
        else:
            text = await self._page.text_content("body")
        # Truncate long text
        if text and len(text) > 10000:
            text = text[:10000] + "\n... (truncated)"
        return ToolOutput(success=True, result=text or "(empty)")

    async def _get_html(self) -> ToolOutput:
        html = await self._page.content()
        if len(html) > 50000:
            html = html[:50000] + "\n... (truncated)"
        return ToolOutput(success=True, result=html)

    def schema_xml(self) -> str:
        return """<tool name="browser">
  <description>Automate a headless web browser. Navigate, screenshot, click, type, and extract text from web pages.</description>
  <args>
    <arg name="operation" type="string" required="true">One of: "navigate", "screenshot", "click", "type_text", "get_text", "get_html"</arg>
    <arg name="url" type="string" required="false">URL to navigate to (required for "navigate")</arg>
    <arg name="selector" type="string" required="false">CSS selector for element targeting (required for "click", "type_text")</arg>
    <arg name="text" type="string" required="false">Text to type (required for "type_text")</arg>
  </args>
</tool>"""


class _MockPage:
    """Minimal mock for testing when Playwright is not available."""

    def __init__(self) -> None:
        self._url = "about:blank"
        self._title = "Mock Page"
        self._content = "<html><body>Mock page content</body></html>"

    async def goto(self, url: str, **kwargs: Any) -> None:
        self._url = url
        self._title = f"Mock: {url}"

    async def title(self) -> str:
        return self._title

    async def screenshot(self, **kwargs: Any) -> bytes:
        # Return a minimal valid 1x1 PNG
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
            "2mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

    async def click(self, selector: str) -> None:
        pass

    async def fill(self, selector: str, text: str) -> None:
        pass

    async def query_selector(self, selector: str) -> Any:
        return _MockElement()

    async def text_content(self, selector: str = "body") -> str:
        return "Mock page content"

    async def content(self) -> str:
        return self._content


class _MockElement:
    async def text_content(self) -> str:
        return "Mock element text"
