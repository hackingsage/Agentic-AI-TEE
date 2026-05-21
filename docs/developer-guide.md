# Developer Guide

## How to Extend Enclave

### Adding a New Tool

1. Create a new file in `enclave/tools/`:

```python
# enclave/tools/my_tool.py
from __future__ import annotations
import logging
from typing import Any
from enclave.agent.models import ToolOutput
from enclave.tools.base import BaseTool

logger = logging.getLogger(__name__)

class MyTool(BaseTool):
    name = "my_tool"  # Must be snake_case, unique
    description = "Description shown to the LLM in the system prompt."

    def validate_args(self, args: dict[str, Any]) -> str | None:
        if not args.get("required_arg"):
            return "Missing required argument: 'required_arg'"
        return None

    async def run(self, **kwargs: Any) -> ToolOutput:
        try:
            result = do_something(kwargs["required_arg"])
            return ToolOutput(success=True, result=result)
        except Exception as exc:
            return ToolOutput(success=False, error=str(exc))

    def schema_xml(self) -> str:
        return """<tool name="my_tool">
  <description>Your tool description here.</description>
  <args>
    <arg name="required_arg" type="string" required="true">Description</arg>
  </args>
</tool>"""
```

2. Register it in the controller setup:

```python
registry = ToolRegistry()
registry.register(MyTool())
```

3. Write tests in `tests/test_tools.py`.

### Running the Development Server

```bash
source .venv/bin/activate
uvicorn host.api.main:app --reload --port 8000
```

### Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v              # All tests
pytest tests/ -k "test_tools" # Just tool tests
pytest tests/ --cov=enclave   # With coverage
```

### Code Quality Checks

```bash
ruff check enclave/ host/     # Lint
mypy enclave/ host/           # Type check
bandit -r enclave/ host/ -ll  # Security scan
```

### Project Conventions

- **All async**: No blocking I/O in the event loop
- **Dataclasses**: All data structures, never raw dicts between components
- **Structured logging**: `logger.info("event_name", extra={...})` — never `print()`
- **Type annotations**: Every function signature
- **Never log secrets**: Only task_id, step_number, token counts, latency
