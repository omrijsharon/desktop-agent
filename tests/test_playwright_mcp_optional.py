from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(os.environ.get("RUN_PLAYWRIGHT_MCP") != "1", reason="set RUN_PLAYWRIGHT_MCP=1 to run")
def test_playwright_mcp_can_list_tools() -> None:
    # This is intentionally optional: it depends on Node/npm availability and may download packages/browsers.
    import asyncio

    from agents.mcp import MCPServerStdio, MCPServerStdioParams

    async def run() -> list[str]:
        params = MCPServerStdioParams(command="cmd.exe", args=["/c", "npx", "-y", "@playwright/mcp@latest", "--headless"])
        srv = MCPServerStdio(params, client_session_timeout_seconds=60)
        async with srv:
            tools = await srv.list_tools()
            return sorted([t.name for t in tools])

    names = asyncio.run(run())
    assert "browser_navigate" in names
    assert "browser_take_screenshot" in names

