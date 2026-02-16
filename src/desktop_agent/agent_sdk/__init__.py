"""desktop_agent.agent_sdk

Experimental integration layer for the OpenAI Agents SDK.

This package is intentionally small: the legacy `ChatSession` remains the default
engine for the UI. The Agents SDK path is provided behind a toggle so we can
incrementally migrate features (notably MCP tool servers) without destabilizing
the app.
"""

from __future__ import annotations

from .session import AgentsSdkSession

__all__ = ["AgentsSdkSession"]

