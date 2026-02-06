#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI


def _load_dotenv_best_effort() -> None:
    """Best-effort `.env` loader (matches src/desktop_agent/config.py behavior)."""
    try:
        env_path = Path.cwd() / ".env"
        if not env_path.exists() or not env_path.is_file():
            return

        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if not key:
                continue
            os.environ.setdefault(key, val)
    except Exception:
        return


def _extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and isinstance(response.output_text, str):
        return response.output_text
    if isinstance(response, dict) and isinstance(response.get("output_text"), str):
        return response["output_text"]
    return ""


def _extract_sources(response: Any) -> list[str]:
    sources: list[str] = []
    # SDK objects expose .output[] items; web search tool calls include an action.sources list.
    out_items = getattr(response, "output", None)
    if out_items is None and isinstance(response, dict):
        out_items = response.get("output")
    if not isinstance(out_items, list):
        return sources

    for item in out_items:
        if isinstance(item, dict):
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                for s in action.get("sources") or []:
                    if isinstance(s, dict) and isinstance(s.get("url"), str):
                        sources.append(s["url"])
        else:
            # Pydantic model objects: try attribute access
            if getattr(item, "type", None) == "web_search_call":
                action = getattr(item, "action", None)
                action_sources = getattr(action, "sources", None) if action else None
                if isinstance(action_sources, list):
                    for s in action_sources:
                        url = getattr(s, "url", None)
                        if isinstance(url, str):
                            sources.append(url)
    # De-dupe preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for u in sources:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def main(argv: list[str]) -> int:
    _load_dotenv_best_effort()

    ap = argparse.ArgumentParser(description="OpenAI Responses API demo using the built-in web_search tool.")
    ap.add_argument("query", nargs="?", help="The question to ask (e.g. 'What happened in the news today?').")
    ap.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.2-2025-12-11"),
        help="Model name (default: $env:OPENAI_MODEL or gpt-5.2-2025-12-11)",
    )
    ap.add_argument(
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="medium",
        help="How much context to allocate to web search (default: medium)",
    )
    ap.add_argument(
        "--allowed-domain",
        action="append",
        default=[],
        help="Limit search to a domain (repeatable), e.g. --allowed-domain openai.com",
    )
    ap.add_argument("--show-sources", action="store_true", help="Print URLs the model consulted.")
    ap.add_argument("--raw-json", action="store_true", help="Print the full response as JSON.")
    args = ap.parse_args(argv)

    if not args.query:
        ap.error("query is required")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("Set it in your environment and re-run.", file=sys.stderr)
        return 2

    tools: list[dict[str, Any]] = [
        {
            "type": "web_search",
            "search_context_size": args.search_context_size,
        }
    ]
    if args.allowed_domain:
        tools[0]["filters"] = {"allowed_domains": args.allowed_domain}

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        tools=tools,
        input=args.query,
    )

    if args.raw_json:
        # The python SDK response object is a Pydantic model with .to_json().
        if hasattr(response, "to_json"):
            print(response.to_json())
        else:
            print(json.dumps(response, indent=2))
        return 0

    text = _extract_output_text(response).strip()
    if text:
        print(text)
    else:
        print("(No output_text found; run with --raw-json to inspect the full response.)")

    if args.show_sources:
        sources = _extract_sources(response)
        if sources:
            print("\nSources:")
            for u in sources:
                print(f"- {u}")
        else:
            print("\nSources: (none found in response output items)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
