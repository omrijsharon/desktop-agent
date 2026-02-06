#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from openai import OpenAI

from desktop_agent.tools import (
    ToolRegistry,
    create_and_register_python_tool_spec,
    make_create_and_register_python_tool_handler,
    make_read_file_handler,
    read_file_tool_spec,
    run_responses_with_function_tools,
)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Demo: model can create + register a Python tool during a running loop.")
    ap.add_argument("goal", help="What you want the agent to do (it may create a tool if needed).")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.2-2025-12-11"))
    ap.add_argument("--max-rounds", type=int, default=6)
    args = ap.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    client = OpenAI()

    registry = ToolRegistry(tools={}, handlers={})
    registry.add(tool_spec=read_file_tool_spec(), handler=make_read_file_handler())
    registry.add(
        tool_spec=create_and_register_python_tool_spec(),
        handler=make_create_and_register_python_tool_handler(registry=registry),
    )

    system = (
        "You are an agent that can call function tools.\n"
        "If you cannot complete the goal with existing tools, you may call create_and_register_python_tool to add one.\n"
        "When creating tools, keep code minimal and pure (no file/network/process access).\n"
        "After tools are created, use them to solve the user's goal.\n"
    )

    input_items: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "input_text", "text": system}]},
        {"role": "user", "content": [{"type": "input_text", "text": args.goal}]},
    ]

    resp = run_responses_with_function_tools(
        client=client,
        model=args.model,
        input_items=input_items,
        registry=registry,
        max_rounds=args.max_rounds,
    )

    print(getattr(resp, "output_text", "") or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
