from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any
from pathlib import Path


def _repo_root() -> Path:
    # scripts/check_models.py -> scripts -> repo root
    return Path(__file__).resolve().parents[1]


@dataclass
class Result:
    model: str
    ok: bool
    elapsed_s: float
    mode: str
    dropped_params: list[str]
    error: str | None = None


def _is_computer_use_model(model: str) -> bool:
    m = (model or "").lower()
    return "computer" in m or "computer_use" in m or "computer-use" in m


def _run_one(*, model: str, use_stream: bool) -> Result:
    from desktop_agent.chat_session import ChatConfig, ChatSession  # noqa: WPS433
    from desktop_agent.config import load_config  # noqa: WPS433
    from desktop_agent.model_caps import ModelCapsStore  # noqa: WPS433

    app_cfg = load_config()
    if not (app_cfg.openai_api_key or os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("No OpenAI API key configured (set OPENAI_API_KEY or update src/desktop_agent/config.py).")

    caps_path = (_repo_root() / "chat_history" / "model_caps.json").resolve()

    ccfg = ChatConfig(
        model=model,
        enable_web_search=False,
        enable_file_search=False,
        include_file_search_results=False,
        # Intentionally set these to common defaults; the runner should drop unsupported ones.
        temperature=0.7,
        top_p=0.95,
        max_output_tokens=128,
        max_tool_calls=1,
        tool_base_dir=_repo_root(),
        allow_model_set_system_prompt=False,
        allow_model_create_tools=False,
        allow_model_propose_tools=False,
        allow_read_file=False,
        allow_write_files=False,
        allow_python_sandbox=False,
        allow_model_create_analysis_tools=False,
        allow_submodels=False,
    )

    s = ChatSession(api_key=app_cfg.openai_api_key, config=ccfg)

    t0 = time.monotonic()
    try:
        if use_stream:
            for _ev in s.send_stream("Reply with exactly: ok"):
                pass
        else:
            _ = s.send("Reply with exactly: ok")
        ok = True
        err = None
    except Exception as e:  # noqa: BLE001
        ok = False
        err = f"{type(e).__name__}: {e}"
    elapsed = time.monotonic() - t0

    dropped: list[str] = []
    try:
        store = ModelCapsStore(path=caps_path)
        snap = store.snapshot()
        rec = snap.get(model) if isinstance(snap, dict) else None
        if isinstance(rec, dict):
            up = rec.get("unsupported_params")
            if isinstance(up, list):
                dropped = [str(x) for x in up if str(x).strip()]
    except Exception:
        dropped = []

    return Result(model=model, ok=ok, elapsed_s=elapsed, mode="stream" if use_stream else "nonstream", dropped_params=dropped, error=err)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check which configured models work with this repo's ChatSession.")
    ap.add_argument("--stream", action="store_true", help="Use streaming path (send_stream) instead of send().")
    ap.add_argument("--json", action="store_true", help="Emit JSON lines output.")
    ap.add_argument("--write-caps", action="store_true", help="Ensure chat_history/model_caps.json exists (populated on-the-fly while running).")
    args = ap.parse_args(argv)

    # Ensure repo root is on sys.path.
    sys.path.insert(0, str(_repo_root() / "src"))

    from desktop_agent.config import SUPPORTED_MODELS  # noqa: WPS433

    models = [m for m in SUPPORTED_MODELS if isinstance(m, str) and m.strip()]
    models = [m for m in models if not _is_computer_use_model(m)]

    results: list[Result] = []
    for m in models:
        try:
            r = _run_one(model=m, use_stream=bool(args.stream))
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 2
        results.append(r)
        if not args.json:
            status = "OK" if r.ok else "FAIL"
            dropped = f" dropped={','.join(r.dropped_params)}" if r.dropped_params else ""
            print(f"{status:4}  {r.mode:8}  {r.elapsed_s:6.2f}s  {r.model}{dropped}")
            if r.error:
                print(f"      error: {r.error}")
        else:
            print(json.dumps(r.__dict__, ensure_ascii=False))

    ok_n = sum(1 for r in results if r.ok)
    if bool(args.write_caps):
        out_path = (_repo_root() / "chat_history" / "model_caps.json").resolve()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                out_path.write_text("{}\n", encoding="utf-8")
            if not args.json:
                print(f"Model caps path: {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed to prepare model caps: {type(e).__name__}: {e}", file=sys.stderr)
    if not args.json:
        print(f"\nSummary: {ok_n}/{len(results)} ok (excluded computer-use models).")
    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
