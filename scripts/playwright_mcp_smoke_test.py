from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Deterministic smoke test for Playwright MCP integration.")
    ap.add_argument("--url", default="https://example.com", help="URL to navigate to (default: https://example.com)")
    ap.add_argument("--headless", action="store_true", default=True, help="Run headless (default: true)")
    ap.add_argument("--headed", dest="headless", action="store_false", help="Run headed (shows a browser window)")
    ap.add_argument("--skip-install", action="store_true", help="Skip browser_install")
    ap.add_argument("--timeout-s", type=float, default=180.0, help="Per-call timeout (seconds)")
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from desktop_agent.tools import make_playwright_browser_handler  # noqa: WPS433

    out_dir = (repo_root / "chat_history" / "browser_smoke").resolve()
    cmd = ["cmd.exe", "/c", "npx", "-y", "@playwright/mcp@latest"]
    if args.headless:
        cmd += ["--headless"]

    handler, shutdown = make_playwright_browser_handler(
        cmd=cmd,
        repo_root=repo_root,
        image_out_dir=out_dir,
        startup_timeout_s=60.0,
        call_timeout_s=float(args.timeout_s),
        auto_install=not bool(args.skip_install),
    )

    def call(action: str, params: dict) -> dict:
        s = handler({"action": action, "params": params})
        try:
            d = json.loads(s)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Non-JSON tool output for {action}: {s[:2000]}") from e
        if not isinstance(d, dict):
            raise RuntimeError(f"Unexpected tool output type for {action}: {type(d).__name__}")
        if not d.get("ok", False):
            raise RuntimeError(f"{action} failed: {d}")
        return d

    try:
        if not args.skip_install:
            call("browser_install", {})
        call("browser_tabs", {"action": "new"})
        call("browser_navigate", {"url": args.url})
        shot = call("browser_take_screenshot", {"fullPage": True})
    finally:
        shutdown()

    imgs = shot.get("image_paths") if isinstance(shot, dict) else None
    if not isinstance(imgs, list) or not any(isinstance(p, str) and p.strip() for p in imgs):
        print("OK, but no image_paths returned:", shot)
        return 2

    # Validate files exist.
    ok = True
    for p in imgs:
        if not isinstance(p, str) or not p.strip():
            continue
        abs_path = (repo_root / p).resolve()
        if not abs_path.exists():
            ok = False
            print("MISSING:", abs_path)
        else:
            print("WROTE:", abs_path)

    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

