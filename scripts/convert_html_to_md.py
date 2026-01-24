r"""Convert an HTML file to Markdown.

Usage (PowerShell):
  .\.venv\Scripts\python scripts\convert_html_to_md.py --in docs_gpt5_2_prompting_guide.html --out docs\gpt-5-2_prompting_guide.md

This is a best-effort conversion tailored for the OpenAI cookbook HTML export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bs4 import BeautifulSoup  # type: ignore
from markdownify import markdownify as md  # type: ignore


def _extract_main_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Drop scripts/styles/nav noise.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # The cookbook is a Next.js page; the main content is usually under <main>
    # or a large container with prose classes.
    main = soup.find("main")
    if main is not None:
        return str(main)

    # Sometimes the content is wrapped in a div with prose classes.
    prose = soup.select_one("div.prose")
    if prose is not None:
        return str(prose)

    # Fallback: concatenate all articles.
    articles = soup.find_all("article")
    if articles:
        return "\n\n".join(str(a) for a in articles)

    # Last resort: body.
    body = soup.find("body")
    return str(body) if body is not None else html


def convert(html_text: str) -> str:
    main = _extract_main_html(html_text)
    out = md(
        main,
        heading_style="ATX",
        bullets="-",
        strong_em_symbol="**",
        code_language="",
    )

    # Light cleanup.
    lines = [ln.rstrip() for ln in out.splitlines()]
    cleaned: list[str] = []
    blank = 0
    for ln in lines:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                cleaned.append("")
        else:
            blank = 0
            cleaned.append(ln)

    return "\n".join(cleaned).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out)

    html_text = inp.read_text(encoding="utf-8", errors="ignore")
    md_text = convert(html_text)

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
