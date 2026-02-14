from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class WifiConfig:
    ssid: str
    password: str


@dataclass(frozen=True)
class PiSshConfig:
    user: str
    host: str
    fallback_host: str
    use_x11: bool = True


@dataclass(frozen=True)
class RemoteConfig:
    webapp_cmd: str
    logs_dir: str


@dataclass(frozen=True)
class LocalConfig:
    logs_dir: str
    archive_root: str


@dataclass(frozen=True)
class AnalysisConfig:
    analysis_base_dir: str
    model: str
    system_prompt: str
    debug_mapping: list[str]
    betaflight_snippet: str
    control_params: JsonDict
    files: list[str]
    extra_context: str = ""


@dataclass(frozen=True)
class RunConfig:
    pi_ap: WifiConfig
    hotspot: WifiConfig
    pi_ssh: PiSshConfig
    remote: RemoteConfig
    local: LocalConfig
    analysis: AnalysisConfig


def build_analysis_system_prompt(*, cfg: AnalysisConfig, saved_analysis_tools: list[str] | None = None) -> str:
    """Build the system prompt for the automated calibration analysis session.

    Design goal:
    - Everything in the config menu (debug mapping, Betaflight snippet, control params, extra context)
      should live in the system prompt, so user turns can be short (e.g., just file paths).
    """

    debug = "\n".join(cfg.debug_mapping or []).strip()
    bf = (cfg.betaflight_snippet or "").strip()
    extra = (cfg.extra_context or "").strip()
    ctrl = cfg.control_params or {}
    tools = list(saved_analysis_tools or [])

    parts: list[str] = []
    parts.append((cfg.system_prompt or "").strip() or "You are a helpful assistant.")

    parts.append(
        "You are analyzing EKF/PID calibration logs from a drone.\n"
        "When the user provides file paths, start the analysis immediately.\n"
        "Prefer using `python_sandbox` (copy_from_repo/copy_globs) for parsing and plotting.\n"
        "Output concrete recommendations: which parameters to change and suggested values/ranges, with reasoning.\n"
        "If more Betaflight code is required, ask for the specific missing function/section.\n"
    )

    if tools:
        parts.append(
            "Reusable analysis tools may already exist (created earlier via `create_and_register_analysis_tool`).\n"
            "Prefer calling an existing analysis tool before rewriting scripts.\n"
            "Saved analysis tool names:\n- " + "\n- ".join(tools)
        )

    if debug:
        parts.append("Debug value mapping:\n" + debug)

    parts.append("Control parameters / gains (JSON):\n" + json.dumps(ctrl, indent=2, ensure_ascii=False))

    if bf:
        parts.append("Relevant Betaflight snippet(s):\n" + bf)

    if extra:
        parts.append("Extra context:\n" + extra)

    return "\n\n".join(p for p in parts if p.strip()).strip() + "\n"


def _must_str(d: JsonDict, key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return v


def _must_list_str(d: JsonDict, key: str) -> list[str]:
    v = d.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise ValueError(f"{key} must be a list of strings")
    out: list[str] = []
    for x in v:
        if isinstance(x, str) and x.strip():
            out.append(x)
    return out


def load_run_config(path: str | Path) -> RunConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("run_config must be a JSON object")

    pi_ap_d = data.get("pi_ap") or {}
    hotspot_d = data.get("hotspot") or {}
    pi_ssh_d = data.get("pi_ssh") or {}
    remote_d = data.get("remote") or {}
    local_d = data.get("local") or {}
    analysis_d = data.get("analysis") or {}

    if not isinstance(pi_ap_d, dict) or not isinstance(hotspot_d, dict) or not isinstance(pi_ssh_d, dict):
        raise ValueError("pi_ap/hotspot/pi_ssh must be objects")
    if not isinstance(remote_d, dict) or not isinstance(local_d, dict) or not isinstance(analysis_d, dict):
        raise ValueError("remote/local/analysis must be objects")

    # Decode file representation ("\\n") back to real newlines for UI + analysis.
    bf_snip = analysis_d.get("betaflight_snippet")
    if isinstance(bf_snip, str) and "\\n" in bf_snip:
        analysis_d = dict(analysis_d)
        analysis_d["betaflight_snippet"] = bf_snip.replace("\\n", "\n")

    pi_ap = WifiConfig(ssid=_must_str(pi_ap_d, "ssid"), password=_must_str(pi_ap_d, "password"))
    hotspot = WifiConfig(ssid=_must_str(hotspot_d, "ssid"), password=_must_str(hotspot_d, "password"))

    use_x11 = pi_ssh_d.get("use_x11", True)
    pi_ssh = PiSshConfig(
        user=_must_str(pi_ssh_d, "user"),
        host=_must_str(pi_ssh_d, "host"),
        fallback_host=str(pi_ssh_d.get("fallback_host") or "10.42.0.1"),
        use_x11=bool(use_x11),
    )

    remote = RemoteConfig(webapp_cmd=_must_str(remote_d, "webapp_cmd"), logs_dir=_must_str(remote_d, "logs_dir"))
    local = LocalConfig(logs_dir=_must_str(local_d, "logs_dir"), archive_root=_must_str(local_d, "archive_root"))

    analysis = AnalysisConfig(
        analysis_base_dir=_must_str(analysis_d, "analysis_base_dir"),
        model=str(analysis_d.get("model") or "").strip() or "gpt-5.2-2025-12-11",
        system_prompt=_must_str(analysis_d, "system_prompt"),
        debug_mapping=_must_list_str(analysis_d, "debug_mapping"),
        betaflight_snippet=str(analysis_d.get("betaflight_snippet") or ""),
        control_params=(analysis_d.get("control_params") if isinstance(analysis_d.get("control_params"), dict) else {}),
        files=_must_list_str(analysis_d, "files"),
        extra_context=str(analysis_d.get("extra_context") or ""),
    )

    return RunConfig(pi_ap=pi_ap, hotspot=hotspot, pi_ssh=pi_ssh, remote=remote, local=local, analysis=analysis)


def default_run_config_path() -> Path:
    # src/desktop_agent -> src -> repo root -> ui/automated_calibration/run_config.json
    return Path(__file__).resolve().parents[2] / "ui" / "automated_calibration" / "run_config.json"


def example_run_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "ui" / "automated_calibration" / "run_config.example.json"


def save_run_config(path: str | Path, data: JsonDict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Keep `betaflight_snippet` JSON-friendly and easy to diff: represent newlines as
    # literal "\\n" sequences in the file. (We decode back to real newlines on load.)
    d = dict(data) if isinstance(data, dict) else {"data": data}
    analysis = d.get("analysis")
    if isinstance(analysis, dict):
        a2 = dict(analysis)
        bf = a2.get("betaflight_snippet")
        if isinstance(bf, str):
            bf = bf.replace("\r\n", "\n").replace("\n", "\\n")
            a2["betaflight_snippet"] = bf
        d["analysis"] = a2

    text = json.dumps(d, indent=2, ensure_ascii=False).replace("\r\n", "\n") + "\n"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
