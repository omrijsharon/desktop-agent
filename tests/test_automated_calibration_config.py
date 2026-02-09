from __future__ import annotations

import json
from pathlib import Path


def test_load_run_config_smoke(tmp_path: Path) -> None:
    from desktop_agent.automated_calibration_config import load_run_config  # noqa: WPS433

    cfg_path = tmp_path / "run_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "pi_ap": {"ssid": "PI", "password": "pw"},
                "hotspot": {"ssid": "HS", "password": "pw"},
                "pi_ssh": {"user": "u", "host": "h", "fallback_host": "10.0.0.1", "use_x11": False},
                "remote": {"webapp_cmd": "echo hi", "logs_dir": "~/logs"},
                "local": {"logs_dir": "C:/tmp/logs", "archive_root": "C:/tmp/archive"},
                "analysis": {
                    "analysis_base_dir": "C:/tmp",
                    "model": "gpt-5.2",
                    "system_prompt": "x",
                    "debug_mapping": ["0: a"],
                    "betaflight_snippet": "",
                    "control_params": {},
                    "files": [],
                    "extra_context": "",
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = load_run_config(cfg_path)
    assert cfg.pi_ap.ssid == "PI"
    assert cfg.hotspot.ssid == "HS"
    assert cfg.remote.logs_dir == "~/logs"
