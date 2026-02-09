from __future__ import annotations

import os


def test_automated_calibration_ui_import_smoke() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from desktop_agent.automated_calibration_ui import CalibrationRunnerWindow  # noqa: WPS433
    from desktop_agent.automated_calibration_analysis_ui import CalibrationAnalysisWindow  # noqa: WPS433

    w = CalibrationRunnerWindow()
    assert w is not None
    try:
        w.close()
    except Exception:
        pass

    # Analysis window needs a config path; use the example file so it doesn't require secrets.
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "ui" / "automated_calibration" / "run_config.example0.json"
    w2 = CalibrationAnalysisWindow(run_config_path=p)
    assert w2 is not None
    try:
        w2.close()
    except Exception:
        pass
