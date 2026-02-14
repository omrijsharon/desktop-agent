from __future__ import annotations


def test_build_analysis_system_prompt_includes_config_fields() -> None:
    from desktop_agent.automated_calibration_config import AnalysisConfig, build_analysis_system_prompt

    cfg = AnalysisConfig(
        analysis_base_dir=".",
        model="gpt-4.1-mini",
        system_prompt="BASE",
        debug_mapping=["0: a", "1: b"],
        betaflight_snippet="void f() {\n  return;\n}\n",
        control_params={"k": 1},
        files=[],
        extra_context="extra",
    )

    sp = build_analysis_system_prompt(cfg=cfg, saved_analysis_tools=["tool_a", "tool_b"])
    assert "BASE" in sp
    assert "Debug value mapping" in sp
    assert "0: a" in sp
    assert "Control parameters / gains" in sp
    assert '"k": 1' in sp
    assert "Relevant Betaflight snippet" in sp
    assert "void f()" in sp
    assert "Extra context" in sp
    assert "Saved analysis tool names" in sp
    assert "- tool_a" in sp

