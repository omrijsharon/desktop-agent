"""desktop_agent.main

Entry-point wiring for MVP v0.

Run:
    python -m desktop_agent.main

This starts a PySide6 UI and runs the planner in a worker thread.
"""

from __future__ import annotations

import threading

from .config import load_config
from .controller import WindowsControls
from .executor import Executor, ExecutorConfig
from .llm import LLMConfig, PlannerLLM
from .planner import EventApprovalGate, Planner, PlannerConfig
from .ui import DesktopAgentUI, UIConfig
from .vision import ScreenCapture


def main() -> None:
    cfg = load_config()

    controls = WindowsControls()

    # Vision/LLM/Executor/Planner
    vision = ScreenCapture()

    approval_gate = EventApprovalGate()

    # Placeholder bounds/rate-limit; can be tightened later using vision metrics.
    executor = Executor(
        controls,
        ExecutorConfig(max_actions_per_second=10.0, batch_timeout_s=10.0),
    )

    planner_cfg = PlannerConfig(max_iters=50)

    planner_holder: dict[str, Planner] = {}
    worker_holder: dict[str, threading.Thread] = {}

    def on_run(goal: str, step_mode: bool, model: str) -> None:
        # Prevent multiple runs.
        if worker_holder.get("t") and worker_holder["t"].is_alive():
            return

        ui.set_controls_enabled(running=True)
        ui.post_status(state="PLANNING", high_level="Starting")

        llm = PlannerLLM(config=LLMConfig(model=model))

        planner = Planner(
            vision=vision,
            llm=llm,
            executor=executor,
            config=planner_cfg,
            approval_gate=approval_gate,
            step_mode=step_mode,
            on_status=lambda st: ui.post_status(state=st.state.value, high_level=st.high_level),
            on_message=lambda msg: ui.post_message(msg),
        )
        planner_holder["p"] = planner

        def work() -> None:
            try:
                planner.run(goal=goal)
            except Exception as e:
                ui.post_message(f"Error: {e}")
            finally:
                ui.set_controls_enabled(running=False)

        t = threading.Thread(target=work, daemon=True)
        worker_holder["t"] = t
        t.start()

    def on_stop() -> None:
        p = planner_holder.get("p")
        if p is not None:
            p.stop()
        try:
            executor.stop()
        except Exception:
            pass
        ui.set_controls_enabled(running=False)
        ui.post_status(state="DONE", high_level="Stopped")

    def on_approve() -> None:
        approval_gate.approve()

    ui = DesktopAgentUI(
        config=UIConfig(always_on_top=cfg.always_on_top),
        on_run=on_run,
        on_stop=on_stop,
        on_approve=on_approve,
    )

    if not cfg.openai_api_key:
        ui.post_message("[SYSTEM] FAKE MODE enabled (no OPENAI_API_KEY). Set OPENAI_API_KEY to use the real model.")

    ui.on_close(on_stop)
    ui.show()
    ui.exec()


if __name__ == "__main__":
    main()
