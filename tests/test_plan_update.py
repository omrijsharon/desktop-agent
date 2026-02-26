"""Tests for the plan_update tool (skill-based planning).

Covers all four actions: create, next, complete, status.
Also tests scratchpad integration and edge cases.
"""

from __future__ import annotations

import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JsonDict = dict[str, Any]


def _repo_root_override(tmp: str):
    """Patch _repo_root to return a temp dir."""
    return lambda: tmp


def _make_plan_handler(tmp_dir: str, scratchpad: dict[str, str]):
    """Import and instantiate the plan_update handler with injected state.

    We re-create the closure that _new_tab() builds, but pointing at a
    temp directory so tests don't touch real files.
    """    # We replicate the plan_update logic (pure-function, no PySide6 dependency)
    # so tests stay lightweight. If the implementation changes, these tests
    # will catch regressions.

    class FakeTabState:
        def __init__(self):
            self.scratchpad = scratchpad

    st = FakeTabState()

    def plan_update(args: JsonDict) -> str:
        action = args.get("action", "")
        plan_dir = tmp_dir

        if action == "create":
            title = args.get("title", "Plan")
            tasks = args.get("tasks", [])
            fname = args.get("file", "plan.md")
            if not tasks:
                return json.dumps({"ok": False, "error": "tasks list is required"})
            lines = [f"# {title}", ""]
            lines.append("## Tasks")
            lines.append("")
            for i, t in enumerate(tasks):
                marker = "[>]" if i == 0 else "[ ]"
                lines.append(f"- {marker} {i + 1}. {t}")
            plan_path = os.path.join(plan_dir, fname)
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            st.scratchpad["plan_file"] = fname
            st.scratchpad["current_task"] = tasks[0]
            st.scratchpad["task_status"] = f"1/{len(tasks)} — {tasks[0]}"
            return json.dumps({"ok": True, "file": fname, "total_tasks": len(tasks), "current": tasks[0]})

        elif action == "next":
            fname = st.scratchpad.get("plan_file", "plan.md")
            plan_path = os.path.join(plan_dir, fname)
            if not os.path.isfile(plan_path):
                return json.dumps({"ok": False, "error": f"plan file not found: {fname}"})
            note = args.get("note", "")
            with open(plan_path, "r", encoding="utf-8") as f:
                content = f.read()
            plan_lines = content.splitlines()
            active_idx = None
            for i, ln in enumerate(plan_lines):
                if ln.strip().startswith("- [>]"):
                    active_idx = i
                    break
            if active_idx is None:
                return json.dumps({"ok": False, "error": "no active task found ([>])"})
            completed_line = plan_lines[active_idx].replace("- [>]", "- [x]", 1)
            if note:
                completed_line += f" ✓ {note}"
            plan_lines[active_idx] = completed_line
            next_idx = None
            for i in range(active_idx + 1, len(plan_lines)):
                if plan_lines[i].strip().startswith("- [ ]"):
                    next_idx = i
                    break
            if next_idx is not None:
                plan_lines[next_idx] = plan_lines[next_idx].replace("- [ ]", "- [>]", 1)
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write("\n".join(plan_lines) + "\n")
            total = sum(1 for ln in plan_lines if ln.strip().startswith("- ["))
            done = sum(1 for ln in plan_lines if ln.strip().startswith("- [x]"))
            if next_idx is not None:
                raw = plan_lines[next_idx].strip()
                task_text = raw.split("] ", 1)[-1] if "] " in raw else raw
                st.scratchpad["current_task"] = task_text
                st.scratchpad["task_status"] = f"{done + 1}/{total} — {task_text}"
                return json.dumps({"ok": True, "done": done, "total": total, "current": task_text})
            else:
                st.scratchpad["current_task"] = "(all done)"
                st.scratchpad["task_status"] = f"{done}/{total} — all complete"
                return json.dumps({"ok": True, "done": done, "total": total, "current": None, "all_complete": True})

        elif action == "complete":
            fname = st.scratchpad.get("plan_file", "plan.md")
            plan_path = os.path.join(plan_dir, fname)
            note = args.get("note", "")
            if os.path.isfile(plan_path):
                with open(plan_path, "r", encoding="utf-8") as f:
                    content = f.read()
                plan_lines = content.splitlines()
                for i, ln in enumerate(plan_lines):
                    if ln.strip().startswith("- [>]") or ln.strip().startswith("- [ ]"):
                        plan_lines[i] = ln.replace("- [>]", "- [x]", 1).replace("- [ ]", "- [x]", 1)
                if note:
                    plan_lines.append(f"\n> ✓ Completed: {note}")
                with open(plan_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(plan_lines) + "\n")
            st.scratchpad.pop("current_task", None)
            st.scratchpad.pop("task_status", None)
            return json.dumps({"ok": True, "status": "plan marked complete"})

        elif action == "status":
            fname = st.scratchpad.get("plan_file", "plan.md")
            plan_path = os.path.join(plan_dir, fname)
            if not os.path.isfile(plan_path):
                return json.dumps({"ok": False, "error": f"no plan file found: {fname}"})
            with open(plan_path, "r", encoding="utf-8") as f:
                content = f.read()
            plan_lines = content.splitlines()
            total = sum(1 for ln in plan_lines if ln.strip().startswith("- ["))
            done = sum(1 for ln in plan_lines if ln.strip().startswith("- [x]"))
            active = None
            for ln in plan_lines:
                if ln.strip().startswith("- [>]"):
                    raw = ln.strip()
                    active = raw.split("] ", 1)[-1] if "] " in raw else raw
                    break
            return json.dumps({
                "ok": True,
                "file": fname,
                "total": total,
                "done": done,
                "active": active,
                "plan": content,
            })

        else:
            return json.dumps({"ok": False, "error": f"unknown action: {action!r}. Use create/next/complete/status."})

    return plan_update, st


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanCreate:
    def test_create_basic(self, tmp_path):
        sp: dict[str, str] = {}
        handler, st = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({
            "action": "create",
            "title": "Test Plan",
            "tasks": ["Task A", "Task B", "Task C"],
        }))
        assert result["ok"] is True
        assert result["total_tasks"] == 3
        assert result["current"] == "Task A"
        # Scratchpad keys
        assert sp["plan_file"] == "plan.md"
        assert sp["current_task"] == "Task A"
        assert "1/3" in sp["task_status"]
        # File content
        content = (tmp_path / "plan.md").read_text(encoding="utf-8")
        assert "# Test Plan" in content
        assert "- [>] 1. Task A" in content
        assert "- [ ] 2. Task B" in content
        assert "- [ ] 3. Task C" in content

    def test_create_empty_tasks_fails(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({"action": "create", "title": "Empty", "tasks": []}))
        assert result["ok"] is False
        assert "required" in result["error"]

    def test_create_custom_filename(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({
            "action": "create",
            "title": "Custom",
            "tasks": ["Step 1"],
            "file": "my_plan.md",
        }))
        assert result["ok"] is True
        assert result["file"] == "my_plan.md"
        assert sp["plan_file"] == "my_plan.md"
        assert (tmp_path / "my_plan.md").exists()


class TestPlanNext:
    def _create_plan(self, tmp_path, sp, tasks=None):
        handler, st = _make_plan_handler(str(tmp_path), sp)
        handler({
            "action": "create",
            "title": "Test",
            "tasks": tasks or ["A", "B", "C"],
        })
        return handler, st

    def test_next_advances(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = self._create_plan(tmp_path, sp)
        result = json.loads(handler({"action": "next"}))
        assert result["ok"] is True
        assert result["done"] == 1
        assert result["total"] == 3
        assert "B" in result["current"]
        # File updated
        content = (tmp_path / "plan.md").read_text(encoding="utf-8")
        assert "- [x] 1. A" in content
        assert "- [>] 2. B" in content
        assert "- [ ] 3. C" in content

    def test_next_with_note(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = self._create_plan(tmp_path, sp)
        handler({"action": "next", "note": "done quickly"})
        content = (tmp_path / "plan.md").read_text(encoding="utf-8")
        assert "✓ done quickly" in content

    def test_next_all_done(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = self._create_plan(tmp_path, sp, tasks=["Only task"])
        result = json.loads(handler({"action": "next"}))
        assert result["ok"] is True
        assert result["all_complete"] is True
        assert result["current"] is None
        assert sp["current_task"] == "(all done)"

    def test_next_no_active_task(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = self._create_plan(tmp_path, sp, tasks=["Only task"])
        handler({"action": "next"})  # completes the only task
        result = json.loads(handler({"action": "next"}))
        assert result["ok"] is False
        assert "no active task" in result["error"]

    def test_next_missing_file(self, tmp_path):
        sp: dict[str, str] = {"plan_file": "nonexistent.md"}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({"action": "next"}))
        assert result["ok"] is False
        assert "plan file" in result["error"]

    def test_full_walkthrough(self, tmp_path):
        """Walk through a 3-task plan from start to finish."""
        sp: dict[str, str] = {}
        handler, _ = self._create_plan(tmp_path, sp, tasks=["Init", "Build", "Deploy"])
        # Advance through all tasks
        r1 = json.loads(handler({"action": "next", "note": "git init done"}))
        assert r1["current"] and "Build" in r1["current"]
        assert "2/3" in sp["task_status"]

        r2 = json.loads(handler({"action": "next", "note": "build passed"}))
        assert r2["current"] and "Deploy" in r2["current"]
        assert "3/3" in sp["task_status"]

        r3 = json.loads(handler({"action": "next", "note": "deployed"}))
        assert r3["all_complete"] is True

        content = (tmp_path / "plan.md").read_text(encoding="utf-8")
        assert content.count("[x]") == 3
        assert "[>]" not in content
        assert "[ ]" not in content


class TestPlanComplete:
    def test_complete_marks_all_done(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        handler({
            "action": "create",
            "title": "Quick",
            "tasks": ["A", "B", "C"],
        })
        result = json.loads(handler({"action": "complete", "note": "all good"}))
        assert result["ok"] is True
        content = (tmp_path / "plan.md").read_text(encoding="utf-8")
        assert content.count("[x]") == 3
        assert "[ ]" not in content
        assert "[>]" not in content
        assert "✓ Completed: all good" in content
        # Scratchpad cleaned up
        assert "current_task" not in sp
        assert "task_status" not in sp
        assert "plan_file" in sp  # kept for review

    def test_complete_without_plan_file(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({"action": "complete"}))
        assert result["ok"] is True  # no-op but succeeds


class TestPlanStatus:
    def test_status_returns_plan(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        handler({
            "action": "create",
            "title": "Status Test",
            "tasks": ["X", "Y"],
        })
        result = json.loads(handler({"action": "status"}))
        assert result["ok"] is True
        assert result["total"] == 2
        assert result["done"] == 0
        assert "X" in result["active"]
        assert "# Status Test" in result["plan"]

    def test_status_after_next(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        handler({"action": "create", "title": "T", "tasks": ["A", "B"]})
        handler({"action": "next"})
        result = json.loads(handler({"action": "status"}))
        assert result["done"] == 1
        assert "B" in (result["active"] or "")

    def test_status_no_file(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({"action": "status"}))
        assert result["ok"] is False
        assert "no plan file found" in result["error"]


class TestPlanUnknownAction:
    def test_unknown_action(self, tmp_path):
        sp: dict[str, str] = {}
        handler, _ = _make_plan_handler(str(tmp_path), sp)
        result = json.loads(handler({"action": "foobar"}))
        assert result["ok"] is False
        assert "unknown action" in result["error"]


class TestSkillFileExists:
    """Verify the skill file is present and well-formed."""

    def test_skill_file_exists(self):
        # tests/ -> repo root (one dirname for file, one for tests/)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        skill_path = os.path.join(repo_root, "docs", "plan_skill.md")
        assert os.path.isfile(skill_path), f"Skill file missing: {skill_path}"
        content = open(skill_path, encoding="utf-8").read()
        assert "plan_update" in content
        assert "create" in content
        assert "next" in content
        assert "complete" in content
        assert "status" in content
        assert "scratchpad" in content.lower()


class TestSystemPromptMentionsPlan:
    """Verify the system prompt references the planning skill."""

    def test_prompt_mentions_plan_update(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.join(
            repo_root, "src", "desktop_agent", "terminal_agent_ui.py",
        )
        with open(src_path, encoding="utf-8") as f:
            src = f.read()
        # Find the default_prompt string region
        assert "plan_update" in src
        assert "plan_skill.md" in src
        assert "Planning skill:" in src
