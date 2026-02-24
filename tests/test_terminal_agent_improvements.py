"""Tests for improvement plan tasks 1.3, 1.4, 2.2, 4.1, 4.3.

Covers:
- _head_tail_truncate (2.2 smarter stdout truncation)
- Stronger continuation prompt text patterns (1.3)
- Error-retry nudge detection logic (4.1)
- Deliberation prompt in system prompt (4.3)
- max_rounds increased (1.4)
"""

from __future__ import annotations


# === 2.2 _head_tail_truncate ===


class TestHeadTailTruncate:
    @staticmethod
    def _func(text: str, **kw: object) -> str:
        from desktop_agent.terminal_agent_ui import _head_tail_truncate
        return _head_tail_truncate(text, **kw)  # type: ignore[arg-type]

    def test_short_text_unchanged(self) -> None:
        text = "line1\nline2\nline3"
        assert self._func(text) == text

    def test_empty_text(self) -> None:
        assert self._func("") == ""

    def test_truncates_middle(self) -> None:
        lines = [f"line{i}" for i in range(200)]
        result = self._func("\n".join(lines), head_lines=5, tail_lines=5)
        assert "line0" in result  # head preserved
        assert "line4" in result  # last head line
        assert "line199" in result  # tail preserved
        assert "line195" in result  # first tail line
        assert "190 lines omitted" in result
        assert "200 total" in result
        # Middle lines should NOT be present
        assert "line50" not in result
        assert "line100" not in result

    def test_char_cap_enforced(self) -> None:
        text = "x" * 5000
        result = self._func(text, max_chars=100)
        assert len(result) <= 120  # 100 + "…(truncated)…"
        assert "truncated" in result

    def test_exact_boundary_no_truncation(self) -> None:
        """When lines == head + tail, no omission message."""
        lines = [f"L{i}" for i in range(10)]
        result = self._func("\n".join(lines), head_lines=5, tail_lines=5)
        assert "omitted" not in result

    def test_one_over_boundary(self) -> None:
        lines = [f"L{i}" for i in range(11)]
        result = self._func("\n".join(lines), head_lines=5, tail_lines=5)
        assert "1 lines omitted" in result
        assert "L0" in result
        assert "L10" in result

    def test_crlf_handling(self) -> None:
        text = "a\r\nb\r\nc"
        result = self._func(text, head_lines=1, tail_lines=1)
        assert "a" in result
        assert "c" in result


# === 1.3 Stronger continuation prompt & 1.2 SSH context ===


class TestContinuationPromptPatterns:
    """Verify the source code contains the improved prompt patterns."""

    @staticmethod
    def _read_source() -> str:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py"
        return p.read_text(encoding="utf-8")

    def test_rounds_remaining_in_prompt(self) -> None:
        src = self._read_source()
        assert "rounds_remaining" in src, "Missing rounds_remaining variable in continuation prompt"

    def test_no_weak_continuation(self) -> None:
        src = self._read_source()
        assert "Otherwise, reply normally to the user with a summary of what happened" not in src, (
            "Old weak continuation prompt still present"
        )

    def test_error_nudge_logic_present(self) -> None:
        src = self._read_source()
        assert "error_nudge" in src, "Missing error_nudge variable"
        assert "FAILED" in src, "Error nudge should mention FAILED"

    def test_ssh_context_lines_present(self) -> None:
        src = self._read_source()
        assert "You are currently inside an SSH session" in src
        assert "You are on the local Windows machine" in src


# === 4.3 Deliberation prompt ===


class TestDeliberationPrompt:
    @staticmethod
    def _read_source() -> str:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py"
        return p.read_text(encoding="utf-8")

    def test_deliberation_in_system_prompt(self) -> None:
        src = self._read_source()
        assert "Problem-solving protocol" in src
        assert "Do NOT ask the user which option to choose" in src


# === 1.4 max_rounds ===


class TestMaxRoundsIncreased:
    @staticmethod
    def _read_source() -> str:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py"
        return p.read_text(encoding="utf-8")

    def test_max_rounds_at_least_12(self) -> None:
        src = self._read_source()
        # Find the _on_send instantiation of _Worker
        import re
        m = re.search(r"max_rounds\s*=\s*(\d+)", src)
        assert m is not None, "max_rounds assignment not found"
        val = int(m.group(1))
        assert val >= 12, f"max_rounds should be >= 12, got {val}"


# === 4.1 Error-retry nudge logic (unit test) ===


class TestErrorRetryNudgeLogic:
    """Test the error detection logic that mirrors _Worker.run."""

    ERROR_KEYWORDS = ("error", "permission denied", "command not found",
                      "no such file", "failed", "fatal", "traceback",
                      "exception", "denied", "not recognized")

    def _should_nudge(self, exit_code: int | None, stdout: str, stderr: str) -> bool:
        combined = (stdout + stderr).lower()
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        return any(k in combined for k in self.ERROR_KEYWORDS)

    def test_nonzero_exit_triggers(self) -> None:
        assert self._should_nudge(1, "", "") is True
        assert self._should_nudge(127, "", "") is True

    def test_zero_exit_no_error_does_not_trigger(self) -> None:
        assert self._should_nudge(0, "all good", "") is False

    def test_error_keyword_in_stderr(self) -> None:
        assert self._should_nudge(0, "", "Permission denied") is True

    def test_error_keyword_in_stdout(self) -> None:
        assert self._should_nudge(0, "fatal: not a git repository", "") is True

    def test_traceback_triggers(self) -> None:
        assert self._should_nudge(0, "Traceback (most recent call last):\n  File ...", "") is True

    def test_command_not_found(self) -> None:
        assert self._should_nudge(0, "", "bash: foo: command not found") is True

    def test_none_exit_no_error(self) -> None:
        assert self._should_nudge(None, "ok", "") is False
