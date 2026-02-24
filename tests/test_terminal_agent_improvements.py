"""Tests for improvement plan tasks 1.3, 1.4, 2.2, 2.3, 3.1, 4.1, 4.2, 4.3.

Covers:
- _head_tail_truncate (2.2 smarter stdout truncation)
- _strip_terminal_noise (2.3 strip terminal noise)
- _extract_ssh_target (3.1 SSH context header)
- Stronger continuation prompt text patterns (1.3)
- SSH context header (3.1)
- Self-verification step (4.2)
- Error-retry nudge detection logic (4.1)
- Deliberation prompt in system prompt (4.3)
- max_rounds increased (1.4)
- _compact_conversation (2.1 conversation compaction)
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


# === 2.3 _strip_terminal_noise ===


class TestStripTerminalNoise:
    @staticmethod
    def _func(text: str) -> str:
        from desktop_agent.terminal_agent_ui import _strip_terminal_noise
        return _strip_terminal_noise(text)

    def test_empty(self) -> None:
        assert self._func("") == ""

    def test_strips_progress_bars(self) -> None:
        text = "Starting build\n[========>     ] 60%\nDone"
        result = self._func(text)
        assert "========" not in result
        assert "Starting build" in result
        assert "Done" in result

    def test_strips_ansi_escapes(self) -> None:
        text = "hello \x1b[32mgreen\x1b[0m world"
        result = self._func(text)
        assert "\x1b" not in result
        assert "hello" in result
        assert "green" in result
        assert "world" in result

    def test_strips_powershell_prompt(self) -> None:
        text = "PS C:\\Users\\user> Get-ChildItem\nfile1.txt\nfile2.txt"
        result = self._func(text)
        assert "PS C:\\" not in result
        assert "file1.txt" in result

    def test_strips_pip_progress(self) -> None:
        text = "Downloading requests-2.28.0.tar.gz\nOK"
        result = self._func(text)
        assert "Downloading" not in result
        assert "OK" in result

    def test_collapses_blank_lines(self) -> None:
        text = "line1\n\n\n\n\nline2"
        result = self._func(text)
        assert "\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_preserves_normal_text(self) -> None:
        text = "total 48\ndrwxr-xr-x 2 user user 4096 Feb 25 10:00 .\n-rw-r--r-- 1 user user  123 Feb 25 10:00 file.py"
        result = self._func(text)
        assert "total 48" in result
        assert "file.py" in result


# === 3.1 _extract_ssh_target ===


class TestExtractSshTarget:
    @staticmethod
    def _func(cmd: str) -> str:
        from desktop_agent.terminal_agent_ui import _extract_ssh_target
        return _extract_ssh_target(cmd)

    def test_simple(self) -> None:
        assert self._func("ssh user@host") == "user@host"

    def test_with_flags(self) -> None:
        assert self._func("ssh -X user@host.local") == "user@host.local"

    def test_just_host(self) -> None:
        assert self._func("ssh myserver") == "myserver"

    def test_multiple_flags(self) -> None:
        assert self._func("ssh -o StrictHostKeyChecking=no user@pi.local") == "user@pi.local"

    def test_empty(self) -> None:
        assert self._func("") == ""

    def test_ssh_only(self) -> None:
        # Edge case: just "ssh" with no target
        assert self._func("ssh") == ""

    def test_with_port_flag(self) -> None:
        # -p takes a value argument; parser should skip both -p and 2222
        result = self._func("ssh -p 2222 user@host")
        assert result == "user@host"


# === 3.1 SSH context header in source ===


class TestSshContextHeader:
    @staticmethod
    def _read_source() -> str:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py"
        return p.read_text(encoding="utf-8")

    def test_ssh_context_block_present(self) -> None:
        src = self._read_source()
        assert "[SSH Context]" in src
        assert "[/SSH Context]" in src

    def test_session_status_includes_target(self) -> None:
        src = self._read_source()
        assert "ssh:connected(" in src, "session_status should include target in parens"


# === 4.2 Self-verification step ===


class TestSelfVerificationStep:
    @staticmethod
    def _read_source() -> str:
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py"
        return p.read_text(encoding="utf-8")

    def test_verification_prompt_present(self) -> None:
        src = self._read_source()
        assert "Before finishing, verify your work" in src

    def test_verified_flag_reset(self) -> None:
        src = self._read_source()
        assert "_verified = False" in src, "Verification flag should be reset each run"

    def test_verified_flag_set(self) -> None:
        src = self._read_source()
        assert "_verified = True" in src, "Verification flag should be set after first use"


# === 2.1 _compact_conversation ===


class TestCompactConversation:
    """Test conversation compaction logic."""

    @staticmethod
    def _make_mock_session(conv: list, ctx_tokens: int = 128_000):
        """Create a minimal mock that looks like ChatSession."""

        class _Cfg:
            context_window_tokens = ctx_tokens

        class _Mock:
            cfg = _Cfg()
            _conversation = conv

            def estimate_prompt_tokens(self, *, user_text: str) -> int:
                # Rough estimate: 4 chars per token
                total = sum(
                    len(p.get("text", ""))
                    for item in self._conversation
                    for p in (item.get("content") or [])
                    if isinstance(p, dict)
                )
                return total // 4

        return _Mock()

    @staticmethod
    def _make_turn(role: str, text: str) -> dict:
        return {"role": role, "content": [{"type": "input_text", "text": text}]}

    def test_no_compaction_when_small(self) -> None:
        from desktop_agent.terminal_agent_ui import _compact_conversation

        conv = [
            self._make_turn("user", "hello"),
            self._make_turn("assistant", "hi"),
        ]
        session = self._make_mock_session(conv)
        _compact_conversation(session, keep_pairs=4)
        assert len(conv) == 2  # unchanged

    def test_no_compaction_when_under_threshold(self) -> None:
        from desktop_agent.terminal_agent_ui import _compact_conversation

        conv = [
            self._make_turn("user", f"msg {i}") for i in range(20)
        ]
        # With small messages and 128k context, should be well under 60%
        session = self._make_mock_session(conv, ctx_tokens=128_000)
        _compact_conversation(session, keep_pairs=4)
        assert len(conv) == 20  # unchanged

    def test_compaction_when_over_threshold(self) -> None:
        from desktop_agent.terminal_agent_ui import _compact_conversation

        # Create a conversation that's very large relative to a tiny context window
        big_text = "x" * 2000
        conv = []
        for i in range(20):
            conv.append(self._make_turn("user", f"Q{i}: {big_text}"))
            conv.append(self._make_turn("assistant", f"A{i}: {big_text}"))
        # 40 items, each ~2000 chars = ~10000 tokens; set ctx to 12000 so 60% = 7200 < 10000
        session = self._make_mock_session(conv, ctx_tokens=12_000)
        _compact_conversation(session, keep_pairs=4)
        # Should have: 1 summary + 8 kept items = 9
        assert len(conv) == 9
        # First item should be the summary placeholder
        first_text = conv[0]["content"][0]["text"]
        assert "older conversation turns were dropped" in first_text
        # Last items should be the most recent
        last_text = conv[-1]["content"][0]["text"]
        assert "A19" in last_text

    def test_compaction_keeps_recent_pairs(self) -> None:
        from desktop_agent.terminal_agent_ui import _compact_conversation

        big_text = "y" * 1000
        conv = []
        for i in range(10):
            conv.append(self._make_turn("user", f"U{i} {big_text}"))
            conv.append(self._make_turn("assistant", f"A{i} {big_text}"))
        session = self._make_mock_session(conv, ctx_tokens=8_000)
        _compact_conversation(session, keep_pairs=3)
        # Should keep last 6 items + 1 summary = 7
        assert len(conv) == 7
        # Verify the kept items are the last 6
        assert "U7" in conv[1]["content"][0]["text"]
        assert "A9" in conv[-1]["content"][0]["text"]

    def test_source_has_compact_call(self) -> None:
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py").read_text(encoding="utf-8")
        assert "_compact_conversation" in src
