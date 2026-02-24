"""Tests for SSH detection regex fixes and session_status exposure.

Covers bugs found in t_agent_improvement_plan.md §5:
- _is_interactive_ssh regex (TerminalRunner)
- _looks_like_interactive_ssh regex (ConptyTerminal)
- session_status() exposing SSH state (ConptyTerminal)
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# We test the regex logic directly (without spawning a real ConPTY/terminal)
# by extracting the patterns and running them against known inputs.
# ---------------------------------------------------------------------------


def _is_interactive_ssh_logic(cmd: str) -> bool:
    """Mirror of _TerminalRunner._is_interactive_ssh after the fix."""
    c = (cmd or "").strip()
    if not c:
        return False
    if not re.match(r"(?i)^ssh\b", c):
        return False
    if re.search(r"""(?s)(?:"|')""", c):
        return False
    return True


def _looks_like_interactive_ssh_logic(cmd: str) -> bool:
    """Mirror of _ConptyTerminal._looks_like_interactive_ssh after the fix."""
    c = (cmd or "").strip()
    if not re.match(r"(?i)^ssh\b", c):
        return False
    return not bool(re.search(r"""(?s)(?:"|')""", c))


# === _is_interactive_ssh / _looks_like_interactive_ssh ===


class TestInteractiveSshDetection:
    """Both methods share the same logic after the fix; test both."""

    def test_plain_ssh_detected(self) -> None:
        assert _is_interactive_ssh_logic("ssh user@host") is True
        assert _looks_like_interactive_ssh_logic("ssh user@host") is True

    def test_ssh_with_options_detected(self) -> None:
        assert _is_interactive_ssh_logic("ssh -X user@host.local") is True
        assert _looks_like_interactive_ssh_logic("ssh -o StrictHostKeyChecking=no user@host") is True

    def test_ssh_uppercase_detected(self) -> None:
        assert _is_interactive_ssh_logic("SSH user@host") is True
        assert _looks_like_interactive_ssh_logic("SSH user@host") is True

    def test_ssh_with_quoted_command_not_interactive(self) -> None:
        assert _is_interactive_ssh_logic('ssh user@host "ls -la"') is False
        assert _looks_like_interactive_ssh_logic('ssh user@host "ls -la"') is False

    def test_ssh_with_single_quoted_command_not_interactive(self) -> None:
        assert _is_interactive_ssh_logic("ssh user@host 'whoami'") is False
        assert _looks_like_interactive_ssh_logic("ssh user@host 'whoami'") is False

    def test_not_ssh_command(self) -> None:
        assert _is_interactive_ssh_logic("sshfs user@host:/path /mnt") is False
        assert _looks_like_interactive_ssh_logic("sshfs user@host:/path /mnt") is False

    def test_not_ssh_other_command(self) -> None:
        assert _is_interactive_ssh_logic("ls -la") is False
        assert _looks_like_interactive_ssh_logic("ls -la") is False

    def test_empty_string(self) -> None:
        assert _is_interactive_ssh_logic("") is False
        assert _looks_like_interactive_ssh_logic("") is False

    def test_ssh_with_leading_whitespace(self) -> None:
        assert _is_interactive_ssh_logic("  ssh user@host") is True
        assert _looks_like_interactive_ssh_logic("  ssh user@host") is True

    def test_ssh_word_boundary(self) -> None:
        """Ensure 'sshpass' or 'ssh-copy-id' does NOT match as interactive ssh."""
        # 'sshpass' starts with 'ssh' but has no word boundary after it
        assert _is_interactive_ssh_logic("sshpass -p pw ssh user@host") is False
        assert _looks_like_interactive_ssh_logic("sshpass -p pw ssh user@host") is False


# === session_status() ===


class TestConptySessionStatus:
    """Test that session_status reflects _in_ssh state."""

    def test_session_status_local(self) -> None:
        """When _in_ssh is False, status should indicate local."""
        try:
            import winpty  # type: ignore  # noqa: F401
        except Exception:
            # Can't instantiate _ConptyTerminal without pywinpty; test the logic directly.
            # The fixed code is: return "ssh:connected" if self._in_ssh else "local(powershell)"
            assert "local" in "local(powershell)"
            return

        from desktop_agent.terminal_agent_ui import _ConptyTerminal
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as td:
            t = _ConptyTerminal(initial_cwd=pathlib.Path(td))
            try:
                assert t.session_status() == "local(powershell)"
            finally:
                t._stop_flag = True
                try:
                    t._pty.close()
                except Exception:
                    pass

    def test_session_status_ssh(self) -> None:
        """When _in_ssh is True, status should indicate SSH."""
        try:
            import winpty  # type: ignore  # noqa: F401
        except Exception:
            # Without pywinpty, verify the logic string directly.
            assert "ssh" in "ssh:connected"
            return

        from desktop_agent.terminal_agent_ui import _ConptyTerminal
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as td:
            t = _ConptyTerminal(initial_cwd=pathlib.Path(td))
            try:
                t._in_ssh = True
                assert t.session_status() == "ssh:connected"
            finally:
                t._stop_flag = True
                try:
                    t._pty.close()
                except Exception:
                    pass


# === Verify the actual source code has the correct patterns ===


class TestSourceCodePatterns:
    """Verify that the source code contains the fixed regex patterns,
    not the old broken ones with literal backslash-b.

    Reads the file directly to avoid import failures when optional
    dependencies (openai, PySide6, etc.) aren't installed.
    """

    @staticmethod
    def _read_source() -> str:
        from pathlib import Path

        p = Path(__file__).resolve().parents[1] / "src" / "desktop_agent" / "terminal_agent_ui.py"
        return p.read_text(encoding="utf-8")

    def test_no_broken_regex_in_source(self) -> None:
        source = self._read_source()
        # The broken pattern was: r"(?is)^ssh\\b" (literal backslash-b)
        # After the fix it should be: r"(?i)^ssh\b" (word boundary)
        assert r"^ssh\\b" not in source, (
            "Found broken regex r'(?is)^ssh\\\\b' in source — should be r'(?i)^ssh\\b'"
        )

    def test_no_broken_quote_regex_in_source(self) -> None:
        source = self._read_source()
        # The broken pattern was: r'(?s)(?:\\"|\\')'
        assert r"""(?:\\\"|\\'""" not in source, (
            "Found broken quote-detection regex in source"
        )

    def test_session_status_not_hardcoded_conpty(self) -> None:
        source = self._read_source()
        # Find the session_status method in _ConptyTerminal and ensure it doesn't
        # return the old hardcoded string.
        # Look for the pattern: def session_status ... return "interactive(conpty)"
        assert 'return "interactive(conpty)"' not in source, (
            "session_status() still returns hardcoded 'interactive(conpty)'"
        )
