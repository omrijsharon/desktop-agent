# Terminal Agent – Architecture Review & Improvement Plan

## 1. Architecture Overview

### Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TerminalAgentWindow (Qt)                        │
│  ┌───────────────────────┐   ┌────────────────────────────────────┐ │
│  │  Chat Pane (left)     │   │  Terminal Pane (right, tabbed)     │ │
│  │  - chat bubbles       │   │  - ConPTY interactive PTY          │ │
│  │  - chat input         │   │  - manual command input            │ │
│  │  - Send/Pause/Stop    │   │  - tab per agent (Main, T2, …)    │ │
│  └───────────────────────┘   └────────────────────────────────────┘ │
│                                                                     │
│  _TabState (per tab):  session, terminal, worker, UI widgets        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │         _Worker (QThread)    │  ← the agent loop
            │                              │
            │  for round in max_rounds:    │
            │    1. send_stream(pending)   │──► ChatSession.send_stream()
            │    2. extract <Terminal>     │        │
            │    3. execute blocks         │──► _ConptyTerminal.send_and_collect()
            │    4. format results         │        │
            │    5. set pending_user =     │        │
            │       <TerminalResponse>     │        │
            └──────────────────────────────┘        │
                                                    ▼
            ┌──────────────────────────────────────────────┐
            │           ChatSession                         │
            │  - _conversation: list[JsonDict]  (history)   │
            │  - _system_prompt: str                        │
            │  - registry: ToolRegistry (function tools)    │
            │  - send_stream() → yields deltas              │
            │    calls run_responses_with_function_tools_    │
            │    stream() in tools.py                       │
            └──────────────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │  run_responses_with_         │
            │  function_tools_stream()     │
            │  (tools.py)                  │
            │                              │
            │  for round in max_rounds:    │  ← up to 12 inner rounds
            │    responses.create(stream)  │──► OpenAI API
            │    process tool calls        │
            │    append function_call_     │
            │    output to input_list      │
            └──────────────────────────────┘
```

### How the Model Interacts with the Terminal

1. **User sends a message** → `_Worker.run()` starts, sets `pending_user = user_text`.
2. **Outer loop** (up to `max_rounds=6`):
   a. Before each round, `_sanitize_terminal_blocks_in_conversation` and `_sanitize_terminal_responses_in_conversation` trim old `<Terminal>` / `<TerminalResponse>` blocks from history to only keep the latest.
   b. `session.send_stream(pending_user)` is called. Inside, `ChatSession` appends the user message to `_conversation` and calls the OpenAI Responses API (streaming).
   c. The Responses API may invoke **function tools** (handled inside `run_responses_with_function_tools_stream` for up to 12 inner tool-call rounds): `wait`, `ssh_read_file`, `ssh_write_file`, `ssh_replace_line`, `peer_agent_ask`, `peer_terminal_run`, `web_search`, `read_file`, `write_file`, `python_sandbox`, etc.
   d. The assistant response text is streamed back to the UI.
3. **Terminal block extraction**: after the full response is received, `extract_terminal_blocks(full_raw)` finds any `<Terminal>…</Terminal>` blocks.
4. **Execution**: each block is sent to `terminal.send_and_collect()` which types it into the ConPTY PowerShell (or into an active SSH session if `_in_ssh` is true).
5. **Result formatting**: stdout/stderr are collected (truncated to 6 KB / 2 KB), formatted as a `<TerminalResponse>`, and set as `pending_user` for the next round.
6. **Loop continues** until the model produces no `<Terminal>` blocks or `max_rounds` is exhausted.

### Available Tools (registered on ChatSession.registry)

| Tool | Transport | Description |
|------|-----------|-------------|
| `read_file` | function_call | Read a local file (repo-scoped) |
| `write_file` / `append_file` | function_call | Write/append local files |
| `python_sandbox` | function_call | Run Python code in a subprocess sandbox |
| `render_plot` | function_call | Plot generation (wraps python_sandbox) |
| `web_search` | built-in | OpenAI web search |
| `web_fetch` | function_call | HTTP GET + readability extraction |
| `playwright_browser` | function_call | Playwright MCP browser automation |
| `wait` | function_call | Sleep 0–5 s |
| `ssh_read_file` | function_call | Read remote file via one-shot SSH |
| `ssh_write_file` | function_call | Write remote file via one-shot SSH |
| `ssh_replace_line` | function_call | Exact-line replace in remote file via SSH |
| `peer_agent_ask` | function_call | Ask another agent tab for help |
| `peer_terminal_run` | function_call | Run a command in another agent's terminal |
| `create_terminal_agent` | function_call | Main-only: spawn a new agent tab |
| `set_system_prompt` | function_call | Model self-modifies its system prompt |
| `TerminalExec` | text block | `<Terminal>…</Terminal>` in assistant output |

---

## 2. Root-Cause Analysis of Current Issues

### Issue 1: Token Waste

**Symptoms**: Token usage climbs rapidly; the model re-reads large terminal outputs.

**Root causes**:
- **Every terminal round injects the full `<TerminalResponse>` as a new user message.** Each round adds ~200 lines of stdout to history. Even though sanitization runs, it only keeps the _latest_ block – but the conversation still accumulates all the non-terminal text (explanations, plans) from every round.
- **The system prompt is ~1,200 tokens** of boilerplate that never changes, repeated every API call.
- **stdout truncation is too generous**: 6,000 chars of stdout per command × multiple commands × multiple rounds adds up.
- **No summarization of prior rounds.** Round 3 still carries the full assistant text from rounds 1 and 2 even though only the most recent terminal output matters.
- **Conversation history is never compacted.** Long multi-round sessions accumulate dozens of user/assistant turns. There's no mechanism to summarize or drop stale turns.

### Issue 2: SSH Context Confusion (local vs. remote)

**Symptoms**: The model doesn't know if it's on the Pi or the local machine; runs wrong commands.

**Root causes**:
- **`session_status()` returns a generic string** like `"interactive(conpty)"` even when inside SSH. The ConPTY terminal tracks `_in_ssh` internally but `session_status()` doesn't expose it – it always returns the same string.
- **No explicit "you are now on the Pi" / "you are now local" context** is injected when SSH state changes. The model has to _infer_ from terminal output (prompts, hostnames) whether it's remote.
- **`_in_ssh` heuristic is fragile.** It looks for `"Last login:"` banners or Linux-style prompts. If the SSH handshake takes longer than `idle_ms` (450 ms), the capture returns before the remote prompt appears, and `_in_ssh` never flips to `True`.
- **`_looks_like_interactive_ssh` regex is BROKEN**: the regex `r"(?is)^ssh\\b"` uses a literal backslash-b instead of the regex word boundary `\b`. This means the method _never matches_ any SSH command. The SSH detection only works because `_maybe_update_ssh_state_from_text` heuristic catches the login banner later.
- **`_is_interactive_ssh` in `_TerminalRunner` has the same broken regex.**
- **The `Terminal state:` line in `pending_user` just says `"interactive(conpty)"` or `"unknown"`,** giving the model zero signal about local-vs-remote.

### Issue 3: SSH File Editing Failures

**Symptoms**: The model fails to edit files on the remote Pi.

**Root causes**:
- **`ssh_read_file` / `ssh_write_file` / `ssh_replace_line` use `BatchMode=yes`**, which means they fail silently if SSH key auth isn't set up (no password prompt). But the interactive SSH session in the terminal _does_ have an authenticated connection. The tools open a _separate_ SSH connection each time.
- **`ssh_replace_line` does exact-line matching**, which is brittle. If the model gets a single character wrong (whitespace, tab vs spaces), it silently fails with `no_match`.
- **No `ssh_patch_file` or `ssh_sed` tool** for regex-based edits, which is what users actually need for config files.
- **The model often tries to use `nano` or `vim` inside `<Terminal>` blocks** instead of the SSH tools. Since ConPTY can't drive interactive TUI editors reliably, this fails. The system prompt mentions this but the model forgets.
- **No `scp` / `sftp` tool** for binary file transfers.

### Issue 4: Agent Stops Prematurely / Waits for User Input

**Symptoms**: The agent stops mid-task and asks the user to choose, even when it could decide itself.

**Root causes**:
- **The continuation prompt is weak.** `"If you need to run more commands, respond with more <Terminal>...</Terminal> blocks. Otherwise, reply normally to the user with a summary."` – the "otherwise, reply normally" clause gives the model an easy escape to stop and summarize after _every_ round, even if the task is clearly not done.
- **No task-completion detection.** The loop exits as soon as the model produces no `<Terminal>` blocks, even if the last output shows an error that the model should fix.
- **`max_rounds=6` is often not enough** for complex multi-step SSH tasks (connect → diagnose → install → configure → verify → fix errors → re-verify).
- **No "keep going" nudge.** When the model stops because it hit an error or is unsure, there's no mechanism to tell it "you still have budget, keep trying."
- **The model sees truncated output** (200 lines, 6 KB) and may miss error messages at the beginning of long outputs, making it think the command succeeded.

---

## 3. Improvement Plan

### Phase 1: Critical Fixes (Low effort, high impact)

#### - [x] 1.1 Fix broken SSH detection regex ✅ 2026-02-25T10:00Z

**Files**: `terminal_agent_ui.py`  
**What**: Fix `r"(?is)^ssh\\b"` → `r"(?is)^ssh\b"` in both `_is_interactive_ssh` and `_looks_like_interactive_ssh`. Same fix for the quote-detection regex `r"(?s)(?:\\\"|\\')"` → `r'(?s)(?:"|\')'`.  
**Impact**: SSH sessions will be correctly detected at start, not just via banner heuristic.

#### - [x] 1.2 Expose SSH state in `Terminal state:` context ✅ 2026-02-25T10:00Z (session_status + SSH context line in pending_user)

**Files**: `terminal_agent_ui.py` (`_ConptyTerminal.session_status`, `_Worker.run`)  
**What**:
- `_ConptyTerminal.session_status()` should return `"ssh:connected"` when `_in_ssh` is `True`, and `"local(powershell)"` otherwise.
- When constructing `pending_user`, inject a clear line: `"You are currently inside an SSH session on the remote host."` or `"You are on the local Windows machine."`.  
**Impact**: The model will always know where it is.

#### - [x] 1.3 Stronger continuation prompt ✅ 2026-02-25T11:00Z

**Files**: `terminal_agent_ui.py` (`_Worker.run`)  
**What**: Replace the weak continuation prompt with:
```
Terminal state: {term_state}

<TerminalResponse>
{full_resp}
</TerminalResponse>

IMPORTANT: You have {rounds_remaining} rounds remaining.
- If the task is NOT yet complete, continue by emitting more <Terminal>...</Terminal> blocks.
- If you encountered an error, try to fix it instead of stopping.
- Do NOT ask the user to choose between options – pick the best approach yourself and proceed.
- Only reply with a summary to the user when you are confident the task is fully done.
```
**Impact**: Dramatically reduces premature stops and "which option do you prefer?" questions.

#### - [x] 1.4 Increase `max_rounds` for SSH-heavy workflows ✅ 2026-02-25T11:00Z (6→12)

**Files**: `terminal_agent_ui.py` (`_on_send`)  
**What**: Increase `max_rounds` from 6 to 12 (or make it configurable via env var / system prompt).  
**Impact**: Complex SSH tasks can run to completion without the agent running out of budget.

### Phase 2: Token Efficiency (Medium effort, high impact)

#### - [ ] 2.1 Aggressive conversation compaction

**Files**: `terminal_agent_ui.py` (`_Worker.run`), potentially `chat_session.py`  
**What**:
- Before each `send_stream`, count estimated prompt tokens. If above a threshold (e.g., 60% of context window), summarize older turns.
- Implementation: take all conversation items except the last 2 user/assistant pairs and the system prompt. Feed them to a fast/cheap model with "Summarize the conversation so far in ≤500 tokens, preserving all key facts, file paths, decisions, and current state." Replace those items with a single `[summary]` user message.
- As a simpler first step: just drop all but the last N turns (e.g., keep last 4 user/assistant pairs + system prompt). This alone would cut tokens by 50%+ in long sessions.  
**Impact**: 2–5× token reduction in multi-round sessions.

#### - [x] 2.2 Smarter stdout truncation ✅ 2026-02-25T11:00Z (head+tail with _head_tail_truncate)

**Files**: `terminal_agent_ui.py` (`_Worker.run`)  
**What**:
- Keep the **first 40 lines** + **last 80 lines** (head + tail) instead of just the last 200 lines. Errors often appear at the top (compilation) or bottom (runtime). The middle is usually progress spam.
- Reduce per-command stdout cap from 6,000 → 3,000 chars.
- For commands that produce very large output, add a hint: `"(output truncated; {total_lines} total lines, showing first 40 + last 80)"`.  
**Impact**: Model gets more useful signal per token spent.

#### - [ ] 2.3 Strip terminal noise from history

**Files**: `terminal_agent_ui.py`  
**What**:
- Before injecting `<TerminalResponse>`, strip PowerShell prompt lines, ANSI leftovers, and blank lines.
- Strip the echoed command itself (ConPTY echoes input; the model already knows what it typed).
- Strip common noise patterns: progress bars (`[=====>    ] 45%`), pip download progress, apt progress, etc.  
**Impact**: 10–30% reduction in terminal response token size.

#### - [ ] 2.4 Don't repeat the system prompt in every round's context

**Files**: `terminal_agent_ui.py` (`_make_session`)  
**What**: The system prompt is already passed via `instructions=` in the API call (not as a conversation item). Verify it's not duplicated. Trim the system prompt itself – remove the SSH example (`ssh -X omrijsharon@omrijsharon.local`) and other user-specific content from the _default_ prompt; put those in the prompt override instead.  
**Impact**: ~200 tokens saved per API call.

### Phase 3: SSH Robustness (Medium effort, high impact)

#### - [ ] 3.1 Persistent SSH context header

**Files**: `terminal_agent_ui.py`  
**What**: When the terminal is in SSH mode, prepend a structured context block to every `pending_user`:
```
[SSH Context]
Host: omrijsharon@omrijsharon.local
Status: connected
Remote OS: Linux (Raspberry Pi)
Remote shell: bash
Remote CWD: /home/omrijsharon
[/SSH Context]
```
Populate this by parsing the SSH command and periodically running `whoami && hostname && pwd` in the background (or after each command). Cache the result.  
**Impact**: Model always knows exactly where it is and what the remote environment looks like.

#### - [ ] 3.2 Add `ssh_run_command` tool (one-shot remote execution)

**Files**: `terminal_agent_ui.py`  
**What**: A new tool that runs a single command on the remote host via a _fresh_ SSH connection (like `ssh_read_file` but for arbitrary commands). Returns stdout/stderr/exit_code. This is more reliable than typing into the ConPTY session for commands that don't need interactivity.
```json
{
  "name": "ssh_run_command",
  "parameters": {
    "host": "string",
    "user": "string",
    "command": "string",
    "sudo": "boolean",
    "timeout_s": "number"
  }
}
```
**Impact**: Model gets a reliable, deterministic way to run remote commands without ConPTY timing/idle heuristics.

#### - [ ] 3.3 Add `ssh_patch_file` tool (regex-based)

**Files**: `terminal_agent_ui.py`  
**What**: Like `ssh_replace_line` but using regex patterns instead of exact-line matching. More robust for config file edits. Implement via `sed -i` or a Python one-liner on the remote.  
**Impact**: Eliminates the most common SSH file editing failure mode (whitespace mismatches).

#### - [ ] 3.4 Improve `_in_ssh` detection robustness

**Files**: `terminal_agent_ui.py` (`_ConptyTerminal`)  
**What**:
- After sending an SSH command, wait up to 10 seconds (not just 450 ms) for the remote prompt to appear before returning.
- Detect SSH exit: if output contains `"Connection to ... closed"` or `"logout"`, set `_in_ssh = False`.
- Periodically (every few seconds while idle), check if the SSH process is still alive by sending a no-op (`echo __TA_SSH_ALIVE__`) and checking for the sentinel.  
**Impact**: Eliminates the race condition where SSH connects but `_in_ssh` stays `False`.

### Phase 4: Autonomous Problem-Solving (Medium effort, very high impact)

#### - [x] 4.1 Error-retry nudge (self-reflection) ✅ 2026-02-25T11:00Z

**Files**: `terminal_agent_ui.py` (`_Worker.run`)  
**What**: After receiving terminal output, check for common failure patterns (non-zero exit code, error keywords like `"Error"`, `"Permission denied"`, `"command not found"`, `"No such file"`). If detected, append an extra nudge to `pending_user`:
```
NOTE: The last command appears to have failed (exit code / error detected).
Analyze the error, determine the fix, and continue. Do NOT stop to ask the user.
```
**Impact**: Model retries failures instead of stopping and asking the user.

#### - [ ] 4.2 Self-verification step

**Files**: `terminal_agent_ui.py` (`_Worker.run`)  
**What**: Before declaring a task done (no `<Terminal>` blocks in output), inject a verification prompt:
```
Before finishing, verify your work:
1. Did all commands succeed?
2. Is the desired outcome achieved? If not, what's missing?
3. If you need to verify, emit a <Terminal> block with a test command.
Only summarize if everything is confirmed working.
```
This uses one extra round but catches incomplete work.  
**Impact**: Dramatically reduces "it says it's done but it's not" scenarios.

#### - [x] 4.3 Single-agent deliberation (internal chain-of-thought before acting) ✅ 2026-02-25T11:00Z

**Files**: `terminal_agent_ui.py` (`_Worker.run` and system prompt)  
**What**: Instead of spawning a second agent for brainstorming, add a structured thinking protocol to the system prompt:
```
When you encounter a problem with multiple possible solutions:
1. Think through the options briefly (2-3 sentences max per option).
2. Pick the most likely to succeed.
3. Execute it.
4. If it fails, try the next option.
Do NOT ask the user which option to choose unless ALL options have been tried and failed.
```
This is lighter than multi-agent and works within the existing single-agent loop.  
**Impact**: Eliminates most "which approach do you prefer?" pauses without any code changes beyond the prompt.

#### - [ ] 4.4 Persistent session scratchpad

**Files**: `terminal_agent_ui.py`, `chat_session.py`  
**What**: Give the model a `scratchpad` tool that stores key-value notes across rounds. The scratchpad is injected into every prompt as a compact block:
```
[Scratchpad]
ssh_host: omrijsharon@omrijsharon.local
task: Install and configure Tailscale on Pi
step: 3/5 - configuring autostart
last_error: systemctl enable failed, trying rc.local approach
[/Scratchpad]
```
The model can update the scratchpad via a tool call (`scratchpad_set(key, value)`, `scratchpad_clear(key)`). This gives the model a persistent memory across conversation compaction/summarization.  
**Impact**: Model maintains task awareness even across token-saving compaction. Eliminates "I forgot what I was doing" after long sessions.

### Phase 5: Quality of Life (Low effort, nice to have)

#### - [ ] 5.1 Show SSH status in UI

**Files**: `terminal_agent_ui.py` (UI)  
**What**: Add an indicator in the terminal tab header or CWD label showing `🟢 SSH: pi@raspberrypi` or `⚪ Local`.  
**Impact**: User always knows terminal state at a glance.

#### - [ ] 5.2 Configurable `max_rounds` via UI

**Files**: `terminal_agent_ui.py`  
**What**: Add a small spinbox or dropdown near the Send button to set rounds (6 / 12 / 20 / unlimited).  
**Impact**: User can give the agent more autonomy for big tasks.

#### - [ ] 5.3 "Continue" button

**Files**: `terminal_agent_ui.py`  
**What**: When the model stops (no `<Terminal>` blocks), show a "Continue" button in the chat that sends `"Continue with the task. If there are remaining steps, execute them."` as the next user message.  
**Impact**: One-click recovery from premature stops.

#### - [ ] 5.4 Token usage chart

**Files**: `terminal_agent_ui.py` (UI)  
**What**: Show a small bar chart or color-coded indicator of token usage relative to the context window. Turn orange at 60%, red at 80%.  
**Impact**: User can see when to expect degraded performance and proactively reset.

---

## 4. Priority & Ordering

| Priority | Item | Effort | Impact | Done |
|----------|------|--------|--------|------|
| 🔴 P0 | 1.1 Fix SSH regex | 5 min | Critical bug fix | ✅ |
| 🔴 P0 | 1.2 Expose SSH state | 30 min | Fixes local/remote confusion | ✅ |
| 🔴 P0 | 1.3 Stronger continuation prompt | 15 min | Fixes premature stops | ✅ |
| 🟠 P1 | 1.4 Increase max_rounds | 5 min | More autonomy | ✅ |
| 🟠 P1 | 4.3 Single-agent deliberation prompt | 15 min | Stops "which option?" pauses | ✅ |
| 🟠 P1 | 2.2 Smarter stdout truncation | 30 min | Token efficiency | ✅ |
| 🟠 P1 | 4.1 Error-retry nudge | 30 min | Auto-recovery from failures | ✅ |
| 🟡 P2 | 2.1 Conversation compaction | 2 hr | Major token savings | |
| 🟡 P2 | 3.1 SSH context header | 1 hr | SSH robustness | |
| 🟡 P2 | 2.3 Strip terminal noise | 1 hr | Token efficiency | |
| 🟡 P2 | 3.2 ssh_run_command tool | 1 hr | Reliable remote commands | |
| 🟡 P2 | 4.2 Self-verification step | 30 min | Task completion quality | |
| 🟢 P3 | 3.3 ssh_patch_file tool | 1 hr | SSH file editing | |
| 🟢 P3 | 3.4 Improve _in_ssh detection | 1 hr | SSH robustness | |
| 🟢 P3 | 4.4 Persistent scratchpad | 2 hr | Long-session memory | |
| 🟢 P3 | 5.1 SSH status in UI | 30 min | UX | |
| 🟢 P3 | 5.3 "Continue" button | 30 min | UX | |
| ⚪ P4 | 5.2 Configurable max_rounds | 15 min | UX | |
| ⚪ P4 | 5.4 Token usage chart | 1 hr | UX | |
| ⚪ P4 | 2.4 System prompt cleanup | 15 min | Minor token savings | |

**Recommended execution order**: P0 items first (can all be done in one session), then P1 (another session), then P2/P3 as needed.

---

## 5. Bugs Found During Review

| Bug | Location | Severity |
|-----|----------|----------|
| `_is_interactive_ssh` regex uses `\\b` (literal backslash-b) instead of `\b` (word boundary) — SSH is **never** correctly detected at command time | `terminal_agent_ui.py:756, 1152` | 🔴 Critical |
| `_is_interactive_ssh` quote-detection regex uses `\\"` and `\\'` which also don't match actual quotes | `terminal_agent_ui.py:759, 1155` | 🟠 High |
| `session_status()` in `_ConptyTerminal` always returns `"interactive(conpty)"` regardless of SSH state | `terminal_agent_ui.py:1198` | 🟠 High |
| `_Worker.run` checks `'critic_txt' in locals()` (from grid demo copy-paste?) — not applicable here but harmless | N/A | ⚪ Info |
