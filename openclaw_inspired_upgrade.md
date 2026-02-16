# OpenClaw-Inspired Upgrade Plan (Desktop Agent)

This plan upgrades `desktop-agent` toward an OpenClaw-like “gateway + channels + online tools” architecture, with a focus on **Go Online** (search + fetch + browser automation via MCP/Playwright) and a **channel adapter** pipeline (WhatsApp/Telegram/Discord) using an “envelope + dispatcher” pattern.

Rules for execution:
- Each task has a checkbox. When completed, mark it `[x]` and add a completion timestamp (UTC).
- Keep changes incremental and keep legacy behavior available behind toggles until the new path is stable.

---

## 0) Groundwork

- [ ] **Document current architecture** (what runs where; legacy vs new agent sdk path). _(done: )_
- [ ] **Add central feature toggles** (legacy vs Agents SDK; MCP browser on/off; channels on/off). _(done: )_
- [ ] **Add structured logging + traces** for: model calls, tool calls, channel events, dispatch steps. _(done: )_

---

## 0.5) Dev Workspace (agent coding sandbox)

Goal: give the chat agent a per-chat project workspace with Python-only execution (venv + pip + localhost preview).

- [x] Add per-chat workspace root under `chat_history/workspaces/<chat_id>/`. _(done: 2026-02-15 08:59:54Z)_
- [x] Add workspace tools (write/read/list, create venv, run python, pip install with approval, http server). _(done: 2026-02-15 08:59:54Z)_
- [x] Add UI controls (enable workspace, allow pip, allow http server) + per-install approval dialog. _(done: 2026-02-15 08:59:54Z)_
- [x] Add docs + tests. _(done: 2026-02-15 08:59:54Z)_

---

## 1) Migrate to OpenAI Agents SDK (keep legacy as fallback)

Goal: use `openai-agents` to gain first-class MCP integration while keeping existing `ChatSession`/Responses runner working.

- [x] **Add Agents SDK integration scaffold**: _(done: 2026-02-15 06:39:21Z)_
  - [x] Add Agents SDK dependency (`openai-agents`). _(done: 2026-02-15 00:16:45Z)_
  - [x] Add minimal wrapper module under `src/desktop_agent/agent_sdk/`. _(done: 2026-02-15 05:50:07Z)_
- [x] **Add minimal Agents SDK session adapter** (`AgentsSdkSession`) compatible with Chat UI streaming. _(done: 2026-02-15 05:50:07Z)_
- [x] **Implement “agent runner” abstraction**: _(done: 2026-02-15 06:14:06Z)_
  - [x] `LegacyRunner` (current `ChatSession` path) _(done: 2026-02-15 06:14:06Z)_
  - [x] `AgentsSdkRunner` (Agents SDK path, session memory, streaming) _(done: 2026-02-15 06:14:06Z)_
  - [x] Unified event stream interface for the UI (assistant deltas, tool events, errors, usage). _(done: 2026-02-15 06:14:06Z)_
- [x] **Wire a UI toggle** “Use Agents SDK (experimental)” that switches the main engine (legacy vs Agents SDK) without breaking chat history. _(done: 2026-02-15 05:50:07Z)_
- [x] **Add regression tests** for engine switching + streamed tool events. _(done: 2026-02-15 06:14:06Z)_

---

## 2) Go Online tool stack (OpenClaw-inspired)

### 2.1 Web search
- [x] Keep existing `web_search` tool support; add provider/config docs. _(done: 2026-02-15 06:39:21Z)_ (see `docs/go_online.md`)

### 2.2 Web fetch + readability
- [x] Add `web_fetch(url, ...)` tool: _(done: 2026-02-14 23:59:09Z)_
  - [x] HTTP fetch with size/time limits, redirects limit, safe UA. _(done: 2026-02-14 23:59:09Z)_
  - [x] Optional readability extraction (“main content”) with safe fallback. _(done: 2026-02-14 23:59:09Z)_
  - [x] Caching (TTL) to reduce repeated requests. _(done: 2026-02-14 23:59:09Z)_
  - [x] Returns text + URL + metadata. _(done: 2026-02-14 23:59:09Z)_
- [x] Add tests for fetch limits and caching. _(done: 2026-02-14 23:59:09Z)_

### 2.3 Browser automation via MCP + Playwright
- [x] Add MCP integration for Playwright (stdio via `cmd.exe /c npx ...`): _(done: 2026-02-15 00:16:45Z)_
  - [x] Local stdio MCP server support via `npx @playwright/mcp@latest` (optional). _(done: 2026-02-15 00:16:45Z)_
  - [x] Approval gating for sensitive tools (navigation/click/type vs eval/run_code). _(done: 2026-02-15 05:27:40Z)_
- [x] UI “watch mode”: _(done: 2026-02-15 05:27:40Z)_
  - [x] Show screenshots inline in chat during browser actions (auto-screenshot mode). _(done: 2026-02-15 05:27:40Z)_
  - [x] Headless vs headed mode selection (applies after restart). _(done: 2026-02-15 05:27:40Z)_
- [x] Add “reconnect / restart browser server” UX for crashy sessions. _(done: 2026-02-15 05:27:40Z)_

---

## 3) Channel adapters (WhatsApp / Telegram / Discord)

Goal: channels are *transports*, not the agent. A single gateway process owns sessions and policies.

- [ ] Adopt OpenClaw’s “gateway owns sessions” pattern:
  - [ ] Separate **channel contract** from **provider runtime** (even if both live in this repo). _(done: )_
  - [ ] Keep secrets/session state in the gateway/relay process only (UI shouldn’t need tokens). _(done: )_
  - [ ] Normalize inbound messages into an “envelope” that is channel-agnostic (id, sender, peer, text, media, reply-to, mentions). _(done: )_
  - [ ] Enforce policies in the gateway (pairing/allowlists/group policy/mention gating), not in the model. _(done: )_
  - [ ] Add provider observability without spamming chat surfaces (structured logs + safe truncation). _(done: )_

- [ ] Define a `Channel` interface + config schema:
  - [ ] `connect() / disconnect()`
  - [ ] `send_message() / send_media() / react()`
  - [ ] events: message, presence, error, reconnect. _(done: )_
- [ ] Implement **Telegram** first (simplest):
  - [x] inbound/outbound (headless relay + UI file IPC bridge). _(done: 2026-02-15 07:25:08Z)_
  - [ ] Durable update offsets + dedupe (so restarts don’t replay old updates). _(done: )_
  - [ ] Polling conflict detection/backoff (avoid “two pollers” problems). _(done: )_
  - [ ] group history policy (how many recent msgs to include; group vs DM). _(done: )_
  - [ ] allowlists/pairing + mention gating (secure default; group bypass policy). _(done: )_
  - [ ] reply threading strategy (reply-to behavior, topics/threads if needed). _(done: )_
- [ ] Implement **Discord**:
  - [ ] inbound/outbound
  - [ ] thread/channel routing
  - [ ] rate limit handling. _(done: )_
- [ ] Implement **WhatsApp**:
  - [ ] Choose integration mode (explicit decision):
    - [ ] **WhatsApp Web / Baileys** (unofficial; logs in as a real account via QR; higher risk of breakage/policy issues). _(done: )_
    - [ ] **Official Cloud API** (compliant; sends as a business number; usually not “your personal user”). _(done: )_
  - [ ] If Baileys-mode:
    - [ ] Gateway-owned session store (multi-file auth state on disk) + corruption hardening/backup. _(done: )_
    - [ ] QR login flow (“start” + “wait”) + UI surface for QR (image) + timeouts. _(done: )_
    - [ ] Inbound pipeline:
      - [ ] dedupe by message id
      - [ ] debounce message bursts into a single prompt
      - [ ] sender identity resolution + group metadata cache
      - [ ] quoted reply context + mentions extraction. _(done: )_
    - [ ] Security/policy:
      - [ ] DM policy defaults to pairing/allowlist (not open)
      - [ ] group policy: open/allowlist/disabled
      - [ ] self-chat mode safety (avoid surprising behaviors like auto read receipts). _(done: )_
    - [ ] Outbound delivery:
      - [ ] WhatsApp-safe chunking limits + mode (direct vs streaming)
      - [ ] media caps (max MB) + safe file handling. _(done: )_
    - [ ] Active-listener guard (must have an active inbox monitor to send). _(done: )_
  - [ ] If Cloud-API mode:
    - [ ] Provider credentials/config + webhook/polling ingestion. _(done: )_
    - [ ] Mapping to/from business sender identity; DM allowlists/pairing by phone. _(done: )_
    - [ ] Delivery chunking + media caps + rate-limit handling. _(done: )_

---

## 4) Automation / reply pipeline (Envelope + Dispatcher)

- [ ] Introduce a canonical `Envelope` type:
  - [ ] message id, channel, sender, chat/group id, timestamp
  - [ ] text, attachments, mentions, quoted message
  - [ ] derived context (group history snippets). _(done: )_
- [ ] Build a `Dispatcher`:
  - [ ] route to an agent (main agent or specialized agents)
  - [ ] buffered block output (chunking rules by channel)
  - [ ] delivery acknowledgements + retries. _(done: )_
- [ ] Add “fast ack reaction” option (OpenClaw-style). _(done: )_

---

## 5) Safety / policy (must-have before wide automation)

- [ ] Central tool policy:
  - [ ] allow/deny list by tool name
  - [ ] per-tool approval mode (“never/always/per tool map”)
  - [ ] audit log. _(done: )_
- [ ] Secrets & credentials:
  - [ ] all tokens/keys in config files are gitignored
  - [ ] redact secrets in logs. _(done: )_

---

## 6) UX + Packaging

- [ ] One “Gateway” app entry-point that can run:
  - [ ] Desktop Chat UI
  - [ ] Channels gateway headless mode
  - [ ] Automated calibration UI. _(done: )_
- [ ] Clear “How to run” docs for:
  - [ ] chat ui
  - [ ] go-online browser
  - [ ] channels gateway. _(done: )_
