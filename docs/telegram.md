# Telegram Relay (Bot API)

This repo supports a **headless Telegram relay** so you can chat with the UI through a Telegram group.

Architecture:
- **Chat UI** is the source-of-truth (agents, history, tools, streaming).
- **Telegram relay** is *only* a transport and never calls OpenAI.
- Communication is via file IPC:
  - inbound: `chat_history/telegram/inbox/*.json`
  - outbound: `chat_history/telegram/outbox/*.json`

## Setup

1) Create a Telegram bot (free) and get a token:
   - In Telegram, open **@BotFather**
   - Run `/newbot`
   - Copy the token it returns

2) Allow the bot to see group messages:
   - In **@BotFather** run `/setprivacy` → choose your bot → **Disable**

3) Put the token in `.env`:

   `TELEGRAM_BOT_TOKEN=123456:AA...`

4) Add the bot to your dedicated group.

## Run

Option A (recommended): one-command launcher (starts relay + chat UI)

`.\.venv\Scripts\python.exe -m desktop_agent.process_manager`

Option B: separate terminals

Start the relay (headless):

`.\.venv\Scripts\python.exe -m desktop_agent.telegram_relay`

In the target group, send this **once** to bind the group id:
- `/allow_here`

Then run the Chat UI:

`.\.venv\Scripts\python.exe -m desktop_agent.chat_ui`

In Chat UI → **Controls** → **Telegram**:
- Enable “Telegram bridge”

## Notes

- The relay persists the allowlisted chat id + update offset in:
  - `chat_history/telegram/relay_state.json`
- For safety, the relay only sends messages to the allowlisted `chat_id`.
