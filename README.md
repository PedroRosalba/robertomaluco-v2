# Minimal Slack Conversational Bot (Python)

This bot responds to:
- mentions in channels (`@your-bot`)
- direct messages (DMs)

Response comes from the selected agent provider (`gpt`, `claude`, or `gemini`).

## 1. Create Slack app

Create an app in Slack and enable:

- **Socket Mode**: On
- **Bot Token Scopes**:
  - `app_mentions:read`
  - `chat:write`
  - `im:history`
- **Event Subscriptions**:
  - `app_mention`
  - `message.im`

Also create an **App-Level Token** with scope:
- `connections:write`

Install/reinstall the app to your workspace.

## 2. Configure environment

Use a local `.env` file:

```bash
cp example.env .env
```

Then set:
- `SLACK_BOT_TOKEN=xoxb-...`
- `SLACK_APP_TOKEN=xapp-...`
- `AGENT_MODEL=gpt` (or `claude` / `gemini`)
- The matching API key for your selected model:
  - `AGENT_MODEL=gpt` -> set `GPT_API_KEY`
  - `AGENT_MODEL=claude` -> set `CLAUDE_API_KEY`
  - `AGENT_MODEL=gemini` -> set `GEMINI_API_KEY`
- For GitHub repo tasks with Gemini tools, set `GITHUB_TOKEN`

## 3. Run with `uv`

```bash
uv sync
set -a
source .env
set +a
uv run app.py
```

## Notes

- In channels, the bot replies when mentioned.
- In DMs, it replies to any user message.
- Bot/system messages are ignored to avoid reply loops.
- `AGENT_MODEL` controls which provider is used by the app.
- Each handled Slack event prints a structured JSON trace to stdout (`TRACE_START`/`TRACE_END`).
- Trace includes nested spans for LLM and tool-calls (including HTTP errors/retries inside a tool-call span).
- Planning mode detection is always enabled and will auto-route coding-oriented prompts to `plan` mode.
- Each provider reads its own model env var override:
  - `GPT_MODEL` (default `gpt-4o-mini`)
  - `CLAUDE_MODEL` (default `claude-3-7-sonnet-latest`)
  - `GEMINI_MODEL` (default `gemini-2.5-pro`)
