import os

from agent import get_provider
from agent.modes import detect_mode
from agent.plan_schema import PlanSchema, parse_plan_response
from agent.provider import AgentRequest
from dotenv import load_dotenv
from logger import TraceStore
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
AGENT_MODEL = os.getenv("AGENT_MODEL")

if not BOT_TOKEN:
    raise RuntimeError("Missing SLACK_BOT_TOKEN")
if not APP_TOKEN:
    raise RuntimeError("Missing SLACK_APP_TOKEN")
if not AGENT_MODEL:
    raise RuntimeError("Missing AGENT_MODEL")
    
app = App(token=BOT_TOKEN)
provider = get_provider(AGENT_MODEL)
trace_store = TraceStore()


def should_ignore_event(event: dict) -> bool:
    # Ignore bot/system messages to avoid loops and noise.
    return bool(event.get("bot_id") or event.get("subtype"))


def build_agent_request(event: dict, mode: str) -> AgentRequest:
    text = event.get("text") or ""
    return AgentRequest(
        prompt=text,
        mode=mode,
        context={
            "user": event.get("user"),
            "channel": event.get("channel"),
            "channel_type": event.get("channel_type"),
            "thread_ts": event.get("thread_ts"),
        },
    )


def handle_event(event: dict, say) -> None:
    text = event.get("text") or ""
    trace = trace_store.create(
        metadata={
            "provider": provider.name,
            "event_type": event.get("type"),
            "user": event.get("user"),
            "channel": event.get("channel"),
            "channel_type": event.get("channel_type"),
        }
    )
    trace.event("request.received", status="ok")
    mode_decision = detect_mode(text)
    trace.event("mode.detected", status="ok", mode=mode_decision.mode, reason=mode_decision.reason)
    request = build_agent_request(event, mode=mode_decision.mode)
    request.context["request_id"] = trace.request_id

    provider_span = trace.span("provider.respond", provider=provider.name, mode=request.mode)
    try:
        response = provider.respond(request, trace=trace)
        provider_span.finish(status="ok", text_length=len(response.text))
        trace.event("response.ready", status="ok", provider=response.provider, text_length=len(response.text))
        outgoing_text = response.text
        if request.mode == "plan":
            plan_span = trace.span("plan.validate")
            try:
                plan = parse_plan_response(response.text)
                outgoing_text = format_plan_for_slack(plan)
                plan_span.finish(status="ok")
            except Exception as exc:
                plan_span.finish(status="error", error=str(exc))
                trace.event("plan.validation.failed", status="warn", error=str(exc))
                outgoing_text = (
                    "Plan mode was requested, but I could not parse a valid plan schema. "
                    "Returning raw model output below:\n\n"
                    f"{response.text}"
                )
    except Exception as exc:
        provider_span.finish(status="error", error=str(exc))
        trace.root.finish(status="error", error=str(exc))
        trace.event("request.error", status="error", error=str(exc))
        trace_store.persist(trace)
        say(f"Sorry, I hit an error while processing your request: {exc}")
        return

    say_span = trace.span("slack.say")
    say(outgoing_text)
    say_span.finish(status="ok")
    trace.event("response.sent", status="ok")
    trace.root.finish(status="ok")
    trace_store.persist(trace)


def format_plan_for_slack(plan: PlanSchema) -> str:
    lines = [
        "*Plan Mode Output*",
        f"*Objective*: {plan.objective}",
        "",
    ]
    if plan.assumptions:
        lines.append("*Assumptions*")
        lines.extend([f"- {item}" for item in plan.assumptions])
        lines.append("")
    if plan.files_to_touch:
        lines.append("*Files To Touch*")
        lines.extend([f"- `{item}`" for item in plan.files_to_touch])
        lines.append("")
    lines.append("*Implementation Steps*")
    for index, step in enumerate(plan.implementation_steps, start=1):
        lines.append(f"{index}. {step.title}: {step.details}")
    if plan.risks:
        lines.append("")
        lines.append("*Risks*")
        lines.extend([f"- {item}" for item in plan.risks])
    if plan.test_plan:
        lines.append("")
        lines.append("*Test Plan*")
        lines.extend([f"- {item}" for item in plan.test_plan])
    if plan.rollback_plan:
        lines.append("")
        lines.append("*Rollback Plan*")
        lines.extend([f"- {item}" for item in plan.rollback_plan])
    return "\n".join(lines)


@app.event("app_mention")
def on_app_mention(event, say):
    if should_ignore_event(event):
        return

    handle_event(event, say)


@app.event("message")
def on_message(event, say):
    if should_ignore_event(event):
        return

    # Only respond in direct messages.
    if event.get("channel_type") != "im":
        return

    handle_event(event, say)


if __name__ == "__main__":
    SocketModeHandler(app, APP_TOKEN).start()
