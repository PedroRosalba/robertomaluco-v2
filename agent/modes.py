from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agent.plan_schema import PlanSchema


class ModeDecision(BaseModel):
    mode: str
    reason: str

    model_config = ConfigDict(extra="forbid")


PLAN_HINTS = (
    "plan mode",
    "make a plan",
    "create a plan",
    "before you code",
    "implementation plan",
    "review before coding",
)

CODE_HINTS = (
    "feature",
    "implement",
    "ship",
    "build",
    "code",
    "refactor",
    "bug",
    "fix",
    "repository",
    "repo",
    "pull request",
    "pr",
)

def detect_mode(prompt: str) -> ModeDecision:
    normalized = prompt.strip().lower()
    for hint in PLAN_HINTS:
        if hint in normalized:
            return ModeDecision(mode="plan", reason=f"matched_hint:{hint}")
    for hint in CODE_HINTS:
        if hint in normalized:
            return ModeDecision(mode="plan", reason=f"matched_code_hint:{hint}")
    return ModeDecision(mode="chat", reason="default")


def build_plan_prompt(user_prompt: str) -> str:
    schema = PlanSchema.model_json_schema()
    return (
        "You are in plan mode. Analyze the request and return only JSON matching this schema. "
        "Do not include markdown fences or commentary.\n\n"
        f"Schema:\n{schema}\n\n"
        f"User request:\n{user_prompt}"
    )
