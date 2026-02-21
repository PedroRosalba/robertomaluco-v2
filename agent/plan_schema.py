from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from json_utils import extract_first_json_object


class PlanStep(BaseModel):
    title: str = Field(min_length=1)
    details: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


class PlanSchema(BaseModel):
    objective: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    files_to_touch: list[str] = Field(default_factory=list)
    implementation_steps: list[PlanStep] = Field(min_length=1)
    risks: list[str] = Field(default_factory=list)
    test_plan: list[str] = Field(default_factory=list)
    rollback_plan: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def parse_plan_response(text: str) -> PlanSchema:
    payload, _trailing = extract_first_json_object(text)
    return PlanSchema.model_validate(payload)
