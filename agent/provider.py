from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from logger import RequestTrace


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mode: str = Field(default="chat")
    context: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class AgentResponse(BaseModel):
    provider: str
    text: str

    model_config = ConfigDict(extra="forbid")


class AgentProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def respond(self, request: AgentRequest, trace: RequestTrace | None = None) -> AgentResponse:
        """Return a provider response for the given request."""
        pass
