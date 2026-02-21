from __future__ import annotations

import json
import os
from urllib import error, request as urllib_request

from logger import RequestTrace
from agent.modes import build_plan_prompt
from agent.provider import AgentProvider, AgentRequest, AgentResponse
from agent.tools import GithubTools


class GPTProvider(AgentProvider):
    def __init__(
        self,
        tools: GithubTools | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.tools = tools or GithubTools()
        self.api_key = api_key or os.getenv("GPT_API_KEY")
        self.model = model or os.getenv("GPT_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise RuntimeError("Missing GPT_API_KEY for AGENT_MODEL=gpt")

    @property
    def name(self) -> str:
        return "gpt"

    def respond(self, request: AgentRequest, trace: RequestTrace | None = None) -> AgentResponse:
        llm_span = trace.span("llm.call", provider=self.name, model=self.model, mode=request.mode) if trace else None
        prompt = request.prompt
        if request.mode == "plan":
            prompt = build_plan_prompt(request.prompt)

        payload = {
            "model": self.model,
            "input": prompt,
        }
        req = urllib_request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            if llm_span:
                llm_span.event("http.request.start", status="ok", endpoint="/v1/responses")
            with urllib_request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
                if llm_span:
                    llm_span.event("http.request.success", status="ok", status_code=response.status)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if llm_span:
                llm_span.event("http.request.error", status="error", status_code=exc.code, detail=detail)
                llm_span.finish(status="error")
            raise RuntimeError(f"OpenAI API error ({exc.code}): {detail}") from exc
        except error.URLError as exc:
            if llm_span:
                llm_span.event("http.request.error", status="error", reason=str(exc.reason))
                llm_span.finish(status="error")
            raise RuntimeError(f"OpenAI network error: {exc.reason}") from exc

        body = json.loads(raw)
        text = body.get("output_text")

        if not text:
            if llm_span:
                llm_span.event("llm.parse.error", status="error", reason="missing_output_text")
                llm_span.finish(status="error")
            raise RuntimeError("OpenAI response missing output_text")

        if llm_span:
            llm_span.finish(status="ok", text_length=len(text))
        return AgentResponse(provider=self.name, text=text)
