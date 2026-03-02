from __future__ import annotations

import json
import os
import re
import socket
import time
from typing import Any
from urllib import error, parse, request as urllib_request

from json_utils import extract_first_json_object
from state.job_store import ExecutionWorkflowState, JobStore
from logger import RequestTrace
from agent.planning.mode_detection import build_plan_prompt
from agent.providers.provider import AgentProvider, AgentRequest, AgentResponse
from agent.tools import GithubTools, PullRequestInput, RepoAccess, WriteFileInput


class GeminiProvider(AgentProvider):
    def __init__(
        self,
        tools: GithubTools | None = None,
        job_store: JobStore | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.tools = tools or GithubTools()
        self.job_store = job_store
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        self.request_timeout_seconds = 60
        self.max_retries = 2

        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY for AGENT_MODEL=gemini")

    @property
    def name(self) -> str:
        return "gemini"

    def respond(self, request: AgentRequest, trace: RequestTrace | None = None) -> AgentResponse:
        span_name = "llm.plan.generate" if request.mode == "plan" else "llm.code.workflow" if request.mode == "code" else "llm.chat.generate"
        llm_span = trace.span(span_name, provider=self.name, model=self.model, mode=request.mode) if trace else None

        if request.mode == "plan":
            text = self._generate_text(build_plan_prompt(request.prompt, request.context), trace_span=llm_span)
            if llm_span:
                llm_span.finish(status="ok", mode="plan", text_length=len(text))
            return AgentResponse(provider=self.name, text=text)

        access = self._extract_repo_access(request.prompt)
        if not access:
            text = self._generate_text(request.prompt, trace_span=llm_span)
            if llm_span:
                llm_span.finish(status="ok", mode="chat", text_length=len(text))
            return AgentResponse(provider=self.name, text=text)

        # Fail fast with a clear message when token cannot push to the target repo.
        with self.tools.with_trace(llm_span):
            self.tools.ensure_repo_write_access(access)

        execution_job_id = request.context.get("execution_job_id")
        workflow_state: ExecutionWorkflowState | None = None
        if isinstance(execution_job_id, str) and self.job_store:
            workflow_state = self.job_store.init_or_resume_workflow(
                job_id=execution_job_id,
                mode="code",
                request_prompt=request.prompt,
                repo_access=access.model_dump(),
                max_steps=12,
            )
            workflow_state["status"] = "running"
            workflow_state["repo_access"] = access.model_dump()
            self.job_store.save_workflow_state(workflow_state)
            self.job_store.append_job_event(
                execution_job_id,
                "workflow.resume" if workflow_state.get("step_index", 0) > 0 else "workflow.start",
                status="ok",
                step_index=int(workflow_state.get("step_index") or 0),
                history_len=len(workflow_state.get("history") or []),
            )

        history: list[dict[str, Any]] = list(workflow_state.get("history") or []) if workflow_state else []
        start_step = int(workflow_state.get("step_index") or 0) if workflow_state else 0
        max_steps = int(workflow_state.get("max_steps") or 12) if workflow_state else 12
        try:
            for step_index in range(start_step, max_steps):
                prompt = self._build_tool_prompt(request.prompt, access, history)
                step_span = llm_span.child("llm.step", index=step_index + 1) if llm_span else None
                model_text = self._generate_text(prompt, trace_span=step_span)
                action = self._parse_action(model_text, trace_span=step_span)
                if isinstance(execution_job_id, str) and self.job_store:
                    self.job_store.append_job_event(
                        execution_job_id,
                        "workflow.action.parsed",
                        status="ok",
                        step_index=step_index + 1,
                        action_type=action.get("type"),
                        tool=action.get("tool", ""),
                    )

                if action["type"] == "final":
                    if workflow_state and self.job_store and isinstance(execution_job_id, str):
                        workflow_state["status"] = "done"
                        workflow_state["step_index"] = step_index + 1
                        self.job_store.save_workflow_state(workflow_state)
                        self.job_store.append_job_event(
                            execution_job_id,
                            "workflow.final",
                            status="ok",
                            step_index=step_index + 1,
                        )
                    if step_span:
                        step_span.finish(status="ok", action_type="final")
                    if llm_span:
                        llm_span.finish(status="ok", mode="tool_workflow")
                    return AgentResponse(provider=self.name, text=action["message"])

                tool_span = step_span.child(
                    "llm.tool_call",
                    tool=action["tool"],
                    arguments=action.get("arguments", {}),
                ) if step_span else None
                tool_result = self._execute_tool(
                    access,
                    action["tool"],
                    action.get("arguments", {}),
                    trace_span=tool_span,
                )
                if workflow_state:
                    self._update_artifacts_from_tool_result(workflow_state, action, tool_result, access)
                if tool_span:
                    tool_span.finish(status="ok")
                history.append(
                    {
                        "assistant": model_text,
                        "tool_result": tool_result,
                    }
                )
                if workflow_state and self.job_store and isinstance(execution_job_id, str):
                    workflow_state["history"] = history
                    workflow_state["step_index"] = step_index + 1
                    workflow_state["status"] = "running"
                    self.job_store.save_workflow_state(workflow_state)
                    self.job_store.append_job_event(
                        execution_job_id,
                        "workflow.tool.completed",
                        status="ok",
                        step_index=step_index + 1,
                        tool=action.get("tool", ""),
                    )
                if step_span:
                    step_span.finish(status="ok", action_type="tool_call")
        except Exception as exc:
            if workflow_state and self.job_store and isinstance(execution_job_id, str):
                workflow_state["status"] = "needs_recovery"
                workflow_state["last_error"] = str(exc)
                workflow_state["history"] = history
                self.job_store.save_workflow_state(workflow_state)
                self.job_store.append_job_event(
                    execution_job_id,
                    "workflow.error",
                    status="error",
                    error=str(exc),
                    step_index=int(workflow_state.get("step_index") or 0),
                )
            raise

        if workflow_state and self.job_store and isinstance(execution_job_id, str):
            workflow_state["status"] = "needs_recovery"
            workflow_state["last_error"] = "step_limit_exceeded"
            workflow_state["history"] = history
            workflow_state["step_index"] = max_steps
            self.job_store.save_workflow_state(workflow_state)
            self.job_store.append_job_event(
                execution_job_id,
                "workflow.step_limit",
                status="error",
                step_index=max_steps,
            )
        if llm_span:
            llm_span.finish(status="error", reason="step_limit_exceeded")
        return AgentResponse(
            provider=self.name,
            text="I couldn't complete the workflow within the step limit.",
        )

    def _update_artifacts_from_tool_result(
        self,
        workflow_state: ExecutionWorkflowState,
        action: dict[str, Any],
        tool_result: dict[str, Any],
        access: RepoAccess,
    ) -> None:
        artifacts = workflow_state.setdefault("artifacts", {"commit_shas": [], "changed_files": []})
        artifacts.setdefault("repo", f"{access.owner}/{access.repo}")
        if action.get("tool") == "create_branch":
            branch = tool_result.get("branch")
            if isinstance(branch, str) and branch:
                artifacts["working_branch"] = branch
        elif action.get("tool") == "write_file":
            commit_sha = tool_result.get("commit_sha")
            path = tool_result.get("path") or action.get("arguments", {}).get("path")
            if isinstance(commit_sha, str) and commit_sha:
                commits = artifacts.setdefault("commit_shas", [])
                if commit_sha not in commits:
                    commits.append(commit_sha)
            if isinstance(path, str) and path:
                changed = artifacts.setdefault("changed_files", [])
                if path not in changed:
                    changed.append(path)
        elif action.get("tool") == "create_pull_request":
            url = tool_result.get("pull_request_url")
            if isinstance(url, str) and url:
                artifacts["pr_url"] = url

    def _generate_text(self, prompt: str, trace_span=None) -> str:
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        encoded_key = parse.quote(self.api_key, safe="")
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={encoded_key}"
        )
        req = urllib_request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        raw = ""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if trace_span:
                    trace_span.event(
                        "gemini.http.start",
                        status="ok",
                        attempt=attempt + 1,
                        endpoint=f"/models/{self.model}:generateContent",
                    )
                with urllib_request.urlopen(req, timeout=self.request_timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    if trace_span:
                        trace_span.event(
                            "gemini.http.success",
                            status="ok",
                            attempt=attempt + 1,
                            status_code=response.status,
                        )
                    last_error = None
                    break
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if trace_span:
                    trace_span.event(
                        "gemini.http.error",
                        status="error",
                        attempt=attempt + 1,
                        status_code=exc.code,
                        detail=detail,
                    )
                raise RuntimeError(f"Gemini API error ({exc.code}): {detail}") from exc
            except (TimeoutError, socket.timeout) as exc:
                if trace_span:
                    trace_span.event(
                        "gemini.http.error",
                        status="error",
                        attempt=attempt + 1,
                        reason="timeout",
                    )
                last_error = RuntimeError(
                    f"Gemini request timed out after {self.request_timeout_seconds}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
            except error.URLError as exc:
                if trace_span:
                    trace_span.event(
                        "gemini.http.error",
                        status="error",
                        attempt=attempt + 1,
                        reason=str(exc.reason),
                    )
                last_error = RuntimeError(f"Gemini network error: {exc.reason}")

            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))

        if last_error:
            raise last_error

        body = json.loads(raw)
        candidates = body.get("candidates", [])
        texts: list[str] = []
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    texts.append(text)

        combined = "\n".join(texts).strip()
        if not combined:
            if trace_span:
                trace_span.event("gemini.parse.error", status="error", reason="missing_text_content")
            raise RuntimeError("Gemini response missing text content")
        return combined

    def _extract_repo_access(self, prompt: str) -> RepoAccess | None:
        cleaned = prompt.replace("<", " ").replace(">", " ")
        match = re.search(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", cleaned)
        if not match:
            return None
        owner = match.group(1)
        repo = match.group(2).removesuffix(".git")
        return RepoAccess(owner=owner, repo=repo, branch="main")

    def _build_tool_prompt(
        self,
        user_prompt: str,
        access: RepoAccess,
        history: list[dict[str, Any]],
    ) -> str:
        system = (
            "You are an autonomous software agent. "
            "You must output only valid JSON with one of these shapes:\n"
            '1) {"type":"tool_call","tool":"<name>","arguments":{...}}\n'
            '2) {"type":"final","message":"<final summary for user>"}\n\n'
            "Available tools:\n"
            "- get_default_branch: {}\n"
            "- create_branch: {new_branch: string, from_branch?: string}\n"
            "- list_files: {branch?: string}\n"
            "- read_file: {path: string, branch?: string}\n"
            "- write_file: {path: string, content: string, commit_message: string, branch?: string}\n"
            "- create_pull_request: {title: string, body: string, head_branch: string, base_branch?: string}\n\n"
            "Workflow guidance:\n"
            "1) Discover files\n"
            "2) Read relevant files\n"
            "3) Create a branch if needed\n"
            "4) Write updated file contents\n"
            "5) Open PR\n"
            "When done, return a final summary including PR URL.\n"
            "Never include markdown code fences."
        )
        return json.dumps(
            {
                "system": system,
                "repo": access.model_dump(),
                "request": user_prompt,
                "history": history,
                "timestamp": int(time.time()),
            }
        )

    def _parse_action(self, model_text: str, trace_span=None) -> dict[str, Any]:
        try:
            payload, trailing = extract_first_json_object(model_text)
        except ValueError as exc:
            if trace_span:
                trace_span.event("gemini.action.parse.error", status="error", error=str(exc), raw=model_text)
            raise RuntimeError(f"Gemini response is not JSON: {model_text}") from exc

        if trailing and trace_span:
            trace_span.event("gemini.action.trailing_text", status="warn", trailing=trailing)

        action_type = payload.get("type")
        if action_type == "tool_call":
            if "tool" not in payload:
                raise RuntimeError(f"Invalid tool_call payload: {payload}")
            payload.setdefault("arguments", {})
            return payload
        if action_type == "final":
            if not payload.get("message"):
                raise RuntimeError(f"Invalid final payload: {payload}")
            return payload
        raise RuntimeError(f"Unsupported action type from Gemini: {payload}")

    def _execute_tool(
        self,
        access: RepoAccess,
        tool_name: str,
        arguments: dict[str, Any],
        trace_span=None,
    ) -> dict[str, Any]:
        with self.tools.with_trace(trace_span):
            return self._execute_tool_inner(access, tool_name, arguments)

    def _execute_tool_inner(
        self,
        access: RepoAccess,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name == "get_default_branch":
            return {"default_branch": self.tools.get_default_branch(access)}

        if tool_name == "create_branch":
            new_branch = arguments["new_branch"]
            from_branch = arguments.get("from_branch")
            created = self.tools.create_branch(access, new_branch, from_branch=from_branch)
            return {"branch": created}

        if tool_name == "list_files":
            files = self.tools.list_files(access, branch=arguments.get("branch"))
            # Keep payload size controlled for model context.
            return {"files": files[:500], "total_files": len(files)}

        if tool_name == "read_file":
            content = self.tools.read_file(
                access,
                path=arguments["path"],
                branch=arguments.get("branch"),
            )
            return {"path": arguments["path"], "content": content}

        if tool_name == "write_file":
            payload = WriteFileInput.model_validate(arguments)
            commit_sha = self.tools.write_file(access, payload)
            return {"path": payload.path, "commit_sha": commit_sha}

        if tool_name == "create_pull_request":
            payload = PullRequestInput.model_validate(arguments)
            url = self.tools.create_pull_request(access, payload)
            return {"pull_request_url": url}

        raise RuntimeError(f"Unknown tool requested by Gemini: {tool_name}")
