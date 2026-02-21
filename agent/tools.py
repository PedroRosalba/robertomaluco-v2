from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
from urllib import error, parse, request as urllib_request

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from logger import TraceSpan


class RepoAccess(BaseModel):
    owner: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    branch: str = Field(default="main", min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("owner", "repo", mode="before")
    @classmethod
    def sanitize_slug(cls, value: str, info: ValidationInfo) -> str:
        cleaned = str(value).strip().strip("<>").strip()
        cleaned = cleaned.split("|", 1)[0]
        if "github.com/" in cleaned:
            tail = cleaned.split("github.com/", 1)[1]
            tail = tail.split("?", 1)[0].split("#", 1)[0].strip("/")
            parts = [part for part in tail.split("/") if part]
            if parts:
                if info.field_name == "owner":
                    cleaned = parts[0]
                elif len(parts) > 1:
                    cleaned = parts[1]
                else:
                    cleaned = parts[0]
        cleaned = cleaned.removesuffix(".git")
        return cleaned


class PullRequestInput(BaseModel):
    title: str = Field(min_length=1)
    body: str = ""
    head_branch: str = Field(min_length=1)
    base_branch: str = Field(default="main", min_length=1)

    model_config = ConfigDict(extra="forbid")


class WriteFileInput(BaseModel):
    path: str = Field(min_length=1)
    content: str
    commit_message: str = Field(min_length=1)
    branch: str | None = None

    model_config = ConfigDict(extra="forbid")


class GithubTools:
    """Minimal GitHub tools surface for the agent providers.

    This is intentionally a thin scaffold. Wire these methods to your
    GitHub client of choice when credentials and repo permissions are ready.
    """

    def __init__(self, token: str | None = None, api_base: str = "https://api.github.com"):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.api_base = api_base.rstrip("/")
        self._active_trace_span: TraceSpan | None = None

    @contextmanager
    def with_trace(self, span: TraceSpan | None):
        previous = self._active_trace_span
        self._active_trace_span = span
        try:
            yield
        finally:
            self._active_trace_span = previous

    def _trace_event(self, name: str, status: str = "info", **data):
        if self._active_trace_span:
            self._active_trace_span.event(name=name, status=status, **data)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        if not self.token:
            self._trace_event("github.auth.error", status="error", reason="missing_token")
            raise RuntimeError("Missing GITHUB_TOKEN for GitHub tools")

        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")

        req = urllib_request.Request(
            f"{self.api_base}{path}",
            data=data,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "robertomaluco-agent",
                "Content-Type": "application/json",
            },
            method=method,
        )

        try:
            self._trace_event("github.http.start", status="ok", method=method, path=path)
            with urllib_request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
                self._trace_event(
                    "github.http.success",
                    status="ok",
                    method=method,
                    path=path,
                    status_code=response.status,
                )
                return json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._trace_event(
                "github.http.error",
                status="error",
                method=method,
                path=path,
                status_code=exc.code,
                detail=detail,
            )
            if exc.code == 404 and path.startswith("/repos/"):
                raise RuntimeError(
                    f"GitHub API error (404) on {method} {path}: {detail}. "
                    "For private repos, verify GITHUB_TOKEN can access this repo "
                    "(repo selected for fine-grained token, Contents/Pull requests permissions, "
                    "and SSO authorization if required)."
                ) from exc
            if exc.code == 403 and path.startswith("/repos/") and method in {"POST", "PUT", "PATCH", "DELETE"}:
                raise RuntimeError(
                    f"GitHub API error (403) on {method} {path}: {detail}. "
                    "This token can likely read the repo but cannot write. "
                    "For this workflow, grant write permissions to Contents and Pull requests "
                    "(and SSO authorize the token if your org requires it)."
                ) from exc
            raise RuntimeError(f"GitHub API error ({exc.code}) on {method} {path}: {detail}") from exc
        except error.URLError as exc:
            self._trace_event(
                "github.http.error",
                status="error",
                method=method,
                path=path,
                reason=str(exc.reason),
            )
            raise RuntimeError(f"GitHub network error on {method} {path}: {exc.reason}") from exc

    def ensure_repo_write_access(self, access: RepoAccess) -> None:
        repo = self._request("GET", f"/repos/{access.owner}/{access.repo}")
        permissions = repo.get("permissions")
        if isinstance(permissions, dict) and not permissions.get("push", False):
            raise RuntimeError(
                "GITHUB_TOKEN has no push access to this repo. "
                "Use a token with repo write permissions (Contents + Pull requests)."
            )

    def get_default_branch(self, access: RepoAccess) -> str:
        repo = self._request("GET", f"/repos/{access.owner}/{access.repo}")
        return repo.get("default_branch") or access.branch

    def create_branch(
        self,
        access: RepoAccess,
        new_branch: str,
        from_branch: str | None = None,
    ) -> str:
        source = from_branch or self.get_default_branch(access)
        source_ref = self._request(
            "GET",
            f"/repos/{access.owner}/{access.repo}/git/ref/heads/{parse.quote(source, safe='')}",
        )
        sha = source_ref["object"]["sha"]
        try:
            self._request(
                "POST",
                f"/repos/{access.owner}/{access.repo}/git/refs",
                {"ref": f"refs/heads/{new_branch}", "sha": sha},
            )
        except RuntimeError as exc:
            # Safe to continue if the branch already exists from a previous run.
            if "GitHub API error (422)" not in str(exc) or "Reference already exists" not in str(exc):
                raise
        return new_branch

    def list_files(self, access: RepoAccess, branch: str | None = None) -> list[str]:
        ref = branch or access.branch
        tree = self._get_tree_with_fallback(access, ref)
        files = []
        for entry in tree.get("tree", []):
            if entry.get("type") == "blob" and entry.get("path"):
                files.append(entry["path"])
        return files

    def read_file(self, access: RepoAccess, path: str, branch: str | None = None) -> str:
        ref = branch or access.branch
        encoded_path = parse.quote(path, safe="/")
        try:
            data = self._request(
                "GET",
                f"/repos/{access.owner}/{access.repo}/contents/{encoded_path}?ref={parse.quote(ref, safe='')}",
            )
        except RuntimeError as exc:
            if "GitHub API error (404)" not in str(exc):
                raise
            fallback = self.get_default_branch(access)
            data = self._request(
                "GET",
                f"/repos/{access.owner}/{access.repo}/contents/{encoded_path}?ref={parse.quote(fallback, safe='')}",
            )
        content = data.get("content", "").replace("\n", "")
        if not content:
            return ""
        return base64.b64decode(content).decode("utf-8")

    def write_file(self, access: RepoAccess, payload: WriteFileInput) -> str:
        branch = payload.branch or access.branch
        encoded_path = parse.quote(payload.path, safe="/")
        body: dict[str, str] = {
            "message": payload.commit_message,
            "content": base64.b64encode(payload.content.encode("utf-8")).decode("utf-8"),
            "branch": branch,
        }

        # If file already exists, include current SHA so GitHub updates instead of creating.
        try:
            existing = self._request(
                "GET",
                f"/repos/{access.owner}/{access.repo}/contents/{encoded_path}?ref={parse.quote(branch, safe='')}",
            )
            if existing.get("sha"):
                body["sha"] = existing["sha"]
        except RuntimeError as exc:
            if "GitHub API error (404)" not in str(exc):
                raise

        response = self._request(
            "PUT",
            f"/repos/{access.owner}/{access.repo}/contents/{encoded_path}",
            body,
        )
        commit = response.get("commit", {})
        return commit.get("sha", "")

    def create_commit(self, access: RepoAccess, message: str) -> str:
        return (
            f"Direct commit creation is not used. "
            f"Use write_file with commit message: {message}"
        )

    def create_pull_request(self, access: RepoAccess, payload: PullRequestInput) -> str:
        response = self._request(
            "POST",
            f"/repos/{access.owner}/{access.repo}/pulls",
            {
                "title": payload.title,
                "body": payload.body,
                "head": payload.head_branch,
                "base": payload.base_branch,
            },
        )
        return response.get("html_url", "")

    def _get_tree_with_fallback(self, access: RepoAccess, ref: str) -> dict:
        try:
            return self._request(
                "GET",
                f"/repos/{access.owner}/{access.repo}/git/trees/{parse.quote(ref, safe='')}?recursive=1",
            )
        except RuntimeError as exc:
            if "GitHub API error (404)" not in str(exc):
                raise
            fallback = self.get_default_branch(access)
            if fallback == ref:
                raise
            return self._request(
                "GET",
                f"/repos/{access.owner}/{access.repo}/git/trees/{parse.quote(fallback, safe='')}?recursive=1",
            )
