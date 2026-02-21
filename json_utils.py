from __future__ import annotations

import json


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_object(text: str) -> tuple[dict, str]:
    candidate = strip_code_fences(text)
    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")

    depth = 0
    in_string = False
    escaped = False
    end = -1

    for index, char in enumerate(candidate[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end == -1:
        raise ValueError("Unterminated JSON object in text")

    raw = candidate[start : end + 1]
    trailing = candidate[end + 1 :].strip()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON payload must be an object")
    return payload, trailing
