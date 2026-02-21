from agent.claude import ClaudeProvider
from agent.gemini import GeminiProvider
from agent.gpt import GPTProvider
from agent.provider import AgentProvider


def get_provider(model: str) -> AgentProvider:
    """Single place to switch agent model in the codebase."""
    normalized = model.strip().lower()

    if normalized == "gpt":
        return GPTProvider()
    if normalized == "claude":
        return ClaudeProvider()
    if normalized == "gemini":
        return GeminiProvider()

    raise ValueError(f"Unsupported provider: {model}")
