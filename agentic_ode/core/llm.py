"""Chat model construction and small helpers shared by the agents."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from agentic_ode.core.config import Settings


class ConfigurationError(RuntimeError):
    pass


def make_llm(settings: Settings) -> BaseChatModel:
    """The chat model named by settings.model, through OpenRouter when an OpenRouter key is set."""
    if not settings.model:
        raise ConfigurationError("MODEL is not set (put it in .env or the environment)")
    common = dict(
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    if settings.api_key:
        return init_chat_model(
            settings.model,
            model_provider="openai",
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.request_timeout_s,
            max_retries=settings.request_retries,
            **common,
        )
    return init_chat_model(settings.model, **common)


def with_retry(runnable, attempts: int):
    """Retry transient provider errors (free endpoints report overload as errors in the body)."""
    if hasattr(runnable, "with_retry"):
        return runnable.with_retry(stop_after_attempt=attempts, wait_exponential_jitter=True)
    return runnable


def text_of(msg: BaseMessage) -> str:
    """Message content as a plain string (some providers return a list of blocks)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
