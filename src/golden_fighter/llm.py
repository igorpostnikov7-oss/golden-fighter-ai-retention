"""LLM setup for the crews: model choice, hard limits, call budget, cost estimate and readable API errors.

The agents only orchestrate tools and write prose, so a small, cheap model is enough. Nothing here is
called from tests: tests inject a mock LLM instead (tests/mock_llm.py).
"""
import logging
import os
from contextlib import contextmanager

from golden_fighter import config

log = logging.getLogger(__name__)


class LLMSetupError(RuntimeError):
    """The LLM cannot be used (no key, unknown model). Raised before any API call is made."""


def build_llm(model: str | None = None):
    """The real Anthropic LLM with the project's limits. Fails fast, without an API call, if no key is set."""
    model = model or config.LLM_MODEL
    if not model.startswith("anthropic/"):
        raise LLMSetupError(f"Only Anthropic models are configured for this project, got '{model}'")
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise LLMSetupError("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example); no API call was made.")
    from crewai import LLM

    # No temperature: Claude 5.5 models reject non-default sampling parameters.
    return LLM(model=model, max_tokens=config.LLM_MAX_TOKENS, timeout=config.LLM_TIMEOUT_SECONDS,
               max_retries=config.LLM_MAX_RETRIES)


def model_id(model: str) -> str:
    return model.split("/", 1)[-1]


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimated USD cost from token counts, or None for a model without a known price (e.g. the mock)."""
    prices = config.LLM_PRICES_PER_MTOK.get(model_id(model))
    if prices is None:
        return None
    return round((prompt_tokens * prices[0] + completion_tokens * prices[1]) / 1_000_000, 6)


@contextmanager
def llm_call_budget(max_calls: int = config.MAX_LLM_CALLS_PER_CREW):
    """Block every model call after `max_calls` within the block. Yields a dict with the live count."""
    from crewai.hooks.llm_hooks import register_before_llm_call_hook, unregister_before_llm_call_hook

    counter = {"calls": 0, "max_calls": max_calls, "exceeded": False}

    def hook(context):
        if counter["calls"] >= max_calls:
            counter["exceeded"] = True
            log.error("LLM call budget reached (%d calls); blocking further calls", max_calls)
            return False
        counter["calls"] += 1  # counts only calls that are allowed to reach the model
        return None

    register_before_llm_call_hook(hook)
    try:
        yield counter
    finally:
        unregister_before_llm_call_hook(hook)


def describe_api_error(exc: BaseException) -> str | None:
    """A readable reason for an Anthropic API failure anywhere in the exception chain, or None.

    The SDK's messages never contain the API key, so they are safe to log.
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is a pinned dependency
        return None
    hints = [
        (anthropic.AuthenticationError, "the API key was rejected: check ANTHROPIC_API_KEY in .env"),
        (anthropic.PermissionDeniedError, "the API key has no access to this model"),
        (anthropic.NotFoundError, "model not found: check GF_LLM_MODEL"),
        (anthropic.RateLimitError, "rate limit reached after retries: wait a minute and run again"),
        (anthropic.APITimeoutError, "the API request timed out after retries"),
        (anthropic.APIConnectionError, "could not reach the Anthropic API (network problem)"),
        (anthropic.BadRequestError, "the API rejected the request (for example: credit balance too low)"),
        (anthropic.InternalServerError, "the Anthropic API had a server error or is overloaded; try again later"),
    ]
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        for cls, hint in hints:
            if isinstance(exc, cls):
                return f"{hint} ({type(exc).__name__}: {str(exc)[:300]})"
        exc = exc.__cause__ or exc.__context__
    return None
