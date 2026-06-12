"""
Central configuration for the Atlas Research Agent.

Everything tunable lives here: provider endpoints, per-agent model tiers,
search/extraction settings, pipeline limits, and cost-tracking tables.
All secrets are read from environment variables (see .env.example).
"""
from __future__ import annotations

import os

# Use the operating-system trust store for TLS verification. This is essential
# on managed / corporate networks that perform TLS inspection with a private
# root CA: that CA lives in the OS certificate store but NOT in certifi's
# bundle, so requests/httpx (and therefore the LLM API calls) would otherwise
# fail with CERTIFICATE_VERIFY_FAILED. Safe no-op on normal networks.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - fall back to default certifi behaviour
    pass

from dotenv import load_dotenv

load_dotenv()


def _load_streamlit_secrets_into_env() -> None:
    """Mirror Streamlit Cloud secrets into environment variables.

    Streamlit Community Cloud exposes dashboard-defined secrets via ``st.secrets``.
    We copy them into ``os.environ`` (without overriding anything already set by a
    local ``.env``) so the same ``os.getenv``-based config works identically in
    local dev and in the cloud. Wrapped in try/except so the evaluation CLI, which
    runs with no Streamlit runtime or secrets file, is unaffected.
    """
    try:
        import streamlit as st

        for key in (
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "BRAVE_API_KEY",
            "JINA_API_KEY",
            "LLM_PROVIDER",
            "APP_PASSWORD",
        ):
            if not os.getenv(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:  # noqa: BLE001 - no Streamlit runtime / no secrets file
        pass


_load_streamlit_secrets_into_env()


# ------------------------------------------------------------------------------
# LLM providers — all expose OpenAI-compatible endpoints, so a single client
# (the `openai` SDK) can talk to any of them just by swapping base_url + key.
# ------------------------------------------------------------------------------
PROVIDERS: dict[str, dict] = {
    "groq": {
        "label": "Groq (Llama 3.3)",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "default_model": "llama-3.3-70b-versatile",
        "fast_model": "llama-3.1-8b-instant",
        # NOTE: api.groq.com is blocked by some corporate proxies (e.g. Zscaler).
        # If requests fail with a 403 / block page, use the "gemini" provider.
    },
    "gemini": {
        "label": "Google Gemini 2.5 Flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "default_model": "gemini-2.5-flash",
        "fast_model": "gemini-2.5-flash-lite",
        # Free-tier quotas are PER MODEL, so rotating across several models when
        # one is rate-limited multiplies effective throughput. The client tries
        # these in order, skipping any that are in cooldown after a 429.
        "fallback_chains": {
            "default": [
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-flash-lite-latest",
            ],
            "fast": [
                "gemini-2.5-flash-lite",
                "gemini-flash-lite-latest",
                "gemini-2.5-flash",
            ],
        },
    },
    "openai": {
        "label": "OpenAI GPT-4o",
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "default_model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
    },
}

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()

# Per-agent model tier. "default" = stronger reasoning, "fast" = cheap/quick.
# This is the core cost-optimisation lever: simple extraction uses the fast
# model, while planning / critique / synthesis use the stronger model.
AGENT_MODEL_TIER: dict[str, str] = {
    "planner": "default",
    "researcher": "fast",
    "critic": "default",
    "synthesizer": "default",
}

# Approximate public pricing (USD per 1M tokens) — used ONLY to display an
# "equivalent cost" estimate. Groq/Gemini free tiers cost $0 in reality, but
# showing the production-equivalent spend is a strong portfolio signal.
COST_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
    "gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

# LLM HTTP behaviour — keep these tight so a blocked or slow endpoint fails fast
# instead of hanging. (The OpenAI SDK default timeout is 600s, which is what made
# a proxy-blocked request appear to "hang" for ~10 minutes.)
LLM_TIMEOUT_SECONDS: float = 60.0
LLM_MAX_ATTEMPTS: int = 3

# Free-tier rate-limit handling:
#  * pace calls so we never burst past the per-minute quota,
#  * honour the server's suggested retry delay on 429 (capped),
#  * after a 429, put that model in cooldown and rotate to the next in the chain.
MIN_SECONDS_BETWEEN_CALLS: float = 2.5
MAX_RETRY_WAIT_SECONDS: float = 25.0
MODEL_COOLDOWN_SECONDS: float = 30.0


# ------------------------------------------------------------------------------
# Web search
# ------------------------------------------------------------------------------
BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")
BRAVE_ENDPOINT: str = "https://api.search.brave.com/res/v1/web/search"
SEARCH_RESULTS_PER_QUERY: int = 5
MAX_SOURCES_PER_SUBQUESTION: int = 2


# ------------------------------------------------------------------------------
# Content extraction (Jina Reader -> clean markdown)
# ------------------------------------------------------------------------------
JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
JINA_READER_BASE: str = "https://r.jina.ai/"
MAX_CONTENT_CHARS: int = 8000  # truncate very long pages before sending to LLM
REQUEST_TIMEOUT: int = 30


# ------------------------------------------------------------------------------
# Pipeline behaviour
# ------------------------------------------------------------------------------
MAX_SUB_QUESTIONS: int = 5
MAX_CRITIC_ITERATIONS: int = 1  # how many refine loops the critic may trigger

# AI/ML "bias": these domains get a credibility boost during research, nudging
# the agent toward authoritative technical sources.
PREFERRED_DOMAINS: list[str] = [
    "arxiv.org",
    "paperswithcode.com",
    "huggingface.co",
    "github.com",
    "ai.googleblog.com",
    "openai.com",
    "anthropic.com",
    "deepmind.com",
    "research.google",
    "microsoft.com",
]


def active_provider() -> dict:
    """Return the configuration dict for the currently selected provider."""
    return PROVIDERS.get(LLM_PROVIDER, PROVIDERS["groq"])


def has_llm_key(provider: str | None = None) -> bool:
    """True if an API key is present for the given (or active) provider."""
    cfg = PROVIDERS.get((provider or LLM_PROVIDER).lower(), {})
    return bool(cfg.get("api_key"))
