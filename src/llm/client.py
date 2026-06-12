"""
Unified LLM client.

Groq, Gemini and OpenAI all expose OpenAI-compatible chat endpoints, so we use
the single `openai` SDK and just swap base_url + key + model. This keeps the
provider abstraction tiny and lets the user pick whichever free key they have.

Responsibilities:
  * retries with exponential backoff (transient 429/5xx),
  * best-effort JSON mode with a graceful fallback for providers that reject
    `response_format`,
  * robust JSON extraction from imperfect model output,
  * token + equivalent-cost accounting for the UI.
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

import config


def _parse_retry_delay(exc: BaseException) -> float | None:
    """Extract the server-suggested retry delay (seconds) from a 429 error.

    Checks the Retry-After header first, then falls back to scraping the message
    body (Gemini embeds e.g. "Please retry in 19.1s" and "'retryDelay': '19s'").
    Returns None if no hint is found.
    """
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "headers"):
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except ValueError:
                pass
    msg = str(exc)
    for pattern in (r"retry in ([\d.]+)s", r"retryDelay['\"]?:\s*['\"]?([\d.]+)s"):
        m = re.search(pattern, msg)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def extract_json(text: str) -> dict | list:
    """Best-effort parse of a JSON object/array from a raw LLM response.

    Tries, in order: direct parse, fenced ```json block, first-brace-to-last.
    Raises ValueError if nothing parses.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from response: {text[:200]!r}")


class LLMClient:
    """Thin, provider-agnostic wrapper with usage tracking."""

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.cfg = config.PROVIDERS[self.provider]
        self._has_key = bool(self.cfg["api_key"])
        self.client = OpenAI(
            api_key=self.cfg["api_key"] or "missing-key",
            base_url=self.cfg["base_url"],
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=0,  # we control retries + model failover ourselves
        )
        # usage accounting, keyed per model for accurate cost math
        self.calls = 0
        self.usage_by_model: dict[str, dict[str, int]] = {}
        # rate-limit state: pace calls + remember which models are cooling down
        self._last_call_ts: float = 0.0
        self._cooldown_until: dict[str, float] = {}

    # -- model selection ----------------------------------------------------
    def model_for(self, tier: str) -> str:
        return self.cfg["fast_model"] if tier == "fast" else self.cfg["default_model"]

    def _candidate_models(self, tier: str) -> list[str]:
        """Ordered models to try for a tier, skipping any still in cooldown.

        Falls back to the full chain if every model is cooling down, so we never
        return an empty list (better to try a throttled model than give up).
        """
        chains = self.cfg.get("fallback_chains", {})
        chain = chains.get(tier) or [self.model_for(tier)]
        now = time.time()
        available = [m for m in chain if self._cooldown_until.get(m, 0.0) <= now]
        return available or chain

    def _pace(self) -> None:
        """Sleep just enough to keep a minimum gap between consecutive calls."""
        gap = config.MIN_SECONDS_BETWEEN_CALLS
        if gap <= 0:
            return
        wait = gap - (time.time() - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()

    # -- accounting ---------------------------------------------------------
    def _record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        bucket = self.usage_by_model.setdefault(model, {"input": 0, "output": 0})
        bucket["input"] += prompt_tokens or 0
        bucket["output"] += completion_tokens or 0

    @property
    def input_tokens(self) -> int:
        return sum(b["input"] for b in self.usage_by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(b["output"] for b in self.usage_by_model.values())

    def estimated_cost_usd(self) -> float:
        total = 0.0
        for model, b in self.usage_by_model.items():
            price = config.COST_PER_1M_TOKENS.get(model)
            if not price:
                continue
            total += b["input"] / 1_000_000 * price["input"]
            total += b["output"] / 1_000_000 * price["output"]
        return round(total, 6)

    # -- core call ----------------------------------------------------------
    def _ensure_key(self) -> None:
        if not self._has_key:
            raise RuntimeError(
                f"No API key configured for provider '{self.provider}'. "
                f"Add the key to your .env file (see .env.example)."
            )

    def _create(self, **kwargs):
        """Call the API, retrying once without response_format if unsupported."""
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - we re-raise after one fallback
            if "response_format" in kwargs and _looks_like_format_error(exc):
                kwargs.pop("response_format", None)
                return self.client.chat.completions.create(**kwargs)
            raise

    def complete(
        self,
        system: str,
        user: str,
        tier: str = "default",
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> str:
        """Call the LLM with pacing, 429-aware retries, and model failover.

        Tries each model in the tier's fallback chain. For each model it retries
        up to LLM_MAX_ATTEMPTS, honouring the server's retry delay on 429 (capped
        at MAX_RETRY_WAIT_SECONDS); if the model stays rate-limited it goes into
        cooldown and we move to the next model. Non-transient errors (auth, 403
        proxy blocks, bad request) are raised immediately by _create.
        """
        self._ensure_key()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_exc: BaseException | None = None

        for model in self._candidate_models(tier):
            for attempt in range(config.LLM_MAX_ATTEMPTS):
                self._pace()
                kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                try:
                    resp = self._create(**kwargs)
                    self.calls += 1
                    if resp.usage:
                        self._record(
                            model, resp.usage.prompt_tokens, resp.usage.completion_tokens
                        )
                    return resp.choices[0].message.content or ""
                except RateLimitError as exc:
                    last_exc = exc
                    delay = _parse_retry_delay(exc)
                    if (
                        delay is not None
                        and delay <= config.MAX_RETRY_WAIT_SECONDS
                        and attempt < config.LLM_MAX_ATTEMPTS - 1
                    ):
                        time.sleep(delay + 0.5)
                        continue  # same model, after the suggested wait
                    self._cooldown_until[model] = time.time() + (
                        delay or config.MODEL_COOLDOWN_SECONDS
                    )
                    break  # move to the next model in the chain
                except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
                    last_exc = exc
                    if attempt < config.LLM_MAX_ATTEMPTS - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    break  # exhausted attempts on this model -> next model

        raise last_exc or RuntimeError("All candidate models failed.")

    def complete_json(
        self, system: str, user: str, tier: str = "default", temperature: float = 0.2
    ) -> dict | list:
        raw = self.complete(system, user, tier=tier, json_mode=True, temperature=temperature)
        return extract_json(raw)

    def stream(
        self,
        system: str,
        user: str,
        tier: str = "default",
        temperature: float = 0.4,
    ) -> Iterator[str]:
        """Yield content tokens as they arrive (used by the Synthesizer).

        Uses the same model-failover strategy as complete(): if a model is rate
        limited before any tokens are produced, it cools down and we try the next
        model in the chain. (A failure mid-stream, after tokens have been yielded,
        is propagated rather than restarted.)
        """
        self._ensure_key()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_exc: BaseException | None = None

        for model in self._candidate_models(tier):
            self._pace()
            base_kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True,
            }
            try:
                # include_usage captures token counts on the final chunk, but not
                # every provider accepts it — fall back without it.
                try:
                    stream = self.client.chat.completions.create(
                        **base_kwargs, stream_options={"include_usage": True}
                    )
                except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError):
                    raise  # handled by the outer except -> failover
                except Exception:  # noqa: BLE001 - provider rejected stream_options
                    stream = self.client.chat.completions.create(**base_kwargs)

                self.calls += 1
                produced = False
                for chunk in stream:
                    if getattr(chunk, "usage", None):
                        self._record(
                            model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                        )
                    if chunk.choices and chunk.choices[0].delta.content:
                        produced = True
                        yield chunk.choices[0].delta.content
                return  # stream finished successfully
            except RateLimitError as exc:
                last_exc = exc
                delay = _parse_retry_delay(exc)
                self._cooldown_until[model] = time.time() + (
                    delay or config.MODEL_COOLDOWN_SECONDS
                )
                continue  # try the next model
            except (APITimeoutError, APIConnectionError, InternalServerError) as exc:
                last_exc = exc
                continue

        if last_exc:
            raise last_exc


def _looks_like_format_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "response_format" in msg or "json" in msg or "not supported" in msg
