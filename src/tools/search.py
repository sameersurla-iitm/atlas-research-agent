"""
Web search tool.

Primary backend: Brave Search API (free 2k/month, high quality).
Fallback backend: DuckDuckGo via the `ddgs` package (free, no key, unlimited).

The fallback is automatic: if no Brave key is configured, or a Brave request
fails, we transparently use DuckDuckGo so the agent always keeps working.
"""
from __future__ import annotations

from urllib.parse import urlparse

import requests

import config
from src.models import Source


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:  # noqa: BLE001
        return ""


def _credibility(url: str) -> float:
    """Heuristic credibility score, boosted for preferred technical domains."""
    domain = _domain(url)
    for preferred in config.PREFERRED_DOMAINS:
        if domain.endswith(preferred):
            return 0.9
    if domain.endswith((".edu", ".gov")):
        return 0.85
    if domain.endswith(".org"):
        return 0.65
    return 0.5


class SearchTool:
    """Returns a ranked list of `Source` objects for a query."""

    def __init__(self) -> None:
        self.brave_key = config.BRAVE_API_KEY
        self.calls = 0

    def search(self, query: str, limit: int | None = None) -> list[Source]:
        limit = limit or config.SEARCH_RESULTS_PER_QUERY
        self.calls += 1
        if self.brave_key:
            try:
                return self._brave(query, limit)
            except Exception:  # noqa: BLE001 - fall back gracefully
                return self._duckduckgo(query, limit)
        return self._duckduckgo(query, limit)

    # -- backends -----------------------------------------------------------
    def _brave(self, query: str, limit: int) -> list[Source]:
        resp = requests.get(
            config.BRAVE_ENDPOINT,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_key,
            },
            params={"q": query, "count": limit},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        return [
            Source(
                url=r.get("url", ""),
                title=r.get("title", ""),
                domain=_domain(r.get("url", "")),
                credibility=_credibility(r.get("url", "")),
            )
            for r in results[:limit]
            if r.get("url")
        ]

    def _duckduckgo(self, query: str, limit: int) -> list[Source]:
        # Imported lazily so a missing optional dep doesn't break import time.
        from ddgs import DDGS

        sources: list[Source] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=limit):
                url = r.get("href") or r.get("link") or ""
                if not url:
                    continue
                sources.append(
                    Source(
                        url=url,
                        title=r.get("title", ""),
                        domain=_domain(url),
                        credibility=_credibility(url),
                    )
                )
        return sources[:limit]
