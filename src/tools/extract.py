"""
Content extraction tool.

Primary: Jina Reader (https://r.jina.ai/<url>) — free, returns clean Markdown,
handles JavaScript-rendered pages, and strips ads/nav automatically.

Fallback: a lightweight BeautifulSoup text extraction for the rare case where
Jina is unavailable. Content is truncated to MAX_CONTENT_CHARS so we never blow
the LLM context window with a single huge page.
"""
from __future__ import annotations

import requests

import config


class ExtractTool:
    """Fetches a URL and returns clean, truncated text content."""

    def __init__(self) -> None:
        self.pages_read = 0

    def extract(self, url: str) -> str:
        self.pages_read += 1
        try:
            text = self._jina(url)
        except Exception:  # noqa: BLE001 - fall back to direct fetch
            try:
                text = self._fallback(url)
            except Exception:  # noqa: BLE001 - give up gracefully
                return ""
        return text[: config.MAX_CONTENT_CHARS]

    # -- backends -----------------------------------------------------------
    def _jina(self, url: str) -> str:
        headers = {"Accept": "text/plain"}
        if config.JINA_API_KEY:
            headers["Authorization"] = f"Bearer {config.JINA_API_KEY}"
        resp = requests.get(
            f"{config.JINA_READER_BASE}{url}",
            headers=headers,
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.text.strip()

    def _fallback(self, url: str) -> str:
        from bs4 import BeautifulSoup

        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AtlasResearchBot/1.0)"},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return "\n".join(
            line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
        )
