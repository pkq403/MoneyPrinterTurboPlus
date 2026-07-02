"""
Tavily (https://tavily.com) integration — optional, opt-in news search.

This module lets MoneyPrinterTurbo fetch the latest news about a video
subject and feed a compact digest into the LLM script prompt, so generated
scripts can reference current events instead of relying solely on the
model's training data.

The integration is fully opt-in and non-breaking:
  * If `tavily_api_key` is not configured, `search_news()` returns an empty
    string, so the script pipeline behaves identically to a build without
    this module.
  * The actual on/off toggle is per-request: callers pass
    `enable_news_search=True` to `llm.generate_script()`. The WebUI exposes
    this as a checkbox in "Advanced Script Settings"; `tavily_search_enabled`
    in config.toml is honored as a default for non-WebUI / API callers.
  * Only the standard `requests` library is used (already a project
    dependency); no extra SDK install is required.
  * Any Tavily outage or malformed response is swallowed and logged, so the
    video generation pipeline can never be broken by a news-search failure.

Config (config.toml, [app] section):
    tavily_api_key = "tvly-xxx"      # required to enable
    tavily_search_enabled = true     # default toggle for API/task callers
    tavily_max_results = 5           # optional override (1-10)
    tavily_topic = "news"            # optional override: "news" or "general"

Configure a Tavily API key from the dashboard (https://app.tavily.com) to
enable this optional integration.
"""

from typing import Optional

import requests
from loguru import logger

from app.config import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
DEFAULT_TOPIC = "news"
MAX_RESULTS_LIMIT = 10
MAX_SNIPPET_LENGTH = 500


def is_enabled() -> bool:
    """True only when a Tavily API key is configured (i.e. the integration *can* run).

    The actual decision to run a search is made by the caller via the
    `enable_news_search` flag on `llm.generate_script()`; this gate only
    guarantees we never call Tavily without credentials.
    """
    return bool(config.app.get("tavily_api_key"))


def _get_tls_verify() -> bool:
    # Mirror app/services/material.py: default to verifying TLS certificates
    # so the Tavily API key and responses cannot be tampered with by a MITM.
    # Only honor tls_verify=false in trusted proxy / self-signed environments.
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")
    return bool(tls_verify)


def _sanitize_error(error: object) -> str:
    """Scrub the Tavily API key from exception text before surfacing it.

    `requests` exceptions usually expose the URL but not the request body,
    so the key should not appear in practice. This is a defensive measure
    to guarantee the key is never leaked via logs or WebUI error messages.
    """
    message = str(error)
    api_key = config.app.get("tavily_api_key") or ""
    if api_key and api_key in message:
        message = message.replace(api_key, "***")
    return message


def _resolve_max_results(max_results: Optional[int]) -> int:
    try:
        value = int(
            max_results
            if max_results is not None
            else config.app.get("tavily_max_results", DEFAULT_MAX_RESULTS)
        )
    except (TypeError, ValueError):
        value = DEFAULT_MAX_RESULTS
    return max(1, min(value, MAX_RESULTS_LIMIT))


def search_news(
    query: str,
    max_results: Optional[int] = None,
    topic: Optional[str] = None,
) -> str:
    """
    Search Tavily for the latest news about `query` and return a formatted
    digest string suitable for injection into an LLM prompt.

    Returns "" when disabled, when `query` is empty, or on any failure, so
    the video pipeline is never broken by a Tavily error.
    """
    if not is_enabled() or not query or not query.strip():
        return ""

    api_key = config.app.get("tavily_api_key")
    topic = topic or config.app.get("tavily_topic", DEFAULT_TOPIC) or DEFAULT_TOPIC
    resolved_max_results = _resolve_max_results(max_results)

    payload = {
        "api_key": api_key,
        "query": query.strip(),
        "topic": topic,
        "max_results": resolved_max_results,
        "search_depth": "basic",
    }

    try:
        logger.info(
            f"Tavily news search: query={query!r}, topic={topic}, "
            f"max_results={resolved_max_results}"
        )
        response = requests.post(
            TAVILY_SEARCH_URL,
            json=payload,
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(15, 30),
        )
        response.raise_for_status()
        data = response.json() or {}
    except Exception as e:  # noqa: BLE001 - never break the pipeline on Tavily errors
        logger.warning(f"Tavily news search failed, skipping: {_sanitize_error(e)}")
        return ""

    results = data.get("results") or []
    if not results:
        logger.info("Tavily news search returned no results")
        return ""

    digest_parts = []
    for index, item in enumerate(results, start=1):
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        published = (item.get("published_date") or "").strip()
        if not title and not content:
            continue
        content = content[:MAX_SNIPPET_LENGTH]
        line = f"{index}. {title}"
        if published:
            line += f" ({published})"
        if content:
            line += f": {content}"
        digest_parts.append(line)

    digest = "\n".join(digest_parts)
    if digest:
        logger.success(
            f"Tavily news digest ({len(digest_parts)} items) prepared for LLM context"
        )
    return digest
