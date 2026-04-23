from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a remote resource cannot be fetched."""


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    status_code: int
    text: str
    content_type: str | None = None


class HttpFetcher:
    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0,
        user_agent: str = "NarrativeNewsBot/0.1 (+https://example.local)",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.user_agent = user_agent

    def fetch(self, url: str) -> FetchResult:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return self._fetch_once(url)
            except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "Fetch attempt %s failed for %s: %s",
                    attempt,
                    url,
                    exc,
                )
                if attempt > self.max_retries:
                    break
                time.sleep(self.retry_delay_seconds)
        raise FetchError(f"Failed to fetch {url}") from last_error

    def _fetch_once(self, url: str) -> FetchResult:
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.5",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            status_code = getattr(response, "status", 200)
            content_type = response.headers.get("Content-Type")
            body = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
            text = body.decode(encoding, errors="replace")
            return FetchResult(
                url=url,
                status_code=status_code,
                text=text,
                content_type=content_type,
            )
