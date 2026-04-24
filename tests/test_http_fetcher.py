from __future__ import annotations

import unittest
from unittest.mock import patch
from urllib.error import URLError

from app.ingestion.fetcher import FetchResult, HttpFetcher


class RetryingFetcher(HttpFetcher):
    def __init__(self, failures_before_success: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.failures_before_success = failures_before_success
        self.calls = 0

    def _fetch_once(self, url: str) -> FetchResult:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise URLError("temporary network problem")
        return FetchResult(url=url, status_code=200, text="ok", content_type="text/html")


class HttpFetcherTests(unittest.TestCase):
    def test_fetch_retries_with_exponential_backoff(self) -> None:
        fetcher = RetryingFetcher(
            failures_before_success=2,
            timeout_seconds=5.0,
            max_retries=3,
            retry_delay_seconds=2.0,
            retry_backoff_multiplier=2.0,
        )

        with patch("app.ingestion.fetcher.http_fetcher.time.sleep") as sleep_mock:
            result = fetcher.fetch("https://example.com/archive")

        self.assertEqual(result.text, "ok")
        self.assertEqual(fetcher.calls, 3)
        self.assertEqual(
            [call.args[0] for call in sleep_mock.call_args_list],
            [2.0, 4.0],
        )

    def test_default_user_agent_looks_like_browser(self) -> None:
        fetcher = HttpFetcher()
        self.assertIn("Mozilla/5.0", fetcher.user_agent)
        self.assertIn("Chrome/", fetcher.user_agent)


if __name__ == "__main__":
    unittest.main()
