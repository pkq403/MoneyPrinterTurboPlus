import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import llm, tavily


class TestTavilyService(unittest.TestCase):
    """
    Tavily 集成是完全 opt-in 的：未配置 tavily_api_key 时 search_news 必须
    返回空字符串，行为与不接入 Tavily 完全一致，绝不破坏视频生成主链路。
    所有用例都用 mock 替换 requests，CI 不依赖真实网络或真实 API key。
    """

    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    # ---------------- disabled / no-op behavior ----------------

    def test_disabled_when_no_api_key(self):
        config.app.pop("tavily_api_key", None)
        self.assertFalse(tavily.is_enabled())
        # search_news must be a no-op returning "" even with a query
        self.assertEqual(tavily.search_news("anything"), "")

    def test_search_news_returns_empty_for_empty_query(self):
        config.app["tavily_api_key"] = "tvly_test"
        self.assertEqual(tavily.search_news(""), "")
        self.assertEqual(tavily.search_news("   "), "")

    def test_request_not_made_when_disabled(self):
        config.app.pop("tavily_api_key", None)
        with patch.object(tavily, "requests") as req:
            tavily.search_news("AI news")
        req.post.assert_not_called()

    # ---------------- enabled behavior ----------------

    def _mock_response(self, payload):
        resp = patch.object(tavily, "requests").start()
        resp.post.return_value.json.return_value = payload
        resp.post.return_value.raise_for_status.return_value = None
        self.addCleanup(patch.stopall)
        return resp

    def test_search_news_returns_formatted_digest(self):
        config.app["tavily_api_key"] = "tvly_test"
        self._mock_response(
            {
                "results": [
                    {
                        "title": "OpenAI launches X",
                        "content": "a big launch",
                        "published_date": "2026-07-01",
                    },
                    {
                        "title": "Second story",
                        "content": "another snippet",
                        "published_date": "",
                    },
                ]
            }
        )
        digest = tavily.search_news("AI news")
        self.assertIn("OpenAI launches X", digest)
        self.assertIn("2026-07-01", digest)
        self.assertIn("Second story", digest)
        self.assertEqual(digest.count("\n"), 1)

    def test_search_news_returns_empty_on_api_failure(self):
        config.app["tavily_api_key"] = "tvly_test"
        resp = self._mock_response({})
        resp.post.side_effect = Exception("boom: tvly_test leaked?")
        # failures must be swallowed and never raise / never leak the key
        self.assertEqual(tavily.search_news("AI news"), "")

    def test_search_news_returns_empty_when_no_results(self):
        config.app["tavily_api_key"] = "tvly_test"
        self._mock_response({"results": []})
        self.assertEqual(tavily.search_news("AI news"), "")

    def test_api_key_never_logged_in_error(self):
        config.app["tavily_api_key"] = "tvly_secret_key"
        resp = self._mock_response({})
        resp.post.side_effect = Exception("request failed: tvly_secret_key")
        with patch.object(tavily.logger, "warning") as warn:
            tavily.search_news("AI news")
        # the sanitized log message must not contain the raw key
        for call in warn.call_args_list:
            logged = str(call)
            self.assertNotIn("tvly_secret_key", logged)

    # ---------------- llm integration ----------------

    def test_generate_script_does_not_search_when_disabled(self):
        config.app.pop("tavily_api_key", None)
        with (
            patch.object(tavily, "search_news") as sn,
            patch.object(llm, "_generate_response", return_value="脚本内容"),
        ):
            llm.generate_script("主题", enable_news_search=False)
        sn.assert_not_called()

    def test_generate_script_searches_when_enabled(self):
        config.app["tavily_api_key"] = "tvly_test"
        with (
            patch.object(tavily, "search_news", return_value="1. News headline") as sn,
            patch.object(llm, "_generate_response", return_value="脚本内容") as gen,
        ):
            llm.generate_script("AI", enable_news_search=True)
        sn.assert_called_once_with("AI")
        # the news context must end up inside the prompt sent to the LLM
        sent_prompt = gen.call_args.kwargs["prompt"]
        self.assertIn("News headline", sent_prompt)
        self.assertIn("Latest News Context", sent_prompt)


if __name__ == "__main__":
    unittest.main()
