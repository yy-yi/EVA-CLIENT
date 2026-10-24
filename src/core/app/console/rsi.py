"""Bounded HTTP proxy to the separately configured RSI evaluation service."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class RsiProxy:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes, str]:
        if not self.url:
            return self._error(503, "EVA_RSI_URL 未配置", configured=False)
        if urlsplit(self.url).scheme not in {"http", "https"}:
            return self._error(503, "EVA_RSI_URL 必须为 HTTP 或 HTTPS 地址")
        allowed = {
            "GET": {"/status", "/frame", "/artifacts"},
            "POST": {"/seek", "/run", "/stop"},
        }
        artifact = method == "GET" and path.startswith("/artifacts/")
        if path not in allowed[method] and not artifact:
            return self._error(404, "RSI 接口不存在")
        request = Request(
            self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        # RSI is a configured local service; external HTTP proxies must not route its frames.
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=5) as response:
                return response.status, response.read(), response.headers.get_content_type()
        except HTTPError as error:
            return error.code, error.read(), error.headers.get_content_type()
        except (URLError, TimeoutError, OSError) as error:
            return self._error(502, f"RSI 服务连接失败：{error}")

    @staticmethod
    def _error(status: int, error: str, *, configured: bool = True) -> tuple[int, bytes, str]:
        return (
            status,
            json.dumps({"ok": False, "configured": configured, "error": error}).encode(),
            "application/json",
        )
