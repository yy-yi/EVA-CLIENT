"""Bounded HTTP proxy to the separately configured RSI evaluation service."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class RsiBackend:
    """Own only the configured RSI child; external services remain independently managed."""

    STOP_TIMEOUT_S = 90.0

    def __init__(self, config: dict | None) -> None:
        self.config = config
        self.url = config["url"] if config is not None else ""
        self.process: subprocess.Popen | None = None
        self.previous_sigterm = None
        self.opener = build_opener(ProxyHandler({}))

    def _status_code(self, timeout: float) -> int | None:
        try:
            with self.opener.open(self.url.rstrip("/") + "/status", timeout=timeout) as response:
                return response.status
        except HTTPError as error:
            error.close()
            return error.code
        except (URLError, OSError):
            return None

    def __enter__(self) -> RsiBackend:
        if self.config is None or self.config["command"] is None:
            return self
        if self._status_code(0.5) is not None:
            raise RuntimeError("RSI URL already has a service; refusing to reuse it: " + self.url)
        self.previous_sigterm = signal.signal(signal.SIGTERM, self._interrupt)
        try:
            self.process = subprocess.Popen(
                self.config["command"], cwd=self.config["cwd"], start_new_session=True,
            )
            deadline = time.monotonic() + self.config["start_timeout_s"]
            while True:
                code = self.process.poll()
                if code is not None:
                    raise RuntimeError(f"RSI backend exited before readiness (exit code {code})")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("RSI backend /status readiness timed out: " + self.url)
                if self._status_code(min(0.5, remaining)) == 200:
                    return self
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        except BaseException:
            self.__exit__(None, None, None)
            raise

    @staticmethod
    def _interrupt(signum, frame) -> None:
        raise KeyboardInterrupt

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=self.STOP_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
        finally:
            if self.previous_sigterm is not None:
                signal.signal(signal.SIGTERM, self.previous_sigterm)


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
