"""Bounded HTTP proxy to the separately configured RSI evaluation service."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
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


def _port_listening_pids(port: int) -> set[int]:
    """PIDs owning a LISTEN socket on *port*; a /proc walk, so Linux-only by design."""
    if not os.path.isdir("/proc/net"):
        return set()
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":  # 0A == TCP_LISTEN
                continue
            if int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    pids: set[int] = set()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            descriptors = list(os.listdir(f"{entry.path}/fd"))
        except OSError:  # other users' processes stay invisible — and untouched
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(f"{entry.path}/fd/{descriptor}")
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                pids.add(int(entry.name))
                break
    return pids


def _process_args(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _matches_backend_command(args: list[str], command: list[str]) -> bool:
    """The process runs the exact backend script we are about to start, for our runtime."""
    script = command[1] if len(command) > 1 else None
    if script is None or not any(arg == script or arg.endswith("/" + Path(script).name) for arg in args):
        return False
    if "--config" in command:
        runtime = command[command.index("--config") + 1]
        return runtime in args
    return True


def _matches_console_command(args: list[str], config_path: str | None) -> bool:
    """An eva console entry point launched with this same --config (execv restarts included)."""
    if not args:
        return False
    if config_path is not None and config_path not in args:
        return False
    return Path(args[0]).name == "eva" or any(arg.endswith("/src/main.py") for arg in args)


def reclaim_stale_services(rsi: dict | None, web_port: int | None,
                           config_path: str | None) -> list[int]:
    """Terminate leftover console/backend processes of this workspace before launch.

    Discovery is by port ownership (whoever listens on the RSI URL and web ports), and a
    candidate is only signalled when its command line is recognizably this workspace's
    console or the backend command we are about to start. Foreign services are left
    untouched; RsiBackend keeps refusing those.
    """
    if rsi is None or not rsi.get("command"):
        return []
    ports: list[int] = []
    try:
        url_port = urlsplit(rsi.get("url", "")).port
    except ValueError:
        url_port = None
    if url_port:
        ports.append(url_port)
    if web_port:
        ports.append(web_port)
    stale: set[int] = set()
    for port in ports:
        for pid in _port_listening_pids(port):
            if pid == os.getpid():
                continue
            args = _process_args(pid)
            if (_matches_backend_command(args, rsi["command"])
                    or _matches_console_command(args, config_path)):
                stale.add(pid)
    if not stale:
        return []

    def _signal(pids: set[int], sig: int) -> None:
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _ports_free() -> bool:
        return not any(_port_listening_pids(port) for port in ports)

    _signal(stale, signal.SIGTERM)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not _ports_free():
        time.sleep(0.2)
    remaining = {pid for pid in stale if Path(f"/proc/{pid}").exists()}
    if remaining:
        _signal(remaining, signal.SIGKILL)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not _ports_free():
            time.sleep(0.1)
    return sorted(stale)


class RsiProxy:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def relay_file(self, path: str, handler) -> None:
        """Stream saved runs without loading complete videos or archives into RAM."""
        opener = build_opener(ProxyHandler({}))
        try:
            response = opener.open(self.url + path, timeout=90)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            status, data, content_type = self._error(502, f"RSI 文件读取失败：{error}")
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
            return
        with response:
            handler.send_response(response.status)
            for name in ("Content-Type", "Content-Length", "Content-Disposition"):
                if response.headers.get(name) is not None:
                    handler.send_header(name, response.headers[name])
            handler.end_headers()
            shutil.copyfileobj(response, handler.wfile, length=1024 * 1024)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, bytes, str]:
        if not self.url:
            return self._error(503, "EVA_RSI_URL 未配置", configured=False)
        if urlsplit(self.url).scheme not in {"http", "https"}:
            return self._error(503, "EVA_RSI_URL 必须为 HTTP 或 HTTPS 地址")
        allowed = {
            "GET": {"/status", "/frame", "/artifacts", "/candidate", "/runs", "/models"},
            "POST": {"/seek", "/run", "/stop", "/replay", "/model"},
        }
        artifact = method == "GET" and path.startswith(("/artifacts/", "/runs/"))
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
