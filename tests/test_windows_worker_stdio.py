"""Exercise the first worker calls while an MCP server reads a Windows stdio pipe."""

import json
import queue
import subprocess
import sys
import threading

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows standard-handle regression")
def test_first_stdio_calls_do_not_wait_for_another_message():
    proc = subprocess.Popen(
        [sys.executable, "-m", "build123d_mcp.cli"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    replies: queue.Queue[str] = queue.Queue()

    def read_replies():
        for line in proc.stdout:
            replies.put(line)

    threading.Thread(target=read_replies, daemon=True).start()

    def send(message):
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def reply_for(request_id, timeout=60):
        while True:
            try:
                line = replies.get(timeout=timeout)
            except queue.Empty:
                pytest.fail(f"MCP response {request_id} did not arrive within {timeout}s")
            response = json.loads(line)
            if response.get("id") == request_id:
                return response

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1"},
                },
            }
        )
        assert "result" in reply_for(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # No ping or second request is sent while waiting for this response.
        send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "execute", "arguments": {"code": "x = 1 + 1\nprint(x)"}},
            }
        )
        execute = reply_for(2)
        assert "result" in execute and not execute["result"].get("isError")
        assert any("2" in item.get("text", "") for item in execute["result"]["content"])

        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "health_check", "arguments": {}},
            }
        )
        health = reply_for(3)
        assert "result" in health and not health["result"].get("isError")
        health_report = json.loads(health["result"]["content"][0]["text"])
        assert health_report["ok"] is True
    finally:
        proc.kill()
        proc.wait(timeout=10)
