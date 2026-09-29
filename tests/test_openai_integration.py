"""Exercise OpenAIJudge through the real ``openai`` SDK against a local mock server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("openai")

from truthlens.judge import OpenAIJudge, build_messages, parse_judge_response  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    requests = []
    fail_first = True

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append((self.path, body))
        if Handler.fail_first:
            Handler.fail_first = False
            self.send_response(500)
            self.end_headers()
            return
        payload = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "system_fingerprint": "fp_test",
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant",
                                            "content": '{"verdict": "FAKE", "justification": "pupils"}'}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_openai_judge_against_local_server(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        judge = OpenAIJudge(model="gpt-4", temperature=0.0, max_retries=2, timeout=10)
        reply = judge.complete(build_messages("The pupils are unusually dilated."))
    finally:
        server.shutdown()
    assert parse_judge_response(reply).verdict == "FAKE"
    assert len(Handler.requests) == 2  # one 500 + one successful retry
    path, body = Handler.requests[-1]
    assert path.endswith("/chat/completions") and body["model"] == "gpt-4" and body["temperature"] == 0.0
    assert body["messages"][1]["content"].endswith("The pupils are unusually dilated.")
    assert judge.last_system_fingerprint == "fp_test"
