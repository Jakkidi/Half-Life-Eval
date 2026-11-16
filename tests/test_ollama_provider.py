"""OllamaProvider against a fake local server speaking the streaming /api/chat protocol."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chl.providers import OllamaProvider

SEEN = []
BEHAVIOUR = {"chunks": ["hello", " there"], "gap": 0.0, "stall_after": None, "status": 200}


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "tiny:latest"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        if BEHAVIOUR["status"] != 200:
            body = b'{"error":"model \\"nope\\" not found"}'
            self.send_response(BEHAVIOUR["status"])
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()  # HTTP/1.0: body runs until the connection closes
        for i, piece in enumerate(BEHAVIOUR["chunks"]):
            if BEHAVIOUR["stall_after"] is not None and i >= BEHAVIOUR["stall_after"]:
                time.sleep(2)  # go silent
                return
            line = {"message": {"role": "assistant", "content": piece}, "done": False}
            self.wfile.write((json.dumps(line) + "\n").encode()); self.wfile.flush()
            time.sleep(BEHAVIOUR["gap"])
        final = {"message": {"role": "assistant", "content": ""}, "done": True,
                 "prompt_eval_count": 42, "eval_count": len(BEHAVIOUR["chunks"])}
        self.wfile.write((json.dumps(final) + "\n").encode())


@pytest.fixture
def server():
    BEHAVIOUR.update(chunks=["hello", " there"], gap=0.0, stall_after=None, status=200)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_chat_round_trip_sets_context_and_seed(server):
    p = OllamaProvider(host=server, num_ctx=8192)
    assert p.list_models() == ["tiny:latest"]
    msgs = [{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}]
    r = p.chat("tiny", msgs, seed=123, temperature=0.2, hint={"ignored": True})
    assert r.text == "hello there" and r.prompt_tokens == 42 and r.completion_tokens == 2
    sent = SEEN[-1]
    assert sent["stream"] is True and sent["messages"] == msgs
    assert sent["options"] == {"temperature": 0.2, "seed": 123, "num_ctx": 8192}


def test_long_reply_outlasting_timeout_still_succeeds(server):
    # The regression behind "qwen fails at episode 40": a reply taking longer in
    # total than the timeout must not fail while tokens keep arriving.
    BEHAVIOUR.update(chunks=["a", "b", "c", "d", "e"], gap=0.25)
    p = OllamaProvider(host=server, timeout=0.6, retries=1)
    t0 = time.perf_counter()
    r = p.chat("tiny", [{"role": "user", "content": "hi"}], seed=0, temperature=0)
    assert r.text == "abcde" and time.perf_counter() - t0 > 0.6


def test_silent_model_times_out_with_helpful_error(server):
    BEHAVIOUR.update(chunks=["a", "b"], stall_after=1)
    p = OllamaProvider(host=server, timeout=0.3, retries=1)
    with pytest.raises(RuntimeError, match="--timeout"):
        p.chat("tiny", [{"role": "user", "content": "hi"}], seed=0, temperature=0)


def test_missing_model_fails_fast_without_retries(server):
    BEHAVIOUR.update(status=404)
    p = OllamaProvider(host=server, retries=3)
    n = len(SEEN)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        p.chat("nope", [{"role": "user", "content": "hi"}], seed=0, temperature=0)
    assert len(SEEN) == n + 1


def test_unreachable_server_gives_actionable_error():
    p = OllamaProvider(host="http://127.0.0.1:9", retries=1, timeout=1)
    with pytest.raises(RuntimeError, match="ollama serve"):
        p.chat("tiny", [{"role": "user", "content": "hi"}], seed=0, temperature=0)


def test_cli_auto_model_runs_end_to_end(server, tmp_path):
    from chl.cli import main
    out = tmp_path / "run.jsonl"
    main(["run", "--models", "auto", "--host", server, "--constraints", "no_digits",
          "--pressures", "benign", "--episodes", "1", "--turns", "2",
          "--interventions", "none", "--out", str(out), "--boot", "20"])
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["model"] == "tiny:latest" and rec["first_violation"] is None
    assert (tmp_path / "run.html").exists()
