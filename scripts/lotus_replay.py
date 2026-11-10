# -*- coding: utf-8 -*-
"""COHORT over an existing semantic engine (LOTUS): replay endpoint + judge adapter.

LOTUS talks to its language model through LiteLLM, i.e. an OpenAI-compatible HTTP endpoint. We start a tiny
local endpoint that answers every chat request with the bit stored in the frozen label tape for the record named
in the prompt, and let LOTUS' own ``sem_filter`` operator run unmodified against it. ``LotusJudge`` is then the
only thing COHORT sees: a ``judge(ids, texts) -> (n, K)`` callback, exactly like the local-GPU operator.

No model runs here; the tape was produced earlier by Qwen3-8B. The experiment therefore checks the plumbing
(interface, request accounting, token accounting), not LOTUS' semantic accuracy.
"""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from cohortrag.semantics import SemanticContract

ID_RE = re.compile(r"\[(\d+)\]")

# one natural-language predicate per judged bit (relevant, exact)
LANGEX = (
    "The product is at least partially relevant to the search query.",
    "The product is an exact match for the search query.",
)


class Tape:
    """Frozen label tape plus request counters (thread-safe)."""

    def __init__(self, labels):
        self.labels = labels
        self.lock = threading.Lock()
        self.requests = [0] * 2
        self.prompt_chars = 0

    def answer(self, messages):
        user = [m["content"] for m in messages if m.get("role") == "user"][-1]
        if isinstance(user, list):
            user = " ".join(p.get("text", "") for p in user if isinstance(p, dict))
        rid = int(ID_RE.search(user).group(1))
        claim = user.split("Claim:")[-1].lower()
        j = 1 if "exact match" in claim else 0
        with self.lock:
            self.requests[j] += 1
            self.prompt_chars += len(json.dumps(messages))
        return "True" if int(self.labels[rid][j]) else "False"


def make_server(tape, port=0):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            text = tape.answer(body["messages"])
            plen = max(1, len(json.dumps(body["messages"])) // 4)
            out = {"id": "chatcmpl-replay", "object": "chat.completion", "created": 0,
                   "model": body.get("model", "replay"),
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": plen, "completion_tokens": 1, "total_tokens": plen + 1}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class LotusJudge(SemanticContract):
    """COHORT's judge callback implemented with LOTUS ``sem_filter`` (one call per judged bit)."""

    def __init__(self, lm, langex=LANGEX, name="lotus-sem_filter"):
        super().__init__(name=name, K=len(langex))
        self.lm = lm
        self.langex = langex
        self.n_records = 0

    def judge(self, ids, texts):
        from lotus.sem_ops.sem_filter import sem_filter
        docs = [{"text": f"[{int(i)}] {t}"} for i, t in zip(ids, texts)]
        out = np.zeros((len(ids), self.K), dtype=np.int8)
        for j, instr in enumerate(self.langex):
            res = sem_filter(docs, self.lm, instr, show_progress_bar=False, default=False)
            out[:, j] = np.asarray(res.outputs, dtype=np.int8)
        self.n_records += len(ids)
        return out
