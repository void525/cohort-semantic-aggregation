# -*- coding: utf-8 -*-
"""Minimal probe: can LOTUS sem_filter run against the local replay endpoint on this machine?"""
import faulthandler, os, sys, time
faulthandler.dump_traceback_later(150, exit=True)          # if anything hangs, print where and exit
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
t0 = time.time()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lotus_replay as LR
print("import lotus_replay ok", round(time.time() - t0, 1), "s", flush=True)
from lotus.models import LM
print("import LM ok", round(time.time() - t0, 1), "s", flush=True)
tape = LR.Tape({1: [1, 0], 2: [1, 1], 3: [0, 0]})
srv = LR.make_server(tape)
model_name = os.environ.get("PROBE_MODEL", "openai/gpt-4")
lm = LM(model=model_name, api_base=f"http://127.0.0.1:{srv.server_port}/v1", api_key="none", max_batch_size=8)
print("LM built ok", round(time.time() - t0, 1), "s", flush=True)
judge = LR.LotusJudge(lm)
out = judge.judge([1, 2, 3], ["q a\nproduct a", "q b\nproduct b", "q c\nproduct c"])
print("judge ok", out.tolist(), "requests per bit:", tape.requests, round(time.time() - t0, 1), "s", flush=True)
