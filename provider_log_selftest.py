"""Offline check that every chat completion records its provider_metadata. No network.

    .venv/bin/python provider_log_selftest.py

The real OpenAI client, _ClineUnwrapTransport and timed_llm_call run unchanged;
only httpx.HTTPTransport.handle_request is replaced, returning a response body
captured from a real ClinePass call on 2026-09-14. Pass criteria (all must hold):
  1. success: the llm_logs entry and the JSONL line carry call_id, resolvedProvider,
     finalProvider, modelAttemptCount and the full provider_metadata
  2. HTTP 500: the failure llm_logs entry carries status 500 and the error body
  3. 8 concurrent calls: each call's record has its own generation_id (no thread mix-up)
  4. a direct client call outside timed_llm_call (as BulletpointAnalyzer makes) is still
     logged, with call_id None - the previous call's label does not leak
  5. JSONL lines == HTTP requests sent; no request left the process
"""
import copy
import glob
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

REPO = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="provider_log_selftest_")
os.environ["ACE_PROVIDER_LOG"] = os.path.join(TMP, "provider.jsonl")
os.environ["ACE_MAX_RETRIES"] = "1"
os.environ["CLINE_BASE_URL"] = "http://selftest.invalid/api/v1"
os.environ.setdefault("CLINE_API_KEY", "selftest-dummy")
sys.path.insert(0, REPO)

import utils  # noqa: E402
from llm import timed_llm_call  # noqa: E402

MODEL = "cline-pass/deepseek-v4-flash"
SAVED = json.load(open(glob.glob(os.path.join(
    REPO, "results/format_trial/identity_20260914/identity_out/*_identity.json"))[0], encoding="utf-8"))["body"]
sent = []
sent_lock = threading.Lock()


def fake_handle_request(self, request):
    with sent_lock:
        sent.append(str(request.url))
    tag = json.loads(request.content)["messages"][0]["content"]
    time.sleep(0.02)
    if tag.startswith("FAIL"):
        body = {"error": "empty response content", "success": False}
        return httpx.Response(500, headers={"content-type": "application/json"},
                              content=json.dumps(body).encode(), request=request)
    body = copy.deepcopy(SAVED)
    body["data"]["id"] = f"gen_{tag}"
    body["data"]["choices"][0]["message"]["provider_metadata"]["gateway"]["routing"]["finalProvider"] = f"prov_{tag}"
    return httpx.Response(200, headers={"content-type": "application/json"},
                          content=json.dumps(body).encode(), request=request)


httpx.HTTPTransport.handle_request = fake_handle_request
client = utils.initialize_clients("clinepass")[0].with_options(max_retries=0)
log_dir = os.path.join(TMP, "llm_logs")


def logged(call_id):
    files = glob.glob(os.path.join(log_dir, f"*_{call_id}_*.json"))
    assert len(files) == 1, (call_id, files)
    return json.load(open(files[0], encoding="utf-8"))


def lines():
    with open(os.environ["ACE_PROVIDER_LOG"], encoding="utf-8") as f:
        return [json.loads(x) for x in f]


# 1. success
text, info = timed_llm_call(client, "clinepass", MODEL, "A", "generator", "test_selftest_a", log_dir=log_dir)
assert text == SAVED["data"]["choices"][0]["message"]["content"]
rec = logged("test_selftest_a")["provider"]
expected_meta = copy.deepcopy(SAVED["data"]["choices"][0]["message"]["provider_metadata"])
expected_meta["gateway"]["routing"]["finalProvider"] = "prov_A"
assert rec["call_id"] == "test_selftest_a" and rec["role"] == "generator"
assert (rec["resolvedProvider"], rec["finalProvider"], rec["modelAttemptCount"]) == ("fireworks", "prov_A", 1), rec
assert rec["provider_metadata"] == expected_meta
assert rec["generation_id"] == "gen_A" and rec["usage"] == SAVED["data"]["usage"]
assert lines()[-1] == rec
print("1 success: PASS")

# 2. HTTP 500
try:
    timed_llm_call(client, "clinepass", MODEL, "FAIL", "curator", "test_selftest_fail", log_dir=log_dir)
    raise AssertionError("500 did not raise")
except AssertionError:
    raise
except Exception:
    pass
rec = logged("test_selftest_fail")["provider"]
assert rec["http_status"] == 500 and "empty response content" in rec["error_body"], rec
assert rec["call_id"] == "test_selftest_fail" and rec["provider_metadata"] is None
print("2 http 500: PASS")

# 3. concurrency
tags = [f"T{i}" for i in range(8)]
with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(lambda t: timed_llm_call(client, "clinepass", MODEL, t, "generator",
                                                     f"test_selftest_{t}", log_dir=log_dir), tags))
for t, (_, info) in zip(tags, results):
    assert info["provider"]["generation_id"] == f"gen_{t}", (t, info["provider"]["generation_id"])
    assert info["provider"]["call_id"] == f"test_selftest_{t}"
    assert info["provider"]["finalProvider"] == f"prov_{t}"
concurrent_lines = [x for x in lines() if (x["call_id"] or "").startswith("test_selftest_T")]
assert len(concurrent_lines) == 8
for line in concurrent_lines:
    assert line["generation_id"] == "gen_" + line["call_id"].rsplit("_", 1)[1], line
print("3 concurrency: PASS")

# 4. direct call outside timed_llm_call
client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "U"}])
last = lines()[-1]
assert last["generation_id"] == "gen_U" and last["call_id"] is None and last["finalProvider"] == "prov_U", last
print("4 unlabelled direct call: PASS")

# 5. completeness, no network
assert len(lines()) == len(sent) == 11, (len(lines()), len(sent))
assert all(u.startswith("http://selftest.invalid/") for u in sent)
print("5 completeness: PASS")
print("ALL PASS -", TMP)
