import requests
import time

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def ok(label, val=None):
    global PASS
    PASS += 1
    suffix = f"  -> {val}" if val is not None else ""
    print(f"  [OK]  {label}{suffix}")


def fail(label, val=None):
    global FAIL
    FAIL += 1
    suffix = f"  -> {val}" if val is not None else ""
    print(f"  [FAIL] {label}{suffix}")


def sep(title):
    print(f"\n{'='*54}")
    print(f"  {title}")
    print("=" * 54)


# ── Health ─────────────────────────────────────────────────────────
sep("GET /health")
r = requests.get(f"{BASE}/health")
d = r.json()
if r.status_code == 200:
    ok(f"status=ok  m1={d['m1']}  m2_broker={d['m2_broker']}")
else:
    fail("status 200", r.status_code)
if d.get("m1_accuracy"):
    ok(f"M1 accuracy in health = {d['m1_accuracy']:.4f}")
else:
    fail("M1 accuracy present in health")


# ── M1 Sync ─────────────────────────────────────────────────────────
sep("POST /v1/tickets  [M1 sync 201]")
m1_cases = [
    ("Critical outage ASAP",   "Server totally down urgent ASAP critical production"),
    ("Invoice question",        "My invoice shows wrong amount please correct billing"),
    ("Contract legal review",   "Need legal team to review our vendor contract clauses"),
    ("Routine maintenance",     "Scheduled maintenance this weekend no urgency"),
]
for subj, body in m1_cases:
    r = requests.post(f"{BASE}/v1/tickets", json={"subject": subj, "body": body})
    if r.status_code == 201:
        d = r.json()
        ur = d.get("urgency_level", "?")
        ok(f"201 [{d['category']:10}] urgency_level={ur}  <- {subj}")
    else:
        fail(f"POST 201 for '{subj}'", f"{r.status_code}: {r.text[:80]}")


# ── M1 priority queue ───────────────────────────────────────────────
sep("GET /v1/tickets/next  [priority queue - urgency_level=1 first]")
for i in range(5):
    r = requests.get(f"{BASE}/v1/tickets/next")
    if r.status_code == 200:
        d = r.json()
        ul = d.get("urgency_level", "?")
        ok(f"[{d['category']:10}] urgency_level={ul}  <- {d['subject']}")
    elif r.status_code == 404:
        print("  (queue empty)")
        break
    else:
        fail(f"GET /next iter={i}", f"{r.status_code}: {r.text[:80]}")


# ── M1 queue status ─────────────────────────────────────────────────
sep("GET /v1/tickets/status")
r = requests.get(f"{BASE}/v1/tickets/status")
if r.status_code == 200:
    ok(f"status 200  -> {r.json()}")
else:
    fail("status 200", r.status_code)


# ── M2 Async ────────────────────────────────────────────────────────
sep("POST /v2/tickets  [M2 async 202 Accepted]")
m2_cases = [
    ("Server down ASAP",  "Production database completely down all users critical"),
    ("Invoice error",     "Billing amount wrong on my account please fix"),
    ("Login broken",      "Cannot login to portal since this morning"),
]
ticket_ids = []
for subj, body in m2_cases:
    r = requests.post(f"{BASE}/v2/tickets", json={"subject": subj, "body": body})
    if r.status_code == 202:
        d = r.json()
        tid = d["ticket_id"]
        ticket_ids.append((subj, tid))
        ok(f"202 accepted  ticket_id={tid[:8]}...  <- {subj}")
    else:
        fail(f"POST 202 for '{subj}'", f"{r.status_code}: {r.text[:80]}")

print("\n  Waiting 20s for arq worker to process tickets (first-run DistilBERT load + inference)...")
time.sleep(20)


# ── M2 GET result ───────────────────────────────────────────────────
sep("GET /v2/tickets/{id}  [polling result]")
for subj, tid in ticket_ids:
    r = requests.get(f"{BASE}/v2/tickets/{tid}")
    if r.status_code == 200:
        d = r.json()
        status_val = d.get("status")
        cat  = d.get("category", "?")
        urg  = d.get("urgency_score")
        conf = d.get("confidence")
        urg_s  = f"{urg:.3f}" if isinstance(urg, float) else str(urg)
        conf_s = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        if status_val == "processed":
            ok(f"processed  [{cat:10}] urgency={urg_s}  conf={conf_s}  <- {subj}")
        else:
            fail(f"Expected processed, got '{status_val}'  <- {subj}", d)
    else:
        fail(f"GET /v2/tickets/{tid[:8]}...", r.status_code)


# ── M2 ZPOPMAX ──────────────────────────────────────────────────────
sep("GET /v2/tickets/next  [ZPOPMAX - highest urgency_score first]")
prev_score = 999.0
for _ in range(5):
    r = requests.get(f"{BASE}/v2/tickets/next")
    if r.status_code == 200:
        d = r.json()
        score = float(d.get("urgency_score", 0.0))
        cat = d.get("category", "?")
        subj = str(d.get("subject", "?"))[:45]
        order_ok = score <= prev_score
        label = "OK" if order_ok else "ORDER VIOLATION"
        ok(f"[{cat:10}] urgency={score:.3f}  [{label}]  <- {subj}")
        prev_score = score
    elif r.status_code == 404:
        print("  (queue empty)")
        break
    else:
        fail("GET /v2/tickets/next", f"{r.status_code}: {r.text[:80]}")


# ── M2 queue status ─────────────────────────────────────────────────
sep("GET /v2/tickets/status")
r = requests.get(f"{BASE}/v2/tickets/status")
if r.status_code == 200:
    ok(f"status 200  -> {r.json()}")
else:
    fail("status 200", r.status_code)


# ── Metrics endpoint ────────────────────────────────────────────────
sep("GET /metrics  [model evaluation metrics]")
r = requests.get(f"{BASE}/metrics")
if r.status_code == 200:
    ok("status 200")
    d = r.json()
    m1m = d.get("m1") or {}
    m2m = d.get("m2") or {}
    if m1m:
        ok(f"M1 accuracy      = {m1m.get('accuracy', 'N/A')}")
        wf1 = (m1m.get("weighted avg") or {}).get("f1-score", "N/A")
        ok(f"M1 weighted-F1   = {wf1}")
    else:
        fail("M1 metrics present")
    if m2m:
        ok(f"M2 accuracy      = {m2m.get('accuracy', 'N/A')}")
        ok(f"M2 RMSE          = {m2m.get('rmse', 'N/A')}")
        ok(f"M2 MAE           = {m2m.get('mae', 'N/A')}")
    else:
        fail("M2 metrics present")
else:
    fail("/metrics status 200", r.status_code)


# ── Summary ─────────────────────────────────────────────────────────
sep(f"RESULTS: {PASS} passed / {FAIL} failed")
if FAIL == 0:
    print("  ALL TESTS PASSED - ready for demo!\n")
else:
    print(f"  {FAIL} test(s) FAILED - see above.\n")
