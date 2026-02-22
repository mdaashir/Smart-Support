"""
Smart-Support Demo Script — Milestone 1 + 2
============================================
Showcases all major API capabilities for a live demo.

Prerequisites:
    # Terminal 1 - Redis
    docker run --name smart-support-redis -p 6379:6379 -d redis:7-alpine

    # Terminal 2 - API
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Terminal 3 - Worker
    uv run arq api.worker.WorkerSettings

    # Terminal 4 - Demo
    uv run python demo.py
"""
import requests
import time
import json

BASE = "http://localhost:8000"
SEP = "=" * 60


def banner(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def pretty(d: dict) -> str:
    return json.dumps(d, indent=4)


# ─────────────────────────────────────────────────────────────
# 1. Health check
# ─────────────────────────────────────────────────────────────
banner("1. Health Check  GET /health")
h = requests.get(f"{BASE}/health").json()
print(f"  Status      : {h['status']}")
print(f"  M1 model    : {h['m1']}")
print(f"  M2 broker   : {h['m2_broker']}")
print(f"  M1 accuracy : {h.get('m1_accuracy', 'N/A')}")
print(f"  Redis queue : {h.get('redis', {})}")

input("\n  [Press Enter to continue]\n")

# ─────────────────────────────────────────────────────────────
# 2. Milestone 1 — Sync Classification (POST /v1/tickets)
# ─────────────────────────────────────────────────────────────
banner("2. Milestone 1 — Sync Classification  POST /v1/tickets")
print("  M1 uses TF-IDF char n-grams (3-5) + LinearSVC.")
print("  Returns 201 immediately with category + urgency_level.\n")

m1_tickets = [
    {
        "subject": "URGENT: Production database down",
        "body": "Our main production database is completely down. All users affected. Critical revenue impact. Fix ASAP!",
        "expected_category": "Technical",
    },
    {
        "subject": "Invoice discrepancy",
        "body": "My invoice dated last month shows an incorrect charge. Please review and correct my billing.",
        "expected_category": "Billing",
    },
    {
        "subject": "Employee contract review needed",
        "body": "We need legal team to review the new vendor contract before signing. Please advise on clauses 7 and 12.",
        "expected_category": "Legal",
    },
    {
        "subject": "Scheduled system maintenance",
        "body": "Routine database maintenance is planned for Sunday 2am-4am EST. No action required.",
        "expected_category": "Technical",
    },
]

print(f"  {'Subject':<40} {'Category':<12} {'Urgency'}")
print(f"  {'-'*40} {'-'*12} {'-'*7}")
for t in m1_tickets:
    r = requests.post(f"{BASE}/v1/tickets", json={"subject": t["subject"], "body": t["body"]})
    d = r.json()
    urgency_label = "HIGH" if d["urgency_level"] == 1 else "NORMAL"
    match = "✓" if d["category"] == t["expected_category"] else "?"
    print(f"  {t['subject'][:40]:<40} {d['category']:<12} {urgency_label}  {match}")

input("\n  [Press Enter to continue]\n")

# ─────────────────────────────────────────────────────────────
# 3. Milestone 1 — Priority Queue (GET /v1/tickets/next)
# ─────────────────────────────────────────────────────────────
banner("3. Milestone 1 — Priority Queue  GET /v1/tickets/next")
print("  In-memory heapq — urgent tickets (level=1) pop first.\n")

print(f"  {'Order':<6} {'Subject':<40} {'Category':<12} {'Urgency'}")
print(f"  {'-'*6} {'-'*40} {'-'*12} {'-'*7}")
i = 0
while True:
    r = requests.get(f"{BASE}/v1/tickets/next")
    if r.status_code == 404:
        print("  Queue empty.")
        break
    d = r.json()
    i += 1
    urgency_label = "HIGH" if d["urgency_level"] == 1 else "normal"
    print(f"  #{i:<5d} {d['subject'][:40]:<40} {d['category']:<12} {urgency_label}")

input("\n  [Press Enter to continue]\n")

# ─────────────────────────────────────────────────────────────
# 4. Milestone 2 — Async Inference (POST /v2/tickets → 202)
# ─────────────────────────────────────────────────────────────
banner("4. Milestone 2 — Async Inference  POST /v2/tickets")
print("  M2 uses frozen DistilBERT [CLS] + LogReg + Ridge urgency regressor.")
print("  Returns 202 immediately; worker processes in background.\n")

m2_tickets = [
    {
        "subject": "CRITICAL: All services unreachable",
        "body": "Complete outage! Production systems are down. Immediate response needed. Revenue loss every second!",
    },
    {
        "subject": "Overcharged on subscription",
        "body": "I was charged twice for November subscription. Please refund the duplicate payment.",
    },
    {
        "subject": "Login portal completely broken",
        "body": "I cannot log in to the customer portal. Getting 500 error since this morning. Please fix urgently.",
    },
]

job_ids = []
print(f"  {'Subject':<45} {'Job ID'}")
print(f"  {'-'*45} {'-'*36}")
for t in m2_tickets:
    r = requests.post(f"{BASE}/v2/tickets", json={"subject": t["subject"], "body": t["body"]})
    d = r.json()
    job_ids.append((t["subject"], d["ticket_id"]))
    print(f"  {t['subject'][:45]:<45} {d['ticket_id']}")
print("\n  Tickets queued! Worker is processing them asynchronously via arq + Redis.")

wait = 20
print(f"\n  Waiting {wait}s for DistilBERT inference to complete...")
for remaining in range(wait, 0, -5):
    time.sleep(5)
    print(f"  ...{remaining - 5}s remaining")

input("\n  [Press Enter to see results]\n")

# ─────────────────────────────────────────────────────────────
# 5. Milestone 2 — Poll Results (GET /v2/tickets/{id})
# ─────────────────────────────────────────────────────────────
banner("5. Milestone 2 — Poll Results  GET /v2/tickets/{id}")
print("  Each result includes category, continuous urgency score S∈[0,1], confidence.\n")

print(f"  {'Subject':<45} {'Cat':<12} {'Urgency':<9} {'Conf':<7} {'Status'}")
print(f"  {'-'*45} {'-'*12} {'-'*9} {'-'*7} {'-'*10}")
for subj, jid in job_ids:
    r = requests.get(f"{BASE}/v2/tickets/{jid}")
    d = r.json()
    cat    = d.get("category", "pending")
    urg    = d.get("urgency_score")
    conf   = d.get("confidence")
    status = d.get("status", "?")
    urg_s  = f"{urg:.3f}" if isinstance(urg, float) else "---"
    conf_s = f"{conf:.2f}" if isinstance(conf, float) else "---"
    webhook = " *** WEBHOOK ALERT ***" if isinstance(urg, float) and urg > 0.8 else ""
    print(f"  {subj[:45]:<45} {cat:<12} {urg_s:<9} {conf_s:<7} {status}{webhook}")

input("\n  [Press Enter to continue]\n")

# ─────────────────────────────────────────────────────────────
# 6. Milestone 2 — Urgency-Sorted Dequeue (ZPOPMAX)
# ─────────────────────────────────────────────────────────────
banner("6. Milestone 2 — Priority Dequeue  GET /v2/tickets/next")
print("  Redis ZPOPMAX — highest urgency_score dequeued first.\n")

print(f"  {'Order':<6} {'Subject':<45} {'Category':<12} {'Urgency Score'}")
print(f"  {'-'*6} {'-'*45} {'-'*12} {'-'*13}")
i = 0
while True:
    r = requests.get(f"{BASE}/v2/tickets/next")
    if r.status_code == 404:
        print("  Queue empty.")
        break
    d = r.json()
    i += 1
    subj = d.get("subject", "?")[:45]
    cat  = d.get("category", "?")
    urg  = float(d.get("urgency_score", 0))
    print(f"  #{i:<5d} {subj:<45} {cat:<12} {urg:.3f}")

input("\n  [Press Enter to continue]\n")

# ─────────────────────────────────────────────────────────────
# 7. Model Metrics (GET /metrics)
# ─────────────────────────────────────────────────────────────
banner("7. Model Metrics  GET /metrics")
print("  Shows training/evaluation metrics for both models.\n")

r = requests.get(f"{BASE}/metrics")
d = r.json()

m1m = d.get("m1") or {}
m2m = d.get("m2") or {}

print("  --- M1: TF-IDF char_wb(3,5) + LinearSVC ---")
print(f"  Accuracy           : {m1m.get('accuracy', 'N/A'):.4f}")
print(f"  Weighted avg F1    : {(m1m.get('weighted avg') or {}).get('f1-score', 'N/A'):.4f}")
print(f"  Train size         : {m1m.get('train_size', 'N/A')} | Eval size: {m1m.get('eval_size', 'N/A')}")

print()
print("  --- M2: DistilBERT [CLS] + LogReg + Ridge ---")
print(f"  Category accuracy  : {m2m.get('accuracy', 'N/A'):.4f}")
print(f"  Urgency RMSE       : {m2m.get('rmse', 'N/A'):.4f}")
print(f"  Urgency MAE        : {m2m.get('mae', 'N/A'):.4f}")
print(f"  Train sample       : {m2m.get('train_sample', 'N/A')} | Test sample: {m2m.get('test_sample', 'N/A')}")

print()
per_class = {
    "Billing":   m1m.get("Billing") or {},
    "Legal":     m1m.get("Legal") or {},
    "Technical": m1m.get("Technical") or {},
}
print("  M1 Per-class F1:")
for cls, v in per_class.items():
    f1 = v.get("f1-score", "N/A")
    if isinstance(f1, float):
        print(f"    {cls:<12}: {f1:.3f}")

input("\n  [Press Enter to finish]\n")

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
banner("Demo Complete!")
print("  Milestone 1: TF-IDF + LinearSVC  |  sync 201  |  in-memory heapq")
print(f"  M1 Accuracy: {m1m.get('accuracy', 'N/A'):.1%}  |  Weighted F1: {(m1m.get('weighted avg') or {}).get('f1-score', 0):.1%}")
print()
print("  Milestone 2: DistilBERT + LogReg + Ridge  |  async 202  |  Redis ZPOPMAX")
print(f"  M2 Category accuracy: {m2m.get('accuracy', 0):.1%}  |  Urgency RMSE: {m2m.get('rmse', 0):.4f}")
print()
print("  Highlights:")
print("    - 6-band urgency labels (0.10 / 0.30 / 0.45 / 0.65 / 0.80 / 1.00)")
print("    - Confidence score via predict_proba on LogReg classifier")
print("    - Slack/webhook alert triggered when urgency_score > 0.8")
print("    - Atomic Redis lock prevents duplicate processing")
print("    - /metrics endpoint exposes both model evaluation reports")
print()
