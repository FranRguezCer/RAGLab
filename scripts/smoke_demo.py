"""Authenticated golden-query smoke test for a deployed demo container."""

from __future__ import annotations

import json
import os
import urllib.request

token = os.environ["RAGLAB_DEMO_TOKEN"]
body = json.dumps(
    {
        "query": "How do I configure a headless Raspberry Pi?",
        "collection": "rpi-computers",
        "history": [],
    }
).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/query",
    data=body,
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
    payload = json.load(response)
if payload.get("abstained") or not payload.get("sources"):
    raise SystemExit("golden query did not return a cited answer")
print("golden query: passed")
