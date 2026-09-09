"""Smoke-test the public, trace-backed portfolio API without a bearer token."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from typing import Any


def get_json(base_url: str, path: str) -> Any:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-build-sha", required=True)
    args = parser.parse_args()

    get_json(args.base_url, "/health/ready")
    status = get_json(args.base_url, "/v1/demo/status")
    actual_build_sha = (
        status.get("build_sha")
        or status.get("build", {}).get("sha")
        or status.get("release", {}).get("build_sha")
    )
    if actual_build_sha != args.expected_build_sha:
        raise SystemExit(
            f"status build mismatch: expected {args.expected_build_sha}, got {actual_build_sha}"
        )

    cases_payload = get_json(args.base_url, "/v1/demo/cases")
    cases = cases_payload.get("cases") if isinstance(cases_payload, dict) else cases_payload
    if not isinstance(cases, list) or not cases:
        raise SystemExit("demo did not expose any curated cases")
    for case in cases:
        case_id = case.get("id") or case.get("case_id")
        if not case_id:
            raise SystemExit("demo case is missing its identifier")
        detail = get_json(
            args.base_url,
            f"/v1/demo/cases/{urllib.parse.quote(str(case_id), safe='')}",
        )
        if not isinstance(detail, dict):
            raise SystemExit(f"demo case {case_id} did not return an object")

    print(f"portfolio smoke: passed ({len(cases)} curated cases)")


if __name__ == "__main__":
    main()
