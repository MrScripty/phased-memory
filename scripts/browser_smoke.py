"""Opt-in browser verification using installed Playwright and Chromium.

Runs this app with a temporary SQLite database and human/synthetic fixture inputs.
No simulated inference server is installed and no model inference is requested.
"""
import argparse
import json
import re
import signal
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def run(binary, chromium):
    checks = []
    with tempfile.TemporaryDirectory(prefix="phased-memory-browser-") as work:
        work = Path(work)
        config = work / "config.json"
        config.write_text(json.dumps({"endpoint": "http://127.0.0.1:8080",
                                      "extraction": None, "embedding": None,
                                      "decision": None, "decision_adapter": None}))
        process = subprocess.Popen([str(binary), "serve", str(work / "memory.sqlite"),
                                    str(config), "0"], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        try:
            line = process.stdout.readline()
            address = re.search(r"http://127\.0\.0\.1:\d+", line)
            assert address, f"Server did not announce loopback URL: {line}"
            url = address.group()
            with sync_playwright() as pw:
                browser = pw.chromium.launch(executable_path=chromium, headless=True,
                                             args=["--no-sandbox"])
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url)
                    assert page.title() == "Phased Memory"
                    assert page.locator("#status").inner_text() == "Ready. No inference has run."
                    checks.append("page loads with explicit unrun inference status")

                    def click_response(button):
                        with page.expect_response(lambda r: r.url == url + "/api") as pending:
                            page.locator(button).click()
                        response = pending.value
                        body = response.json()
                        assert response.ok, body
                        page.wait_for_function("!document.getElementById('search').disabled")
                        return body

                    page.locator("#batch").fill((ROOT / "fixtures/authored/corpus.json").read_text())
                    result = click_response("#ingest")
                    assert result["claims"] == 10
                    checks.append("human claims and explicitly synthetic vectors persist atomically")
                    page.locator("#synthetic").check()
                    page.locator("#query").fill("Latest evidence of Alice trusting Bob")
                    filters = {"subject": "Alice", "predicate": "trusts", "object": "Bob",
                               "perspective": "world", "continuity": "main",
                               "timeline": "story", "latest": True}
                    page.locator("#filters").fill(json.dumps(filters))
                    result = click_response("#search")
                    assert {h["claim"]["id"] for h in result["structured"]} == {"m02", "m09"}
                    assert result["reranked"] is None
                    assert page.locator("#structured .hit").count() == 2
                    checks.append("latest story state preserves conflicting evidence; rerank stays unavailable")

                    filters.pop("latest")
                    filters["chronological"] = True
                    page.locator("#filters").fill(json.dumps(filters))
                    result = click_response("#search")
                    assert [h["claim"]["id"] for h in result["structured"]] == ["m03", "m01", "m02", "m09"]
                    checks.append("late-ingested flashback orders by story time")

                    page.locator("#filters").fill(json.dumps({"subject": "Bob", "predicate": "trusts", "object": "Alice"}))
                    result = click_response("#search")
                    assert [h["claim"]["id"] for h in result["structured"]] == ["m04"]
                    checks.append("directed relationship filter distinguishes subject from object")

                    relation = {"id": "browser-relation", "a": "m02", "b": "m09",
                                "labels": {"relationship": "contradicts", "temporal_order": "same_time", "update_type": "none"},
                                "confidence": None,
                                "provenance": {"kind": "human", "note": "browser smoke annotation"}}
                    page.locator("#relation").fill(json.dumps(relation))
                    click_response("#save-relation")
                    result = click_response("#search")
                    assert result["relations"][0]["id"] == "browser-relation"
                    checks.append("human relationship annotation remains in history")

                    page.locator("#filters").fill(json.dumps({"subject": "no-such-character"}))
                    page.locator("#rerank").check()
                    result = click_response("#search")
                    assert result["structured"] == []
                    assert result["reranked"] is None
                    assert "unavailable" in result["rerank_status"]
                    checks.append("empty candidate set cannot claim unconfigured model reranking succeeded")

                    forbidden = page.request.post(url + "/api", headers={"Origin": "https://example.invalid"},
                                                  data={"action": "discover"})
                    assert forbidden.status == 403
                    checks.append("foreign-origin mutation is refused")
                    assert not errors, errors
                    checks.append("browser workflow has no JavaScript exceptions")
                finally:
                    browser.close()
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise AssertionError("App failed to stop its idle listener after SIGINT")
            stderr = process.stderr.read()
            assert process.returncode == 0, stderr
    return {"status": "passed", "checks": checks, "check_count": len(checks),
            "evidence_kind": "real Chromium; human claims; synthetic vectors; no model inference",
            "model_acceptance": "unrun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/phased-memory")
    parser.add_argument("--chromium", default="/usr/bin/chromium")
    args = parser.parse_args()
    print(json.dumps(run(args.binary.resolve(), args.chromium), indent=2))
