#!/usr/bin/env python3
"""Batch runner: submit every *.json in a folder to POST /api/v1/broll/jobs, poll, download outputs.

Usage: python tools/batch_run.py examples/test_pack --out results --api http://localhost:8000 [--api-key KEY]
Each input JSON is a BrollJobIn body. Writes results/<name>/{files...} and results/summary.json.
"""
import argparse, json, pathlib, sys, time
import urllib.request, urllib.error


def call(method, url, body=None, key=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["X-API-Key"] = key
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in sorted(pathlib.Path(a.inputs).glob("*.json")):
        t0 = time.time()
        rec = {"input": f.name}
        code, raw = call("POST", f"{a.api}/api/v1/broll/jobs", json.loads(f.read_text()), a.api_key)
        body = json.loads(raw or b"{}")
        if code != 202:
            rec.update(state="rejected", http=code, error=body.get("error"))
            summary.append(rec); print(f.name, "rejected", code); continue
        jid = body["job_id"]
        while time.time() - t0 < a.timeout:
            _, raw = call("GET", f"{a.api}/api/v1/jobs/{jid}", key=a.api_key)
            st = json.loads(raw)
            if st.get("state") in ("succeeded", "failed"):
                break
            time.sleep(3)
        else:
            st = {"state": "timeout"}
        rec.update(job_id=jid, state=st.get("state"), error=st.get("error"), seconds=round(time.time() - t0, 1))
        if st.get("state") == "succeeded":
            d = out / f.stem; d.mkdir(exist_ok=True)
            for name in st.get("files", {}):
                c, data = call("GET", f"{a.api}/api/v1/jobs/{jid}/files/{name}", key=a.api_key)
                if c == 200:
                    (d / name).write_bytes(data)
        summary.append(rec); print(f.name, rec["state"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.exit(0 if all(r["state"] == "succeeded" for r in summary) else 1)


if __name__ == "__main__":
    main()
