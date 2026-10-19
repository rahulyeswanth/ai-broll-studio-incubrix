"""Browser regression test for the overlapping-text defect and general layout. Skipped without Playwright + Chromium.

Root cause being guarded: a global `font-family` on bare span/div replaced Streamlit's icon font, so ligature names such
as "keyboard_arrow_right" were printed over expander labels. We assert (1) icon spans keep the Material font and are
glyph-sized, (2) the type scale is applied, (3) no horizontal overflow, (4) long text is clamped, at 4 widths."""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from .conftest import ROOT
from .fixtures import LONG, scene

pw = pytest.importorskip("playwright.sync_api")
JOB = "job123456789"
WIDTHS = (1366, 1440, 1920, 390)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def stack():
    long_scene = scene(1, "matched", LONG)
    long_scene["selection_reason"] = LONG
    long_scene["clip"]["title"] = "T" * 300
    timeline = {"scenes": [long_scene, scene(2, "review", start=4, end=8), scene(3, "unmatched", start=8, end=12)]}
    manifest = {"status": "completed", "outputs": {"video": {"duration_seconds": 12.0, "width": 1920, "height": 1080, "fps": 30}},
                "diagnostics": {"summary": [{"level": "warning", "text": "1 scene has no matching footage."}], "items": [], "counts": {}},
                "providers": {}, "matching": {}}
    routes = {"/api/v1/health": {"status": "healthy"}, "/api/v1/jobs/config": {"pixabay_key_configured": True, "visual_verifier": "metadata"},
              "/api/v1/jobs/voices": {"voices": [{"id": "v1"}]}, f"/api/v1/jobs/{JOB}/files/manifest.json": manifest,
              f"/api/v1/jobs/{JOB}/files/timeline.json": timeline,
              f"/api/v1/jobs/{JOB}": {"job_id": JOB, "status": "completed", "progress": 100, "step": "Done", "files": {}}}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = routes.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body if body is not None else {}).encode())

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send_response(202)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"job_id": JOB, "status": "queued"}).encode())

        def log_message(self, *a):
            pass

    api_port, ui_port = free_port(), free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", api_port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {**os.environ, "BROLL_API_URL": f"http://127.0.0.1:{api_port}/api/v1"}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port", str(ui_port), "--server.headless", "true",
                             "--browser.gatherUsageStats", "false"], cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.time() + 40
    while time.time() < end:
        try:
            socket.create_connection(("127.0.0.1", ui_port), 0.5).close()
            break
        except OSError:
            time.sleep(0.5)
    yield f"http://127.0.0.1:{ui_port}"
    proc.terminate()
    srv.shutdown()


def open_page(p, url, width):
    try:
        b = p.chromium.launch()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Chromium unavailable: {exc}")
    pg = b.new_page(viewport={"width": width, "height": 1000})
    pg.goto(url)
    pg.wait_for_selector("text=Create a video", timeout=30000)
    return b, pg


@pytest.mark.parametrize("width", WIDTHS)
def test_empty_state_layout(stack, width):
    with pw.sync_playwright() as p:
        b, pg = open_page(p, stack, width)
        pg.wait_for_timeout(800)
        info = pg.evaluate("""()=>{const c=s=>{const e=document.querySelector(s);return e?getComputedStyle(e):null};
          const t=c('.page-title');return {over:document.documentElement.scrollWidth-innerWidth, title:t&&t.fontSize, family:t&&t.fontFamily,
          icons:[...document.querySelectorAll('[data-testid=stIconMaterial]')].map(e=>({f:getComputedStyle(e).fontFamily,w:e.getBoundingClientRect().width}))}}""")
        b.close()
    assert info["over"] <= 1, "horizontal overflow"
    assert info["title"] == ("24px" if width <= 640 else "30px")
    assert "Inter" in info["family"]
    assert info["icons"], "expected Streamlit icon spans"
    assert all("Material" in i["f"] and i["w"] < 48 for i in info["icons"]), info["icons"]


@pytest.mark.parametrize("width", WIDTHS)
def test_completed_state_long_text_does_not_break_layout(stack, width):
    with pw.sync_playwright() as p:
        b, pg = open_page(p, stack, width)
        ta = pg.locator("textarea").first
        ta.fill("A farmer waters tomato plants in a sunny field.")
        ta.press("Control+Enter")
        pg.wait_for_timeout(600)
        pg.get_by_role("button", name="Generate Video").click()
        pg.wait_for_selector("text=Scene review", timeout=30000)
        pg.wait_for_timeout(1200)
        info = pg.evaluate("""()=>{const q=s=>[...document.querySelectorAll(s)];
          return {over:document.documentElement.scrollWidth-innerWidth,
                  textH:q('.sc-text').map(e=>e.getBoundingClientRect().height), lineH:parseFloat(getComputedStyle(q('.sc-text')[0]).lineHeight),
                  clipClipped:q('.sc-clip').map(e=>e.getBoundingClientRect().height),
                  tooWide:q('.sc-text,.sc-clip,.sc-why,.notice,.kv').filter(e=>e.scrollWidth>e.clientWidth+1 && getComputedStyle(e).overflow==='visible').length,
                  icons:q('[data-testid=stIconMaterial]').map(e=>({f:getComputedStyle(e).fontFamily,w:e.getBoundingClientRect().width})),
                  timeline:q('.tl-seg').length, cards:q('.sc-head').length,
                  diagOpen:(q('[data-testid=stExpander] details').find(d=>d.textContent.includes('Generation details'))||{}).open}}""")
        b.close()
    assert info["over"] <= 1, info
    assert info["cards"] == 3 and info["timeline"] == 3
    assert max(info["textH"]) <= info["lineH"] * 2 + 2, "2,000-character narration must be clamped to two lines"
    assert max(info["clipClipped"]) <= 28, "a 300-character clip title must stay on one line"
    assert info["tooWide"] == 0
    assert all("Material" in i["f"] and i["w"] < 48 for i in info["icons"]), info["icons"]
    assert info["diagOpen"] is False, "diagnostics must be collapsed by default"
