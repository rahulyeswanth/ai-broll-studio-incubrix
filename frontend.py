"""AI B-roll Studio - Streamlit frontend.

Layout: app header (brand, backend/provider status, settings) -> page.  The default page, "Studio", is the complete
script/audio -> video workflow: input workspace, live progress, result workspace, scene review, collapsible diagnostics.
The older step-by-step pages are kept under "Tools".

Design notes (see theme.css):
* Fonts are set on text containers only. A global ``span { font-family }`` rule used to override Streamlit's icon font,
  which printed ligature names such as "keyboard_arrow_down" on top of expander labels.
* Every ``st.markdown`` HTML snippet is self-contained. Opening a <div> in one call and closing it in another does not
  work in Streamlit (each call is its own DOM subtree); bordered ``st.container`` is used for grouping instead.
* All dynamic text is HTML-escaped (ui_helpers.esc).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st

import ui_helpers as ui

API_URL = os.environ.get("BROLL_API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="AI B-roll Studio", layout="wide", initial_sidebar_state="auto")
st.markdown(f"<style>{(HERE / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

DEFAULTS = {"page": "workflow", "script": "", "ingestion": None, "matches": None, "narration": None, "captions": None,
            "render_result": None, "wf_job": None, "wf_job_id": None, "wf_started": None, "backend_ok": None,
            "cfg_aspect": "16:9", "cfg_provider": "pixabay", "count_scripts": 0, "count_scenes": 0,
            "count_matches": 0, "count_renders": 0, "sel_scene": 1, "submit_error": None}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# =====================================================================================================================
# HTTP helpers
# =====================================================================================================================
def api_json(path: str, method: str = "GET", payload: Optional[dict] = None, timeout: float = 20, quiet: bool = True):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout) if method == "GET" else requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
        try:
            data = r.json()
        except ValueError:
            data = {"detail": r.text}
        if r.ok:
            return data
        if not quiet:
            st.error(f"The backend returned an error ({r.status_code}) for {path}.")
            with st.expander("Error details"):
                st.json(data)
        return None
    except requests.RequestException as exc:
        if not quiet:
            st.error(f"Cannot connect to the backend at {API_URL}.")
            with st.expander("Error details"):
                st.code(str(exc))
        return None


def api_bytes(path: str, timeout: float = 120) -> Optional[bytes]:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        return r.content if r.ok else None
    except requests.RequestException:
        return None


def check_backend() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        return False


def backend_ok() -> bool:
    if st.session_state["backend_ok"] is None:
        st.session_state["backend_ok"] = check_backend()
    return bool(st.session_state["backend_ok"])


def pipeline_cfg() -> Dict[str, Any]:
    if not st.session_state.get("pipeline_cfg"):
        st.session_state["pipeline_cfg"] = (api_json("/jobs/config", timeout=4) or {}) if backend_ok() else {}
    return st.session_state["pipeline_cfg"]


def safe_int(v, d=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


aspect_ratio = st.session_state["cfg_aspect"]
provider = st.session_state["cfg_provider"]


# =====================================================================================================================
# Shared widgets
# =====================================================================================================================
def chip(label: str, value: str = "", dot: str = "") -> str:
    d = f'<span class="dot {dot}"></span>' if dot else ""
    v = f" <b>{ui.esc(value)}</b>" if value else ""
    return f'<span class="chip">{d}{ui.esc(label)}{v}</span>'


def section(title: str, sub: str = "") -> None:
    s = f'<p class="section-sub">{ui.esc(sub)}</p>' if sub else ""
    st.markdown(f'<h2 class="section-title">{ui.esc(title)}</h2>{s}', unsafe_allow_html=True)


def page_header(title: str, sub: str) -> None:
    st.markdown(f'<h1 class="page-title">{ui.esc(title)}</h1><p class="page-sub">{ui.esc(sub)}</p>', unsafe_allow_html=True)


def render_header() -> None:
    ok = backend_ok()
    cfg = pipeline_cfg() if ok else {}
    prov = st.session_state["cfg_provider"]
    key_ok = cfg.get(f"{prov}_key_configured") if prov in ("pixabay", "pexels") and cfg else None
    prov_dot = "bad" if key_ok is False else ("ok" if key_ok else "")
    mark = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="3"/><path d="m10 9 5 3-5 3z"/></svg>')
    c1, c2, c3 = st.columns([4, 5, 1.4], vertical_alignment="center")
    with c1:
        st.markdown(f'<div class="brand"><div class="brand-mark">{mark}</div><div><div class="brand-name">AI B-roll Studio</div>'
                    f'<div class="brand-sub">Script to cinematic video</div></div></div>', unsafe_allow_html=True)
    with c2:
        chips = chip("Backend", "online" if ok else "offline", "ok" if ok else "bad")
        chips += chip("Footage", prov + (" · key missing" if key_ok is False else ""), prov_dot)
        chips += chip("Output", st.session_state["cfg_aspect"])
        if cfg:
            chips += chip("Visual check", "frames" if cfg.get("visual_verifier") == "clip" else "tags only")
        st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)
    with c3:
        with st.popover("Settings", use_container_width=True):
            st.markdown("**Connection**")
            st.code(API_URL, language=None)
            if st.button("Refresh connection", key="btn_refresh_backend", use_container_width=True):
                st.session_state["backend_ok"] = check_backend()
                st.session_state["pipeline_cfg"] = None
                st.rerun()
            if cfg:
                st.markdown("**Pipeline**")
                st.caption(f"Narration: {cfg.get('tts_provider')} · Match threshold: {cfg.get('match_threshold')} · "
                           f"Review floor: {cfg.get('review_floor', '-')} · Policy: {cfg.get('fallback_policy', '-')}")
                st.caption(f"Pixabay key: {'configured' if cfg.get('pixabay_key_configured') else 'missing'} · "
                           f"Pexels key: {'configured' if cfg.get('pexels_key_configured') else 'missing'}")
            st.caption("Keys live in backend/.env and are never sent to the browser.")
    st.markdown('<div style="border-bottom:1px solid var(--border-soft);margin:6px 0 20px"></div>', unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        pages = [("workflow", "Studio"), ("overview", "Overview")]
        tools = [("script", "Script Studio"), ("broll", "B-roll Library"), ("narration", "Narration"),
                 ("captions", "Captions"), ("render", "Render & Editor")]
        st.markdown('<div class="sb-label">Workspace</div>', unsafe_allow_html=True)
        for key, label in pages:
            if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if st.session_state["page"] == key else "secondary"):
                st.session_state["page"] = key
                st.rerun()
        st.markdown('<div class="sb-label">Tools</div>', unsafe_allow_html=True)
        for key, label in tools:
            if st.button(label, key=f"nav_{key}", use_container_width=True, type="primary" if st.session_state["page"] == key else "secondary"):
                st.session_state["page"] = key
                st.rerun()


# =====================================================================================================================
# STUDIO (one-click workflow)
# =====================================================================================================================
def submit_job(data: dict, files) -> None:
    st.session_state["submit_error"] = None
    try:
        r = requests.post(f"{API_URL}/jobs", data=data, files=files, timeout=300)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
    except requests.RequestException as exc:
        st.session_state["submit_error"] = f"Cannot reach the backend at {API_URL}. ({type(exc).__name__})"
        return
    if r.status_code == 202:
        st.session_state.update(wf_job_id=body["job_id"], wf_job=None, wf_started=time.time())
    else:
        detail = body.get("detail", body)
        st.session_state["submit_error"] = (detail.get("message") if isinstance(detail, dict) else str(detail)) or "The job could not be started."


def render_input_workspace() -> None:
    busy = bool(st.session_state.get("wf_job_id"))
    with st.container(border=True):
        mode = st.segmented_control("Input", ["Script", "Upload audio"], default="Script", key="wf_mode", disabled=busy)
        script_mode = mode != "Upload audio"
        wf_script, wf_audio = "", None
        if script_mode:
            wf_script = st.text_area("Narration script", value=st.session_state.get("script", ""), height=190, key="wf_script", disabled=busy,
                                     placeholder="Paste your narration. Each sentence becomes a scene, so concrete, visual sentences give the best footage.")
            n = len(wf_script)
            st.caption(f"{len(wf_script.split())} words · {n:,} characters" + (" · over the 10,000 character limit" if n > 10000 else ""))
        else:
            wf_audio = st.file_uploader("Narration audio", type=["mp3", "wav", "m4a", "aac", "ogg", "flac"], key="wf_audio_file", disabled=busy,
                                        help="Speech is transcribed locally; the audio decides all timing.")
            wf_script = st.text_area("Optional script (improves spelling; the audio still decides timing)", value="", height=90, key="wf_script_opt", disabled=busy)

        voices = api_json("/jobs/voices") if backend_ok() else None
        voice_ids = [v["id"] for v in (voices or {}).get("voices", [])] or ["en-US-AriaNeural"]
        c1, c2, c3 = st.columns(3)
        with c1:
            wf_voice = st.selectbox("Narration voice", voice_ids, key="wf_voice", disabled=busy or not script_mode,
                                    help="Used only when narration is generated from a script.")
        with c2:
            st.segmented_control("Aspect ratio", ["16:9", "9:16", "1:1"], key="cfg_aspect", disabled=busy)
        with c3:
            st.selectbox("Footage provider", ["pixabay", "pexels"], key="cfg_provider", disabled=busy)
        o1, o2, _ = st.columns([1.2, 1.4, 2])
        with o1:
            wf_caps = st.toggle("Generate captions", value=True, key="wf_caps", disabled=busy)
        with o2:
            wf_burn = st.toggle("Burn captions into video", value=True, key="wf_burn", disabled=busy or not wf_caps)

        clicked = st.button("Generate Video", type="primary", key="btn_generate", disabled=busy or not backend_ok(), use_container_width=True)
        if not backend_ok():
            st.caption("The backend is offline. Start it, then use Settings → Refresh connection.")
        if clicked:
            asp, prov = st.session_state["cfg_aspect"], st.session_state["cfg_provider"]
            if script_mode and not wf_script.strip():
                st.session_state["submit_error"] = "Enter a script first."
            elif not script_mode and wf_audio is None:
                st.session_state["submit_error"] = "Upload an audio file first."
            elif len(wf_script) > 10000:
                st.session_state["submit_error"] = "The script is longer than 10,000 characters."
            else:
                data = {"aspect_ratio": asp or "16:9", "voice": wf_voice if script_mode else "", "captions": str(bool(wf_caps)).lower(),
                        "burn_captions": str(bool(wf_caps and wf_burn)).lower(), "footage_provider": prov}
                if wf_script.strip():
                    data["script"] = wf_script
                files = None if script_mode else {"audio": (wf_audio.name, wf_audio.getvalue(), wf_audio.type or "application/octet-stream")}
                submit_job(data, files)
                if st.session_state["wf_job_id"]:
                    st.rerun()
        if st.session_state.get("submit_error"):
            st.markdown(ui.notice_html("error", st.session_state["submit_error"]), unsafe_allow_html=True)


@st.fragment(run_every=1.5)
def progress_panel() -> None:
    jid = st.session_state.get("wf_job_id")
    if not jid:
        return
    job = api_json(f"/jobs/{jid}", timeout=10)
    started = st.session_state.get("wf_started") or time.time()
    with st.container(border=True):
        if job is None:
            st.markdown('<div class="prog-title">Reconnecting…</div><div class="prog-sub">Lost contact with the backend; retrying.</div>', unsafe_allow_html=True)
            return
        pct = int(min(100, max(0, job.get("progress") or 0)))
        step = job.get("step") or job.get("status", "").title()
        sp = ui.scene_progress(step)
        sub = f"Elapsed {ui.elapsed_text(time.time() - started)}" + (f" · scene {sp[0]} of {sp[1]}" if sp else "")
        st.markdown(f'<div class="prog-title">{ui.esc(step)}</div><div class="prog-sub">{sub} · {pct}%</div>', unsafe_allow_html=True)
        st.progress(pct)
        st.markdown(ui.stages_html(pct, step, failed=job.get("status") == "failed"), unsafe_allow_html=True)
    if job.get("status") in ("completed", "failed"):
        st.session_state.update(wf_job=job, wf_job_id=None, wf_started=None)
        if job["status"] == "completed":
            st.session_state["count_renders"] += 1
        st.rerun()


def load_bundle(jid: str) -> Dict[str, Any]:
    key = f"bundle_{jid}"
    if key in st.session_state:
        return st.session_state[key]
    for k in [k for k in st.session_state if str(k).startswith("bundle_")]:
        del st.session_state[k]
    manifest = api_json(f"/jobs/{jid}/files/manifest.json", timeout=30) or {}
    timeline = api_json(f"/jobs/{jid}/files/timeline.json", timeout=30) or {}
    bundle = {"manifest": manifest, "timeline": timeline, "video": api_bytes(f"/jobs/{jid}/files/video.mp4"),
              "srt": api_bytes(f"/jobs/{jid}/files/captions.srt"), "thumbs": {}}
    for s in timeline.get("scenes", []):
        th = (s.get("clip") or {}).get("thumbnail")
        if th:
            bundle["thumbs"][s["scene_id"]] = api_bytes(f"/jobs/{jid}/files/{th}", timeout=30)
    st.session_state[key] = bundle
    return bundle


def render_failed(job: Dict[str, Any]) -> None:
    user_msg, tech = ui.job_error_text(job)
    st.markdown(ui.notice_html("error", user_msg), unsafe_allow_html=True)
    err = job.get("error") or {}
    with st.expander("Generation details", expanded=False):
        st.caption(f"Request id: {job.get('job_id')} · error code: {err.get('code', 'unknown')}")
        if tech and tech != user_msg:
            st.markdown(f'<div class="wrap-any meta">{ui.esc(tech)}</div>', unsafe_allow_html=True)
        for sc in (err.get("details") or {}).get("scenes", []):
            st.markdown(f'<div class="wrap-any"><b>{ui.esc(sc.get("scene_id"))}</b> · {ui.esc(sc.get("status"))}: {ui.esc(sc.get("reason"))}'
                        f'<div class="meta">Queries tried: {ui.esc(", ".join(sc.get("queries") or []))}</div></div>', unsafe_allow_html=True)
        if job.get("outputs"):
            st.caption("Partial files kept: " + ", ".join(job["outputs"]))
        st.json(err, expanded=False)


def render_results(job: Dict[str, Any]) -> None:
    jid = job["job_id"]
    b = load_bundle(jid)
    manifest, timeline = b["manifest"], b["timeline"]
    scenes = timeline.get("scenes", [])
    diag = manifest.get("diagnostics") or {}
    out = manifest.get("outputs") or {}
    vid = out.get("video") or {}

    section("Your video", "Review the result, download it, then check how each scene was matched.")
    notices = diag.get("summary") or [{"level": "success", "text": "Video generated successfully."}]
    if manifest.get("status") == "completed_with_validation_errors":
        notices = [{"level": "error", "text": "The video was rendered but failed validation. See Generation details."}] + notices
    st.markdown("".join(ui.notice_html(n["level"], n["text"]) for n in notices), unsafe_allow_html=True)

    cnt = ui.counts(scenes)
    left, right = st.columns([3, 2], gap="large")
    with left:
        if b["video"]:
            st.video(b["video"])
        else:
            st.markdown(ui.notice_html("warning", "The video file could not be fetched from the backend."), unsafe_allow_html=True)
    with right:
        dur = vid.get("duration_seconds")
        kv = [("Duration", f"{dur:.1f}s" if isinstance(dur, (int, float)) else "-", ""),
              ("Resolution", f"{vid.get('width', '-')}×{vid.get('height', '-')}", f"{vid.get('fps', '-')} fps"),
              ("Scenes", str(cnt["all"]), ""),
              ("Matched", f"{cnt['matched']} of {cnt['all']}", f"{cnt['review']} to review · {cnt['unmatched']} without footage")]
        st.markdown('<div class="card"><div class="kv">' + "".join(
            f'<div><div class="k">{ui.esc(k)}</div><div class="v">{ui.esc(v)}</div><div class="s">{ui.esc(s)}</div></div>' for k, v, s in kv) + '</div></div>',
            unsafe_allow_html=True)
        st.write("")
        if b["video"]:
            st.download_button("Download MP4", b["video"], file_name=f"{jid}_video.mp4", mime="video/mp4", key="dl_mp4", type="primary", use_container_width=True)
        with st.popover("More downloads", use_container_width=True):
            if b["srt"]:
                st.download_button("Captions (SRT)", b["srt"], file_name=f"{jid}_captions.srt", key="dl_srt", use_container_width=True)
            for name in (n for n in (job.get("files") or {}) if n.startswith("narration.")):
                ab = api_bytes(f"/jobs/{jid}/files/{name}")
                if ab:
                    st.download_button("Narration audio", ab, file_name=f"{jid}_{name}", key="dl_audio", use_container_width=True)
            st.download_button("timeline.json", json.dumps(timeline, indent=2).encode(), file_name=f"{jid}_timeline.json", mime="application/json", key="dl_tl", use_container_width=True)
            st.download_button("manifest.json", json.dumps(manifest, indent=2).encode(), file_name=f"{jid}_manifest.json", mime="application/json", key="dl_mf", use_container_width=True)

    section("Scene timeline", "Segment widths are proportional to scene duration. Amber = needs review, hatched red = no footage.")
    n = len(scenes)
    if n:
        sel = st.session_state.get("sel_scene") or 1
        sel = sel if 1 <= sel <= n else 1
        picked = st.segmented_control("Selected scene", list(range(1, n + 1)), default=sel, key="sel_scene_ctl", label_visibility="collapsed")
        sel = picked or sel
        st.session_state["sel_scene"] = sel
        st.markdown(ui.timeline_html(scenes, sel), unsafe_allow_html=True)

    render_scene_review(scenes, b, st.session_state.get("sel_scene") or 1)
    render_diagnostics(job, manifest, timeline)


def scene_card(scene: Dict[str, Any], idx: int, bundle: Dict[str, Any], selected: bool) -> None:
    status = ui.scene_status(scene)
    clip = scene.get("clip") or {}
    label, small = ui.relevance_text(scene)
    with st.container(border=True):
        c_thumb, c_main, c_side = st.columns([1.5, 5, 2], vertical_alignment="top")
        with c_thumb:
            tb = bundle["thumbs"].get(scene["scene_id"])
            if tb:
                st.image(tb, use_container_width=True)
            else:
                st.markdown('<div class="thumb-empty">No preview</div>', unsafe_allow_html=True)
        with c_main:
            sel = '<span class="chip"><b>Selected</b></span>' if selected else ""
            st.markdown(
                f'<div class="sc-head"><span class="sc-num">{idx}</span><span class="sc-time">{ui.esc(ui.fmt_range(scene.get("start"), scene.get("end")))}'
                f' · {ui.esc(ui.fmt_time(scene.get("duration")))}</span>{ui.badge_html(status)}{sel}</div>'
                f'<div class="sc-text clamp-2" title="{ui.esc(scene.get("narration_text"))}">{ui.esc(scene.get("narration_text"))}</div>',
                unsafe_allow_html=True)
            if clip:
                url = clip.get("source_page_url") or ""
                title = ui.esc(ui.clamp(clip.get("title") or clip.get("clip_id"), 70))
                link = f'<a href="{ui.esc(url)}" target="_blank" rel="noopener noreferrer">{title}</a>' if url.startswith(("http://", "https://")) else title
                st.markdown(f'<div class="sc-clip clamp-1">{link} · {ui.esc(str(clip.get("provider", "")).capitalize())}'
                            f'{" · alternate clip" if scene.get("used_alternate") else ""}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sc-why clamp-2" title="{ui.esc(scene.get("selection_reason"))}">{ui.esc(ui.why_selected(scene))}</div>', unsafe_allow_html=True)
        with c_side:
            st.markdown(f'<span class="lvl {ui.esc(label)}">{ui.esc(label)}</span><div class="meta" style="margin-top:4px">{ui.esc(small)}</div>', unsafe_allow_html=True)
            with st.popover("Details", use_container_width=True):
                scene_details(scene)


def scene_details(scene: Dict[str, Any]) -> None:
    va = scene.get("visual_analysis") or {}
    st.markdown(f'<div class="wrap-any"><b>Narration</b><br>{ui.esc(scene.get("narration_text"))}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta wrap-any">Subject: {ui.esc(va.get("subject") or "-")} · Action: {ui.esc(va.get("action") or "-")} · '
                f'Objects: {ui.esc(", ".join(va.get("objects") or []) or "-")} · Setting: {ui.esc(va.get("setting") or "-")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="meta wrap-any">Visual description: {ui.esc(scene.get("visual_description"))}</div>', unsafe_allow_html=True)
    qs = scene.get("queries_detail") or []
    if qs:
        st.markdown("**Searches tried**")
        for q in qs:
            st.markdown(f'<div class="meta wrap-any">“{ui.esc(q.get("query"))}” — {q.get("returned", 0)} returned, {q.get("kept", 0)} usable</div>', unsafe_allow_html=True)
    alts = scene.get("alternatives") or []
    if alts:
        st.markdown("**Other candidates**")
        for a in alts:
            note = f" — {ui.esc(a.get('gated'))}" if a.get("gated") else ""
            st.markdown(f'<div class="meta wrap-any">{ui.esc(ui.clamp(a.get("title"), 60))} · {ui.esc(a.get("label", ""))} · score {a.get("final_score")}{note}</div>', unsafe_allow_html=True)
    for d in scene.get("download_attempts") or []:
        if d.get("outcome") != "ok":
            st.markdown(f'<div class="meta wrap-any">Skipped {ui.esc(d.get("clip"))}: {ui.esc(d.get("reason"))}</div>', unsafe_allow_html=True)


def render_scene_review(scenes, bundle, selected: int) -> None:
    section("Scene review", "How each scene was matched. Review amber scenes; they may not show exactly what is narrated.")
    if not scenes:
        st.markdown(ui.notice_html("info", "This job has no scene records."), unsafe_allow_html=True)
        return
    cnt = ui.counts(scenes)
    opts = [f"All ({cnt['all']})", f"Matched ({cnt['matched']})", f"Needs review ({cnt['review']})", f"No footage ({cnt['unmatched']})"]
    choice = st.segmented_control("Filter scenes", opts, default=opts[0], key="scene_filter_ctl", label_visibility="collapsed") or opts[0]
    shown = ui.filter_scenes(scenes, choice)
    if not shown:
        st.markdown(ui.notice_html("info", "No scenes in this category."), unsafe_allow_html=True)
    for s in shown:
        scene_card(s, s.get("index", 0), bundle, selected == s.get("index"))


def render_diagnostics(job: Dict[str, Any], manifest: Dict[str, Any], timeline: Dict[str, Any]) -> None:
    diag = manifest.get("diagnostics") or {}
    items = diag.get("items") or []
    st.write("")
    with st.expander("Generation details", expanded=False):
        t1, t2, t3, t4 = st.tabs(["Notices", "Scene decisions", "Providers & limits", "Raw data"])
        with t1:
            groups = ui.group_diagnostics(items)
            if not items:
                st.caption("No notices.")
            for sev in ("error", "warning", "info"):
                for it in groups.get(sev, []):
                    where = f" · {it['scene_id']}" if it.get("scene_id") else ""
                    st.markdown(f'<div class="notice n-{sev}"><span class="n-ico">{"i" if sev == "info" else "!"}</span>'
                                f'<span class="n-txt"><b>{ui.esc(it.get("title"))}</b>{ui.esc(where)}<br>{ui.esc(it.get("message"))}</span></div>', unsafe_allow_html=True)
        with t2:
            rows = []
            for s in timeline.get("scenes", []):
                rel = s.get("relevance") or {}
                clip = s.get("clip") or {}
                rows.append({"Scene": s.get("index"), "Status": ui.STATUS_LABEL[ui.scene_status(s)], "Label": rel.get("label") or "-",
                             "Score": rel.get("score"), "Method": rel.get("scoring_method") or "-",
                             "Clip": f"{clip.get('provider', '')}:{clip.get('clip_id', '')}" if clip else "-",
                             "Searches": s.get("retry_attempts", 0) + 1, "Reason": ui.clamp(s.get("selection_reason"), 140)})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        with t3:
            prov, matching = manifest.get("providers") or {}, manifest.get("matching") or {}
            vv = prov.get("visual_verifier") or {}
            st.markdown(f"- **Narration:** {(prov.get('tts') or {}).get('name') or 'uploaded audio'}\n"
                        f"- **Footage provider:** {(prov.get('footage') or {}).get('name')}\n"
                        f"- **Visual verification:** {'frames inspected' if matching.get('visual_ai_verification_performed') else 'not performed (titles/tags only)'}"
                        f"{' — ' + str(vv.get('unavailable_reason')) if vv.get('unavailable_reason') else ''}\n"
                        f"- **Match threshold / review floor:** {matching.get('threshold')} / {matching.get('review_floor')}\n"
                        f"- **Scoring:** relevance scores are uncalibrated rankings, not probabilities; High/Medium/Low use documented thresholds.\n"
                        f"- **Licences:** {(manifest.get('licensing') or {}).get('disclaimer', '')}\n"
                        f"- **Request id:** `{job.get('job_id')}`")
            st.caption("Processing seconds: " + ", ".join(f"{k} {v}" for k, v in (manifest.get("processing_durations_seconds") or {}).items()))
        with t4:
            st.json(manifest, expanded=False)


def render_workflow() -> None:
    page_header("Create a video", "Turn a script or narration audio into a captioned video with B-roll matched to what each scene says.")
    render_input_workspace()
    if st.session_state.get("wf_job_id"):
        progress_panel()
        return
    job = st.session_state.get("wf_job")
    if not job:
        st.markdown('<div class="card" style="margin-top:16px"><div class="prog-title">Your video will appear here</div>'
                    '<div class="prog-sub">Paste a script above and press Generate Video. Concrete sentences such as “A farmer waters tomato plants” match best.</div></div>',
                    unsafe_allow_html=True)
        return
    if job.get("status") == "failed":
        render_failed(job)
    else:
        render_results(job)


# =====================================================================================================================
# Overview + legacy step pages (kept working)
# =====================================================================================================================
def kpi(label, value, sub):
    st.markdown(f'<div class="card"><div class="meta" style="text-transform:uppercase;letter-spacing:.06em">{ui.esc(label)}</div>'
                f'<div style="font-size:26px;font-weight:650">{ui.esc(value)}</div><div class="meta">{ui.esc(sub)}</div></div>', unsafe_allow_html=True)


def render_overview() -> None:
    page_header("Overview", "Session metrics for the step-by-step tools.")
    c = st.columns(4)
    for col, (l, v, s) in zip(c, [("Scripts analysed", st.session_state["count_scripts"], "Ingest calls"), ("Scenes detected", st.session_state["count_scenes"], "Segmentation"),
                                  ("B-roll matched", st.session_state["count_matches"], "Candidate footage"), ("Videos rendered", st.session_state["count_renders"], "MP4 outputs")]):
        with col:
            kpi(l, v, s)
    st.write("")
    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    if b1.button("Open Studio", type="primary", use_container_width=True):
        st.session_state["page"] = "workflow"
        st.rerun()
    if b2.button("Analyse a script", use_container_width=True):
        st.session_state["page"] = "script"
        st.rerun()
    if b3.button("Open renderer", use_container_width=True):
        st.session_state["page"] = "render"
        st.rerun()
    if st.session_state["ingestion"]:
        with st.expander("Latest scene breakdown"):
            st.markdown(ui.timeline_html(st.session_state["ingestion"].get("scenes", [])), unsafe_allow_html=True)
            st.json(st.session_state["ingestion"], expanded=False)


def simple_scene_card(i, sid, rng, text):
    with st.container(border=True):
        st.markdown(f'<div class="sc-head"><span class="sc-num">{i}</span><span class="sc-time">{ui.esc(sid)} · {ui.esc(rng)}</span></div>'
                    f'<div class="sc-text wrap-any">{ui.esc(text)}</div>', unsafe_allow_html=True)


def render_script() -> None:
    page_header("Script Studio", "Paste your narration and segment it into scenes.")
    script = st.text_area("Narration script", value=st.session_state.get("script") or "Coral reefs are home to colorful marine life.\nSea turtles swim through the ocean.\nProtecting our oceans is important.",
                          height=200, key="studio_script_input")
    wc = len(script.split())
    c1, c2, c3 = st.columns(3)
    c1.metric("Word count", wc)
    c2.metric("Est. duration", f"{wc / 2.6:.1f}s" if wc else "-")
    c3.metric("Aspect ratio", st.session_state["cfg_aspect"])
    if st.button("Analyse script", type="primary", key="btn_analyze"):
        if not script.strip():
            st.warning("Please enter a script.")
        else:
            with st.spinner("Analysing script…"):
                result = api_json("/scripts/ingest", "POST", {"text": script, "target_aspect_ratio": st.session_state["cfg_aspect"]}, timeout=180, quiet=False)
            if result is not None:
                st.session_state.update(script=script, ingestion=result)
                st.session_state["count_scripts"] += 1
                st.session_state["count_scenes"] += len(result.get("scenes", []))
                st.success("Script analysed.")
    ing = st.session_state["ingestion"]
    if ing and ing.get("scenes"):
        section("Scene breakdown")
        st.markdown(ui.timeline_html(ing["scenes"]), unsafe_allow_html=True)
        for i, sc in enumerate(ing["scenes"], 1):
            simple_scene_card(i, sc.get("scene_id", f"scene-{i:03d}"), ui.fmt_range(sc.get("start_time"), sc.get("end_time")), sc.get("text", ""))


def render_broll() -> None:
    page_header("B-roll Library", "Search candidate footage for each scene (legacy keyword matcher; use Studio for action-aware matching).")
    if st.button("Match B-roll", type="primary", key="btn_match"):
        current = st.session_state.get("script") or st.session_state.get("studio_script_input", "")
        if not current.strip():
            st.warning("Analyse a script first.")
        else:
            with st.spinner(f"Matching via {st.session_state['cfg_provider']}…"):
                result = api_json("/broll/match", "POST", {"script": {"text": current, "scene_hints": []}, "target_aspect_ratio": st.session_state["cfg_aspect"],
                                                          "provider": st.session_state["cfg_provider"]}, timeout=300, quiet=False)
            if result is not None:
                st.session_state["matches"] = result
                st.session_state["count_matches"] += safe_int(result.get("matched_scenes", 0))
    m = st.session_state.get("matches")
    if not m:
        st.info("Click “Match B-roll” to find footage.")
        return
    matched, total = safe_int(m.get("matched_scenes")), safe_int(m.get("total_scenes"))
    k = st.columns(3)
    with k[0]:
        kpi("Matched", matched, "Scenes with footage")
    with k[1]:
        kpi("Total scenes", total, "From the script")
    with k[2]:
        kpi("Unmatched", max(total - matched, 0), "Needs manual review")
    for i, item in enumerate(m.get("matches") or m.get("results") or [], 1):
        asset = item.get("asset") or item.get("selected_asset") or {}
        with st.container(border=True):
            st.markdown(f'<div class="sc-head"><span class="sc-num">{i}</span><span class="sc-time">{ui.esc(item.get("scene_id", f"scene-{i:03d}"))}</span></div>'
                        f'<div class="meta wrap-any">Query: {ui.esc(item.get("search_query") or item.get("query") or "-")}</div>', unsafe_allow_html=True)
            if isinstance(asset, dict) and asset:
                ca, cb = st.columns([1, 3])
                prev = asset.get("preview_url") or asset.get("thumbnail_url")
                if prev:
                    try:
                        ca.image(prev, use_container_width=True)
                    except Exception:  # noqa: BLE001
                        pass
                for key in ("title", "author", "source", "license"):
                    if asset.get(key):
                        cb.markdown(f'<div class="wrap-any"><span class="meta">{key.capitalize()}</span> {ui.esc(asset[key])}</div>', unsafe_allow_html=True)


def render_narration() -> None:
    page_header("Narration", "Plan narration timing and synchronisation.")
    script = st.session_state.get("script") or st.session_state.get("studio_script_input", "")
    if not script:
        st.info("Analyse a script first.")
        return
    narr_provider = st.selectbox("TTS provider", ["edge", "mock"], index=0, key="narr_provider",
                                 help="edge = real narration audio (needs internet + edge-tts). mock = timing plan only, NO audio.")
    if st.button("Generate narration", type="primary", key="btn_narr"):
        with st.spinner("Generating narration…"):
            result = api_json("/audio/narrate", "POST", {"script": script, "provider": narr_provider, "target_aspect_ratio": st.session_state["cfg_aspect"]}, timeout=300, quiet=False)
        if result is not None:
            st.session_state["narration"] = result
    n = st.session_state.get("narration")
    if not n:
        return
    for i, s in enumerate(n.get("scenes") or n.get("segments") or [], 1):
        simple_scene_card(i, s.get("scene_id", f"scene-{i:03d}"), ui.fmt_range(s.get("start_time"), s.get("end_time")), s.get("text", ""))
    if n.get("audio_is_placeholder"):
        st.info("This was a placeholder plan (mock provider): no audio exists. Choose the edge provider, or use Studio, to get real narration.")
    else:
        st.success(f"Narration audio saved on the server: {n.get('audio_file')}")


def render_captions() -> None:
    page_header("Captions", "Generate captions from a JSON request.")
    default_json = json.dumps({"scenes": [{"scene_id": "scene-001", "start_time": 0, "end_time": 4, "narration_text": "Colorful coral reefs provide shelter."},
                                          {"scene_id": "scene-002", "start_time": 4, "end_time": 8, "narration_text": "Sea turtles glide through the ocean."}],
                               "format": "srt", "max_words_per_caption": 4}, indent=2)
    caption_json = st.text_area("Caption request JSON", value=default_json, height=250, key="studio_caption_json")
    if st.button("Generate captions", type="primary", key="btn_caps"):
        try:
            payload = json.loads(caption_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            result = api_json("/captions/generate", "POST", payload, timeout=60, quiet=False)
            if result is not None:
                st.session_state["captions"] = result
    c = st.session_state.get("captions")
    if c and isinstance(c, dict) and c.get("subtitle_text"):
        st.code(c["subtitle_text"], language="text")
        st.download_button("Download subtitles", data=c["subtitle_text"].encode("utf-8"), file_name=f"captions.{c.get('format', 'srt')}", mime="text/plain")


def display_video_result(result, key_prefix, output_name="rendered_video.mp4") -> None:
    if not isinstance(result, dict):
        st.markdown(f'<div class="card" style="text-align:center"><div class="prog-title">Video preview</div><div class="meta wrap-any">{ui.esc(output_name)}</div></div>', unsafe_allow_html=True)
        return
    out_path = result.get("output_path", "")
    out_name = result.get("output_filename") or output_name
    dur, w, h = result.get("duration_seconds"), result.get("width"), result.get("height")
    c = st.columns(4)
    c[0].metric("Duration", f"{dur:.2f}s" if isinstance(dur, (int, float)) else "-")
    c[1].metric("Resolution", f"{w}x{h}" if w and h else "-")
    c[2].metric("Clips", result.get("clip_count", "-"))
    c[3].metric("File size", f"{os.path.getsize(out_path) / 1048576:.2f} MB" if out_path and os.path.isfile(out_path) else "-")
    if out_path and os.path.isfile(out_path):
        try:
            data = Path(out_path).read_bytes()
        except OSError as exc:
            st.error(f"Could not read video file: {exc}")
            return
        st.video(data)
        st.download_button("Download MP4", data=data, file_name=out_name, mime="video/mp4", key=f"{key_prefix}_dl")
    else:
        st.warning("Rendered file not found on disk.")


def render_render() -> None:
    page_header("Render & Editor", "Render local clips into an MP4 (legacy tool).")
    clip_names = st.text_input("Clip filenames (comma-separated)", value="coral_reef.mp4, sea_turtle.mp4, ocean_waves.mp4", key="r_clips")
    durs_txt = st.text_input("Durations (comma-separated, seconds)", value="3.5, 3.0, 2.428", key="r_durs")
    c1, c2 = st.columns(2)
    with c1:
        output_name = st.text_input("Output filename", value="ocean_video_fixed.mp4", key="r_out")
        audio_name = st.text_input("External audio filename (optional)", value="", key="r_audio_ext", placeholder="my_audio.mp3")
    with c2:
        include_audio = st.checkbox("Keep audio from clips", value=False, key="r_audio_clips")
        burn = st.checkbox("Burn generated SRT captions", value=True, key="r_burn")
    if st.button("Render video", type="primary", key="btn_render"):
        clips = [n.strip() for n in clip_names.split(",") if n.strip()]
        try:
            durations = [float(v.strip()) for v in durs_txt.split(",") if v.strip()]
        except ValueError:
            durations = []
        if not clips or len(durations) != len(clips) or not output_name.lower().endswith(".mp4"):
            st.warning("Please check clips, durations, and the output filename.")
        else:
            caps = st.session_state.get("captions") or {}
            subs = caps.get("subtitle_text") if burn and caps.get("format") == "srt" else None
            if burn and not subs:
                st.warning("Generate SRT captions first, or untick burn.")
            else:
                with st.spinner("Rendering video…"):
                    result = api_json("/video/render", "POST", {"clips": clips, "clip_durations": durations, "target_aspect_ratio": st.session_state["cfg_aspect"],
                                                                "output_filename": output_name, "audio_filename": audio_name.strip() or None,
                                                                "include_audio_from_clips": include_audio, "subtitle_text": subs}, timeout=600, quiet=False)
                if result is not None:
                    st.session_state["render_result"] = result
                    st.session_state["count_renders"] += 1
    display_video_result(st.session_state["render_result"], "render_page", output_name)


# =====================================================================================================================
# Router
# =====================================================================================================================
render_sidebar()
render_header()
PAGES = {"workflow": render_workflow, "overview": render_overview, "script": render_script, "broll": render_broll,
         "narration": render_narration, "captions": render_captions, "render": render_render}
PAGES.get(st.session_state.get("page", "workflow"), render_workflow)()
