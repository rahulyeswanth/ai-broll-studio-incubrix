#!/usr/bin/env python3
"""Hard-case, swap, timing and language checks against the live API (stdlib only).

  python checks.py hard                      # hard cases A-G -> hardcase_results/
  python checks.py swap JOB_ID [SCENE_ID]    # swap one scene, print timings
  python checks.py timing                    # ~60 s narration job, prints elapsed seconds
  python checks.py lang                      # Hindi + Tamil jobs
  python checks.py audio                     # upload speech_clear.mp3/.wav/.m4a -> audio_results/
Options: --api URL  --audio-dir PATH (folder with silent.wav, not_audio.mp3, music_only.mp3)
"""
import argparse, json, pathlib, sys, threading, time, uuid
import urllib.request, urllib.error

API = "https://ai-broll-studio-api-840142298487.asia-south1.run.app"


def req(method, url, body=None, headers=None, timeout=120):
    h = dict(headers or {})
    data = None
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode(); h["Content-Type"] = "application/json"
    elif body is not None:
        data = body
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def multipart(fields, file_field=None, path=None):
    b = uuid.uuid4().hex
    out = b""
    for k, v in fields.items():
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    if path:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
                f'Content-Type: application/octet-stream\r\n\r\n').encode() + path.read_bytes() + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return out, {"Content-Type": f"multipart/form-data; boundary={b}"}


def err(raw):
    try:
        e = json.loads(raw).get("error") or {}
        return f'{e.get("code")}: {e.get("message")}'
    except Exception:
        return raw[:200].decode("utf8", "replace")


def wait(api, jid, outdir, timeout=1800):
    t0 = time.time(); st = {}
    while time.time() - t0 < timeout:
        _, raw = req("GET", f"{api}/api/v1/jobs/{jid}")
        st = json.loads(raw)
        if st.get("state") in ("succeeded", "failed"):
            break
        time.sleep(3)
    secs = round(time.time() - t0, 1)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "status.json").write_text(json.dumps(st, indent=2))
    if st.get("state") == "succeeded":
        for name in st.get("files", {}):
            c, data = req("GET", f"{api}/api/v1/jobs/{jid}/files/{name}")
            if c == 200:
                (outdir / name).write_bytes(data)
    return st, secs


def run_script(api, name, body, outroot):
    t0 = time.time()
    code, raw = req("POST", f"{api}/api/v1/broll/jobs", body)
    if code != 202:
        print(f"{name:22} REJECTED http={code} {err(raw)}")
        (outroot / name).mkdir(parents=True, exist_ok=True)
        (outroot / name / "response.json").write_bytes(raw)
        return {"case": name, "result": f"rejected {code}", "error": err(raw)}
    jid = json.loads(raw)["job_id"]
    st, secs = wait(api, jid, outroot / name)
    msg = (st.get("error") or {})
    print(f"{name:22} {st.get('state'):10} {secs:6}s job={jid} {msg.get('code', '') if isinstance(msg, dict) else msg}")
    return {"case": name, "job_id": jid, "result": st.get("state"), "seconds": secs, "error": st.get("error")}


def run_audio(api, name, path, outroot):
    if not path.is_file():
        print(f"{name:22} SKIP (missing {path})"); return {"case": name, "result": "skipped"}
    body, h = multipart({"aspect_ratio": "16:9"}, "audio", path)
    code, raw = req("POST", f"{api}/api/v1/jobs", body, h)
    (outroot / name).mkdir(parents=True, exist_ok=True)
    if code != 202:
        print(f"{name:22} REJECTED http={code} {err(raw)}")
        (outroot / name / "response.json").write_bytes(raw)
        return {"case": name, "result": f"rejected {code}", "error": err(raw)}
    jid = json.loads(raw)["job_id"]
    st, secs = wait(api, jid, outroot / name)
    print(f"{name:22} {st.get('state'):10} {secs:6}s job={jid} {st.get('error') or ''}")
    return {"case": name, "job_id": jid, "result": st.get("state"), "seconds": secs, "error": st.get("error")}


SCRIPT_60 = ("Every morning, millions of people start their day with a cup of coffee. "
             "It begins on a farm, where coffee cherries ripen on green hillside plants. "
             "Farmers pick the cherries by hand and spread the beans out to dry in the sun. "
             "Then the beans travel by truck and ship to roasters around the world. "
             "In the roastery, hot drums turn the green beans brown and fill the air with aroma. "
             "Baristas grind the roasted beans fresh and pour hot water over them. "
             "Steam rises from the cup as the first sip warms a cold morning. "
             "Friends meet in busy cafes to talk, laugh and share their plans for the day. "
             "Students open laptops at quiet tables and focus on their work. "
             "From a small seed to a warm cup, coffee connects farmers, workers and neighbours. "
             "Next time you drink a cup, remember the long journey behind every sip.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hard", "swap", "timing", "lang", "audio"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--api", default=API)
    ap.add_argument("--audio-dir", default="backend/tests/fixtures/audio")
    a = ap.parse_args()
    api = a.api.rstrip("/")

    if a.cmd == "hard":
        out = pathlib.Path("hardcase_results"); out.mkdir(exist_ok=True)
        ad = pathlib.Path(a.audio_dir)
        rows = []
        rows.append(run_script(api, "A_named_person_brand", {"script": "Elon Musk announced a new Tesla update. Apple released the iPhone at an event. Nike sponsors athletes around the world.", "aspect_ratios": ["16:9"]}, out))
        rows.append(run_script(api, "B_sad_news", {"script": "A deadly flood swept through the village last night. Families lost their homes and many people are still missing. Rescue teams are working through the night to find survivors.", "aspect_ratios": ["16:9"]}, out))
        rows.append(run_script(api, "C_empty_script", {"script": "   "}, out))
        rows.append(run_audio(api, "D_silent_audio", ad / "silent.wav", out))
        rows.append(run_audio(api, "E_wrong_file_type", ad / "not_audio.mp3", out))
        rows.append(run_audio(api, "F_music_only", ad / "music_only.mp3", out))
        rows.append(run_script(api, "G1_double_meaning", {"script": "The bank of the river was calm. The bank approved the loan. A mouse ran across the cloud computing server room. A python slithered through the grass.", "aspect_ratios": ["16:9"]}, out))
        res = {}
        def go(n, txt):
            res[n] = run_script(api, n, {"script": txt, "aspect_ratios": ["16:9"]}, out)
        t1 = threading.Thread(target=go, args=("G2_concurrent_user1", "Mountains rise above the clouds at sunrise. Hikers climb a narrow trail through the pines."))
        t2 = threading.Thread(target=go, args=("G3_concurrent_user2", "Waves crash on a sandy beach at sunset. Surfers paddle out into the blue water."))
        t1.start(); t2.start(); t1.join(); t2.join()
        rows += [res["G2_concurrent_user1"], res["G3_concurrent_user2"]]
        (out / "summary.json").write_text(json.dumps(rows, indent=2))
        print("\nSaved hardcase_results/summary.json")

    elif a.cmd == "swap":
        if not a.args:
            sys.exit("usage: checks.py swap JOB_ID [SCENE_ID]")
        jid = a.args[0]; scene = a.args[1] if len(a.args) > 1 else "scene-001"
        code, raw = req("POST", f"{api}/api/v1/broll/jobs/{jid}/scenes/{scene}/swap", {}, timeout=300)
        print(code); print(json.dumps(json.loads(raw), indent=2))
        pathlib.Path(f"swap_{jid}_{scene}.json").write_bytes(raw)

    elif a.cmd == "timing":
        out = pathlib.Path("timing_results"); out.mkdir(exist_ok=True)
        print("Words:", len(SCRIPT_60.split()))
        r = run_script(api, "script_60s", {"script": SCRIPT_60, "aspect_ratios": ["16:9"]}, out)
        (out / "summary.json").write_text(json.dumps(r, indent=2))

    elif a.cmd == "audio":
        out = pathlib.Path("audio_results"); out.mkdir(exist_ok=True)
        ad = pathlib.Path(a.audio_dir)
        rows = [run_audio(api, p.stem + p.suffix.replace(".", "_"), p, out) for p in sorted(ad.glob("speech_clear.*"))]
        (out / "summary.json").write_text(json.dumps(rows, indent=2))

    elif a.cmd == "lang":
        out = pathlib.Path("lang_results"); out.mkdir(exist_ok=True)
        rows = [
            run_script(api, "hindi", {"script": "समुद्र हमारी धरती का बड़ा हिस्सा ढकता है। लहरें किनारे से टकराती हैं। मछुआरे सुबह नाव लेकर निकलते हैं।", "voice": "hi-IN-SwaraNeural", "language": "hi", "aspect_ratios": ["16:9"]}, out),
            run_script(api, "tamil", {"script": "கடல் நம் பூமியின் பெரும்பகுதியை மூடியுள்ளது. அலைகள் கரையைத் தொடுகின்றன. மீனவர்கள் காலையில் படகில் செல்கிறார்கள்.", "voice": "ta-IN-PallaviNeural", "language": "ta", "aspect_ratios": ["16:9"]}, out),
        ]
        (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
