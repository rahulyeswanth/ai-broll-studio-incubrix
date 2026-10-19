"""API-level checks added with the review/diagnostics work: duplicate submissions, thumbnails, summaries, friendly errors."""
import asyncio

import pytest

from tests.test_jobs_api import services, wait_done  # noqa: F401  (fixture re-use)


@pytest.mark.asyncio
async def test_identical_submission_while_running_returns_the_same_job(async_client, services):
    data = {"script": "Sea turtles swim through the ocean.", "aspect_ratio": "16:9"}
    first = await async_client.post("/api/v1/jobs", data=data)
    second = await async_client.post("/api/v1/jobs", data=data)
    assert first.status_code == 202 and second.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.json().get("deduplicated") is True
    body = await wait_done(async_client, first.json()["job_id"])
    assert body["status"] == "completed"
    again = await async_client.post("/api/v1/jobs", data=data)       # finished jobs do not block a deliberate re-run
    assert again.json()["job_id"] != first.json()["job_id"]
    await wait_done(async_client, again.json()["job_id"])


@pytest.mark.asyncio
async def test_different_settings_are_not_deduplicated(async_client, services):
    a = await async_client.post("/api/v1/jobs", data={"script": "Sea turtles swim through the ocean.", "aspect_ratio": "16:9"})
    b = await async_client.post("/api/v1/jobs", data={"script": "Sea turtles swim through the ocean.", "aspect_ratio": "9:16"})
    assert a.json()["job_id"] != b.json()["job_id"]
    await wait_done(async_client, a.json()["job_id"])
    await wait_done(async_client, b.json()["job_id"])


@pytest.mark.asyncio
async def test_completed_job_exposes_summary_thumbnail_and_diagnostics(async_client, services):
    r = await async_client.post("/api/v1/jobs", data={"script": "Sea turtles swim through the ocean.", "aspect_ratio": "16:9"})
    body = await wait_done(async_client, r.json()["job_id"])
    assert body["status"] == "completed"
    assert body["summary"] and body["summary"][0]["level"] in ("success", "warning")
    jid = body["job_id"]
    manifest = (await async_client.get(f"/api/v1/jobs/{jid}/files/manifest.json")).json()
    assert {"summary", "items", "counts"} <= set(manifest["diagnostics"])
    tl = (await async_client.get(f"/api/v1/jobs/{jid}/files/timeline.json")).json()
    thumb = tl["scenes"][0]["clip"]["thumbnail"]
    img = await async_client.get(f"/api/v1/jobs/{jid}/files/{thumb}")
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/") and len(img.content) > 500


@pytest.mark.asyncio
async def test_download_whitelist_still_blocks_arbitrary_files(async_client, services):
    r = await async_client.post("/api/v1/jobs", data={"script": "Sea turtles swim through the ocean."})
    body = await wait_done(async_client, r.json()["job_id"])
    for bad in ("thumb_scene-001.png", "thumb_../x.jpg", "thumb_scene-abc.jpg"):
        assert (await async_client.get(f"/api/v1/jobs/{body['job_id']}/files/{bad}")).status_code in (400, 404, 422)


@pytest.mark.asyncio
async def test_config_endpoint_reports_key_status_but_never_key_values(async_client, services):
    r = await async_client.get("/api/v1/jobs/config")
    assert r.status_code == 200
    text = r.text.lower()
    assert "pixabay_key_configured" in text
    for secret_word in ("api_key\":", "secret", "password"):
        assert secret_word not in text
