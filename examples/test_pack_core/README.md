# Core test pack

Eleven script inputs for `POST /api/v1/broll/jobs`, from about 20 s to about 4 min of narration: a tutorial, a news item,
a product pitch, a story, an explainer, travel, health, a sad news story (sensitive-topic rules), a text with brand names
(generic stand-ins, no logos or real people), history, and one long mixed script (city and countryside).

Run: `python tools/batch_run.py examples/test_pack_core --out results_core --api <URL>`
Audio uploads: `python checks.py audio`. Ratings and results go into REPORT.md.
