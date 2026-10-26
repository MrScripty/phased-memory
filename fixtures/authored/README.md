# Newly authored development fixtures

These independently authored development fixtures provide **additional regressions**, separate from the byte-verified reference starter pack in `fixtures/original/`. They are not copies of its cases or labels.

Run `python3 scripts/create_authored_fixtures.py` to reproduce 24 normalized claim pairs, their raw-text counterparts, ten memories, eight retrieval queries, and the memory-ID inventory. Labels are proposed development annotations requiring independent review; they are not a held-out benchmark. The raw-text counterparts state scope explicitly to support a separate extraction check, and are not representative screenplay scenes.

The corpus intentionally has two conflicting world assertions at the latest story position. Both are relevant evidence; retrieval must preserve the conflict. A flashback is ingested after later story states. Belief, speech, continuity, subject and object each remain distinct. Ties in story time do not imply an ingestion-based temporal relationship.

Every attached vector is `[1, 0, 0]` in `synthetic-smoke-v1`, explicitly marked synthetic. These vectors check storage, filtering and UI plumbing only. They provide **no semantic embedding, inference, model-quality or retrieval-improvement evidence**. For real Stage C evaluation, create embeddings using the configured Pumas model in a fresh database and preserve its actual provenance. Never compare this synthetic smoke space with a real model space.

Gold labels stay in the scorer's case files. Runners must project only claims or raw text into inference. Keep extraction failures separate from relationship failures, and retain missing/error outputs in evaluation denominators. Independent scenes and scenario-separated evaluation data are required before drawing conclusions about models.
