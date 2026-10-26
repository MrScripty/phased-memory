"""Reproduce authored development cases distinct from the original starter pack.

All vectors are deliberately identical synthetic smoke inputs. They do not
represent semantic embeddings or evidence of any model's retrieval quality.
"""
import copy
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "fixtures" / "authored"


def claim(cid, subject="Alice", predicate="trusts", obj="Bob", polarity="affirmed",
          time=9, perspective="world", continuity="main", qualifiers=None):
    return {"id": cid, "normalized": f"{subject} {predicate} {obj} ({polarity})",
            "triple": {"subject": subject, "predicate": predicate, "object": obj,
                       "polarity": polarity}, "entities": [subject, obj],
            "source": f"Authored source: {subject} {predicate} {obj} ({polarity}).",
            "perspective": perspective, "continuity": continuity,
            "story_time": None if time is None else {"timeline": "story", "position": time},
            "qualifiers": qualifiers or []}


cases = []


def pair(name, a=None, b=None, relationship="compatible", temporal="same_time", update="none"):
    a = copy.deepcopy(a or claim("a"))
    b = copy.deepcopy(b or claim("b"))
    # Inference sees claim IDs, so keep those neutral even when scorer case IDs
    # use descriptive names. No expected relation is encoded in source text/IDs.
    sequence = len(cases) + 1
    a["id"], b["id"] = f"pair-{sequence:03}-a", f"pair-{sequence:03}-b"
    cases.append({"id": name, "a": a, "b": b,
                  "expected": {"relationship": relationship, "temporal_order": temporal,
                               "update_type": update}})


pair("same-time-negation", b=claim("b", polarity="negated"), relationship="contradicts")
pair("later-trust-change", b=claim("b", polarity="negated", time=10), temporal="b_later", update="state_change")
pair("late-ingested-flashback", claim("a", "gate", "is", "locked", time=10),
     claim("b", "gate", "is", "locked", polarity="negated", time=8), temporal="b_earlier", update="state_change")
pair("author-correction", claim("a", "car", "color", "red"),
     claim("b", "car", "color", "blue", qualifiers=["explicit_correction"]),
     relationship="contradicts", update="explicit_correction")
pair("lying-dialogue", claim("a", "safe", "contains", "diamond", polarity="negated"),
     claim("b", "safe", "contains", "diamond", perspective="Bob:speech", qualifiers=["lie"]))
pair("belief-and-world", claim("a", "Bob", "located", "city", perspective="Alice:belief"),
     claim("b", "Bob", "located", "city", polarity="negated"))
pair("reversed-direction", b=claim("b", "Bob", "trusts", "Alice"))
pair("intended-move", claim("a", "key", "located", "drawer"),
     claim("b", "key", "located", "box", perspective="Alice:intention", qualifiers=["intention"]))
pair("missing-time", claim("a", "gate", "is", "open", time=None),
     claim("b", "gate", "is", "open", polarity="negated"),
     relationship="insufficient_information", temporal="unknown", update="undetermined")
pair("supporting-paraphrase", relationship="supports")
cases[-1]["b"]["normalized"] = "Bob is trusted by Alice."
pair("exact-count-conflict", claim("a", "room", "occupant_count", "2"),
     claim("b", "room", "occupant_count", "3"), relationship="contradicts")
pair("exclusive-location", claim("a", "key", "located", "drawer"),
     claim("b", "key", "located", "box"), relationship="contradicts")
pair("ambiguous-identity", claim("a", "Alex#unknown", "located", "city"),
     claim("b", "Alex#2", "located", "city", polarity="negated"),
     relationship="insufficient_information", update="undetermined")
pair("separate-continuities", b=claim("b", polarity="negated", continuity="alternate"), temporal="unknown")
pair("missing-speaker", claim("a", perspective="unknown:speech"),
     claim("b", polarity="negated"), relationship="insufficient_information", update="undetermined")
pair("wrong-person-negation", b=claim("b", "Charlie", polarity="negated"))
pair("same-negative-support", claim("a", polarity="negated"),
     claim("b", polarity="negated"), relationship="supports")
pair("past-stable-state", claim("a", time=8), claim("b", time=9), temporal="b_later")
pair("opposite-beliefs", claim("a", perspective="Alice:belief"),
     claim("b", perspective="Charlie:belief", polarity="negated"))
pair("question-not-fact", b=claim("b", polarity="unknown", qualifiers=["question"]),
     relationship="insufficient_information", update="undetermined")
pair("same-time-mutual-exclusion", claim("a", "lamp", "is", "on"),
     claim("b", "lamp", "is", "on", polarity="negated"), relationship="contradicts")
pair("earlier-stable-state", claim("a", time=10), claim("b", time=8), temporal="b_earlier")
pair("unrelated-predicates", b=claim("b", predicate="knows"))
pair("correction-does-not-move-time", claim("a", "car", "color", "red", time=8),
     claim("b", "car", "color", "blue", time=8, qualifiers=["explicit_correction"]),
     relationship="contradicts", update="explicit_correction")

memories = [
    claim("m01", time=1),
    claim("m02", polarity="negated", time=2),
    claim("m03", time=0),  # Deliberately ingested after later story state.
    claim("m04", "Bob", "trusts", "Alice", time=2),
    claim("m05", "Charlie", "trusts", "Bob", time=2),
    claim("m06", time=2, continuity="alternate"),
    claim("m07", time=2, perspective="Alice:belief"),
    claim("m08", time=2, perspective="Bob:speech", qualifiers=["lie"]),
    claim("m09", time=2),  # Contradictory world evidence at same story time.
    claim("m10", "key", "located", "drawer", time=2),
]
note = "Authored development fixtures distinct from the original starter pack; labels need independent review."
corpus = {"raw_text": "Ten new authored smoke memories. See each claim's source and qualifiers.",
          "claims": memories, "provenance": {"kind": "human", "note": note},
          "embeddings": [{"claim_id": m["id"], "embedding": {
              "space": "synthetic-smoke-v1", "vector": [1.0, 0.0, 0.0],
              "provenance": {"kind": "synthetic", "note": "Identical vectors for plumbing only; no semantic or model-quality evidence."}}}
                         for m in memories]}
base = {"subject": "Alice", "predicate": "trusts", "object": "Bob", "perspective": "world", "continuity": "main", "timeline": "story"}
queries = []


def query(qid, text, filters, expected, misleading, chronology=None):
    queries.append({"id": qid, "query_text": text, "filters": filters,
                    "expected_ids": expected, "misleading_ids": misleading,
                    "chronology": chronology or []})


query("latest", "Latest world evidence about Alice trusting Bob", {**base, "latest": True}, ["m02", "m09"], ["m03", "m06", "m07", "m08"])
query("historical", "Alice's trust in Bob at story position one", {**base, "at_or_before": 1, "latest": True}, ["m01"], ["m02", "m09", "m06"])
query("history", "History of Alice's trust in Bob", {**base, "chronological": True}, ["m03", "m01", "m02", "m09"], ["m04", "m05", "m06", "m07", "m08"], ["m03", "m01", "m02", "m09"])
query("contradictory", "Same-time contradictory world evidence at position two", {**base, "at_or_before": 2, "latest": True}, ["m02", "m09"], ["m03", "m07", "m08"])
query("direction", "Bob trusts Alice", {"subject": "Bob", "predicate": "trusts", "object": "Alice", "perspective": "world", "continuity": "main"}, ["m04"], ["m01", "m02", "m09"])
query("character", "Charlie trusts Bob", {"subject": "Charlie", "predicate": "trusts", "object": "Bob", "continuity": "main"}, ["m05"], ["m01", "m02"])
query("belief", "Alice's belief about trusting Bob", {**base, "perspective": "Alice:belief"}, ["m07"], ["m02", "m09", "m08"])
query("alternate", "Alice trusts Bob in the alternate draft", {**base, "continuity": "alternate"}, ["m06"], ["m02", "m09"])

raw_pairs = [{"id": c["id"], "a_raw": c["a"]["normalized"],
              "b_raw": c["b"]["normalized"], "expected": c["expected"]} for c in cases]
# Preserve scope as visible input; labels remain in expected, excluded by runner.
for raw, c in zip(raw_pairs, cases):
    for side in ("a", "b"):
        x = c[side]
        raw[side + "_raw"] += f". Perspective: {x['perspective']}; continuity: {x['continuity']}; story time: {x['story_time']}; qualifiers: {x['qualifiers']}."

OUT.mkdir(parents=True, exist_ok=True)
for name, value in (("pairs.json", cases), ("raw-pairs.json", raw_pairs),
                    ("corpus.json", corpus), ("queries.json", queries),
                    ("corpus-ids.json", [m["id"] for m in memories])):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")
print(f"Wrote {len(cases)} NEW pairs, {len(memories)} memories, {len(queries)} queries.")
