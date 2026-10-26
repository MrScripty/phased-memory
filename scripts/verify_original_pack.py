"""Integrity and arithmetic checks for the unmodified original development pack.

Gold-echo outputs below are scorer oracles, never model predictions or evidence
of inference. This script performs no inference and changes no fixture labels.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from zipfile import ZipFile

ARCHIVE_SHA256 = "61de57fe311aa32f529acf0e5dd964b9fe058b0df1ba2431ff2eaa71ed505e76"
FILES = {"README.md", "pairs.jsonl", "retrieval_memories.jsonl", "retrieval_queries.jsonl", "score.py", "manifest.json"}


def verify(pack, archive=None):
    count = 0

    def check(condition, message):
        nonlocal count
        if not condition:
            raise AssertionError(message)
        count += 1

    def rejected(callback, message):
        try:
            callback()
        except ValueError:
            check(True, message)
        else:
            raise AssertionError(message)

    manifest = json.loads((pack / "manifest.json").read_text())
    check(manifest["schema_version"] == 1, "original manifest schema")
    for name, digest in manifest["sha256"].items():
        data = (pack / name).read_bytes()
        check(hashlib.sha256(data).hexdigest() == digest, f"unchanged original {name}")
        check(bool(data.decode("utf-8")), f"UTF-8 original {name}")
    if archive:
        data = archive.read_bytes()
        check(len(data) == 14603, "original archive byte count")
        check(hashlib.sha256(data).hexdigest() == ARCHIVE_SHA256, "original archive SHA256")
        with ZipFile(archive) as zipped:
            check(zipped.testzip() is None, "original ZIP CRC")
            check(len(zipped.namelist()) == 6, "six original files")
            for name in FILES:
                check(zipped.read("semantic_memory_starter_tests/" + name) == (pack / name).read_bytes(), f"archive equality {name}")

    spec = importlib.util.spec_from_file_location("original_semantic_memory_scorer", pack / "score.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    pairs = scorer.read_jsonl(pack / "pairs.jsonl")
    memories = scorer.read_jsonl(pack / "retrieval_memories.jsonl")
    queries = scorer.read_jsonl(pack / "retrieval_queries.jsonl")
    check((len(pairs), len(memories), len(queries)) == (24, 10, 8), "original fixture counts")
    for rows in (pairs, memories, queries):
        check(len({row["id"] for row in rows}) == len(rows), "unique source IDs")
    check(manifest["counts"] == {"claim_pairs": 24, "retrieval_memories": 10, "retrieval_queries": 8}, "manifest counts")
    for case in pairs:
        check(set(case["gold"]) == set(scorer.FIELDS), "three separate gold decisions")
        for side in ("a", "b"):
            index = case["comparison_targets"][side + "_claim_index"]
            check(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(case["memory_" + side]["normalized_claims"]), "selected primary claim exists")
        check(all(case["gold"][field] in values for field, values in scorer.FIELDS.items()), "valid original labels")
    memory_ids = {memory["id"] for memory in memories}
    for query in queries:
        relevant = query["relevant_memory_ids"]
        negative = query["hard_negative_memory_ids"]
        check(bool(relevant) and set(relevant + negative) <= memory_ids, "known retrieval gold IDs")
        check(not set(relevant) & set(negative), "disjoint relevant/hard-negative sets")
        if "expected_chronological_ids" in query:
            check(set(query["expected_chronological_ids"]) <= set(relevant), "chronology evidence is relevant")

    gold_echo = [{"id": case["id"], **case["gold"]} for case in pairs]
    perfect = scorer.score_pairs(pairs, gold_echo)
    check(perfect["joint_accuracy"] == 1 and perfect["relation_macro_f1"] == 1, "gold-echo scorer arithmetic only")
    missing = scorer.score_pairs(pairs, [])
    check(missing["expected_cases"] == 24 and missing["valid_complete_rows"] == 0 and missing["joint_accuracy"] == 0, "missing outputs retain full pair denominator")
    fixed = [{"id": case["id"], "relation": "compatible", "temporal_order": "unknown", "update_type": "none"} for case in pairs]
    check(math.isclose(scorer.score_pairs(pairs, fixed)["accuracy"]["relation"], 11 / 24), "untuned compatible baseline")
    check(perfect["multiclass_brier"] is None and perfect["probability_coverage_cases"] == 0, "unavailable probabilities stay unavailable")
    rejected(lambda: scorer.score_pairs(pairs, [gold_echo[0], gold_echo[0]]), "duplicate pair output IDs rejected")
    rejected(lambda: scorer.score_pairs(pairs, [{"id": "unknown-case"}]), "unknown pair IDs rejected")
    for invalid in (None, {"supports": 1}, {key: 0.0 for key in scorer.RELATIONS},
                    {key: float("nan") for key in scorer.RELATIONS},
                    {key: True for key in scorer.RELATIONS},
                    {key: -0.25 for key in scorer.RELATIONS},
                    {key: 2.0 for key in scorer.RELATIONS}):
        if invalid is None:
            continue  # null is explicitly unsupported/unavailable, not malformed.
        output = copy.deepcopy(gold_echo[0])
        output["relation_probabilities"] = invalid
        rejected(lambda: scorer.score_pairs(pairs, [output]), "invalid probability vector rejected")
    one_hot = copy.deepcopy(gold_echo)
    for row in one_hot:
        row["relation_probabilities"] = {key: float(key == row["relation"]) for key in scorer.RELATIONS}
    check(scorer.score_pairs(pairs, one_hot)["multiclass_brier"] == 0.0, "synthetic one-hot arithmetic oracle only")

    relevance_echo = [{"id": query["id"], "ranked_ids": query["relevant_memory_ids"],
                       **({"chronological_ids": query["expected_chronological_ids"]} if "expected_chronological_ids" in query else {})}
                      for query in queries]
    retrieval = scorer.score_retrieval(queries, memories, relevance_echo)
    check(retrieval["top1_accuracy"] == 1 and retrieval["mean_recall_at_5"] == 1, "retrieval arithmetic oracle only")
    check(retrieval["timeline_exact_match_fraction"] == 1, "chronology separate from relevance ranking")
    empty = scorer.score_retrieval(queries, memories, [])
    check(empty["expected_queries"] == 8 and empty["prediction_rows"] == 0 and empty["top1_accuracy"] == 0, "missing retrieval keeps full denominator")
    rejected(lambda: scorer.score_retrieval(queries, memories, [relevance_echo[0], relevance_echo[0]]), "duplicate retrieval output IDs rejected")
    rejected(lambda: scorer.score_retrieval(queries, memories, [{"id": "unknown-query"}]), "unknown query IDs rejected")
    rejected(lambda: scorer.score_retrieval(queries, memories, [{"id": queries[0]["id"], "ranked_ids": ["unknown-memory"]}]), "unknown ranked memory rejected")
    repeated = queries[0]["relevant_memory_ids"][0]
    rejected(lambda: scorer.score_retrieval(queries, memories, [{"id": queries[0]["id"], "ranked_ids": [repeated, repeated]}]), "duplicate ranked memory rejected")
    timeline_query = next(query for query in queries if "expected_chronological_ids" in query)
    rejected(lambda: scorer.score_retrieval(queries, memories, [{"id": timeline_query["id"], "ranked_ids": [], "chronological_ids": timeline_query["expected_chronological_ids"]}]), "chronology cannot invent unretrieved evidence")
    return {"status": "passed", "checks": count, "pairs": 24, "memories": 10, "queries": 8,
            "archive_verified": archive is not None,
            "evidence_kind": "original-fixture integrity and synthetic scorer arithmetic only",
            "model_inference": "unrun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path(__file__).resolve().parents[1] / "fixtures/original")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.pack.resolve(), args.archive), indent=2))
