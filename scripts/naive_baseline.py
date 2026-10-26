"""Fixed, untuned baseline for Stage A; no model inference or gold-label reads."""
import argparse
import json
from pathlib import Path


def predict(cases, original=False):
    seen = set()
    predictions = []
    for case in cases:
        cid = case["id"]
        if not isinstance(cid, str) or not cid.strip() or len(cid.encode()) > 256 or cid in seen:
            raise ValueError("Invalid or duplicate case ID")
        seen.add(cid)
        if original:
            predictions.append({"id": cid, "relation": "compatible",
                                "temporal_order": "unknown", "update_type": "none"})
            continue
        predictions.append({
            "id": cid,
            "labels": {"relationship": "compatible", "temporal_order": "unknown", "update_type": "none"},
            "error": None,
            "provenance": {"kind": "synthetic", "note": "Fixed compatible/unknown/none naive baseline; no inference or label tuning."},
            "native_probabilities": None,
        })
    if not predictions:
        raise ValueError("Empty baseline corpus")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-jsonl", action="store_true",
                        help="Use the original pack's JSONL input/output contract; IDs only.")
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    # Only IDs are accessed by predict. Expected labels never affect predictions.
    if args.original_jsonl:
        cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    else:
        cases = json.loads(args.cases.read_text())
    predictions = predict(cases, args.original_jsonl)
    with args.output.open("x") as output:
        if args.original_jsonl:
            for prediction in predictions:
                output.write(json.dumps(prediction) + "\n")
        else:
            json.dump(predictions, output, indent=2)
        output.write("\n")
