#!/usr/bin/env python3
"""Score saved predictions against the included synthetic development fixtures.

This program performs no model inference. It uses only Python's standard library.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RELATIONS = ('supports', 'contradicts', 'compatible', 'insufficient_information')
FIELDS = {
    'relation': RELATIONS,
    'temporal_order': ('same_time', 'b_later', 'b_earlier', 'unknown'),
    'update_type': ('none', 'state_change', 'explicit_correction', 'undetermined'),
}
INVALID = '<missing_or_invalid>'


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path}:{number}: invalid JSON: {exc.msg}') from exc
        if not isinstance(row, dict) or not isinstance(row.get('id'), str):
            raise ValueError(f'{path}:{number}: expected an object with a string id')
        rows.append(row)
    return rows


def index_predictions(rows: list[dict[str, Any]], allowed: set[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row['id']
        if key not in allowed:
            raise ValueError(f'Unexpected prediction id: {key}')
        if key in result:
            raise ValueError(f'Duplicate prediction id: {key}')
        result[key] = row
    return result


def score_pairs(gold: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = index_predictions(rows, {g['id'] for g in gold})
    total = len(gold)
    confusion = {a: {b: 0 for b in (*RELATIONS, INVALID)} for a in RELATIONS}
    correct = {field: 0 for field in FIELDS}
    joint = valid = 0
    probability_errors: list[float] = []
    errors: list[dict[str, Any]] = []
    for case in gold:
        p = predictions.get(case['id'], {})
        all_valid = all(isinstance(p.get(f), str) and p[f] in labels for f, labels in FIELDS.items())
        valid += int(all_valid)
        field_matches = {f: p.get(f) == case['gold'][f] for f in FIELDS}
        for field, matched in field_matches.items():
            correct[field] += int(matched)
        joint += int(all(field_matches.values()))
        prediction = p.get('relation')
        if not isinstance(prediction, str) or prediction not in RELATIONS:
            prediction = INVALID
        confusion[case['gold']['relation']][prediction] += 1
        if not all(field_matches.values()):
            errors.append({'id': case['id'], 'expected': case['gold'],
                           'predicted': {f: p.get(f) for f in FIELDS}})
        probs = p.get('relation_probabilities')
        if probs is not None:
            if not isinstance(probs, dict) or set(probs) != set(RELATIONS):
                raise ValueError(f"{case['id']}: probabilities must contain exactly the four relation labels")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
                raise ValueError(f"{case['id']}: probabilities must be finite numbers in [0, 1]")
            if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4):
                raise ValueError(f"{case['id']}: probabilities must sum to one")
            probability_errors.append(sum((probs[r] - int(r == case['gold']['relation'])) ** 2
                                          for r in RELATIONS))
    per_class: dict[str, Any] = {}
    for label in RELATIONS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in RELATIONS if other != label)
        fn = sum(confusion[label][other] for other in (*RELATIONS, INVALID) if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {'precision': precision, 'recall': recall, 'f1': f1,
                            'support': sum(confusion[label].values())}
    return {'expected_cases': total, 'prediction_rows': len(predictions), 'valid_complete_rows': valid,
            'accuracy': {f: n / total for f, n in correct.items()},
            'joint_accuracy': joint / total,
            'relation_macro_f1': sum(x['f1'] for x in per_class.values()) / len(RELATIONS),
            'per_class': per_class, 'confusion': confusion,
            'multiclass_brier': sum(probability_errors) / len(probability_errors) if probability_errors else None,
            'probability_coverage_cases': len(probability_errors), 'errors': errors}


def checked_ids(value: Any, allowed: set[str], case_id: str, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f'{case_id}: {field} must be a list of memory IDs')
    if len(set(value)) != len(value) or not set(value) <= allowed:
        raise ValueError(f'{case_id}: {field} has duplicate or unknown memory IDs')
    return value


def score_retrieval(gold: list[dict[str, Any]], memories: list[dict[str, Any]],
                    rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = index_predictions(rows, {g['id'] for g in gold})
    allowed = {m['id'] for m in memories}
    per_query: list[dict[str, Any]] = []
    timeline_scores: list[int] = []
    for query in gold:
        p = predictions.get(query['id'], {})
        ranked = checked_ids(p.get('ranked_ids', []), allowed, query['id'], 'ranked_ids')
        relevant = set(query['relevant_memory_ids'])
        negatives = set(query['hard_negative_memory_ids'])
        first_rank = next((i for i, key in enumerate(ranked, 1) if key in relevant), None)
        row: dict[str, Any] = {
            'id': query['id'],
            'top1_relevant': int(bool(ranked) and ranked[0] in relevant),
            'reciprocal_rank': 1 / first_rank if first_rank is not None else 0.0,
            'recall_at_5': len(set(ranked[:5]) & relevant) / len(relevant),
            'hard_negative_in_top5': int(bool(set(ranked[:5]) & negatives)),
            'hard_negative_ids_at5': [key for key in ranked[:5] if key in negatives],
        }
        if 'expected_chronological_ids' in query:
            chronological = checked_ids(p.get('chronological_ids', []), allowed, query['id'], 'chronological_ids')
            if not set(chronological) <= set(ranked):
                raise ValueError(f"{query['id']}: chronological_ids must come from retrieved ranked_ids")
            matched = int(chronological == query['expected_chronological_ids'])
            row['timeline_exact_match'] = matched
            timeline_scores.append(matched)
        per_query.append(row)
    n = len(gold)
    return {'expected_queries': n, 'prediction_rows': len(predictions),
            'top1_accuracy': sum(x['top1_relevant'] for x in per_query) / n,
            'mean_reciprocal_rank': sum(x['reciprocal_rank'] for x in per_query) / n,
            'mean_recall_at_5': sum(x['recall_at_5'] for x in per_query) / n,
            'queries_with_hard_negative_at5_fraction': sum(x['hard_negative_in_top5'] for x in per_query) / n,
            'timeline_exact_match_fraction': sum(timeline_scores) / len(timeline_scores) if timeline_scores else None,
            'timeline_query_count': len(timeline_scores), 'per_query': per_query}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('pairs', 'retrieval'))
    parser.add_argument('predictions', type=Path, help='Saved prediction JSONL, never the gold fixture file')
    args = parser.parse_args()
    try:
        predictions = read_jsonl(args.predictions)
        if args.mode == 'pairs':
            result = score_pairs(read_jsonl(ROOT / 'pairs.jsonl'), predictions)
        else:
            result = score_retrieval(read_jsonl(ROOT / 'retrieval_queries.jsonl'),
                                     read_jsonl(ROOT / 'retrieval_memories.jsonl'), predictions)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
