use phased_memory::{
    domain::*,
    eval::*,
    pumas::{endpoint, validate_capabilities, Capabilities, Selection},
    store::{Filters, Store},
};
use std::collections::{BTreeMap, BTreeSet};
fn claim(id: &str, subject: &str, object: &str, time: i64) -> Claim {
    Claim {
        id: id.into(),
        normalized: format!("{subject} lives in {object}"),
        model_local_id: None,
        state_key: Some("residence".into()),
        triple: Triple {
            subject: subject.into(),
            predicate: "lives_in".into(),
            object: object.into(),
            polarity: Polarity::Affirmed,
        },
        entities: vec![subject.into(), object.into()],
        source: "test author".into(),
        perspective: "world".into(),
        continuity: "main".into(),
        story_time: Some(StoryTime {
            timeline: "story".into(),
            position: time,
        }),
        qualifiers: vec![],
    }
}
fn human() -> Provenance {
    Provenance::Human {
        note: "authored fixture".into(),
    }
}
fn embedding(vector: Vec<f32>) -> Embedding {
    Embedding {
        space: "synthetic:test".into(),
        vector,
        provenance: Provenance::Synthetic {
            note: "test plumbing only".into(),
        },
    }
}
fn ingest(claims: Vec<Claim>) -> Ingest {
    Ingest {
        raw_text: "original raw text remains exact\n".into(),
        embeddings: claims
            .iter()
            .map(|c| ClaimEmbedding {
                claim_id: c.id.clone(),
                embedding: embedding(vec![1.0, 0.0]),
            })
            .collect(),
        claims,
        provenance: human(),
    }
}
fn labels() -> Labels {
    Labels {
        relationship: Relationship::Supports,
        temporal_order: TemporalOrder::SameTime,
        update_type: UpdateType::None,
    }
}
fn pair() -> PairCase {
    PairCase {
        id: "case-1".into(),
        a: claim("a", "A", "Paris", 1),
        b: claim("b", "A", "Paris", 1),
        expected: labels(),
    }
}
fn prediction() -> PairPrediction {
    PairPrediction {
        id: "case-1".into(),
        labels: Some(labels()),
        error: None,
        provenance: Some(human()),
        native_probabilities: None,
    }
}
#[test]
fn cosine_exact_dimensions_zero_and_extreme_values() {
    assert_eq!(cosine(&[1.0, 0.0], &[-1.0, 0.0]).unwrap(), -1.0);
    assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0);
    assert!(cosine(&[1.0], &[1.0, 0.0])
        .unwrap_err()
        .to_string()
        .contains("dimension mismatch"));
    assert!(cosine(&[0.0], &[1.0])
        .unwrap_err()
        .to_string()
        .contains("zero embedding"));
    assert!(validate_vector(&[f32::NAN]).is_err());
    assert!((cosine(&[f32::MAX, f32::MAX], &[f32::MAX, f32::MAX]).unwrap() - 1.0).abs() < 1e-12);
}
#[test]
fn history_uses_story_time_not_ingestion_and_retains_raw() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.sqlite");
    let mut s = Store::open(&path).unwrap();
    s.ingest(&ingest(vec![claim("new", "Alice", "Rome", 5)]))
        .unwrap();
    s.ingest(&ingest(vec![
        claim("old", "Alice", "Paris", 1),
        claim("other", "Bob", "Alice", 9),
    ]))
    .unwrap();
    let mut f = Filters {
        subject: Some("Alice".into()),
        timeline: Some("story".into()),
        latest: true,
        ..Default::default()
    };
    let hits = s.compare(&embedding(vec![1.0, 0.0]), &f).unwrap();
    assert_eq!(hits.structured[0].claim.id, "new");
    assert_eq!(hits.structured.len(), 1);
    f.at_or_before = Some(2);
    assert_eq!(
        s.compare(&embedding(vec![1.0, 0.0]), &f)
            .unwrap()
            .structured[0]
            .claim
            .id,
        "old"
    );
    f.latest = false;
    f.at_or_before = None;
    f.chronological = true;
    let ids: Vec<_> = s
        .compare(&embedding(vec![1.0, 0.0]), &f)
        .unwrap()
        .structured
        .into_iter()
        .map(|h| h.claim.id)
        .collect();
    assert_eq!(ids, vec!["old", "new"]);
    assert_eq!(s.claims().unwrap().len(), 3);
    drop(s);
    let c = rusqlite::Connection::open(path).unwrap();
    let raw: String = c
        .query_row(
            "SELECT raw_text FROM sources ORDER BY id LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(raw, "original raw text remains exact\n");
}
#[test]
fn transactions_preserve_original_on_duplicate_and_space_mismatch() {
    let mut s = Store::open(":memory:").unwrap();
    s.ingest(&ingest(vec![claim("a", "A", "Paris", 1)]))
        .unwrap();
    let bad = ingest(vec![
        claim("b", "B", "Rome", 2),
        claim("a", "A", "Berlin", 3),
    ]);
    assert!(s.ingest(&bad).is_err());
    assert_eq!(s.claims().unwrap().len(), 1);
    let mut bad = ingest(vec![claim("c", "C", "Rome", 2)]);
    bad.embeddings[0].embedding.vector = vec![1.0];
    assert!(s
        .ingest(&bad)
        .unwrap_err()
        .to_string()
        .contains("dimension changed"));
    assert_eq!(s.claims().unwrap().len(), 1);
}
#[test]
fn perspective_continuity_direction_and_relations_are_not_collapsed() {
    let mut s = Store::open(":memory:").unwrap();
    let a = claim("a", "Alice", "Bob", 1);
    let b = claim("b", "Bob", "Alice", 1);
    let mut belief = a.clone();
    belief.id = "belief".into();
    belief.perspective = "alice:belief".into();
    belief.qualifiers = vec!["intention".into()];
    s.ingest(&ingest(vec![a, b, belief])).unwrap();
    s.add_relation(&Relation {
        id: "r".into(),
        a: "a".into(),
        b: "b".into(),
        labels: Labels {
            relationship: Relationship::Compatible,
            ..labels()
        },
        confidence: Some(0.6),
        provenance: human(),
    })
    .unwrap();
    let f = Filters {
        subject: Some("Alice".into()),
        perspective: Some("world".into()),
        relationship: Some(Relationship::Compatible),
        ..Default::default()
    };
    let c = s.compare(&embedding(vec![1.0, 0.0]), &f).unwrap();
    assert_eq!(c.structured.len(), 1);
    assert_eq!(c.structured[0].claim.id, "a");
    assert_eq!(c.relations.len(), 1);
    let mut invalid = c.relations[0].clone();
    invalid.id = "missing-relation".into();
    invalid.b = "absent".into();
    assert!(s.add_relation(&invalid).is_err());
    assert_eq!(s.relations().unwrap().len(), 1);
}
#[test]
fn unsupported_schema_never_resets_authoritative_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("future.sqlite");
    let c = rusqlite::Connection::open(&path).unwrap();
    c.execute_batch("CREATE TABLE precious(value TEXT);INSERT INTO precious VALUES('keep');PRAGMA user_version=999;").unwrap();
    assert!(Store::open(&path)
        .err()
        .unwrap()
        .to_string()
        .contains("unsupported"));
    assert_eq!(
        c.query_row("SELECT value FROM precious", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "keep"
    );
}
#[test]
fn pair_missing_predictions_keep_denominator_and_confusion() {
    let mut b = pair();
    b.id = "case-2".into();
    let score = score_pairs(&[pair(), b], &[prediction()]).unwrap();
    assert_eq!(score.total, 2);
    assert_eq!(score.completed, 1);
    assert_eq!(score.relationship_accuracy, 0.5);
    assert_eq!(score.temporal_accuracy, 0.5);
    assert_eq!(score.update_accuracy, 0.5);
    assert_eq!(score.joint_accuracy, 0.5);
    assert_eq!(score.confusion["supports"]["missing"], 1);
    assert!((score.relationship_macro_f1 - 1.0 / 6.0).abs() < 1e-12);
    assert!(score.brier.is_none());
}
#[test]
fn pair_duplicate_unknown_ids_and_probabilities_rejected() {
    assert!(score_pairs(&[pair()], &[prediction(), prediction()])
        .unwrap_err()
        .to_string()
        .contains("duplicate"));
    let mut p = prediction();
    p.id = "unknown".into();
    assert!(score_pairs(&[pair()], &[p])
        .unwrap_err()
        .to_string()
        .contains("unknown"));
    let mut p = prediction();
    p.native_probabilities = Some(NativeProbabilities {
        values: BTreeMap::from([(Relationship::Supports, 1.0)]),
        provenance: human(),
        native_evidence: "not native".into(),
    });
    assert!(score_pairs(&[pair()], &[p])
        .unwrap_err()
        .to_string()
        .contains("provenance"));
    let json = r#"{"values":{"supports":0.5,"supports":0.5},"provenance":{"kind":"human","note":"x"},"native_evidence":"x"}"#;
    assert!(serde_json::from_str::<NativeProbabilities>(json)
        .unwrap_err()
        .to_string()
        .contains("duplicate probability"));
}
#[test]
fn brier_requires_complete_native_vectors_not_confidence() {
    let mut p = prediction();
    let prob = NativeProbabilities {
        values: Relationship::ALL
            .into_iter()
            .map(|r| {
                (
                    r,
                    if r == Relationship::Supports {
                        1.0
                    } else {
                        0.0
                    },
                )
            })
            .collect(),
        provenance: Provenance::Pumas {
            model: "operator-import-test".into(),
            profile: "test".into(),
            instance_generation: "instance".into(),
            service_generation: "service".into(),
            request_id: "request".into(),
        },
        native_evidence: "authored scorer fixture; not real runtime evidence".into(),
    };
    p.native_probabilities = Some(prob);
    assert_eq!(
        score_pairs(&[pair()], &[p.clone()]).unwrap().brier,
        Some(0.0)
    );
    p.native_probabilities
        .as_mut()
        .unwrap()
        .values
        .insert(Relationship::Compatible, 0.5);
    assert!(score_pairs(&[pair()], &[p])
        .unwrap_err()
        .to_string()
        .contains("range or sum"));
}
fn retrieval() -> RetrievalCase {
    RetrievalCase {
        id: "q".into(),
        query_text: "test".into(),
        filters: Filters::default(),
        expected_ids: vec!["a".into(), "b".into()],
        misleading_ids: vec!["c".into()],
        chronology: vec!["a".into(), "b".into()],
    }
}
#[test]
fn retrieval_scores_known_ranking_and_missing_pessimistically() {
    let ids = BTreeSet::from(["a".into(), "b".into(), "c".into()]);
    let p = RetrievalPrediction {
        id: "q".into(),
        ranked_ids: Some(vec!["c".into(), "a".into(), "b".into()]),
        error: None,
    };
    let score = score_retrieval(&[retrieval()], &[p], &ids).unwrap();
    assert_eq!(score.top_accuracy, 0.0);
    assert_eq!(score.mrr, 0.5);
    assert_eq!(score.recall_at_5, 1.0);
    assert_eq!(score.misleading_at_5, 0.2);
    assert_eq!(score.chronology_accuracy, Some(1.0));
    let missing = score_retrieval(&[retrieval()], &[], &ids).unwrap();
    assert_eq!(missing.completed, 0);
    assert_eq!(missing.misleading_at_5, 1.0);
    assert_eq!(missing.chronology_accuracy, Some(0.0));
}
#[test]
fn retrieval_rejects_duplicate_and_unknown_rankings() {
    let ids = BTreeSet::from(["a".into(), "b".into(), "c".into()]);
    for ranked in [vec!["a".into(), "a".into()], vec!["unknown".into()]] {
        let p = RetrievalPrediction {
            id: "q".into(),
            ranked_ids: Some(ranked),
            error: None,
        };
        assert!(score_retrieval(&[retrieval()], &[p], &ids)
            .unwrap_err()
            .to_string()
            .contains("unknown or duplicate ranked"));
    }
}
#[test]
fn endpoint_validation_and_capability_identity_are_strict() {
    for bad in [
        "https://127.0.0.1:8080",
        "http://localhost:8080",
        "http://192.0.2.1:8080",
        "http://127.0.0.1:8080/a",
        "http://u:p@127.0.0.1:8080",
        "http://127.0.0.1:0",
    ] {
        assert!(endpoint(bad).is_err(), "{bad}");
    }
    assert_eq!(endpoint("http://[::1]:8080/").unwrap(), "http://[::1]:8080");
    let s = Selection {
        model: "exact".into(),
        profile: "p".into(),
        capability: "text_embedding".into(),
    };
    let mut c:Capabilities=serde_json::from_value(serde_json::json!({"supported_contract_versions":[1],"model":"exact","profile":"p","max_request_bytes":1024,"max_response_bytes":1024,"capabilities":[{"capability":"text_embedding","semantic_task":"text_embedding","input_formats":["text"],"output_formats":["embeddings_float32"],"availability":{"state":"available"},"option_bounds":[]}]})).unwrap();
    validate_capabilities(&c, &s).unwrap();
    c.profile = "other".into();
    assert!(validate_capabilities(&c, &s)
        .unwrap_err()
        .to_string()
        .contains("identity mismatch"));
    c.profile = "p".into();
    c.capabilities.push(c.capabilities[0].clone());
    assert!(validate_capabilities(&c, &s)
        .unwrap_err()
        .to_string()
        .contains("duplicate capability"));
}
#[test]
fn latest_filters_after_snapshot_and_does_not_merge_distinct_objects() {
    let mut s = Store::open(":memory:").unwrap();
    let mut old = claim("old", "Alice", "Bob", 1);
    old.state_key = None;
    let mut new = old.clone();
    new.id = "new".into();
    new.story_time.as_mut().unwrap().position = 2;
    new.triple.polarity = Polarity::Negated;
    let mut other = claim("other", "Alice", "Carol", 3);
    other.state_key = None;
    s.ingest(&ingest(vec![old, new, other])).unwrap();
    let f = Filters {
        subject: Some("Alice".into()),
        object: Some("Bob".into()),
        polarity: Some(Polarity::Affirmed),
        latest: true,
        ..Default::default()
    };
    assert!(s
        .compare(&embedding(vec![1.0, 0.0]), &f)
        .unwrap()
        .structured
        .is_empty());
    let f = Filters {
        latest: true,
        ..Default::default()
    };
    let c = s.compare(&embedding(vec![1.0, 0.0]), &f).unwrap();
    assert_eq!(c.structured.len(), 2);
    assert!(c.structured.iter().any(|h| h.claim.id == "new"));
    assert!(c.structured.iter().any(|h| h.claim.id == "other"));
}
#[test]
fn explicit_state_key_does_not_resurrect_superseded_value() {
    let mut s = Store::open(":memory:").unwrap();
    s.ingest(&ingest(vec![
        claim("old", "Alice", "Paris", 1),
        claim("new", "Alice", "Rome", 2),
    ]))
    .unwrap();
    let f = Filters {
        object: Some("Paris".into()),
        latest: true,
        ..Default::default()
    };
    assert!(s
        .compare(&embedding(vec![1.0, 0.0]), &f)
        .unwrap()
        .structured
        .is_empty());
}
#[test]
fn extracted_model_local_ids_get_unique_durable_namespaces() {
    let mut a = vec![claim("claim-1", "A", "Paris", 1)];
    let mut b = a.clone();
    assign_durable_ids(&mut a, "source-a").unwrap();
    assign_durable_ids(&mut b, "source-b").unwrap();
    assert_ne!(a[0].id, b[0].id);
    assert_eq!(a[0].model_local_id.as_deref(), Some("claim-1"));
    let mut s = Store::open(":memory:").unwrap();
    s.ingest(&ingest(a)).unwrap();
    s.ingest(&ingest(b)).unwrap();
    assert_eq!(s.claims().unwrap().len(), 2);
}
#[test]
fn extraction_failure_cannot_be_imported_as_decision_success() {
    let raw = RawPairCase {
        id: "case-1".into(),
        a_raw: "a".into(),
        b_raw: "b".into(),
        expected: labels(),
    };
    let r = ExtractionRecord {
        id: "case-1".into(),
        a: None,
        b: Some(vec![claim("b", "A", "Paris", 1)]),
        a_error: Some("provider unavailable".into()),
        b_error: None,
        a_provenance: None,
        b_provenance: Some(human()),
        decision: prediction(),
    };
    assert!(score_extraction(&[raw], &[r])
        .unwrap_err()
        .to_string()
        .contains("without aligned extraction"));
}
