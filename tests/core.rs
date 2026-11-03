use phased_memory::{
    domain::*,
    eval::*,
    pumas::{
        endpoint, native_decision, operation_result, select_capability, Config, Decision, Pumas,
        ServedModel,
    },
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
fn endpoint_validation_is_strict() {
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
}
fn served(capabilities: serde_json::Value) -> ServedModel {
    serde_json::from_value(serde_json::json!({"model":"exact","profile":"p","model_id":"org/exact","provider":"llama_cpp","load_state":"loaded","profile_required":false,"future_field":{"nested":true},"capabilities":capabilities})).unwrap()
}
#[test]
fn capabilities_are_selected_by_name_and_unknown_descriptors_are_ignored() {
    let embedding = serde_json::json!({"capability":"text_embedding","semantic_task":"text_embedding","input_formats":["text","text_batch","future_format"],"output_formats":["embeddings_float32"],"streaming":false,"availability":{"state":"available"},"option_bounds":[{"option":"dimensions","minimum":1,"maximum":4294967295.0},{"option":"future_option","minimum":"x"}],"future":1});
    // Unknown capabilities, malformed unused descriptors, unknown availability
    // states and extra entries precede the one this consumer uses.
    let s = served(serde_json::json!([
        {"capability":"future_capability","shape":"unknown"},
        {"capability":"decision","semantic_task":"decision","input_formats":["decision"],"output_formats":["decisions"],"availability":{"state":"degraded","reason":"future_reason"},"option_bounds":[]},
        embedding.clone(),
        {"capability":"audio_classification","availability":{"state":"unavailable","reason":"unsupported_adapter"}}
    ]));
    let d = select_capability(&s, &["text_embedding"]).unwrap();
    assert_eq!(d.capability, "text_embedding");
    let e = select_capability(&s, &["decision"])
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("unavailable: selected capability decision: future_reason"),
        "{e}"
    );
    let e = select_capability(&s, &["chat_generation", "text_generation"])
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("chat_generation: not advertised; text_generation: not advertised"),
        "{e}"
    );
    // Only the selected capability must be unambiguous.
    let dup = served(serde_json::json!([embedding.clone(), embedding]));
    assert!(select_capability(&dup, &["text_embedding"])
        .unwrap_err()
        .to_string()
        .contains("duplicate text_embedding"));
    // Preference order: chat first, then text generation.
    let text = served(
        serde_json::json!([{"capability":"text_generation","semantic_task":"text_generation","input_formats":["text"],"output_formats":["text"],"availability":{"state":"available"},"option_bounds":[]}]),
    );
    assert_eq!(
        select_capability(&text, &["chat_generation", "text_generation"])
            .unwrap()
            .capability,
        "text_generation"
    );
}
#[test]
fn typed_errors_are_examined_before_correlation() {
    let body = |v: serde_json::Value| serde_json::to_vec(&v).unwrap();
    // Unreadable request bodies are rejected with a null request_id: still not admitted.
    let e = operation_result(400, &body(serde_json::json!({"contract_version":1,"request_id":null,"error":{"code":"invalid_request","outcome":"not_admitted","message":"request body is not valid JSON"}})), "mine").unwrap_err().to_string();
    assert!(e.starts_with("not_admitted: Pumas operation invalid_request (HTTP 400): request body is not valid JSON"), "{e}");
    assert!(!e.contains("never replay"), "{e}");
    // Unknown codes and extra fields are tolerated; the message is surfaced.
    let e = operation_result(429, &body(serde_json::json!({"contract_version":1,"request_id":"mine","error":{"code":"future_code","outcome":"not_admitted","message":"busy\nretry later","hint":1}})), "mine").unwrap_err().to_string();
    assert!(
        e.starts_with("not_admitted: Pumas operation future_code (HTTP 429): busy retry later"),
        "{e}"
    );
    let e = operation_result(502, &body(serde_json::json!({"contract_version":1,"request_id":"mine","error":{"code":"provider_failure","outcome":"unknown","message":"llama.cpp reported: out of memory"}})), "mine").unwrap_err().to_string();
    assert!(e.starts_with("unknown: Pumas operation provider_failure (HTTP 502): llama.cpp reported: out of memory; never replay"), "{e}");
    // An unrecognized outcome stays conservative.
    let e = operation_result(502, &body(serde_json::json!({"contract_version":1,"request_id":"mine","error":{"code":"x","outcome":"future_outcome","message":"m"}})), "mine").unwrap_err().to_string();
    assert!(e.starts_with("unknown:"), "{e}");
    // A different caller's ID is not trusted, even for not_admitted.
    let e = operation_result(400, &body(serde_json::json!({"contract_version":1,"request_id":"theirs","error":{"code":"invalid_request","outcome":"not_admitted","message":"m"}})), "mine").unwrap_err().to_string();
    assert!(
        e.starts_with("unknown: Pumas error correlation mismatch"),
        "{e}"
    );
    // A typed error with a success status is contradictory.
    let e = operation_result(200, &body(serde_json::json!({"contract_version":1,"request_id":"mine","error":{"code":"invalid_request","outcome":"not_admitted","message":"m"}})), "mine").unwrap_err().to_string();
    assert!(
        e.starts_with("unknown: HTTP 200 contradicts typed error"),
        "{e}"
    );
    // Empty-body fence refusals happen before admission.
    for (status, text) in [
        (412, "stale runtime generation"),
        (503, "not admitting"),
        (400, "fence headers"),
    ] {
        let e = operation_result(status, b"", "mine")
            .unwrap_err()
            .to_string();
        assert!(
            e.starts_with("not_admitted:") && e.contains(text),
            "{status}: {e}"
        );
    }
    assert!(operation_result(502, b"", "mine")
        .unwrap_err()
        .to_string()
        .starts_with("unknown:"));
    assert!(operation_result(200, b"not json", "mine")
        .unwrap_err()
        .to_string()
        .starts_with("unknown: invalid Pumas JSON"));
    // Success responses ignore unknown fields but stay correlated.
    let ok = operation_result(200, &body(serde_json::json!({"contract_version":1,"request_id":"mine","result":{"kind":"embeddings","vectors":[[1.0]],"usage":{}},"timing":{}})), "mine").unwrap();
    assert_eq!(ok["kind"], "embeddings");
    assert!(operation_result(200, &body(serde_json::json!({"contract_version":1,"request_id":"other","result":{"kind":"embeddings"}})), "mine").unwrap_err().to_string().contains("correlation/version mismatch"));
}
#[test]
fn native_decision_answers_map_to_labels_with_uncalibrated_probabilities() {
    let result = serde_json::json!({"kind":"decisions","input_tokens":321,"future":true,"answers":{
        "relationship":{"type":"choice","choice":"contradicts","confidence":0.7,"probabilities":{"supports":0.1,"contradicts":0.7,"compatible":0.15,"insufficient_information":0.05,"future_label":0.0}},
        "temporal_order":{"type":"choice","choice":"same_time","confidence":0.6,"probabilities":{"same_time":0.6,"b_later":0.2,"b_earlier":0.1,"unknown":0.1}},
        "update_type":{"type":"choice","choice":"none","confidence":0.9,"probabilities":{"none":0.9,"state_change":0.05,"explicit_correction":0.03,"undetermined":0.02}},
        "extra_question":{"type":"noul","noul":0.5}}});
    let (labels, answers) = native_decision(&result).unwrap();
    assert_eq!(
        labels,
        Labels {
            relationship: Relationship::Contradicts,
            temporal_order: TemporalOrder::SameTime,
            update_type: UpdateType::None
        }
    );
    assert_eq!(answers["relationship"].probabilities.len(), 4);
    let decision = Decision {
        labels,
        adapter: "decision_native_v1".into(),
        confidence: Some(answers["relationship"].confidence),
        native_answers: Some(answers),
    };
    let p = decision.relationship_probabilities().unwrap();
    assert_eq!(p[&Relationship::Contradicts], 0.7);
    for bad in [
        serde_json::json!({"type":"choice","choice":"maybe","confidence":0.5,"probabilities":{}}),
        serde_json::json!({"type":"score","score":1.0,"confidence":0.5,"probabilities":{}}),
        serde_json::json!({"type":"choice","choice":"supports","confidence":1.5,"probabilities":{}}),
        serde_json::json!({"type":"choice","choice":"supports","confidence":0.5,"probabilities":{"supports":2.0}}),
    ] {
        let mut r = result.clone();
        r["answers"]["relationship"] = bad;
        assert!(native_decision(&r)
            .unwrap_err()
            .to_string()
            .starts_with("unknown:"));
    }
    let mut r = result.clone();
    r["answers"].as_object_mut().unwrap().remove("update_type");
    assert!(native_decision(&r).is_err());
}
#[test]
fn config_pins_are_validated_on_the_request_side() {
    let config = |embedding: serde_json::Value,
                  decision: serde_json::Value,
                  adapter: serde_json::Value| {
        serde_json::from_value::<Config>(serde_json::json!({"endpoint":"http://127.0.0.1:8080","extraction":null,"embedding":embedding,"decision":decision,"decision_adapter":adapter})).map_err(anyhow::Error::from).and_then(Pumas::new)
    };
    let gemma = serde_json::json!({"model":"embeddinggemma-2","query_prefix":"task: search result | query: ","document_prefix":"title: none | text: "});
    config(
        gemma.clone(),
        serde_json::json!({"model":"kev"}),
        serde_json::json!("decision_native_v1"),
    )
    .unwrap();
    // Legacy configs with explicit profile and capability still parse as pins.
    config(
        serde_json::json!({"model":"e","profile":"p","capability":"text_embedding"}),
        serde_json::Value::Null,
        serde_json::Value::Null,
    )
    .unwrap();
    for (embedding, decision, adapter) in [
        (
            gemma.clone(),
            serde_json::json!({"model":"kev","capability":"decision"}),
            serde_json::json!("decision_json_v1"),
        ),
        (
            gemma.clone(),
            serde_json::json!({"model":"chat","capability":"chat_generation"}),
            serde_json::json!("decision_native_v1"),
        ),
        (
            gemma.clone(),
            serde_json::json!({"model":"kev"}),
            serde_json::json!("guessed_adapter"),
        ),
        (
            serde_json::json!({"model":"e","capability":"chat_generation"}),
            serde_json::Value::Null,
            serde_json::Value::Null,
        ),
        (
            serde_json::json!({"model":"e","query_prefix":"bad\u{7}"}),
            serde_json::Value::Null,
            serde_json::Value::Null,
        ),
        (
            serde_json::json!({"model":"e","prefix":"typo"}),
            serde_json::Value::Null,
            serde_json::Value::Null,
        ),
        (
            gemma.clone(),
            serde_json::json!({"model":"kev","query_prefix":"q: "}),
            serde_json::json!("decision_native_v1"),
        ),
    ] {
        assert!(
            config(embedding.clone(), decision.clone(), adapter.clone()).is_err(),
            "{embedding} {decision} {adapter}"
        );
    }
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
