//! Deterministic arithmetic controls are not semantic-quality or live-model evidence.
use phased_memory::{
    domain::{Polarity, StoryTime},
    phase::*,
    pumas::{Config, Pumas},
};
use serde_json::{json, Value};
use std::path::PathBuf;
fn semantics() -> Semantics {
    serde_json::from_value(json!({"subject":"alice","predicate":"trusts","object":"bob","polarity":"affirmed","perspective":{"holder":"narrator","modality":"fact"},"continuity":"main","entities":[],"qualifiers":[],"state_key":null,"story_time":null})).unwrap()
}
fn input(corpus: Vec<Semantics>, queries: Vec<Semantics>) -> ExperimentInput {
    ExperimentInput {
        schema_version: 1,
        data_kind: DataKind::AuthoredDevelopment,
        candidate_count: corpus.len(),
        corpus: corpus
            .into_iter()
            .enumerate()
            .map(|(i, semantics)| Item {
                id: format!("m{i:02}"),
                semantics,
            })
            .collect(),
        queries: queries
            .into_iter()
            .enumerate()
            .map(|(i, semantics)| Item {
                id: format!("q{i:02}"),
                semantics,
            })
            .collect(),
    }
}
fn fixture() -> ExperimentInput {
    serde_json::from_slice(
        &std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/phase/control.json"),
        )
        .unwrap(),
    )
    .unwrap()
}
fn unavailable() -> Pumas {
    Pumas::new(Config {
        endpoint: "http://127.0.0.1:8080".into(),
        extraction: None,
        embedding: None,
        decision: None,
        decision_adapter: None,
    })
    .unwrap()
}
#[test]
fn tuple_negation_is_antiphase_and_unknown_is_independent_imaginary_channel() {
    let affirmed = semantics();
    let mut negated = affirmed.clone();
    negated.polarity = Polarity::Negated;
    let mut unknown = affirmed.clone();
    unknown.polarity = Polarity::Unknown;
    let experiment = input(
        vec![affirmed.clone(), negated.clone(), unknown.clone()],
        vec![affirmed.clone()],
    );
    let frozen = freeze(&experiment).unwrap();
    let tuple = frozen
        .vocabulary
        .iter()
        .position(|s| s.starts_with("tuple:"))
        .unwrap();
    let a = frozen.encode(&affirmed).unwrap();
    let b = frozen.encode(&negated).unwrap();
    let u = frozen.encode(&unknown).unwrap();
    assert_eq!(a.real[tuple], -b.real[tuple]);
    assert_eq!(a.imaginary[tuple], 0.0);
    assert_eq!(u.real[tuple], 0.0);
    assert_eq!(u.imaginary[tuple], a.real[tuple]);
    assert!(complex_score(&a, &b).unwrap().abs() < 1e-15);
    assert!((complex_score(&a, &u).unwrap() - 0.5).abs() < 1e-15);
}
#[test]
fn direction_scope_and_json_types_get_distinct_tuple_slots() {
    let a = semantics();
    let mut reversed = a.clone();
    reversed.subject = Some("bob".into());
    reversed.object = json!("alice");
    let mut belief = a.clone();
    belief.perspective = json!({"modality":"belief","holder":"alice"});
    let mut number = a.clone();
    number.object = json!(2);
    let mut string = a.clone();
    string.object = json!("2");
    let x = input(
        vec![a.clone(), reversed.clone(), belief, number, string],
        vec![a],
    );
    let frozen = freeze(&x).unwrap();
    assert_eq!(
        frozen
            .vocabulary
            .iter()
            .filter(|s| s.starts_with("tuple:"))
            .count(),
        5
    );
    let scores = control(&x).unwrap();
    assert_eq!(scores.numeric_queries[0].phase[0].id, "m00");
    assert!(
        scores.numeric_queries[0]
            .phase
            .iter()
            .find(|s| s.id == "m01")
            .unwrap()
            .score
            < 1.0
    );
}
#[test]
fn time_scale_is_frozen_over_corpus_and_queries_with_known_and_unknown_clock_slots() {
    let mut a = semantics();
    a.story_time = Some(StoryTime {
        timeline: "story".into(),
        position: -5,
    });
    let mut query = a.clone();
    query.story_time.as_mut().unwrap().position = 20;
    let unknown = semantics();
    let experiment = input(vec![a.clone(), unknown.clone()], vec![query.clone()]);
    let frozen = freeze(&experiment).unwrap();
    assert_eq!(frozen.time_scale, 20.0);
    let numeric = frozen
        .vocabulary
        .iter()
        .position(|s| s.starts_with("time_value:"))
        .unwrap();
    let presence = frozen
        .vocabulary
        .iter()
        .position(|s| s.starts_with("timeline:"))
        .unwrap();
    let unknown_slot = frozen
        .vocabulary
        .iter()
        .position(|s| s == "unknown_time")
        .unwrap();
    let encoded = frozen.encode(&a).unwrap();
    assert!((encoded.real[numeric] / encoded.real[presence] + 0.25).abs() < 1e-15);
    assert_eq!(encoded.real[unknown_slot], 0.0);
    let encoded = frozen.encode(&unknown).unwrap();
    assert!(encoded.real[unknown_slot] > 0.0);
    assert_eq!(encoded.real[presence], 0.0);
    query.story_time.as_mut().unwrap().position = i64::MIN;
    assert!(control(&input(vec![a], vec![query]))
        .unwrap_err()
        .to_string()
        .contains("exact f64 integer range"));
}
#[test]
fn canonical_projection_sorts_recursive_keys_and_deduplicates_category_sets_for_all_arms() {
    let mut a = semantics();
    a.entities = vec!["bob".into(), "alice".into(), "bob".into()];
    a.qualifiers = vec!["lie".into(), "belief".into(), "lie".into()];
    a.object = json!({"z":{"z":true,"a":2},"a":["ordered",false]});
    let mut b = a.clone();
    b.entities = vec!["alice".into(), "bob".into()];
    b.qualifiers = vec!["belief".into(), "lie".into()];
    b.object = serde_json::from_str(r#"{"a":["ordered",false],"z":{"a":2,"z":true}}"#).unwrap();
    assert_eq!(a.canonical_json().unwrap(), b.canonical_json().unwrap());
    let output = control(&input(vec![a], vec![b])).unwrap();
    assert_eq!(
        output.frozen.corpus[0].canonical_semantics,
        output.frozen.queries[0].canonical_semantics
    );
    assert!((output.numeric_queries[0].phase[0].score - 1.0).abs() < 1e-12);
    let value: Value = serde_json::from_str(&output.frozen.corpus[0].canonical_semantics).unwrap();
    assert_eq!(value["entities"], json!(["alice", "bob"]));
    assert_eq!(value["qualifiers"], json!(["belief", "lie"]));
}
#[test]
fn duplicate_ids_wrong_candidate_budgets_unknown_fields_and_missing_nullable_fields_fail() {
    let mut x = input(vec![semantics(), semantics()], vec![semantics()]);
    x.corpus[1].id = x.corpus[0].id.clone();
    assert!(control(&x)
        .unwrap_err()
        .to_string()
        .contains("duplicate phase id"));
    x.corpus[1].id = "other".into();
    x.candidate_count = 1;
    assert!(control(&x)
        .unwrap_err()
        .to_string()
        .contains("entire corpus"));
    let mut value = serde_json::to_value(semantics()).unwrap();
    value["confidence"] = json!(0.99);
    assert!(serde_json::from_value::<Semantics>(value)
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
    let mut value = serde_json::to_value(semantics()).unwrap();
    value.as_object_mut().unwrap().remove("story_time");
    assert!(serde_json::from_value::<Semantics>(value)
        .unwrap_err()
        .to_string()
        .contains("missing field"));
    let mut value = serde_json::to_value(input(vec![semantics()], vec![semantics()])).unwrap();
    value["gold"] = json!({"expected":"m00"});
    assert!(serde_json::from_value::<ExperimentInput>(value).is_err());
}
#[test]
fn exact_ties_are_broken_by_ids_and_all_candidates_are_ranked() {
    let mut x = input(
        vec![semantics(), semantics(), semantics()],
        vec![semantics()],
    );
    x.corpus[0].id = "z".into();
    x.corpus[1].id = "b".into();
    x.corpus[2].id = "a".into();
    let result = control(&x).unwrap();
    assert_eq!(
        result.numeric_queries[0]
            .phase
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "z"]
    );
    assert_eq!(result.numeric_queries[0].real_concatenation.len(), 3);
    assert!(result.numeric_queries[0].rankings_equal);
    assert!(result.pumas_embeddings.is_none());
}
#[test]
fn authored_ten_candidate_control_obeys_expected_equivalence_without_claiming_quality() {
    let x = fixture();
    let result = control(&x).unwrap();
    assert_eq!(result.frozen.candidate_count, 10);
    assert_eq!(result.numeric_queries.len(), 4);
    assert!(result.equivalence_passed);
    assert!(result.maximum_absolute_error <= 1e-12);
    for query in &result.numeric_queries {
        assert_eq!(query.phase.len(), 10);
        assert_eq!(query.real_concatenation.len(), 10);
    }
    assert!(result.claim.contains("no phase advantage"));
    let renamed = ExperimentInput {
        corpus: x
            .corpus
            .iter()
            .enumerate()
            .map(|(i, item)| Item {
                id: format!("renamed-{i}"),
                semantics: item.semantics.clone(),
            })
            .collect(),
        ..x.clone()
    };
    let renamed = control(&renamed).unwrap();
    assert_eq!(result.frozen.vocabulary, renamed.frozen.vocabulary);
    assert_eq!(result.frozen.time_scale, renamed.frozen.time_scale);
    assert_eq!(
        result
            .frozen
            .corpus
            .iter()
            .map(|r| &r.canonical_semantics)
            .collect::<Vec<_>>(),
        renamed
            .frozen
            .corpus
            .iter()
            .map(|r| &r.canonical_semantics)
            .collect::<Vec<_>>()
    );
}
#[tokio::test]
async fn unavailable_pumas_keeps_every_case_and_never_substitutes_numeric_vectors() {
    let x = fixture();
    let result = compare(&unavailable(), &x).await.unwrap();
    assert!(result.equivalence_passed);
    let arm = result.pumas_embeddings.unwrap();
    assert_eq!(arm.status, "unrun");
    assert_eq!(arm.corpus.len(), 10);
    assert_eq!(arm.queries.len(), 4);
    assert!(arm.space.is_none());
    assert!(arm.corpus.iter().all(|r| r.status == "unrun"
        && r.error
            .as_deref()
            .unwrap()
            .contains("embedding identity not configured")));
    assert!(arm.queries.iter().all(|r| r.status == "unrun"
        && r.ranked.is_none()
        && r.provenance.is_none()
        && r.error
            .as_deref()
            .unwrap()
            .contains("no inference attempted")));
}
#[test]
fn near_tie_score_equivalence_does_not_claim_identical_raw_rankings() {
    let mut z = semantics();
    z.story_time = Some(StoryTime {
        timeline: "story".into(),
        position: 0,
    });
    let mut a = z.clone();
    a.story_time.as_mut().unwrap().position = 100;
    let mut scale = z.clone();
    scale.object = json!("unrelated");
    scale.story_time.as_mut().unwrap().position = 245;
    let mut query = z.clone();
    query.story_time.as_mut().unwrap().position = 49;
    let mut experiment = input(vec![z, a, scale], vec![query]);
    experiment.corpus[0].id = "z".into();
    experiment.corpus[1].id = "a".into();
    let result = control(&experiment).unwrap();
    assert!(result.equivalence_passed);
    assert!(result.maximum_absolute_error < 1e-12);
    assert!(result.equivalence_scope.contains("pairwise score"));
    let q = &result.numeric_queries[0];
    assert!(!q.rankings_equal, "{q:?}");
    assert_eq!(q.phase[0].id, "z");
    assert_eq!(q.real_concatenation[0].id, "a");
}

#[derive(Clone)]
struct Mock {
    endpoint: String,
    texts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    fail_first: bool,
}
async fn mock_discovery(axum::extract::State(s): axum::extract::State<Mock>) -> axum::Json<Value> {
    axum::Json(
        json!({"advertisement_schema_version":1,"endpoint":s.endpoint,"service_generation":"phase-service","instance":{"generation":"phase-instance"},"build_info":{"component":"test-only","package_version":"test","schemas":[{"name":"pumas.http-admission-fence","version":1}]}}),
    )
}
async fn mock_capabilities() -> axum::Json<Value> {
    axum::Json(
        json!({"supported_contract_versions":[1],"model":"test-only","profile":"p","max_request_bytes":65536,"max_response_bytes":65536,"capabilities":[{"capability":"text_embedding","semantic_task":"text_embedding","input_formats":["text"],"output_formats":["embeddings_float32"],"availability":{"state":"available"},"option_bounds":[]}]}),
    )
}
async fn mock_operation(
    axum::extract::State(s): axum::extract::State<Mock>,
    axum::Json(request): axum::Json<Value>,
) -> impl axum::response::IntoResponse {
    let text = request["input"]["text"].as_str().unwrap();
    let first = {
        let mut texts = s.texts.lock().unwrap();
        texts.push(text.into());
        texts.len() == 1
    };
    if s.fail_first && first {
        (
            axum::http::StatusCode::BAD_GATEWAY,
            axum::Json(
                json!({"contract_version":1,"request_id":request["request_id"],"error":{"code":"provider_failure","outcome":"unknown"}}),
            ),
        )
    } else {
        (
            axum::http::StatusCode::OK,
            axum::Json(
                json!({"contract_version":1,"request_id":request["request_id"],"result":{"kind":"embeddings","vectors":[[1.0,0.0]]}}),
            ),
        )
    }
}
async fn mock(fail_first: bool) -> (Pumas, Mock, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let state = Mock {
        endpoint: endpoint.clone(),
        texts: Default::default(),
        fail_first,
    };
    let app = axum::Router::new()
        .route("/.well-known/pumas", axum::routing::get(mock_discovery))
        .route("/v1/capabilities", axum::routing::get(mock_capabilities))
        .route("/v1/model-operations", axum::routing::post(mock_operation))
        .with_state(state.clone());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let pumas = Pumas::new(Config {
        endpoint,
        extraction: None,
        embedding: Some(phased_memory::pumas::Selection {
            model: "test-only".into(),
            profile: "p".into(),
            capability: "text_embedding".into(),
        }),
        decision: None,
        decision_adapter: None,
    })
    .unwrap();
    (pumas, state, task)
}
#[tokio::test]
async fn pumas_receives_exact_canonical_semantics_without_ids_or_prose_and_returns_full_budget() {
    let (pumas, state, task) = mock(false).await;
    let experiment = fixture();
    let result = compare(&pumas, &experiment).await.unwrap();
    let arm = result.pumas_embeddings.as_ref().unwrap();
    assert_eq!(arm.status, "completed");
    let expected: Vec<_> = result
        .frozen
        .corpus
        .iter()
        .chain(&result.frozen.queries)
        .map(|item| item.canonical_semantics.clone())
        .collect();
    assert_eq!(*state.texts.lock().unwrap(), expected);
    for text in state.texts.lock().unwrap().iter() {
        let value: Value = serde_json::from_str(text).unwrap();
        assert!(value.get("id").is_none());
        assert!(value.get("source").is_none());
        assert!(value.get("confidence").is_none());
        assert!(value.get("gold").is_none());
    }
    for query in &arm.queries {
        assert_eq!(query.ranked.as_ref().unwrap().len(), 10);
        assert_eq!(query.vector.as_deref(), Some([1.0, 0.0].as_slice()));
    }
    assert!(arm
        .corpus
        .iter()
        .all(|r| r.provenance.is_some() && r.vector.is_some()));
    task.abort();
}
#[tokio::test]
async fn incomplete_corpus_never_retries_failed_input_or_dispatches_query_embeddings() {
    let (pumas, state, task) = mock(true).await;
    let experiment = fixture();
    let result = compare(&pumas, &experiment).await.unwrap();
    let arm = result.pumas_embeddings.unwrap();
    assert_eq!(arm.status, "unavailable");
    assert_eq!(state.texts.lock().unwrap().len(), 10);
    assert_eq!(arm.corpus.iter().filter(|r| r.status == "error").count(), 1);
    assert!(arm
        .queries
        .iter()
        .all(|q| q.ranked.is_none() && q.vector.is_none()));
    task.abort();
}
