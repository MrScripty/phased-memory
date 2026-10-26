use phased_memory::{
    domain::*,
    original::*,
    pumas::{Config, Pumas},
    store::{Filters, Store},
};
use serde_json::{json, Value};
use std::{path::PathBuf, process::Command};
fn pack() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/original")
}
fn pairs() -> Vec<PairCase> {
    read_jsonl(pack().join("pairs.jsonl")).unwrap()
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
fn all_original_rows_preserve_primary_fields_context_and_extra_claims() {
    let cases = pairs();
    assert_eq!(cases.len(), 24);
    for case in &cases {
        let input = case.decision_input().unwrap();
        assert_eq!(input.context, case.context);
        assert_eq!(
            input.primary_a,
            case.memory_a.normalized_claims[case.comparison_targets.a_claim_index]
        );
        assert_eq!(
            input.primary_b,
            case.memory_b.normalized_claims[case.comparison_targets.b_claim_index]
        );
    }
    let s = cases.iter().find(|c| c.id == "S04").unwrap();
    let input = s.decision_input().unwrap();
    assert_eq!(input.primary_b.subject.as_deref(), Some("alice"));
    assert_eq!(
        input.evidence_b.additional_claims[0].subject.as_deref(),
        Some("bob")
    );
    let mut changed = s.clone();
    changed.comparison_targets.b_claim_index = 1;
    assert_eq!(
        changed
            .decision_input()
            .unwrap()
            .primary_b
            .subject
            .as_deref(),
        Some("bob")
    );
}
#[test]
fn decision_projection_cannot_leak_gold_diagnostics_or_raw_source() {
    let mut case = pairs().remove(1);
    let expected = case.decision_input().unwrap();
    case.id = "CASE_ID_SENTINEL".into();
    case.group = "GROUP_SENTINEL".into();
    case.split = "SPLIT_SENTINEL".into();
    case.gold.relation = Relationship::InsufficientInformation;
    case.rationale = "RATIONALE_SENTINEL".into();
    case.extraction_checks = vec!["CHECK_SENTINEL".into()];
    case.memory_a.text = "RAW_A_SENTINEL".into();
    case.memory_b.text = "RAW_B_SENTINEL".into();
    let input = case.decision_input().unwrap();
    assert_eq!(expected, input);
    let prompt = decision_prompt(&input).unwrap();
    for forbidden in [
        "CASE_ID_SENTINEL",
        "GROUP_SENTINEL",
        "SPLIT_SENTINEL",
        "RATIONALE_SENTINEL",
        "CHECK_SENTINEL",
        "RAW_A_SENTINEL",
        "RAW_B_SENTINEL",
        "gold",
        "comparison_targets",
    ] {
        assert!(!prompt.contains(forbidden), "{forbidden}");
    }
    assert!(prompt.contains("exactly one brass key"));
}
#[test]
fn unknown_scope_nested_values_and_intention_clock_roundtrip_without_defaults() {
    let cases = pairs();
    let unknown = cases
        .iter()
        .find(|c| c.id == "U01")
        .unwrap()
        .decision_input()
        .unwrap();
    assert!(unknown.primary_a.valid_time.is_none());
    let scope = cases
        .iter()
        .find(|c| c.id == "U03")
        .unwrap()
        .decision_input()
        .unwrap();
    assert_eq!(scope.primary_b.modality.as_deref(), Some("unknown"));
    assert_eq!(scope.primary_b.holder.as_deref(), Some("unknown"));
    let continuity = cases
        .iter()
        .find(|c| c.id == "U04")
        .unwrap()
        .decision_input()
        .unwrap();
    assert_eq!(continuity.primary_b.continuity.as_deref(), Some("unknown"));
    let plan = cases
        .iter()
        .find(|c| c.id == "N04")
        .unwrap()
        .decision_input()
        .unwrap()
        .primary_b;
    assert_eq!(plan.valid_time.as_ref().unwrap().minute_of_day, 540);
    assert_eq!(plan.object["target_time"]["minute_of_day"], 600);
    let app = plan.application_claim("plan", "original source").unwrap();
    let restored: NormalizedClaim = serde_json::from_str(&app.normalized).unwrap();
    assert_eq!(restored, plan);
    assert_eq!(app.story_time.unwrap().position, 1980);
    let count = cases
        .iter()
        .find(|c| c.id == "C03")
        .unwrap()
        .decision_input()
        .unwrap()
        .primary_a;
    assert!(count.object.is_number());
    let fact = cases
        .iter()
        .find(|c| c.id == "P02")
        .unwrap()
        .decision_input()
        .unwrap()
        .primary_a;
    assert!(fact.object.is_boolean());
    let mut null = plan;
    null.subject = None;
    null.holder = None;
    null.continuity = None;
    null.valid_time = None;
    null.object = Value::Null;
    let round: NormalizedClaim =
        serde_json::from_str(&serde_json::to_string(&null).unwrap()).unwrap();
    assert_eq!(round, null);
}
#[test]
fn belief_and_world_scope_remain_separate() {
    let p = pairs()
        .into_iter()
        .find(|c| c.id == "P03")
        .unwrap()
        .decision_input()
        .unwrap();
    let a = p.primary_a.application_claim("belief", "raw").unwrap();
    let b = p.primary_b.application_claim("world", "raw").unwrap();
    assert_ne!(a.perspective, b.perspective);
    assert_eq!(p.primary_a.holder.as_deref(), Some("alice"));
    assert_eq!(p.primary_b.holder.as_deref(), Some("narrator"));
}
#[test]
fn stage_b_primary_is_model_selected_and_wrong_extraction_is_not_repaired() {
    let case = pairs().into_iter().find(|c| c.id == "S04").unwrap();
    let raw = case.extraction_input();
    let a = Extraction {
        claims: vec![QuotedClaim {
            claim: case.memory_a.normalized_claims[0].clone(),
            source_quote: "Alice knows the vault code".into(),
        }],
        primary_claim_index: 0,
    };
    let mut b = Extraction {
        claims: vec![
            QuotedClaim {
                claim: case.memory_b.normalized_claims[0].clone(),
                source_quote: "Alice knows the vault code".into(),
            },
            QuotedClaim {
                claim: case.memory_b.normalized_claims[1].clone(),
                source_quote: "Bob does not".into(),
            },
        ],
        primary_claim_index: 1,
    };
    b.claims[1].claim.polarity = Some(Polarity::Affirmed);
    let input = extracted_decision_input(&raw, &a, &b).unwrap();
    assert_eq!(input.primary_b.subject.as_deref(), Some("bob"));
    assert_eq!(input.primary_b.polarity, Some(Polarity::Affirmed));
    assert_eq!(input.evidence_b.additional_claims.len(), 1);
    let prompt = decision_prompt(&input).unwrap();
    assert!(!prompt.contains("Bob does not"));
    b.claims[1].source_quote = "not an exact source quote".into();
    assert!(extracted_decision_input(&raw, &a, &b)
        .unwrap_err()
        .to_string()
        .contains("exact source substring"));
}
#[tokio::test]
async fn stage_b_never_reads_reference_target_and_missing_models_keep_all_ids() {
    let mut cases = pairs();
    for c in &mut cases {
        c.comparison_targets.a_claim_index = usize::MAX;
        c.comparison_targets.b_claim_index = usize::MAX;
        c.memory_a.normalized_claims.clear();
        c.memory_b.normalized_claims.clear();
    }
    let records = run_pairs(&unavailable(), &cases, true).await.unwrap();
    assert_eq!(records.len(), 24);
    for row in records {
        assert_eq!(row.status, "invalid_output");
        let error = row.trace.error.unwrap();
        assert!(
            error.contains("extraction_failure")
                && error.contains("extraction identity not configured"),
            "{error}"
        );
        assert!(row.relation.is_none());
    }
}
#[tokio::test]
async fn original_scorer_accepts_emitted_missing_rows_and_rejects_unknown_ids() {
    let cases = pairs();
    let predictions = run_pairs(&unavailable(), &cases, false).await.unwrap();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("predictions.jsonl");
    write_jsonl(&path, &predictions).unwrap();
    let output = Command::new("python3")
        .arg(pack().join("score.py"))
        .arg("pairs")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let score: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(score["expected_cases"], 24);
    assert_eq!(score["prediction_rows"], 24);
    assert_eq!(score["valid_complete_rows"], 0);
    assert_eq!(score["joint_accuracy"], 0.0);
    assert!(score["multiclass_brier"].is_null());
    let unknown = temp.path().join("unknown.jsonl");
    std::fs::write(
        &unknown,
        "{\"id\":\"UNKNOWN\",\"status\":\"invalid_output\"}\n",
    )
    .unwrap();
    let output = Command::new("python3")
        .arg(pack().join("score.py"))
        .arg("pairs")
        .arg(&unknown)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Unexpected prediction id"));
}
fn synthetic() -> Embedding {
    Embedding {
        space: "synthetic-original-boundary".into(),
        vector: vec![1.0, 0.0],
        provenance: Provenance::Synthetic {
            note: "test-only vectors, not inference".into(),
        },
    }
}
#[test]
fn all_ten_candidates_survive_before_top_five_and_chronology_is_separate() {
    let memories: Vec<Memory> = read_jsonl(pack().join("retrieval_memories.jsonl")).unwrap();
    let mut store = Store::open(":memory:").unwrap();
    for m in &memories {
        store
            .ingest(&Ingest {
                raw_text: m.text.clone(),
                claims: vec![m
                    .normalized_claim
                    .application_claim(&m.id, &m.text)
                    .unwrap()],
                provenance: Provenance::Human {
                    note: "original human refs".into(),
                },
                embeddings: vec![ClaimEmbedding {
                    claim_id: m.id.clone(),
                    embedding: synthetic(),
                }],
            })
            .unwrap();
    }
    assert_eq!(
        store
            .compare_all(&synthetic(), &Filters::default())
            .unwrap()
            .vector
            .len(),
        10
    );
    assert_eq!(
        store
            .compare(&synthetic(), &Filters::default())
            .unwrap()
            .vector
            .len(),
        5
    );
    let all: Vec<_> = memories
        .iter()
        .map(|m| Candidate {
            id: m.id.clone(),
            text: m.text.clone(),
            normalized_claim: m.normalized_claim.clone(),
            cosine: 0.5,
        })
        .collect();
    let plan:QueryPlan=serde_json::from_value(json!({"subject":"alice","predicate":"trusts","object":"bob","polarity":null,"modality":"fact","holder":"narrator","continuity":"main","valid_time":{"story_day":2,"minute_of_day":540},"time_mode":"through","chronological":true})).unwrap();
    let selected = select_candidates(&all, &plan).unwrap();
    assert_eq!(
        selected.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        vec!["M01", "M02", "M06"]
    );
    assert_eq!(
        presentation_chronology(&selected),
        vec!["M06", "M01", "M02"]
    );
    let ranking = Ranking {
        ranked_ids: selected.iter().map(|c| c.id.clone()).collect(),
        chronological_ids: presentation_chronology(&selected),
    };
    validate_ranking(&ranking, &selected, true).unwrap();
    assert_ne!(ranking.ranked_ids, ranking.chronological_ids);
    let prompt = retrieval_prompt("Query without labels", &all, true).unwrap();
    for m in memories {
        assert!(prompt.contains(&m.id));
    }
    assert!(!prompt.contains("relevant_memory_ids"));
}
#[tokio::test]
async fn original_retrieval_missing_index_keeps_eight_queries_and_scorer_semantics() {
    let memories: Vec<Memory> = read_jsonl(pack().join("retrieval_memories.jsonl")).unwrap();
    let queries: Vec<Query> = read_jsonl(pack().join("retrieval_queries.jsonl")).unwrap();
    let records = run_retrieval(
        &Store::open(":memory:").unwrap(),
        &unavailable(),
        &memories,
        &queries,
        "vector",
    )
    .await
    .unwrap();
    assert_eq!(records.len(), 8);
    assert!(records
        .iter()
        .all(|r| r.status == "invalid_output" && r.ranked_ids.is_empty()));
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("retrieval.jsonl");
    write_jsonl(&path, &records).unwrap();
    let output = Command::new("python3")
        .arg(pack().join("score.py"))
        .arg("retrieval")
        .arg(path)
        .output()
        .unwrap();
    assert!(output.status.success());
    let score: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(score["expected_queries"], 8);
    assert_eq!(score["prediction_rows"], 8);
    assert_eq!(score["top1_accuracy"], 0.0);
    assert_eq!(score["queries_with_hard_negative_at5_fraction"], 0.0);
    assert_eq!(score["timeline_exact_match_fraction"], 0.0);
}

// Local protocol simulator: synthetic outputs qualify adapters only, never real models.
#[derive(Clone)]
struct Mock {
    endpoint: String,
    prompts: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}
async fn mock_discovery(axum::extract::State(s): axum::extract::State<Mock>) -> axum::Json<Value> {
    axum::Json(
        json!({"advertisement_schema_version":1,"endpoint":s.endpoint,"service_generation":"service-original","instance":{"generation":"instance-original"},"build_info":{"component":"test-only","package_version":"test","schemas":[{"name":"pumas.http-admission-fence","version":1}]}}),
    )
}
async fn mock_capabilities() -> axum::Json<Value> {
    axum::Json(
        json!({"supported_contract_versions":[1],"model":"test-only","profile":"p","max_request_bytes":65536,"max_response_bytes":65536,"capabilities":[{"capability":"text_embedding","semantic_task":"text_embedding","input_formats":["text"],"output_formats":["embeddings_float32"],"availability":{"state":"available"},"option_bounds":[]},{"capability":"text_generation","semantic_task":"text_generation","input_formats":["text"],"output_formats":["text"],"availability":{"state":"available"},"option_bounds":[]}]}),
    )
}
async fn mock_operation(
    axum::extract::State(s): axum::extract::State<Mock>,
    axum::Json(request): axum::Json<Value>,
) -> axum::Json<Value> {
    let result = if request["capability"] == "text_embedding" {
        json!({"kind":"embeddings","vectors":[[1.0,0.0]]})
    } else {
        let prompt = request["input"]["text"].as_str().unwrap();
        s.prompts.lock().unwrap().push(prompt.into());
        let value = if prompt.starts_with("Parse this retrieval") {
            json!({"subject":"does_not_exist","predicate":null,"object":null,"polarity":null,"modality":null,"holder":null,"continuity":null,"valid_time":null,"time_mode":"any","chronological":false})
        } else if prompt.starts_with("Rank every") {
            json!({"ranked_ids":(1..=10).rev().map(|i|format!("M{i:02}")).collect::<Vec<_>>(),"chronological_ids":[]})
        } else {
            json!({"relationship":"compatible","temporal_order":"unknown","update_type":"none"})
        };
        json!({"kind":"text","text":serde_json::to_string(&value).unwrap(),"finish_reason":"stop"})
    };
    axum::Json(json!({"contract_version":1,"request_id":request["request_id"],"result":result}))
}
async fn mock() -> (Pumas, Mock, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let state = Mock {
        endpoint: endpoint.clone(),
        prompts: Default::default(),
    };
    let app = axum::Router::new()
        .route("/.well-known/pumas", axum::routing::get(mock_discovery))
        .route("/v1/capabilities", axum::routing::get(mock_capabilities))
        .route("/v1/model-operations", axum::routing::post(mock_operation))
        .with_state(state.clone());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let generation = phased_memory::pumas::Selection {
        model: "test-only".into(),
        profile: "p".into(),
        capability: "text_generation".into(),
    };
    let embedding = phased_memory::pumas::Selection {
        capability: "text_embedding".into(),
        ..generation.clone()
    };
    let pumas = Pumas::new(Config {
        endpoint,
        extraction: Some(generation.clone()),
        embedding: Some(embedding),
        decision: Some(generation),
        decision_adapter: Some("original_json_v1".into()),
    })
    .unwrap();
    (pumas, state, task)
}
#[tokio::test]
async fn original_pair_http_prompt_has_context_and_no_scene_gold_or_case_identity() {
    let (pumas, state, task) = mock().await;
    let case = pairs().remove(1);
    let predictions = run_pairs(&pumas, &[case], false).await.unwrap();
    assert_eq!(predictions[0].status, "completed");
    let prompts = state.prompts.lock().unwrap();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("exactly one brass key"));
    for forbidden in [
        "C02",
        "same_time_contradiction",
        "The brass key is in the wooden box.",
        "rationale",
        "gold",
        "extraction_checks",
        "comparison_targets",
    ] {
        assert!(!prompts[0].contains(forbidden), "{forbidden}");
    }
    task.abort();
}
#[tokio::test]
async fn original_reranker_receives_all_ten_even_when_query_plan_excludes_everyone() {
    let (pumas, state, task) = mock().await;
    let memories: Vec<Memory> = read_jsonl(pack().join("retrieval_memories.jsonl")).unwrap();
    let queries: Vec<Query> = read_jsonl(pack().join("retrieval_queries.jsonl")).unwrap();
    let mut store = Store::open(":memory:").unwrap();
    let records = index_memories(&mut store, &pumas, &memories, false)
        .await
        .unwrap();
    assert!(records.iter().all(|r| r.status == "stored"));
    let predictions = run_retrieval(&store, &pumas, &memories, &queries[..1], "reranked")
        .await
        .unwrap();
    let prediction = &predictions[0];
    assert_eq!(prediction.status, "completed");
    assert_eq!(prediction.trace.eligible_candidates, 0);
    assert_eq!(prediction.trace.rerank_candidates, 10);
    assert_eq!(
        prediction.ranked_ids,
        vec!["M10", "M09", "M08", "M07", "M06"]
    );
    assert!(prediction.trace.metadata_source.contains("human_reference"));
    let prompts = state.prompts.lock().unwrap();
    let rank = prompts
        .iter()
        .find(|p| p.starts_with("Rank every"))
        .unwrap();
    for memory in memories {
        assert!(rank.contains(&memory.id));
    }
    for forbidden in [
        "R01",
        "relevant_memory_ids",
        "hard_negative_memory_ids",
        "rationale",
    ] {
        assert!(!rank.contains(forbidden), "{forbidden}");
    }
    task.abort();
}
