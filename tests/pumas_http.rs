//! Test-only public-contract simulator. These assertions do not qualify models.
//!
//! Responses mirror the Pumas v0.8 typed HTTP shapes, including fields,
//! descriptors and enum values this consumer does not use.
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use phased_memory::{
    domain::*,
    pumas::{Config, Pumas, Selection, TextRole},
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
#[derive(Clone)]
struct Fixture {
    endpoint: String,
    posts: Arc<AtomicUsize>,
    mode: usize,
    generation: Arc<AtomicUsize>,
    models: Value,
    requests: Arc<Mutex<Vec<Value>>>,
}
/// A real-server-shaped descriptor list: every capability is listed, most are
/// unavailable with a reason, and one future capability/format/option is added.
fn descriptors(available: &[&str], unavailable_reason: &str) -> Value {
    let state = |c: &str| {
        if available.contains(&c) {
            json!({"state":"available"})
        } else {
            json!({"state":"unavailable","reason":unavailable_reason})
        }
    };
    let text_bounds = json!([{"option":"max_tokens","minimum":1.0,"maximum":8192.0},{"option":"temperature","minimum":0.0,"maximum":2.0},{"option":"top_p","minimum":0.0,"maximum":1.0}]);
    let image_bounds = json!([{"option":"image_bytes","minimum":1.0,"maximum":8388608.0},{"option":"image_pixels","minimum":1.0,"maximum":4194304.0},{"option":"image_count","minimum":0.0,"maximum":4.0}]);
    json!([
        {"capability":"chat_generation","semantic_task":"chat_generation","input_formats":["messages_text"],"output_formats":["text"],"streaming":true,"availability":state("chat_generation"),"option_bounds":text_bounds},
        {"capability":"text_generation","semantic_task":"text_generation","input_formats":["text"],"output_formats":["text"],"streaming":true,"availability":state("text_generation"),"option_bounds":text_bounds},
        {"capability":"text_embedding","semantic_task":"text_embedding","input_formats":["text","text_batch"],"output_formats":["embeddings_float32"],"streaming":false,"availability":state("text_embedding"),"option_bounds":[{"option":"dimensions","minimum":1.0,"maximum":4294967295.0},{"option":"input_count","minimum":1.0,"maximum":1.0},{"option":"input_characters","minimum":1.0,"maximum":1000.0}]},
        {"capability":"image_generation","semantic_task":"text_to_image","input_formats":["text"],"output_formats":["png_base64"],"streaming":false,"availability":state("image_generation"),"option_bounds":[]},
        {"capability":"audio_transcription","semantic_task":"speech_to_text","input_formats":["pcm_s16le","pcm_f32le"],"output_formats":["text"],"streaming":false,"availability":{"state":"unavailable","reason":"unqualified_audio_runtime"},"option_bounds":[{"option":"max_output_tokens","minimum":512.0,"maximum":512.0}]},
        {"capability":"audio_classification","semantic_task":"audio_classification","input_formats":["pcm_s16le"],"output_formats":["labels"],"streaming":false,"availability":{"state":"unavailable","reason":"unsupported_adapter"},"option_bounds":[]},
        {"capability":"image_to_text","semantic_task":"image_to_text","input_formats":["png_base64","jpeg_base64","messages_image"],"output_formats":["text"],"streaming":false,"availability":state("image_to_text"),"option_bounds":image_bounds},
        {"capability":"multimodal_embedding","semantic_task":"multimodal_embedding","input_formats":["png_base64","jpeg_base64","content_batch"],"output_formats":["embeddings_float32"],"streaming":false,"availability":state("multimodal_embedding"),"option_bounds":image_bounds},
        {"capability":"decision","semantic_task":"decision","input_formats":["decision"],"output_formats":["decisions"],"streaming":false,"availability":state("decision"),"option_bounds":image_bounds},
        {"capability":"future_capability","semantic_task":"future_task","input_formats":["future_format"],"output_formats":["future_output"],"streaming":false,"availability":{"state":"future_state","reason":"future_reason"},"option_bounds":[{"option":"future_option","minimum":0.0,"maximum":1.0}],"future_field":{}}
    ])
}
fn entry(model: &str, profile: &str, required: bool, available: &[&str]) -> Value {
    json!({"model":model,"profile":profile,"model_id":format!("org/{model}"),"provider":"llama_cpp","load_state":"loaded","context_tokens":8192,"profile_required":required,"capabilities":descriptors(available, "model_task_mismatch"),"future_field":"ignored"})
}
fn default_models(embedding_reason: Option<&str>) -> Value {
    let embedding = match embedding_reason {
        None => entry("embeddinggemma-2", "default", false, &["text_embedding"]),
        Some(reason) => {
            json!({"model":"embeddinggemma-2","profile":"default","model_id":"org/e","provider":"llama_cpp","load_state":"loading","profile_required":false,"capabilities":descriptors(&[], reason)})
        }
    };
    json!([
        embedding,
        entry("chat", "gpu", true, &["chat_generation", "text_generation"]),
        entry("chat", "cpu", true, &["chat_generation", "text_generation"]),
        entry("kev-4b", "default", false, &["decision"]),
        // A malformed entry for another alias must not break selection.
        {"model":"other","unexpected":true}
    ])
}
fn fenced(req: &Request) {
    assert_eq!(req.headers()["pumas-instance-generation"], "instance-1");
    assert_eq!(req.headers()["pumas-service-generation"], "service-0");
}
async fn discovery(State(s): State<Fixture>) -> impl IntoResponse {
    Json(
        json!({"advertisement_schema_version":1,"endpoint":s.endpoint,"service_generation":format!("service-{}",s.generation.load(Ordering::SeqCst)),"instance":{"generation":"instance-1"},"build_info":{"component":"test-only","package_version":"test","schemas":[{"name":"pumas.http-admission-fence","version":1},{"name":"pumas.future-schema","version":3}]}}),
    )
}
async fn served_models(State(s): State<Fixture>, req: Request) -> Response {
    fenced(&req);
    if s.mode == 20 {
        return StatusCode::PRECONDITION_FAILED.into_response();
    }
    Json(json!({"supported_contract_versions":[1],"max_request_bytes":33554432,"max_response_bytes":33554432,"max_stream_event_bytes":262144,"models":s.models})).into_response()
}
fn decisions(request: &Value) -> Value {
    let questions = request["input"]["questions"].as_object().unwrap();
    let mut answers = serde_json::Map::new();
    for (id, q) in questions {
        let options: Vec<&String> = q["criteria"].as_object().unwrap().keys().collect();
        let chosen = match id.as_str() {
            "relationship" => "contradicts",
            "temporal_order" => "b_later",
            _ => "state_change",
        };
        let rest = 0.2 / (options.len() - 1) as f64;
        let probabilities: serde_json::Map<String, Value> = options
            .iter()
            .map(|o| ((*o).clone(), json!(if *o == chosen { 0.8 } else { rest })))
            .collect();
        answers.insert(id.clone(), json!({"type":"choice","choice":chosen,"probabilities":probabilities,"confidence":0.8,"future_field":1}));
    }
    json!({"kind":"decisions","answers":answers,"input_tokens":412})
}
async fn operation(State(s): State<Fixture>, req: Request) -> Response {
    fenced(&req);
    s.posts.fetch_add(1, Ordering::SeqCst);
    let bytes: Bytes = axum::body::to_bytes(req.into_body(), 1 << 20)
        .await
        .unwrap();
    let request: Value = serde_json::from_slice(&bytes).unwrap();
    s.requests.lock().unwrap().push(request.clone());
    assert_eq!(request["contract_version"], 1);
    assert_eq!(request["stream"], false);
    let id = request["request_id"].clone();
    let reply = |status: StatusCode, v: Value| (status, Json(v)).into_response();
    match s.mode {
        2 => reply(
            StatusCode::BAD_GATEWAY,
            json!({"contract_version":1,"request_id":id,"error":{"code":"provider_failure","outcome":"unknown","message":"llama.cpp reported: slot crashed"}}),
        ),
        3 => reply(
            StatusCode::OK,
            json!({"contract_version":1,"request_id":"wrong-id","result":{"kind":"embeddings","vectors":[[1.0,0.0]]}}),
        ),
        4 => reply(
            StatusCode::OK,
            json!({"contract_version":1,"request_id":id,"result":{"kind":"embeddings","vectors":[[1.0],[2.0]]}}),
        ),
        5 => reply(
            StatusCode::TEMPORARY_REDIRECT,
            json!({"redirect":"must not follow"}),
        ),
        // Unreadable body as Pumas sees it: null request_id, not admitted.
        6 => reply(
            StatusCode::BAD_REQUEST,
            json!({"contract_version":1,"request_id":null,"error":{"code":"invalid_request","outcome":"not_admitted","message":"request body is not valid JSON: EOF while parsing"}}),
        ),
        7 => reply(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"contract_version":1,"request_id":id,"error":{"code":"capability_unavailable","outcome":"not_admitted","message":"runtime_unavailable: the serving runtime is not ready for this model"}}),
        ),
        // Generation fence refusal: empty body before admission.
        8 => Response::builder()
            .status(StatusCode::PRECONDITION_FAILED)
            .body(Body::empty())
            .unwrap(),
        9 => reply(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"contract_version":1,"request_id":id,"error":{"code":"future_code","outcome":"not_admitted","message":"a reason from a newer Pumas","detail":{}}}),
        ),
        _ => {
            let result = match request["capability"].as_str().unwrap() {
                "text_embedding" => {
                    json!({"kind":"embeddings","vectors":[[1.0,0.0]],"usage":{"tokens":3}})
                }
                "decision" => decisions(&request),
                _ => {
                    json!({"kind":"text","text":"{\"relationship\":\"supports\",\"temporal_order\":\"same_time\",\"update_type\":\"none\"}","finish_reason":"stop"})
                }
            };
            reply(
                StatusCode::OK,
                json!({"contract_version":1,"request_id":id,"result":result,"timing":{"ms":1}}),
            )
        }
    }
}
async fn start_with(mode: usize, models: Value, config: Config) -> (Pumas, Fixture) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let fixture = Fixture {
        endpoint: endpoint.clone(),
        posts: Arc::new(AtomicUsize::new(0)),
        mode,
        generation: Arc::new(AtomicUsize::new(0)),
        models,
        requests: Default::default(),
    };
    let app = Router::new()
        .route("/.well-known/pumas", get(discovery))
        .route("/v1/served-models", get(served_models))
        .route("/v1/model-operations", post(operation))
        .with_state(fixture.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let p = Pumas::new(Config { endpoint, ..config }).unwrap();
    (p, fixture)
}
fn gemma() -> Selection {
    Selection {
        model: "embeddinggemma-2".into(),
        query_prefix: "task: search result | query: ".into(),
        document_prefix: "title: none | text: ".into(),
        ..Default::default()
    }
}
fn config(
    embedding: Option<Selection>,
    decision: Option<Selection>,
    adapter: Option<&str>,
) -> Config {
    Config {
        endpoint: String::new(),
        extraction: None,
        embedding,
        decision,
        decision_adapter: adapter.map(Into::into),
    }
}
async fn start(mode: usize) -> (Pumas, Fixture) {
    start_with(
        mode,
        default_models(None),
        config(Some(gemma()), None, None),
    )
    .await
}
fn claim(id: &str, object: &str, time: i64) -> Claim {
    Claim {
        id: id.into(),
        normalized: format!("Alice lives in {object}"),
        model_local_id: None,
        state_key: Some("residence".into()),
        triple: Triple {
            subject: "alice".into(),
            predicate: "lives_in".into(),
            object: object.into(),
            polarity: Polarity::Affirmed,
        },
        entities: vec!["alice".into()],
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
#[tokio::test]
async fn discovered_profile_prefixed_text_and_shared_space() {
    let (p, f) = start(0).await;
    let doc = p
        .embed("Alice lives in Paris", TextRole::Document)
        .await
        .unwrap();
    let query = p.embed("where is Alice", TextRole::Query).await.unwrap();
    assert_eq!(doc.vector, vec![1.0, 0.0]);
    assert_eq!(
        doc.space, query.space,
        "queries and documents share one space"
    );
    assert!(doc
        .space
        .contains(":embeddinggemma-2:default:instance-1:service-0:"));
    assert!(doc.space.ends_with(
        r#"text_prefixes:{"query":"task: search result | query: ","document":"title: none | text: "}"#
    ));
    match &doc.provenance {
        Provenance::Pumas { profile, .. } => assert_eq!(profile, "default"),
        other => panic!("{other:?}"),
    }
    let requests = f.requests.lock().unwrap();
    assert_eq!(
        requests[0]["input"]["text"],
        "title: none | text: Alice lives in Paris"
    );
    assert_eq!(
        requests[1]["input"]["text"],
        "task: search result | query: where is Alice"
    );
    for r in requests.iter() {
        assert_eq!(r["model"], "embeddinggemma-2");
        assert!(r.get("profile").is_none(), "profile only when required");
        assert_eq!(r["capability"], "text_embedding");
        assert_eq!(r["output"], "embeddings_float32");
    }
    assert_eq!(f.posts.load(Ordering::SeqCst), 2);
}
#[tokio::test]
async fn prefix_choice_is_part_of_space_identity() {
    let (plain, _) = start_with(
        0,
        default_models(None),
        config(
            Some(Selection {
                model: "embeddinggemma-2".into(),
                ..Default::default()
            }),
            None,
            None,
        ),
    )
    .await;
    let (prefixed, _) = start(0).await;
    let a = plain.embed("x", TextRole::Document).await.unwrap();
    let b = prefixed.embed("x", TextRole::Document).await.unwrap();
    let strip = |s: &str| s.split(":text_prefixes:").nth(1).unwrap().to_owned();
    assert_ne!(strip(&a.space), strip(&b.space));
    assert!(a
        .space
        .ends_with(r#"text_prefixes:{"query":"","document":""}"#));
}
#[tokio::test]
async fn profile_is_sent_only_when_required_or_pinned() {
    let chat = |profile: Option<&str>| Selection {
        model: "chat".into(),
        profile: profile.map(Into::into),
        ..Default::default()
    };
    let (p, f) = start_with(
        0,
        default_models(None),
        config(None, Some(chat(None)), Some("decision_json_v1")),
    )
    .await;
    let e = p
        .decide(&claim("a", "Paris", 1), &claim("b", "Rome", 2))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("served by several profiles [gpu, cpu]"), "{e}");
    assert_eq!(f.posts.load(Ordering::SeqCst), 0);

    let (p, f) = start_with(
        0,
        default_models(None),
        config(None, Some(chat(Some("cpu"))), Some("decision_json_v1")),
    )
    .await;
    let o = p
        .decide(&claim("a", "Paris", 1), &claim("b", "Rome", 2))
        .await
        .unwrap();
    assert_eq!(o.value.labels.relationship, Relationship::Supports);
    assert_eq!(o.value.adapter, "decision_json_v1");
    assert!(o.value.confidence.is_none() && o.value.native_answers.is_none());
    let r = f.requests.lock().unwrap()[0].clone();
    assert_eq!(r["profile"], "cpu");
    assert_eq!(r["capability"], "chat_generation");
    assert_eq!(r["input"]["kind"], "messages");

    // Pinning a profile also sends it when the alias has only one.
    let (p, f) = start_with(
        0,
        default_models(None),
        config(
            Some(Selection {
                profile: Some("default".into()),
                ..gemma()
            }),
            None,
            None,
        ),
    )
    .await;
    p.embed("x", TextRole::Query).await.unwrap();
    assert_eq!(f.requests.lock().unwrap()[0]["profile"], "default");
}
#[tokio::test]
async fn unserved_alias_and_unavailable_capability_prevent_dispatch() {
    let (p, f) = start_with(
        0,
        default_models(Some("runtime_unavailable")),
        config(Some(gemma()), None, None),
    )
    .await;
    let e = p
        .embed("semantic input", TextRole::Query)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("unavailable: selected capability text_embedding: runtime_unavailable"),
        "{e}"
    );
    let (p, g) = start_with(
        0,
        default_models(None),
        config(
            Some(Selection {
                model: "missing".into(),
                ..Default::default()
            }),
            None,
            None,
        ),
    )
    .await;
    let e = p
        .embed("semantic input", TextRole::Query)
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not serve model missing"), "{e}");
    assert!(e.contains("kev-4b/default"), "{e}");
    assert_eq!(
        f.posts.load(Ordering::SeqCst) + g.posts.load(Ordering::SeqCst),
        0
    );
}
#[tokio::test]
async fn unknown_failure_correlation_count_and_redirect_never_retry() {
    for (mode, diagnostic) in [
        (
            2,
            "unknown: Pumas operation provider_failure (HTTP 502): llama.cpp reported: slot crashed; never replay",
        ),
        (3, "correlation/version mismatch"),
        (4, "embedding count"),
        (5, "correlation/version mismatch"),
    ] {
        let (p, f) = start(mode).await;
        let e = p
            .embed("semantic input", TextRole::Query)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains(diagnostic), "{mode}: {e}");
        assert!(e.starts_with("unknown:"), "{mode}: {e}");
        assert_eq!(f.posts.load(Ordering::SeqCst), 1);
    }
}
#[tokio::test]
async fn not_admitted_rejections_surface_message_and_are_not_uncertain() {
    for (mode, diagnostic) in [
        (
            6,
            "not_admitted: Pumas operation invalid_request (HTTP 400): request body is not valid JSON: EOF while parsing",
        ),
        (
            7,
            "not_admitted: Pumas operation capability_unavailable (HTTP 503): runtime_unavailable: the serving runtime is not ready for this model",
        ),
        (8, "not_admitted: stale runtime generation (HTTP 412)"),
        (
            9,
            "not_admitted: Pumas operation future_code (HTTP 503): a reason from a newer Pumas",
        ),
    ] {
        let (p, f) = start(mode).await;
        let e = p
            .embed("semantic input", TextRole::Query)
            .await
            .unwrap_err()
            .to_string();
        assert!(e.starts_with(diagnostic), "{mode}: {e}");
        assert!(!e.contains("never replay"), "{mode}: {e}");
        assert_eq!(f.posts.load(Ordering::SeqCst), 1, "no retry");
    }
}
#[tokio::test]
async fn fenced_served_models_refusal_is_not_admitted() {
    let (p, f) = start(20).await;
    let e = p
        .embed("semantic input", TextRole::Query)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        e.starts_with("not_admitted: stale runtime generation"),
        "{e}"
    );
    assert_eq!(f.posts.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn changed_generation_never_silently_rebinds() {
    let (p, f) = start(0).await;
    p.embed("first", TextRole::Document).await.unwrap();
    f.generation.store(1, Ordering::SeqCst);
    let e = p
        .embed("second", TextRole::Query)
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("stale runtime generation"), "{e}");
    assert_eq!(f.posts.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn native_decision_maps_labels_and_uncalibrated_probabilities() {
    let kev = Selection {
        model: "kev-4b".into(),
        ..Default::default()
    };
    let (p, f) = start_with(
        0,
        default_models(None),
        config(None, Some(kev), Some("decision_native_v1")),
    )
    .await;
    let o = p
        .decide(&claim("a", "Paris", 1), &claim("b", "Rome", 2))
        .await
        .unwrap();
    assert_eq!(
        o.value.labels,
        Labels {
            relationship: Relationship::Contradicts,
            temporal_order: TemporalOrder::BLater,
            update_type: UpdateType::StateChange,
        }
    );
    assert_eq!(o.value.adapter, "decision_native_v1");
    assert_eq!(o.value.confidence, Some(0.8));
    let probabilities = o.value.relationship_probabilities().unwrap();
    assert_eq!(probabilities.len(), 4);
    assert!((probabilities.values().sum::<f64>() - 1.0).abs() < 1e-9);
    let requests = f.requests.lock().unwrap();
    let r = &requests[0];
    assert_eq!(r["capability"], "decision");
    assert_eq!(r["output"], "decisions");
    assert_eq!(r["options"], json!({"kind":"decision"}));
    assert!(r.get("profile").is_none());
    assert_eq!(r["input"]["kind"], "decision");
    assert_eq!(r["input"]["state"]["a"]["id"], "a");
    assert_eq!(r["input"]["state"]["b"]["id"], "b");
    let questions = r["input"]["questions"].as_object().unwrap();
    assert_eq!(
        questions.keys().collect::<Vec<_>>(),
        ["relationship", "temporal_order", "update_type"]
    );
    for q in questions.values() {
        assert_eq!(q["type"], "choice");
        assert!(q["instructions"].is_string());
    }
    let criteria: Vec<&String> = questions["relationship"]["criteria"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    assert_eq!(
        criteria,
        [
            "compatible",
            "contradicts",
            "insufficient_information",
            "supports"
        ]
    );
}
#[tokio::test]
async fn decision_adapter_must_match_what_the_model_advertises() {
    let sel = |model: &str| Selection {
        model: model.into(),
        profile: (model == "chat").then(|| "gpu".into()),
        ..Default::default()
    };
    for (model, adapter, diagnostic) in [
        (
            "kev-4b",
            "decision_json_v1",
            "advertises native decision; qualify decision_native_v1",
        ),
        (
            "chat",
            "decision_native_v1",
            "does not advertise an available decision capability (model_task_mismatch); no fallback",
        ),
    ] {
        let (p, f) = start_with(
            0,
            default_models(None),
            config(None, Some(sel(model)), Some(adapter)),
        )
        .await;
        let e = p
            .decide(&claim("a", "Paris", 1), &claim("b", "Rome", 2))
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains(diagnostic), "{model}: {e}");
        assert_eq!(f.posts.load(Ordering::SeqCst), 0, "no adapter fallback");
    }
    // Without qualification nothing is observed or dispatched.
    let (p, f) = start_with(
        0,
        default_models(None),
        config(None, Some(sel("kev-4b")), None),
    )
    .await;
    let e = p
        .decide(&claim("a", "Paris", 1), &claim("b", "Rome", 2))
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("operator-qualified"), "{e}");
    assert_eq!(f.posts.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn probe_reports_discovered_identity_without_inference() {
    let kev = Selection {
        model: "kev-4b".into(),
        ..Default::default()
    };
    let (p, f) = start_with(
        0,
        default_models(None),
        config(Some(gemma()), Some(kev), Some("decision_native_v1")),
    )
    .await;
    let (fence, _) = p.discover().await.unwrap();
    let e = p
        .observe_role(phased_memory::pumas::Role::Embedding, &fence)
        .await
        .unwrap();
    assert_eq!(e["profile"], "default");
    assert_eq!(e["sends_profile"], false);
    assert_eq!(e["capability"], "text_embedding");
    assert_eq!(e["context_tokens"], 8192);
    let d = p
        .observe_role(phased_memory::pumas::Role::Decision, &fence)
        .await
        .unwrap();
    assert_eq!(d["capability"], "decision");
    assert_eq!(f.posts.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn native_pair_runs_retain_probabilities_for_brier_scoring() {
    let kev = Selection {
        model: "kev-4b".into(),
        ..Default::default()
    };
    let (p, _) = start_with(
        0,
        default_models(None),
        config(None, Some(kev), Some("decision_native_v1")),
    )
    .await;
    let cases = vec![phased_memory::eval::PairCase {
        id: "pair-1".into(),
        a: claim("a", "Paris", 1),
        b: claim("b", "Rome", 2),
        expected: Labels {
            relationship: Relationship::Contradicts,
            temporal_order: TemporalOrder::BLater,
            update_type: UpdateType::StateChange,
        },
    }];
    let predictions = phased_memory::eval::run_pairs(&p, &cases).await.unwrap();
    let native = predictions[0].native_probabilities.as_ref().unwrap();
    assert!(native.native_evidence.contains("not calibrated"));
    let score = phased_memory::eval::score_pairs(&cases, &predictions).unwrap();
    assert_eq!(score.joint_accuracy, 1.0);
    // (0.8-1)^2 + 3 * (0.2/3)^2
    assert!((score.brier.unwrap() - (0.04 + 3.0 * (0.2f64 / 3.0).powi(2))).abs() < 1e-9);
}
