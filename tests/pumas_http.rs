//! Test-only public-contract simulator. These assertions do not qualify models.
use axum::{
    body::Bytes,
    extract::{Request, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use phased_memory::pumas::{Config, Pumas, Selection};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
#[derive(Clone)]
struct Fixture {
    endpoint: String,
    posts: Arc<AtomicUsize>,
    mode: usize,
    generation: Arc<AtomicUsize>,
}
async fn discovery(State(s): State<Fixture>) -> impl IntoResponse {
    Json(
        json!({"advertisement_schema_version":1,"endpoint":s.endpoint,"service_generation":format!("service-{}",s.generation.load(Ordering::SeqCst)),"instance":{"generation":"instance-1"},"build_info":{"component":"test-only","package_version":"test","schemas":[{"name":"pumas.http-admission-fence","version":1}]}}),
    )
}
async fn capabilities(State(s): State<Fixture>, req: Request) -> impl IntoResponse {
    assert_eq!(req.headers()["pumas-instance-generation"], "instance-1");
    assert_eq!(req.headers()["pumas-service-generation"], "service-0");
    assert!(req.uri().query().unwrap().contains("model=exact-model"));
    assert!(req.uri().query().unwrap().contains("profile=exact-profile"));
    Json(
        json!({"supported_contract_versions":[1],"model":"exact-model","profile":"exact-profile","max_request_bytes":65536,"max_response_bytes":65536,"capabilities":[{"capability":"text_embedding","semantic_task":"text_embedding","input_formats":["text"],"output_formats":["embeddings_float32"],"availability":if s.mode==1 {json!({"state":"unavailable","reason":"runtime_unavailable"})}else{json!({"state":"available"})},"option_bounds":[{"option":"input_count","minimum":1,"maximum":1},{"option":"input_characters","minimum":1,"maximum":1000}]}]}),
    )
}
async fn operation(State(s): State<Fixture>, req: Request) -> impl IntoResponse {
    assert_eq!(req.headers()["pumas-instance-generation"], "instance-1");
    assert_eq!(req.headers()["pumas-service-generation"], "service-0");
    s.posts.fetch_add(1, Ordering::SeqCst);
    let bytes: Bytes = axum::body::to_bytes(req.into_body(), 65536).await.unwrap();
    let request: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(request["model"], "exact-model");
    assert_eq!(request["profile"], "exact-profile");
    assert_eq!(request["capability"], "text_embedding");
    assert_eq!(request["stream"], false);
    match s.mode {
        2 => (
            StatusCode::BAD_GATEWAY,
            Json(
                json!({"contract_version":1,"request_id":request["request_id"],"error":{"code":"provider_failure","outcome":"unknown"}}),
            ),
        ),
        3 => (
            StatusCode::OK,
            Json(
                json!({"contract_version":1,"request_id":"wrong-id","result":{"kind":"embeddings","vectors":[[1.0,0.0]]}}),
            ),
        ),
        4 => (
            StatusCode::OK,
            Json(
                json!({"contract_version":1,"request_id":request["request_id"],"result":{"kind":"embeddings","vectors":[[1.0],[2.0]]}}),
            ),
        ),
        5 => (
            StatusCode::TEMPORARY_REDIRECT,
            Json(json!({"redirect":"must not follow"})),
        ),
        _ => (
            StatusCode::OK,
            Json(
                json!({"contract_version":1,"request_id":request["request_id"],"result":{"kind":"embeddings","vectors":[[1.0,0.0]]}}),
            ),
        ),
    }
}
async fn start(mode: usize) -> (Pumas, Fixture, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let fixture = Fixture {
        endpoint: endpoint.clone(),
        posts: Arc::new(AtomicUsize::new(0)),
        mode,
        generation: Arc::new(AtomicUsize::new(0)),
    };
    let app = Router::new()
        .route("/.well-known/pumas", get(discovery))
        .route("/v1/capabilities", get(capabilities))
        .route("/v1/model-operations", post(operation))
        .with_state(fixture.clone());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let p = Pumas::new(Config {
        endpoint,
        extraction: None,
        embedding: Some(Selection {
            model: "exact-model".into(),
            profile: "exact-profile".into(),
            capability: "text_embedding".into(),
        }),
        decision: None,
        decision_adapter: None,
    })
    .unwrap();
    (p, fixture, task)
}
#[tokio::test]
async fn actual_http_fences_exact_identity_and_typed_embedding() {
    let (p, f, t) = start(0).await;
    let e = p.embed("semantic input").await.unwrap();
    assert_eq!(e.vector, vec![1.0, 0.0]);
    assert!(e.space.ends_with("instance-1:service-0"));
    assert_eq!(f.posts.load(Ordering::SeqCst), 1);
    t.abort();
}
#[tokio::test]
async fn unavailable_capability_prevents_inference_dispatch() {
    let (p, f, t) = start(1).await;
    let e = p.embed("semantic input").await.unwrap_err().to_string();
    assert!(e.contains("unavailable: selected capability"), "{e}");
    assert_eq!(f.posts.load(Ordering::SeqCst), 0);
    t.abort();
}
#[tokio::test]
async fn unknown_failure_correlation_count_and_redirect_never_retry() {
    for (mode, diagnostic) in [
        (2, "unknown: Pumas operation provider_failure"),
        (3, "correlation/version mismatch"),
        (4, "embedding count"),
        (5, "correlation/version mismatch"),
    ] {
        let (p, f, t) = start(mode).await;
        let e = p.embed("semantic input").await.unwrap_err().to_string();
        assert!(e.contains(diagnostic), "{mode}: {e}");
        assert_eq!(f.posts.load(Ordering::SeqCst), 1);
        t.abort();
    }
}
#[tokio::test]
async fn changed_generation_never_silently_rebinds() {
    let (p, f, t) = start(0).await;
    p.embed("first").await.unwrap();
    f.generation.store(1, Ordering::SeqCst);
    let e = p.embed("second").await.unwrap_err().to_string();
    assert!(e.contains("stale runtime generation"), "{e}");
    assert_eq!(f.posts.load(Ordering::SeqCst), 1);
    t.abort();
}
