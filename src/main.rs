use anyhow::{bail, ensure, Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use phased_memory::{
    domain::*,
    eval::*,
    pumas::{Config, Pumas},
    store::{Comparison, DecisionEvidence, Filters, Store},
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Arc, Mutex},
};
use tokio::sync::Semaphore;

fn read<T: DeserializeOwned>(p: impl AsRef<Path>) -> Result<T> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(p)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "invalid: input file exceeds 16 MiB"
    );
    Ok(serde_json::from_slice(&bytes)?)
}
fn write(p: &str, v: &impl Serialize) -> Result<()> {
    let mut out = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(p)
        .context("output must be a new path; existing evidence is never overwritten")?;
    serde_json::to_writer_pretty(&mut out, v)?;
    Ok(())
}
fn print(v: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}
fn runtime(path: &str) -> Result<Pumas> {
    Pumas::new(read::<Config>(path)?)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("serve") if args.len() == 5 => serve(&args[2], &args[3], args[4].parse()?).await,
        Some("probe") if args.len() == 3 => print(&observe_runtime(&runtime(&args[2])?).await?),
        Some("ingest") if args.len() == 4 || args.len() == 5 => {
            let mut batch: Ingest = read(&args[3])?;
            if let Some(config) = args.get(4) { embed_batch(&runtime(config)?, &mut batch).await?; }
            Store::open(&args[2])?.ingest(&batch)?;
            print(&json!({"status":"stored", "claims":batch.claims.len(), "inference":"inspect provenance; human/synthetic are not model qualification"}))
        }
        Some("stage-a") if args.len() == 5 => {
            ensure_new_output(&args[4])?;
            let cases: Vec<PairCase> = read(&args[3])?;
            let predictions = run_pairs(&runtime(&args[2])?, &cases).await?;
            write(&args[4], &predictions)?;
            let score = score_pairs(&cases, &predictions)?;
            print(&score)?;
            ensure!(score.completed == score.total, "unavailable: incomplete Stage A; output retained");
            Ok(())
        }
        Some("stage-b") if args.len() == 5 => {
            ensure_new_output(&args[4])?;
            let cases: Vec<RawPairCase> = read(&args[3])?;
            let records = run_extraction(&runtime(&args[2])?, &cases).await?;
            write(&args[4], &records)?;
            let score = score_extraction(&cases, &records)?;
            print(&score)?;
            ensure!(score.pair_score.completed == score.total, "unavailable: incomplete Stage B; output retained");
            Ok(())
        }
        Some("stage-c") if args.len() == 7 => stage_c(&args[2], &args[3], &args[4], &args[5], &args[6]).await,
        Some("original-stage-a") | Some("original-stage-b") if args.len()==5 => {
            ensure_new_output(&args[4])?;
            let cases:Vec<phased_memory::original::PairCase>=phased_memory::original::read_jsonl(&args[3])?;
            let predictions=phased_memory::original::run_pairs(&runtime(&args[2])?,&cases,args[1]=="original-stage-b").await?;
            phased_memory::original::write_jsonl(&args[4],&predictions)?;
            let complete=predictions.iter().filter(|p|p.status=="completed").count();
            print(&json!({"expected_cases":cases.len(),"completed":complete,"scorer":"python3 fixtures/original/score.py pairs OUTPUT.jsonl","qualification":"runner completion is not a model-quality claim; inspect provenance and scorer"}))?;
            ensure!(complete==cases.len(),"unavailable: incomplete original pair run; all case outputs retained");Ok(())
        }
        Some("original-index") if args.len()==7 => {
            ensure_new_output(&args[5])?;
            ensure!(["human-reference","extract"].contains(&args[6].as_str()),"invalid: index metadata mode");
            let memories:Vec<phased_memory::original::Memory>=phased_memory::original::read_jsonl(&args[4])?;
            let records=phased_memory::original::index_memories(&mut Store::open(&args[2])?,&runtime(&args[3])?,&memories,args[6]=="extract").await?;
            phased_memory::original::write_jsonl(&args[5],&records)?;
            let complete=records.iter().filter(|r|r.status=="stored").count();
            print(&json!({"expected_memories":10,"stored":complete,"metadata_mode":args[6]}))?;
            ensure!(complete==10,"unavailable: incomplete original index; per-memory failures retained");Ok(())
        }
        Some("original-retrieval") if args.len()==8 => {
            ensure_new_output(&args[6])?;
            let memories:Vec<phased_memory::original::Memory>=phased_memory::original::read_jsonl(&args[4])?;
            let queries:Vec<phased_memory::original::Query>=phased_memory::original::read_jsonl(&args[5])?;
            let predictions=phased_memory::original::run_retrieval(&Store::open(&args[2])?,&runtime(&args[3])?,&memories,&queries,&args[7]).await?;
            phased_memory::original::write_jsonl(&args[6],&predictions)?;
            let complete=predictions.iter().filter(|p|p.status!="invalid_output").count();
            print(&json!({"expected_queries":queries.len(),"completed":complete,"mode":args[7],"scorer":"python3 fixtures/original/score.py retrieval OUTPUT.jsonl"}))?;
            ensure!(complete==queries.len(),"unavailable: incomplete original retrieval run; all query outputs retained");Ok(())
        }
        Some("phase-control") if args.len()==4 => {
            ensure_new_output(&args[3])?;
            let input:phased_memory::phase::ExperimentInput=read(&args[2])?;
            let output=phased_memory::phase::control(&input)?;
            write(&args[3],&output)?;
            print(&json!({"protocol":output.protocol,"candidate_count":output.frozen.candidate_count,"queries":output.numeric_queries.len(),"equivalence_passed":output.equivalence_passed,"maximum_absolute_error":output.maximum_absolute_error,"pumas_embeddings":"unrun"}))
        }
        Some("phase-compare") if args.len()==5 => {
            ensure_new_output(&args[4])?;
            let input:phased_memory::phase::ExperimentInput=read(&args[3])?;
            let output=phased_memory::phase::compare(&runtime(&args[2])?,&input).await?;
            write(&args[4],&output)?;
            let arm=output.pumas_embeddings.as_ref().context("invalid: missing embedding arm")?;
            print(&json!({"protocol":output.protocol,"candidate_count":output.frozen.candidate_count,"equivalence_passed":output.equivalence_passed,"pumas_embeddings":arm.status}))?;
            ensure!(arm.status=="completed","unavailable: Pumas comparison incomplete; full evidence retained");Ok(())
        }
        Some("score-pairs") if args.len() == 4 => print(&score_pairs(&read::<Vec<PairCase>>(&args[2])?, &read::<Vec<PairPrediction>>(&args[3])?)?),
        Some("score-extraction") if args.len() == 4 => print(&score_extraction(&read::<Vec<RawPairCase>>(&args[2])?, &read::<Vec<ExtractionRecord>>(&args[3])?)?),
        Some("score-retrieval") if args.len() == 5 => {
            let ids:BTreeSet<String> = Store::open(&args[2])?.claims()?.into_iter().map(|c|c.id).collect();
            print(&score_retrieval(&read::<Vec<RetrievalCase>>(&args[3])?, &read::<Vec<RetrievalPrediction>>(&args[4])?, &ids)?)
        }
        _ => bail!("usage: phased-memory serve DB CONFIG PORT | probe CONFIG | ingest DB INGEST_JSON [CONFIG_TO_EMBED] | stage-a CONFIG CASES OUTPUT | stage-b CONFIG CASES OUTPUT | stage-c DB CONFIG CASES OUTPUT vector|structured|reranked | score-pairs CASES PREDICTIONS | score-extraction CASES RECORDS | score-retrieval DB CASES PREDICTIONS | original-stage-a|original-stage-b CONFIG PAIRS_JSONL OUTPUT_JSONL | original-index DB CONFIG MEMORIES_JSONL TRACE_JSONL human-reference|extract | original-retrieval DB CONFIG MEMORIES_JSONL QUERIES_JSONL OUTPUT_JSONL vector|structured|reranked | phase-control INPUT_JSON OUTPUT_JSON | phase-compare CONFIG INPUT_JSON OUTPUT_JSON"),
    }
}
fn ensure_new_output(path: &str) -> Result<()> {
    ensure!(
        !Path::new(path).exists(),
        "invalid: output path already exists; no inference attempted"
    );
    Ok(())
}
async fn observe_runtime(p: &Pumas) -> Result<Value> {
    let (fence, build) = p.discover().await?;
    let mut roles = serde_json::Map::new();
    for (name, selection) in [
        ("extraction", &p.config.extraction),
        ("embedding", &p.config.embedding),
        ("decision", &p.config.decision),
    ] {
        let observation = match selection {
            Some(selection) => match p.inspect(selection, &fence).await {
                Ok(_) => {
                    json!({"status":"advertised_only","model":selection.model,"profile":selection.profile})
                }
                Err(error) => json!({"status":"unavailable","error":error.to_string()}),
            },
            None => json!({"status":"unconfigured"}),
        };
        roles.insert(name.into(), observation);
    }
    Ok(json!({"fence":fence,"build":build,"roles":roles,"inference":"unrun"}))
}
async fn stage_c(
    database: &str,
    config: &str,
    input: &str,
    output: &str,
    mode: &str,
) -> Result<()> {
    ensure_new_output(output)?;
    ensure!(
        ["vector", "structured", "reranked"].contains(&mode),
        "invalid: Stage C mode"
    );
    let store = Store::open(database)?;
    let pumas = runtime(config)?;
    let cases: Vec<RetrievalCase> = read(input)?;
    let mut predictions = Vec::new();
    for case in &cases {
        let result = async {
            let embedding = pumas.embed(&case.query_text).await?;
            let mut comparison = store.compare_all(&embedding, &case.filters)?;
            if mode == "reranked" {
                rerank(
                    &pumas,
                    &query_claim(&case.query_text, &case.filters),
                    &mut comparison,
                )
                .await?;
            }
            comparison.limit(5);
            let hits = match mode {
                "vector" => comparison.vector,
                "structured" => comparison.structured,
                _ => comparison.reranked.context("unavailable: reranking")?,
            };
            Ok::<_, anyhow::Error>(hits.into_iter().map(|hit| hit.claim.id).collect())
        }
        .await;
        predictions.push(match result {
            Ok(ids) => RetrievalPrediction {
                id: case.id.clone(),
                ranked_ids: Some(ids),
                error: None,
            },
            Err(error) => RetrievalPrediction {
                id: case.id.clone(),
                ranked_ids: None,
                error: Some(error.to_string()),
            },
        });
    }
    write(output, &predictions)?;
    let ids = store.claims()?.into_iter().map(|claim| claim.id).collect();
    let score = score_retrieval(&cases, &predictions, &ids)?;
    print(&score)?;
    ensure!(
        score.completed == score.total,
        "unavailable: incomplete Stage C; output retained"
    );
    Ok(())
}
async fn embed_batch(p: &Pumas, batch: &mut Ingest) -> Result<()> {
    ensure!(
        batch.embeddings.is_empty(),
        "invalid: runtime embedding import requires no preattached vectors"
    );
    for c in &batch.claims {
        c.validate()?;
        batch.embeddings.push(ClaimEmbedding {
            claim_id: c.id.clone(),
            embedding: p.embed(&c.normalized).await?,
        });
    }
    Ok(())
}
fn query_claim(text: &str, f: &Filters) -> Claim {
    Claim {
        id: "query".into(),
        normalized: text.into(),
        model_local_id: None,
        state_key: None,
        triple: Triple {
            subject: f.subject.clone().unwrap_or_else(|| "unspecified".into()),
            predicate: f.predicate.clone().unwrap_or_else(|| "unspecified".into()),
            object: f.object.clone().unwrap_or_else(|| "unspecified".into()),
            polarity: f.polarity.unwrap_or(Polarity::Unknown),
        },
        entities: f.entity.clone().into_iter().collect(),
        source: "query".into(),
        perspective: f
            .perspective
            .clone()
            .unwrap_or_else(|| "unspecified".into()),
        continuity: f.continuity.clone().unwrap_or_else(|| "unspecified".into()),
        story_time: f
            .timeline
            .as_ref()
            .zip(f.at_or_before)
            .map(|(timeline, position)| StoryTime {
                timeline: timeline.clone(),
                position,
            }),
        qualifiers: f.qualifier.clone().into_iter().collect(),
    }
}
async fn rerank(p: &Pumas, query: &Claim, c: &mut Comparison) -> Result<()> {
    ensure!(
        p.config.decision_adapter.as_deref() == Some("decision_json_v1")
            && p.config.decision.is_some(),
        "unavailable: operator-qualified decision adapter and exact model/profile required"
    );
    let mut hits = c.structured.clone();
    if hits.is_empty() {
        hits.truncate(5);
        c.reranked = Some(hits);
        c.rerank_status = "unrun: no eligible candidates; no decision inference occurred".into();
        return Ok(());
    }

    for h in &mut hits {
        let o = p.decide(query, &h.claim).await?;
        h.decision = Some(DecisionEvidence {
            labels: o.value,
            provenance: o.provenance,
        });
    }
    fn rank(r: Relationship) -> u8 {
        match r {
            Relationship::Supports => 3,
            Relationship::Compatible => 2,
            Relationship::InsufficientInformation => 1,
            Relationship::Contradicts => 0,
        }
    }
    hits.sort_by(|a, b| {
        rank(b.decision.as_ref().unwrap().labels.relationship)
            .cmp(&rank(a.decision.as_ref().unwrap().labels.relationship))
            .then(b.score.total_cmp(&a.score))
            .then(a.claim.id.cmp(&b.claim.id))
    });
    hits.truncate(5);
    c.reranked = Some(hits);
    c.rerank_status="completed: operator-qualified decision_json_v1; ordinal relation ordering, not probabilities".into();
    Ok(())
}
#[derive(Clone)]
struct App {
    store: Arc<Mutex<Store>>,
    pumas: Arc<Pumas>,
    token: Arc<String>,
    host: Arc<String>,
    gate: Arc<Semaphore>,
    shutdown: tokio::sync::watch::Receiver<bool>,
}
impl App {
    fn with_store<T>(&self, operation: impl FnOnce(&mut Store) -> Result<T>) -> Result<T> {
        // The multi-thread runtime retains synchronous completion; cancellation cannot detach writes.
        tokio::task::block_in_place(|| {
            let mut store = self
                .store
                .lock()
                .map_err(|_| anyhow::anyhow!("unavailable: store lock"))?;
            operation(&mut store)
        })
    }
}
#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    Discover,
    Ingest {
        batch: Ingest,
        #[serde(default)]
        embed: bool,
    },
    Extract {
        raw_text: String,
    },
    IngestRaw {
        raw_text: String,
    },
    Relation {
        relation: Relation,
    },
    Query {
        text: String,
        embedding: Option<Embedding>,
        #[serde(default)]
        filters: Filters,
        #[serde(default)]
        rerank: bool,
    },
}
async fn serve(db: &str, config: &str, port: u16) -> Result<()> {
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await?;
    let addr = listener.local_addr()?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let app = App {
        store: Arc::new(Mutex::new(Store::open(db)?)),
        pumas: Arc::new(runtime(config)?),
        token: Arc::new(uuid::Uuid::new_v4().to_string()),
        host: Arc::new(addr.to_string()),
        gate: Arc::new(Semaphore::new(1)),
        shutdown: shutdown_rx,
    };
    let router = Router::new()
        .route("/", get(index))
        .route("/app.js", get(js))
        .route("/style.css", get(css))
        .route("/api", post(api))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .layer(middleware::from_fn_with_state(app.clone(), guard))
        .with_state(app);
    println!("Phased Memory http://{addr} — live model qualification unrun");
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            let _ = shutdown_tx.send(true);
        })
        .await?;
    Ok(())
}
async fn guard(State(s): State<App>, req: Request, next: Next) -> Response {
    let headers = req.headers();
    let valid = headers.get(header::HOST).and_then(|v| v.to_str().ok()) == Some(s.host.as_str())
        && headers
            .get(header::ORIGIN)
            .is_none_or(|v| v.to_str().ok() == Some(format!("http://{}", s.host).as_str()))
        && headers
            .get("sec-fetch-site")
            .is_none_or(|v| v == "same-origin" || v == "none")
        && (req.method() != axum::http::Method::POST
            || (headers.get("x-lab-session").and_then(|v| v.to_str().ok())
                == Some(s.token.as_str())
                && headers
                    .get(header::CONTENT_TYPE)
                    .is_some_and(|v| v == "application/json")));
    if !valid {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"error":"invalid: local browser boundary"})),
        )
            .into_response();
    }
    let mut response = next.run(req).await;
    let h = response.headers_mut();
    h.insert(header::CONTENT_SECURITY_POLICY,axum::http::HeaderValue::from_static("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"));
    h.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        axum::http::HeaderValue::from_static("nosniff"),
    );
    h.insert(
        header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}
async fn index(State(s): State<App>) -> Html<String> {
    Html(include_str!("../public/index.html").replace("SESSION_TOKEN", &s.token))
}
async fn js() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/javascript; charset=utf-8")
        .body(Body::from(include_str!("../public/app.js")))
        .unwrap()
}
async fn css() -> Response {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/css; charset=utf-8")
        .body(Body::from(include_str!("../public/style.css")))
        .unwrap()
}
async fn api(State(s): State<App>, Json(action): Json<Action>) -> Response {
    let Ok(_permit) = s.gate.try_acquire() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({"error":"overload: one active operation; no queue"})),
        )
            .into_response();
    };
    let mut shutdown = s.shutdown.clone();
    let work = async {
        match action {
            Action::Discover => observe_runtime(&s.pumas).await,
            Action::Ingest { mut batch, embed } => {
                ensure!(
                    matches!(batch.provenance, Provenance::Human { .. }),
                    "invalid: submitted claims must be labeled human"
                );
                ensure!(
                    batch
                        .embeddings
                        .iter()
                        .all(|e| matches!(e.embedding.provenance, Provenance::Synthetic { .. })),
                    "invalid: submitted vectors must be labeled synthetic"
                );
                if embed {
                    embed_batch(&s.pumas, &mut batch).await?;
                }
                s.with_store(|store| store.ingest(&batch))?;
                Ok(json!({"status":"stored","claims":batch.claims.len()}))
            }
            Action::IngestRaw { raw_text } => {
                let mut o = s.pumas.extract(&raw_text).await?;
                assign_durable_ids(&mut o.value, &uuid::Uuid::new_v4().to_string())?;
                let mut batch = Ingest {
                    raw_text,
                    claims: o.value,
                    provenance: o.provenance,
                    embeddings: vec![],
                };
                embed_batch(&s.pumas, &mut batch).await?;
                s.with_store(|store| store.ingest(&batch))?;
                Ok(
                    json!({"status":"stored from Pumas extraction and embeddings","claims":batch.claims.len()}),
                )
            }
            Action::Extract { raw_text } => {
                Ok(serde_json::to_value(s.pumas.extract(&raw_text).await?)?)
            }
            Action::Relation { relation } => {
                ensure!(
                    matches!(relation.provenance, Provenance::Human { .. }),
                    "invalid: browser relation provenance must be human"
                );
                s.with_store(|store| store.add_relation(&relation))?;
                Ok(json!({"status":"relation appended"}))
            }
            Action::Query {
                text,
                embedding,
                filters,
                rerank: do_rerank,
            } => {
                let embedding = match embedding {
                    Some(e) => {
                        ensure!(
                            matches!(e.provenance, Provenance::Synthetic { .. }),
                            "invalid: supplied query vectors must be explicitly synthetic"
                        );
                        e
                    }
                    None => s.pumas.embed(&text).await?,
                };
                let mut c = s.with_store(|store| store.compare_all(&embedding, &filters))?;
                if do_rerank {
                    if let Err(e) = rerank(&s.pumas, &query_claim(&text, &filters), &mut c).await {
                        c.rerank_status = e.to_string();
                    }
                }
                c.limit(5);
                Ok(serde_json::to_value(c)?)
            }
        }
    };
    let result: Result<Value> = tokio::select! {result=work=>result,_=shutdown.changed()=>Err(anyhow::anyhow!("unknown: shutdown cancelled owned operation transport; provider cessation is unproven")),};
    match result {
        Ok(v) => Json(v).into_response(),
        Err(e) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"error":e.to_string()})),
        )
            .into_response(),
    }
}
