//! Opt-in expected-equivalence control; this module makes no phase quality claim.
use crate::{
    domain::{bounded, cosine, Polarity, Provenance, StoryTime},
    pumas::Pumas,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const PROTOCOL: &str = "semantic-phase-control-v1";
pub const EQUIVALENCE_TOLERANCE: f64 = 1e-12;
const MAX_ITEMS: usize = 128;
const MAX_VOCABULARY: usize = 8192;
const MAX_EXACT_POSITION: u64 = (1_u64 << 53) - 1;
fn nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::<T>::deserialize(d)
}

/// Every field is explicit. IDs, prose, sources, confidence and gold have no slot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Semantics {
    #[serde(deserialize_with = "nullable")]
    pub subject: Option<String>,
    #[serde(deserialize_with = "nullable")]
    pub predicate: Option<String>,
    pub object: Value,
    pub polarity: Polarity,
    /// String/null, or {holder:string|null,modality:string|null}; no inferred defaults.
    pub perspective: Value,
    #[serde(deserialize_with = "nullable")]
    pub continuity: Option<String>,
    pub entities: Vec<String>,
    pub qualifiers: Vec<String>,
    #[serde(deserialize_with = "nullable")]
    pub state_key: Option<String>,
    #[serde(deserialize_with = "nullable")]
    pub story_time: Option<StoryTime>,
}
impl Semantics {
    pub fn validate(&self) -> Result<()> {
        for text in [
            &self.subject,
            &self.predicate,
            &self.continuity,
            &self.state_key,
        ]
        .into_iter()
        .flatten()
        {
            bounded("semantic category", text, 1024)?;
        }
        ensure!(
            serde_json::to_vec(&self.object)?.len() <= 8192,
            "invalid: semantic object bound"
        );
        match &self.perspective {
            Value::Null => {}
            Value::String(s) => bounded("perspective", s, 1024)?,
            Value::Object(m) => {
                ensure!(
                    m.len() == 2 && m.contains_key("holder") && m.contains_key("modality"),
                    "invalid: perspective needs exactly holder and modality"
                );
                for value in m.values() {
                    match value {
                        Value::Null => {}
                        Value::String(s) => bounded("perspective category", s, 1024)?,
                        _ => {
                            anyhow::bail!("invalid: perspective categories must be strings or null")
                        }
                    }
                }
            }
            _ => anyhow::bail!("invalid: perspective must be a string, scoped object or null"),
        }
        for (name, list) in [
            ("entities", &self.entities),
            ("qualifiers", &self.qualifiers),
        ] {
            ensure!(list.len() <= 128, "invalid: {name} bound");
            for value in list {
                bounded(name, value, 1024)?;
            }
        }
        if let Some(time) = &self.story_time {
            bounded("timeline", &time.timeline, 256)?;
            ensure!(
                time.position.unsigned_abs() <= MAX_EXACT_POSITION,
                "unsupported: phase clock exceeds exact f64 integer range"
            );
        }
        Ok(())
    }
    pub fn canonical_json(&self) -> Result<String> {
        self.validate()?;
        let mut semantic = self.clone();
        semantic.entities.sort();
        semantic.entities.dedup();
        semantic.qualifiers.sort();
        semantic.qualifiers.dedup();
        canonical_json(&serde_json::to_value(semantic)?)
    }
    fn features(&self) -> Result<BTreeMap<String, (f64, f64)>> {
        self.validate()?;
        let mut values = BTreeMap::new();
        let tuple = canonical_json(&json!([
            self.subject,
            self.predicate,
            self.object,
            self.perspective,
            self.continuity
        ]))?;
        let polarity = match self.polarity {
            Polarity::Affirmed => (1.0, 0.0),
            Polarity::Negated => (-1.0, 0.0),
            Polarity::Unknown => (0.0, 1.0),
        };
        values.insert(format!("tuple:{tuple}"), polarity);
        for entity in &self.entities {
            values.insert(
                format!("entity:{}", serde_json::to_string(entity)?),
                (1.0, 0.0),
            );
        }
        for qualifier in &self.qualifiers {
            values.insert(
                format!("qualifier:{}", serde_json::to_string(qualifier)?),
                (1.0, 0.0),
            );
        }
        if let Some(key) = &self.state_key {
            values.insert(
                format!("state_key:{}", serde_json::to_string(key)?),
                (1.0, 0.0),
            );
        }
        if let Some(time) = &self.story_time {
            let timeline = serde_json::to_string(&time.timeline)?;
            values.insert(format!("timeline:{timeline}"), (1.0, 0.0));
            values.insert(
                format!("time_value:{timeline}"),
                (time.position as f64, 0.0),
            );
        } else {
            values.insert("unknown_time".into(), (1.0, 0.0));
        }
        Ok(values)
    }
}
/// Sorted object keys recursively; arrays retain semantic order except category sets above.
/// Escaping and numeric spelling remain owned by serde_json, not a claimed RFC canonicalizer.
pub fn canonical_json(value: &Value) -> Result<String> {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let ordered: BTreeMap<_, _> =
                    map.iter().map(|(k, v)| (k.clone(), sorted(v))).collect();
                Value::Object(ordered.into_iter().collect())
            }
            Value::Array(values) => Value::Array(values.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    Ok(serde_json::to_string(&sorted(value))?)
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DataKind {
    AuthoredDevelopment,
    OperatorSupplied,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Item {
    pub id: String,
    pub semantics: Semantics,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentInput {
    pub schema_version: u32,
    pub data_kind: DataKind,
    pub candidate_count: usize,
    pub corpus: Vec<Item>,
    pub queries: Vec<Item>,
}
impl ExperimentInput {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported: phase input schema");
        ensure!(
            !self.corpus.is_empty()
                && self.corpus.len() <= MAX_ITEMS
                && !self.queries.is_empty()
                && self.queries.len() <= MAX_ITEMS,
            "invalid: phase expects 1..128 corpus items and queries"
        );
        ensure!(
            self.candidate_count == self.corpus.len(),
            "invalid: frozen candidate_count must include the entire corpus"
        );
        for items in [&self.corpus, &self.queries] {
            let mut seen = BTreeSet::new();
            for item in items {
                bounded("phase output id", &item.id, 256)?;
                ensure!(
                    seen.insert(&item.id),
                    "invalid: duplicate phase id within corpus/query namespace"
                );
                item.semantics.validate()?;
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Projected {
    pub id: String,
    pub canonical_semantics: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct Frozen {
    pub vocabulary: Vec<String>,
    pub time_scale: f64,
    pub candidate_count: usize,
    pub corpus: Vec<Projected>,
    pub queries: Vec<Projected>,
    pub tie_rule: String,
    pub representation: String,
}
#[derive(Debug, Clone)]
pub struct Encoding {
    pub real: Vec<f64>,
    pub imaginary: Vec<f64>,
}
impl Frozen {
    pub fn encode(&self, semantic: &Semantics) -> Result<Encoding> {
        let values = semantic.features()?;
        let mut real = vec![0.0; self.vocabulary.len()];
        let mut imaginary = real.clone();
        for (feature, (mut re, im)) in values {
            let index = self.vocabulary.binary_search(&feature).map_err(|_| {
                anyhow::anyhow!("invalid: semantic feature outside frozen vocabulary")
            })?;
            if feature.starts_with("time_value:") {
                re /= self.time_scale;
            }
            real[index] = re;
            imaginary[index] = im;
        }
        let norm = real
            .iter()
            .zip(&imaginary)
            .map(|(re, im)| re * re + im * im)
            .sum::<f64>()
            .sqrt();
        ensure!(norm.is_finite() && norm > 0.0, "invalid: phase norm");
        for value in real.iter_mut().chain(&mut imaginary) {
            *value /= norm;
        }
        Ok(Encoding { real, imaginary })
    }
}
pub fn freeze(input: &ExperimentInput) -> Result<Frozen> {
    input.validate()?;
    let mut vocabulary = BTreeSet::new();
    let mut scale = 1.0_f64;
    for item in input.corpus.iter().chain(&input.queries) {
        vocabulary.extend(item.semantics.features()?.into_keys());
        if let Some(time) = &item.semantics.story_time {
            scale = scale.max(time.position.unsigned_abs() as f64);
        }
    }
    ensure!(
        vocabulary.len() <= MAX_VOCABULARY,
        "invalid: phase vocabulary exceeds 8192 slots"
    );
    fn project(items: &[Item]) -> Result<Vec<Projected>> {
        items
            .iter()
            .map(|item| {
                Ok(Projected {
                    id: item.id.clone(),
                    canonical_semantics: item.semantics.canonical_json()?,
                })
            })
            .collect()
    }
    Ok(Frozen{vocabulary:vocabulary.into_iter().collect(),time_scale:scale,candidate_count:input.candidate_count,corpus:project(&input.corpus)?,queries:project(&input.queries)?,tie_rule:"descending score, then ascending ID; no candidate preselection".into(),representation:"normalized complex channels; tuple polarity affirmed=(1,0), negated=(-1,0), unknown=(0,1); presence=(1,0); time=position/T real; real control=[Re(z),Im(z)]".into()})
}
pub fn complex_score(a: &Encoding, b: &Encoding) -> Result<f64> {
    ensure!(
        a.real.len() == b.real.len()
            && a.real.len() == a.imaginary.len()
            && b.real.len() == b.imaginary.len(),
        "invalid: complex dimensions"
    );
    let value = a
        .real
        .iter()
        .zip(&b.real)
        .zip(a.imaginary.iter().zip(&b.imaginary))
        .map(|((ar, br), (ai, bi))| ar * br + ai * bi)
        .sum::<f64>();
    ensure!(value.is_finite(), "invalid: complex score");
    Ok(value)
}
pub fn real_concatenation_score(a: &Encoding, b: &Encoding) -> Result<f64> {
    ensure!(
        a.real.len() == b.real.len()
            && a.real.len() == a.imaginary.len()
            && b.real.len() == b.imaginary.len(),
        "invalid: real control dimensions"
    );
    let mut dot = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for (a, b) in a
        .real
        .iter()
        .chain(&a.imaginary)
        .zip(b.real.iter().chain(&b.imaginary))
    {
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    ensure!(aa > 0.0 && bb > 0.0, "invalid: real control zero norm");
    let value = dot / (aa.sqrt() * bb.sqrt());
    ensure!(value.is_finite(), "invalid: real control score");
    Ok(value)
}
#[derive(Debug, Clone, Serialize)]
pub struct Scored {
    pub id: String,
    pub score: f64,
}
fn rank(scores: &mut [Scored]) {
    scores.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
}
#[derive(Debug, Clone, Serialize)]
pub struct NumericQuery {
    pub id: String,
    pub phase: Vec<Scored>,
    pub real_concatenation: Vec<Scored>,
    pub maximum_absolute_error: f64,
    pub rankings_equal: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingQuery {
    pub id: String,
    pub status: String,
    pub ranked: Option<Vec<Scored>>,
    pub provenance: Option<Provenance>,
    pub vector: Option<Vec<f32>>,
    pub error: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingCorpus {
    pub id: String,
    pub status: String,
    pub provenance: Option<Provenance>,
    pub vector: Option<Vec<f32>>,
    pub error: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingArm {
    pub status: String,
    pub space: Option<String>,
    pub corpus: Vec<EmbeddingCorpus>,
    pub queries: Vec<EmbeddingQuery>,
}
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentOutput {
    pub schema_version: u32,
    pub protocol: String,
    pub data_kind: DataKind,
    pub frozen: Frozen,
    pub equivalence_scope: String,
    pub equivalence_tolerance: f64,
    pub maximum_absolute_error: f64,
    pub equivalence_passed: bool,
    pub numeric_queries: Vec<NumericQuery>,
    pub pumas_embeddings: Option<EmbeddingArm>,
    pub claim: String,
}
pub fn control(input: &ExperimentInput) -> Result<ExperimentOutput> {
    let frozen = freeze(input)?;
    let corpus: Vec<_> = input
        .corpus
        .iter()
        .map(|item| frozen.encode(&item.semantics))
        .collect::<Result<_>>()?;
    let mut numeric_queries = Vec::new();
    let mut max_error = 0.0_f64;
    for query in &input.queries {
        let encoding = frozen.encode(&query.semantics)?;
        let mut phase = Vec::new();
        let mut real = Vec::new();
        let mut query_error = 0.0_f64;
        for (item, corpus_encoding) in input.corpus.iter().zip(&corpus) {
            let a = complex_score(&encoding, corpus_encoding)?;
            let b = real_concatenation_score(&encoding, corpus_encoding)?;
            query_error = query_error.max((a - b).abs());
            phase.push(Scored {
                id: item.id.clone(),
                score: a,
            });
            real.push(Scored {
                id: item.id.clone(),
                score: b,
            });
        }
        rank(&mut phase);
        rank(&mut real);
        let rankings_equal = phase.iter().map(|s| &s.id).eq(real.iter().map(|s| &s.id));
        max_error = max_error.max(query_error);
        numeric_queries.push(NumericQuery {
            id: query.id.clone(),
            phase,
            real_concatenation: real,
            maximum_absolute_error: query_error,
            rankings_equal,
        });
    }
    ensure!(
        max_error <= EQUIVALENCE_TOLERANCE,
        "invalid: expected-equivalence control exceeded tolerance"
    );
    Ok(ExperimentOutput {
        schema_version: 1,
        protocol: PROTOCOL.into(),
        data_kind: input.data_kind.clone(),
        frozen,
        equivalence_scope: "pairwise score equivalence within tolerance only; raw ranking equality is diagnostic and can differ at floating-point near ties".into(),
        equivalence_tolerance: EQUIVALENCE_TOLERANCE,
        maximum_absolute_error: max_error,
        equivalence_passed: true,
        numeric_queries,
        pumas_embeddings: None,
        claim: "expected-equivalence arithmetic control only; no phase advantage or model-quality claim; no gold labels consumed".into(),
    })
}
/// Embedding arm never substitutes deterministic/synthetic vectors for an unavailable model.
pub async fn compare(pumas: &Pumas, input: &ExperimentInput) -> Result<ExperimentOutput> {
    let mut output = control(input)?;
    if pumas.config.embedding.is_none() {
        let reason = "unavailable: embedding identity not configured; no inference attempted";
        let corpus = output
            .frozen
            .corpus
            .iter()
            .map(|item| EmbeddingCorpus {
                id: item.id.clone(),
                status: "unrun".into(),
                provenance: None,
                vector: None,
                error: Some(reason.into()),
            })
            .collect();
        let queries = output
            .frozen
            .queries
            .iter()
            .map(|item| EmbeddingQuery {
                id: item.id.clone(),
                status: "unrun".into(),
                ranked: None,
                provenance: None,
                vector: None,
                error: Some(reason.into()),
            })
            .collect();
        output.pumas_embeddings = Some(EmbeddingArm {
            status: "unrun".into(),
            space: None,
            corpus,
            queries,
        });
        return Ok(output);
    }
    let mut corpus = Vec::new();
    let mut embeddings = Vec::new();
    let mut space: Option<String> = None;
    for item in &output.frozen.corpus {
        match pumas.embed(&item.canonical_semantics).await {
            Ok(embedding) => {
                if space.as_ref().is_some_and(|s| s != &embedding.space) {
                    corpus.push(EmbeddingCorpus {
                        id: item.id.clone(),
                        status: "error".into(),
                        provenance: Some(embedding.provenance.clone()),
                        vector: Some(embedding.vector.clone()),
                        error: Some(
                            "unavailable: embedding space changed within frozen comparison".into(),
                        ),
                    });
                    embeddings.push(None);
                } else {
                    space = Some(embedding.space.clone());
                    corpus.push(EmbeddingCorpus {
                        id: item.id.clone(),
                        status: "completed".into(),
                        provenance: Some(embedding.provenance.clone()),
                        vector: Some(embedding.vector.clone()),
                        error: None,
                    });
                    embeddings.push(Some(embedding));
                }
            }
            Err(error) => {
                corpus.push(EmbeddingCorpus {
                    id: item.id.clone(),
                    status: "error".into(),
                    provenance: None,
                    vector: None,
                    error: Some(error.to_string()),
                });
                embeddings.push(None);
            }
        }
    }
    let complete = embeddings.iter().all(Option::is_some);
    let mut queries = Vec::new();
    for item in &output.frozen.queries {
        let mut provenance = None;
        let mut vector = None;
        let result = async {
            ensure!(complete, "unavailable: full frozen corpus embeddings required; no candidate reduction or synthetic fallback");
            let query = pumas.embed(&item.canonical_semantics).await?;
            provenance = Some(query.provenance.clone());
            vector = Some(query.vector.clone());
            ensure!(space.as_ref() == Some(&query.space), "unavailable: query embedding space differs from frozen corpus");
            let mut ranked = Vec::new();
            for (memory, embedding) in output.frozen.corpus.iter().zip(&embeddings) {
                ranked.push(Scored {
                    id: memory.id.clone(),
                    score: cosine(&query.vector, &embedding.as_ref().context("unavailable: corpus embedding")?.vector)?,
                });
            }
            rank(&mut ranked);
            Ok::<_, anyhow::Error>(ranked)
        }.await;
        queries.push(match result {
            Ok(ranked) => EmbeddingQuery {
                id: item.id.clone(),
                status: "completed".into(),
                ranked: Some(ranked),
                provenance,
                vector,
                error: None,
            },
            Err(error) => EmbeddingQuery {
                id: item.id.clone(),
                status: if complete { "error" } else { "unrun" }.into(),
                ranked: None,
                provenance,
                vector,
                error: Some(error.to_string()),
            },
        });
    }
    let succeeded = complete && queries.iter().all(|q| q.status == "completed");
    output.pumas_embeddings = Some(EmbeddingArm {
        status: if succeeded {
            "completed"
        } else {
            "unavailable"
        }
        .into(),
        space,
        corpus,
        queries,
    });
    Ok(output)
}
