use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

pub const MAX_TEXT: usize = 65_536;
pub const MAX_DIMENSIONS: usize = 8192;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Relationship {
    Supports,
    Contradicts,
    Compatible,
    InsufficientInformation,
}
impl Relationship {
    pub const ALL: [Self; 4] = [
        Self::Supports,
        Self::Contradicts,
        Self::Compatible,
        Self::InsufficientInformation,
    ];
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum TemporalOrder {
    SameTime,
    BLater,
    BEarlier,
    Unknown,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum UpdateType {
    None,
    StateChange,
    ExplicitCorrection,
    Undetermined,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Labels {
    pub relationship: Relationship,
    pub temporal_order: TemporalOrder,
    pub update_type: UpdateType,
}

/// Story time is author-supplied; ingestion time never becomes story chronology.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StoryTime {
    pub timeline: String,
    pub position: i64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub polarity: Polarity,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    Affirmed,
    Negated,
    Unknown,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Claim {
    pub id: String,
    pub normalized: String,
    #[serde(default)]
    pub model_local_id: Option<String>,
    /// Explicit exclusive state slot; absent means latest per full directed triple.
    #[serde(default)]
    pub state_key: Option<String>,
    pub triple: Triple,
    pub entities: Vec<String>,
    pub source: String,
    pub perspective: String,
    pub continuity: String,
    pub story_time: Option<StoryTime>,
    /// Retains e.g. belief, lie, intention, speaker, direction without truth collapse.
    pub qualifiers: Vec<String>,
}
impl Claim {
    pub fn validate(&self) -> Result<()> {
        for (name, s) in [
            ("id", &self.id),
            ("normalized", &self.normalized),
            ("source", &self.source),
            ("perspective", &self.perspective),
            ("continuity", &self.continuity),
        ] {
            bounded(name, s, MAX_TEXT)?;
        }
        for s in [
            &self.triple.subject,
            &self.triple.predicate,
            &self.triple.object,
        ] {
            bounded("triple field", s, 1024)?;
        }
        if let Some(key) = &self.state_key {
            bounded("state key", key, 1024)?;
        }
        if let Some(id) = &self.model_local_id {
            bounded("model local id", id, 256)?;
        }
        ensure!(self.id.len() <= 256, "invalid: claim id too long");
        ensure!(
            self.entities.len() <= 128 && self.qualifiers.len() <= 128,
            "invalid: too many claim attributes"
        );
        for s in self.entities.iter().chain(&self.qualifiers) {
            bounded("attribute", s, 1024)?;
        }
        if let Some(t) = &self.story_time {
            bounded("timeline", &t.timeline, 256)?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Provenance {
    Human {
        note: String,
    },
    Synthetic {
        note: String,
    },
    Pumas {
        model: String,
        profile: String,
        instance_generation: String,
        service_generation: String,
        request_id: String,
    },
}
impl Provenance {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Human { note } | Self::Synthetic { note } => {
                bounded("provenance note", note, 4096)
            }
            Self::Pumas {
                model,
                profile,
                instance_generation,
                service_generation,
                request_id,
            } => {
                for s in [
                    model,
                    profile,
                    instance_generation,
                    service_generation,
                    request_id,
                ] {
                    bounded("inference provenance", s, 256)?;
                }
                Ok(())
            }
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Embedding {
    pub space: String,
    pub vector: Vec<f32>,
    pub provenance: Provenance,
}
impl Embedding {
    pub fn validate(&self) -> Result<()> {
        bounded("embedding space", &self.space, 1024)?;
        validate_vector(&self.vector)?;
        match &self.provenance {
            Provenance::Synthetic { .. } => ensure!(
                self.space.starts_with("synthetic-") || self.space.starts_with("synthetic:"),
                "invalid: synthetic vectors require a synthetic space"
            ),
            Provenance::Pumas { .. } => ensure!(
                self.space.starts_with("pumas:"),
                "invalid: real vectors require a Pumas space"
            ),
            Provenance::Human { .. } => {
                bail!("invalid: manually supplied vectors must be labeled synthetic")
            }
        }
        self.provenance.validate()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ingest {
    pub raw_text: String,
    pub claims: Vec<Claim>,
    pub provenance: Provenance,
    pub embeddings: Vec<ClaimEmbedding>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimEmbedding {
    pub claim_id: String,
    pub embedding: Embedding,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relation {
    pub id: String,
    pub a: String,
    pub b: String,
    pub labels: Labels,
    pub confidence: Option<f64>,
    pub provenance: Provenance,
}
impl Relation {
    pub fn validate(&self) -> Result<()> {
        for s in [&self.id, &self.a, &self.b] {
            bounded("relation id", s, 256)?;
        }
        ensure!(
            self.a != self.b,
            "invalid: relation requires distinct claims"
        );
        if let Some(c) = self.confidence {
            ensure!(
                c.is_finite() && (0.0..=1.0).contains(&c),
                "invalid: confidence"
            );
        }
        self.provenance.validate()
    }
}
pub fn bounded(name: &str, s: &str, max: usize) -> Result<()> {
    ensure!(
        !s.trim().is_empty() && s.len() <= max,
        "invalid: {name} is empty or exceeds bound"
    );
    Ok(())
}
pub fn validate_vector(v: &[f32]) -> Result<()> {
    ensure!(
        !v.is_empty() && v.len() <= MAX_DIMENSIONS && v.iter().all(|n| n.is_finite()),
        "invalid: embedding dimensions or values"
    );
    ensure!(v.iter().any(|n| *n != 0.0), "invalid: zero embedding");
    Ok(())
}
pub fn cosine(a: &[f32], b: &[f32]) -> Result<f64> {
    validate_vector(a)?;
    validate_vector(b)?;
    ensure!(a.len() == b.len(), "invalid: embedding dimension mismatch");
    let (mut dot, mut aa, mut bb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    let c = dot / (aa.sqrt() * bb.sqrt());
    if !c.is_finite() {
        bail!("invalid: cosine overflow");
    }
    Ok(c.clamp(-1.0, 1.0))
}

/// Assign host identities once when extracted claims enter durable storage.
pub fn assign_durable_ids(claims: &mut [Claim], namespace: &str) -> Result<()> {
    bounded("source namespace", namespace, 128)?;
    for (index, c) in claims.iter_mut().enumerate() {
        c.validate()?;
        c.model_local_id = Some(c.id.clone());
        c.id = format!("{namespace}:{index}");
        c.validate()?;
    }
    Ok(())
}
