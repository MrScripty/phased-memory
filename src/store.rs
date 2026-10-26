use crate::domain::*;
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection};
use serde::Serialize;
use std::{collections::BTreeSet, path::Path};

pub struct Store {
    conn: Connection,
}
#[derive(Debug, Clone, Serialize)]
pub struct DecisionEvidence {
    pub labels: Labels,
    pub provenance: Provenance,
}
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub claim: Claim,
    pub score: f64,
    pub embedding_provenance: Provenance,
    pub decision: Option<DecisionEvidence>,
}
#[derive(Debug, Clone, serde::Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Filters {
    pub perspective: Option<String>,
    pub continuity: Option<String>,
    pub relationship: Option<Relationship>,
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub polarity: Option<Polarity>,
    pub entity: Option<String>,
    pub qualifier: Option<String>,
    pub timeline: Option<String>,
    pub at_or_before: Option<i64>,
    /// Latest per directed triple or explicit state_key, perspective, continuity and timeline.
    #[serde(default)]
    pub latest: bool,
    #[serde(default)]
    pub chronological: bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub vector: Vec<Hit>,
    pub structured: Vec<Hit>,
    pub reranked: Option<Vec<Hit>>,
    pub rerank_status: String,
    pub relations: Vec<Relation>,
}
impl Comparison {
    pub fn limit(&mut self, count: usize) {
        self.vector.truncate(count);
        self.structured.truncate(count);
        if let Some(hits) = &mut self.reranked {
            hits.truncate(count);
        }
    }
}
impl Store {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut conn = Connection::open(path)?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(2))?;
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            version == 0 || version == 1,
            "unsupported: SQLite schema version {version}; preserved without migration"
        );
        if version == 0 {
            let tables:i64=conn.query_row("SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",[],|row|row.get(0))?;
            ensure!(
                tables == 0,
                "unsupported: existing unversioned SQLite state is preserved"
            );
            let tx = conn.transaction()?;
            tx.execute_batch("CREATE TABLE sources(id INTEGER PRIMARY KEY, raw_text TEXT NOT NULL, provenance TEXT NOT NULL, ingested_at TEXT NOT NULL DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now'))); CREATE TABLE claims(id TEXT PRIMARY KEY, source_id INTEGER NOT NULL REFERENCES sources(id), body TEXT NOT NULL); CREATE TABLE embeddings(claim_id TEXT NOT NULL REFERENCES claims(id), space TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(claim_id,space)); CREATE TABLE relations(id TEXT PRIMARY KEY, a TEXT NOT NULL REFERENCES claims(id), b TEXT NOT NULL REFERENCES claims(id), body TEXT NOT NULL, ingested_at TEXT NOT NULL DEFAULT(strftime('%Y-%m-%dT%H:%M:%fZ','now'))); PRAGMA user_version=1;")?;
            tx.commit()?;
        }
        Ok(Self { conn })
    }
    pub fn ingest(&mut self, batch: &Ingest) -> Result<()> {
        bounded("raw text", &batch.raw_text, MAX_TEXT)?;
        batch.provenance.validate()?;
        ensure!(
            !batch.claims.is_empty() && batch.claims.len() <= 128,
            "invalid: expected 1..128 claims"
        );
        let mut ids = BTreeSet::new();
        for c in &batch.claims {
            c.validate()?;
            ensure!(ids.insert(&c.id), "invalid: duplicate claim id");
        }
        let mut embedded = BTreeSet::new();
        for e in &batch.embeddings {
            e.embedding.validate()?;
            ensure!(
                ids.contains(&e.claim_id),
                "invalid: unknown embedding claim id"
            );
            ensure!(
                embedded.insert((&e.claim_id, &e.embedding.space)),
                "invalid: duplicate embedding"
            );
        }
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO sources(raw_text,provenance) VALUES(?1,?2)",
            params![batch.raw_text, serde_json::to_string(&batch.provenance)?],
        )?;
        let source_id = tx.last_insert_rowid();
        for c in &batch.claims {
            tx.execute(
                "INSERT INTO claims(id,source_id,body) VALUES(?1,?2,?3)",
                params![c.id, source_id, serde_json::to_string(c)?],
            )?;
        }
        for e in &batch.embeddings {
            let dimensions: Option<String> = tx
                .query_row(
                    "SELECT body FROM embeddings WHERE space=?1 LIMIT 1",
                    [&e.embedding.space],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(v) = dimensions {
                let old: Embedding = serde_json::from_str(&v)?;
                ensure!(
                    old.vector.len() == e.embedding.vector.len(),
                    "invalid: embedding space dimension changed"
                );
            }
            tx.execute(
                "INSERT INTO embeddings(claim_id,space,body) VALUES(?1,?2,?3)",
                params![
                    e.claim_id,
                    e.embedding.space,
                    serde_json::to_string(&e.embedding)?
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn add_relation(&mut self, r: &Relation) -> Result<()> {
        r.validate()?;
        self.conn.execute(
            "INSERT INTO relations(id,a,b,body) VALUES(?1,?2,?3,?4)",
            params![r.id, r.a, r.b, serde_json::to_string(r)?],
        )?;
        Ok(())
    }
    pub fn claims(&self) -> Result<Vec<Claim>> {
        let claims: Vec<Claim> = self.json_rows("SELECT body FROM claims ORDER BY id")?;
        for claim in &claims {
            claim.validate()?;
        }
        Ok(claims)
    }
    pub fn relations(&self) -> Result<Vec<Relation>> {
        let relations: Vec<Relation> =
            self.json_rows("SELECT body FROM relations ORDER BY rowid")?;
        for relation in &relations {
            relation.validate()?;
        }
        Ok(relations)
    }
    fn json_rows<T: serde::de::DeserializeOwned>(&self, sql: &str) -> Result<Vec<T>> {
        let mut s = self.conn.prepare(sql)?;
        let rows = s.query_map([], |r| r.get::<_, String>(0))?;
        rows.map(|r| Ok(serde_json::from_str(&r?)?)).collect()
    }
    pub fn compare(&self, q: &Embedding, filters: &Filters) -> Result<Comparison> {
        let mut comparison = self.compare_all(q, filters)?;
        comparison.limit(5);
        Ok(comparison)
    }
    pub fn compare_all(&self, q: &Embedding, filters: &Filters) -> Result<Comparison> {
        q.validate()?;
        let claims = self.claims()?;
        let relations = self.relations()?;
        let related: BTreeSet<&str> = relations
            .iter()
            .filter(|r| {
                filters
                    .relationship
                    .is_none_or(|x| r.labels.relationship == x)
            })
            .flat_map(|r| [r.a.as_str(), r.b.as_str()])
            .collect();
        let mut s = self
            .conn
            .prepare("SELECT claim_id,body FROM embeddings WHERE space=?1")?;
        let rows = s.query_map([&q.space], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        let mut vector = Vec::new();
        for row in rows {
            let (id, body) = row?;
            let e: Embedding = serde_json::from_str(&body)?;
            e.validate()?;
            let claim = claims
                .iter()
                .find(|c| c.id == id)
                .context("invalid: orphan embedding")?
                .clone();
            vector.push(Hit {
                claim,
                score: cosine(&q.vector, &e.vector)?,
                embedding_provenance: e.provenance,
                decision: None,
            });
        }
        vector.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.claim.id.cmp(&b.claim.id))
        });
        ensure!(
            filters.at_or_before.is_none() || filters.timeline.is_some(),
            "invalid: story-time cutoff requires timeline"
        );
        let mut selected: Vec<Hit> = vector
            .iter()
            .filter(|h| {
                let c = &h.claim;
                filters
                    .perspective
                    .as_ref()
                    .is_none_or(|x| c.perspective == *x)
                    && filters
                        .continuity
                        .as_ref()
                        .is_none_or(|x| c.continuity == *x)
                    && (filters.relationship.is_none() || related.contains(c.id.as_str()))
                    && filters
                        .subject
                        .as_ref()
                        .is_none_or(|x| c.triple.subject == *x)
                    && filters
                        .predicate
                        .as_ref()
                        .is_none_or(|x| c.triple.predicate == *x)
                    && filters
                        .entity
                        .as_ref()
                        .is_none_or(|x| c.entities.contains(x))
                    && filters
                        .qualifier
                        .as_ref()
                        .is_none_or(|x| c.qualifiers.contains(x))
                    && filters.timeline.as_ref().is_none_or(|x| {
                        c.story_time.as_ref().is_some_and(|t| {
                            t.timeline == *x
                                && filters.at_or_before.is_none_or(|cut| t.position <= cut)
                        })
                    })
            })
            .cloned()
            .collect();
        if filters.latest {
            // Unknown times are excluded: there is no honest latest ordering for them.
            let pool = &claims;
            selected.retain(|h| {
                h.claim.story_time.as_ref().is_some_and(|t| {
                    !pool.iter().any(|other| {
                        let (a, b) = (&h.claim, other);
                        a.triple.subject == b.triple.subject
                            && a.triple.predicate == b.triple.predicate
                            && match (&a.state_key, &b.state_key) {
                                (Some(x), Some(y)) => x == y,
                                (None, None) => a.triple.object == b.triple.object,
                                _ => false,
                            }
                            && a.perspective == b.perspective
                            && a.continuity == b.continuity
                            && b.story_time.as_ref().is_some_and(|u| {
                                u.timeline == t.timeline
                                    && u.position > t.position
                                    && filters.at_or_before.is_none_or(|cut| u.position <= cut)
                            })
                    })
                })
            });
        }
        selected.retain(|h| {
            filters
                .object
                .as_ref()
                .is_none_or(|x| h.claim.triple.object == *x)
                && filters
                    .polarity
                    .is_none_or(|x| h.claim.triple.polarity == x)
        });
        if filters.chronological {
            ensure!(
                filters.timeline.is_some(),
                "invalid: chronological ordering requires an explicit timeline"
            );
            selected.retain(|h| h.claim.story_time.is_some());
            selected.sort_by(|a, b| {
                a.claim
                    .story_time
                    .as_ref()
                    .unwrap()
                    .position
                    .cmp(&b.claim.story_time.as_ref().unwrap().position)
                    .then(a.claim.id.cmp(&b.claim.id))
            });
        }
        let structured = selected;
        Ok(Comparison {
            vector,
            structured,
            reranked: None,
            rerank_status: "unavailable: decision adapter not negotiated".into(),
            relations,
        })
    }
}
use rusqlite::OptionalExtension;
