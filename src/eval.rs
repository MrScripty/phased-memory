//! Scorers own gold labels; runners project only input data into inference.
use crate::{domain::*, pumas::Pumas, store::Filters};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairCase {
    pub id: String,
    pub a: Claim,
    pub b: Claim,
    pub expected: Labels,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeProbabilities {
    #[serde(deserialize_with = "probability_map")]
    pub values: BTreeMap<Relationship, f64>,
    pub provenance: Provenance,
    pub native_evidence: String,
}
fn probability_map<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<BTreeMap<Relationship, f64>, D::Error> {
    struct MapVisitor;
    impl<'de> serde::de::Visitor<'de> for MapVisitor {
        type Value = BTreeMap<Relationship, f64>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("distinct relationship probability keys")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = map.next_entry()? {
                if values.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom("duplicate probability key"));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(MapVisitor)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairPrediction {
    pub id: String,
    pub labels: Option<Labels>,
    pub error: Option<String>,
    pub provenance: Option<Provenance>,
    pub native_probabilities: Option<NativeProbabilities>,
}
#[derive(Debug, Serialize)]
pub struct PairScore {
    pub total: usize,
    pub completed: usize,
    pub relationship_accuracy: f64,
    pub relationship_macro_f1: f64,
    pub temporal_accuracy: f64,
    pub update_accuracy: f64,
    pub joint_accuracy: f64,
    pub confusion: BTreeMap<String, BTreeMap<String, usize>>,
    pub brier: Option<f64>,
    pub brier_covered: usize,
    pub brier_status: String,
}
fn unique<'a>(ids: impl Iterator<Item = &'a str>) -> Result<BTreeSet<&'a str>> {
    let mut set = BTreeSet::new();
    for id in ids {
        bounded("evaluation id", id, 256)?;
        ensure!(set.insert(id), "invalid: duplicate evaluation id {id}");
    }
    ensure!(!set.is_empty(), "invalid: empty evaluation corpus");
    Ok(set)
}
pub fn score_pairs(cases: &[PairCase], predictions: &[PairPrediction]) -> Result<PairScore> {
    let ids = unique(cases.iter().map(|c| c.id.as_str()))?;
    let mut seen = BTreeSet::new();
    for p in predictions {
        ensure!(
            ids.contains(p.id.as_str()),
            "invalid: unknown prediction id {}",
            p.id
        );
        ensure!(
            seen.insert(&p.id),
            "invalid: duplicate prediction id {}",
            p.id
        );
        ensure!(
            p.labels.is_some() != p.error.is_some(),
            "invalid: prediction must contain exactly labels or error"
        );
    }
    let (mut correct, mut temporal, mut update, mut joint, mut completed) = (0, 0, 0, 0, 0);
    let mut confusion: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut brier_sum = 0.0;
    let mut brier_count = 0;
    for c in cases {
        c.a.validate()?;
        c.b.validate()?;
        let p = predictions.iter().find(|p| p.id == c.id);
        let label = p.and_then(|p| p.labels.as_ref());
        if let Some(l) = label {
            completed += 1;
            correct += usize::from(l.relationship == c.expected.relationship);
            temporal += usize::from(l.temporal_order == c.expected.temporal_order);
            update += usize::from(l.update_type == c.expected.update_type);
            joint += usize::from(*l == c.expected);
        }
        let expected = serde_json::to_value(c.expected.relationship)?
            .as_str()
            .unwrap()
            .to_owned();
        let predicted = label
            .map(|l| {
                serde_json::to_value(l.relationship)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .unwrap_or_else(|| "missing".into());
        *confusion
            .entry(expected)
            .or_default()
            .entry(predicted)
            .or_default() += 1;
        if let Some(prob) = p.and_then(|p| p.native_probabilities.as_ref()) {
            ensure!(label.is_some(), "invalid: probabilities without prediction");
            ensure!(
                matches!(prob.provenance, Provenance::Pumas { .. }),
                "invalid: native probability provenance must be Pumas"
            );
            prob.provenance.validate()?;
            bounded("native probability evidence", &prob.native_evidence, 4096)?;
            ensure!(
                prob.values.len() == 4
                    && Relationship::ALL
                        .iter()
                        .all(|k| prob.values.contains_key(k)),
                "invalid: complete native probability vector required"
            );
            ensure!(
                prob.values
                    .values()
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
                    && (prob.values.values().sum::<f64>() - 1.0).abs() < 1e-6,
                "invalid: probability range or sum"
            );
            brier_sum += Relationship::ALL
                .iter()
                .map(|k| (prob.values[k] - f64::from(*k == c.expected.relationship)).powi(2))
                .sum::<f64>();
            brier_count += 1;
        }
    }
    let mut f1 = 0.0;
    for k in Relationship::ALL {
        let key = serde_json::to_value(k)?.as_str().unwrap().to_owned();
        let tp = confusion
            .get(&key)
            .and_then(|r| r.get(&key))
            .copied()
            .unwrap_or(0) as f64;
        let fp = confusion
            .iter()
            .filter(|(e, _)| **e != key)
            .map(|(_, r)| r.get(&key).copied().unwrap_or(0))
            .sum::<usize>() as f64;
        let fn_ = confusion
            .get(&key)
            .map(|r| {
                r.iter()
                    .filter(|(p, _)| **p != key)
                    .map(|(_, n)| *n)
                    .sum::<usize>()
            })
            .unwrap_or(0) as f64;
        let denom = 2.0 * tp + fp + fn_;
        if denom > 0.0 {
            f1 += 2.0 * tp / denom;
        }
    }
    let n = cases.len() as f64;
    Ok(PairScore{total:cases.len(),completed,relationship_accuracy:correct as f64/n,relationship_macro_f1:f1/4.0,temporal_accuracy:temporal as f64/n,update_accuracy:update as f64/n,joint_accuracy:joint as f64/n,confusion,brier:if brier_count==cases.len(){Some(brier_sum/n)}else{None},brier_covered:brier_count,brier_status:if brier_count==cases.len(){"complete native probabilities"}else{"unavailable: every scored case requires evidenced native probabilities; generated confidence is not probability"}.into()})
}
pub async fn run_pairs(pumas: &Pumas, cases: &[PairCase]) -> Result<Vec<PairPrediction>> {
    unique(cases.iter().map(|c| c.id.as_str()))?;
    for c in cases {
        c.a.validate()?;
        c.b.validate()?;
    }
    let mut out = Vec::new();
    for c in cases {
        // Neither expected labels nor case ID is passed to the inference adapter.
        let prediction = match pumas.decide(&c.a, &c.b).await {
            Ok(o) => PairPrediction {
                id: c.id.clone(),
                labels: Some(o.value),
                error: None,
                provenance: Some(o.provenance),
                native_probabilities: None,
            },
            Err(e) => PairPrediction {
                id: c.id.clone(),
                labels: None,
                error: Some(e.to_string()),
                provenance: None,
                native_probabilities: None,
            },
        };
        out.push(prediction);
    }
    Ok(out)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawPairCase {
    pub id: String,
    pub a_raw: String,
    pub b_raw: String,
    pub expected: Labels,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionRecord {
    pub id: String,
    pub a: Option<Vec<Claim>>,
    pub b: Option<Vec<Claim>>,
    pub a_error: Option<String>,
    pub b_error: Option<String>,
    pub a_provenance: Option<Provenance>,
    pub b_provenance: Option<Provenance>,
    pub decision: PairPrediction,
}
#[derive(Debug, Serialize)]
pub struct ExtractionScore {
    pub total: usize,
    pub extraction_completed: usize,
    pub extraction_failures: usize,
    pub extracted_multi_claim_pairs: usize,
    pub pair_score: PairScore,
}
pub async fn run_extraction(pumas: &Pumas, cases: &[RawPairCase]) -> Result<Vec<ExtractionRecord>> {
    unique(cases.iter().map(|c| c.id.as_str()))?;
    let mut out = Vec::new();
    for c in cases {
        let (a, a_error, a_provenance) = match pumas.extract(&c.a_raw).await {
            Ok(o) => (Some(o.value), None, Some(o.provenance)),
            Err(e) => (None, Some(e.to_string()), None),
        };
        let (b, b_error, b_provenance) = match pumas.extract(&c.b_raw).await {
            Ok(o) => (Some(o.value), None, Some(o.provenance)),
            Err(e) => (None, Some(e.to_string()), None),
        };
        // Multi-claim alignment requires explicit annotation, never silently select first.
        let decision = if let (Some(a), Some(b)) = (&a, &b) {
            if a.len() == 1 && b.len() == 1 {
                match pumas.decide(&a[0], &b[0]).await {
                    Ok(o) => PairPrediction {
                        id: c.id.clone(),
                        labels: Some(o.value),
                        error: None,
                        provenance: Some(o.provenance),
                        native_probabilities: None,
                    },
                    Err(e) => failure(&c.id, e.to_string()),
                }
            } else {
                failure(
                    &c.id,
                    "extraction_alignment_unavailable: expected exactly one claim on each side"
                        .into(),
                )
            }
        } else {
            failure(&c.id, "extraction_failure: decision not attempted".into())
        };
        out.push(ExtractionRecord {
            id: c.id.clone(),
            a,
            b,
            a_error,
            b_error,
            a_provenance,
            b_provenance,
            decision,
        });
    }
    Ok(out)
}
fn failure(id: &str, error: String) -> PairPrediction {
    PairPrediction {
        id: id.into(),
        labels: None,
        error: Some(error),
        provenance: None,
        native_probabilities: None,
    }
}
pub fn score_extraction(
    cases: &[RawPairCase],
    records: &[ExtractionRecord],
) -> Result<ExtractionScore> {
    let ids = unique(cases.iter().map(|c| c.id.as_str()))?;
    let mut seen = BTreeSet::new();
    for r in records {
        ensure!(
            ids.contains(r.id.as_str()) && seen.insert(&r.id) && r.decision.id == r.id,
            "invalid: extraction record id"
        );
        ensure!(
            r.decision.labels.is_none()
                || (r.a.as_ref().is_some_and(|v| v.len() == 1)
                    && r.b.as_ref().is_some_and(|v| v.len() == 1)),
            "invalid: decision success without aligned extraction"
        );
        for (claims, error) in [(&r.a, &r.a_error), (&r.b, &r.b_error)] {
            ensure!(
                claims.is_some() != error.is_some(),
                "invalid: extraction side outcome"
            );
            if let Some(cs) = claims {
                ensure!(!cs.is_empty(), "invalid: empty extraction");
                for c in cs {
                    c.validate()?;
                }
            }
        }
    }
    // Scoring does not need claims; a valid placeholder carries no semantic oracle.
    let placeholder = Claim {
        id: "scoring-only".into(),
        normalized: "scoring-only".into(),
        model_local_id: None,
        state_key: None,
        triple: Triple {
            subject: "unknown".into(),
            predicate: "unknown".into(),
            object: "unknown".into(),
            polarity: Polarity::Unknown,
        },
        entities: vec![],
        source: "scoring-only".into(),
        perspective: "unknown".into(),
        continuity: "unknown".into(),
        story_time: None,
        qualifiers: vec![],
    };
    let normalized: Vec<_> = cases
        .iter()
        .map(|c| PairCase {
            id: c.id.clone(),
            a: placeholder.clone(),
            b: placeholder.clone(),
            expected: c.expected.clone(),
        })
        .collect();
    let completed = records
        .iter()
        .filter(|r| r.a.is_some() && r.b.is_some())
        .count();
    let predictions: Vec<_> = records.iter().map(|r| r.decision.clone()).collect();
    Ok(ExtractionScore {
        total: cases.len(),
        extraction_completed: completed,
        extraction_failures: cases.len() - completed,
        extracted_multi_claim_pairs: records
            .iter()
            .filter(|r| {
                r.a.as_ref().is_some_and(|a| a.len() != 1)
                    || r.b.as_ref().is_some_and(|b| b.len() != 1)
            })
            .count(),
        pair_score: score_pairs(&normalized, &predictions)?,
    })
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalCase {
    pub id: String,
    pub query_text: String,
    pub filters: Filters,
    pub expected_ids: Vec<String>,
    pub misleading_ids: Vec<String>,
    pub chronology: Vec<String>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalPrediction {
    pub id: String,
    pub ranked_ids: Option<Vec<String>>,
    pub error: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct RetrievalScore {
    pub total: usize,
    pub completed: usize,
    pub top_accuracy: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub misleading_at_5: f64,
    pub chronology_accuracy: Option<f64>,
    pub chronology_cases: usize,
}
pub fn score_retrieval(
    cases: &[RetrievalCase],
    predictions: &[RetrievalPrediction],
    corpus_ids: &BTreeSet<String>,
) -> Result<RetrievalScore> {
    let ids = unique(cases.iter().map(|c| c.id.as_str()))?;
    let mut seen = BTreeSet::new();
    for p in predictions {
        ensure!(
            ids.contains(p.id.as_str()) && seen.insert(&p.id),
            "invalid: duplicate or unknown retrieval prediction id"
        );
        ensure!(
            p.ranked_ids.is_some() != p.error.is_some(),
            "invalid: retrieval outcome"
        );
        if let Some(ranked) = &p.ranked_ids {
            let mut seen = BTreeSet::new();
            for id in ranked {
                ensure!(
                    corpus_ids.contains(id) && seen.insert(id),
                    "invalid: unknown or duplicate ranked memory id"
                );
            }
        }
    }
    let (mut top, mut mrr, mut recall, mut misleading, mut chrono) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut completed, mut chronology_cases) = (0, 0);
    for c in cases {
        ensure!(!c.expected_ids.is_empty(), "invalid: empty relevance gold");
        for list in [&c.expected_ids, &c.misleading_ids, &c.chronology] {
            let mut seen = BTreeSet::new();
            for id in list {
                ensure!(
                    corpus_ids.contains(id) && seen.insert(id),
                    "invalid: unknown or duplicate gold memory id"
                );
            }
        }
        ensure!(
            !c.expected_ids.iter().any(|i| c.misleading_ids.contains(i)),
            "invalid: relevant/misleading overlap"
        );
        if c.chronology.len() >= 2 {
            chronology_cases += 1;
        }
        let ranked = predictions
            .iter()
            .find(|p| p.id == c.id)
            .and_then(|p| p.ranked_ids.as_ref());
        if ranked.is_none() {
            misleading += 1.0;
        }
        if let Some(ranked) = ranked {
            completed += 1;
            top += f64::from(ranked.first().is_some_and(|i| c.expected_ids.contains(i)));
            if let Some(rank) = ranked.iter().position(|i| c.expected_ids.contains(i)) {
                mrr += 1.0 / (rank + 1) as f64;
            }
            recall += ranked
                .iter()
                .take(5)
                .filter(|id| c.expected_ids.contains(id))
                .count() as f64
                / c.expected_ids.len() as f64;
            // Fixed five slots: empty/missing slots cannot improve this metric's denominator.
            misleading += ranked
                .iter()
                .take(5)
                .filter(|id| c.misleading_ids.contains(id))
                .count() as f64
                / 5.0;
            if c.chronology.len() >= 2 {
                let positions: Option<Vec<usize>> = c
                    .chronology
                    .iter()
                    .map(|id| ranked.iter().position(|x| x == id))
                    .collect();
                chrono += f64::from(positions.is_some_and(|ps| ps.windows(2).all(|w| w[0] < w[1])));
            }
        }
    }
    let n = cases.len() as f64;
    Ok(RetrievalScore {
        total: cases.len(),
        completed,
        top_accuracy: top / n,
        mrr: mrr / n,
        recall_at_5: recall / n,
        misleading_at_5: misleading / n,
        chronology_accuracy: if chronology_cases > 0 {
            Some(chrono / chronology_cases as f64)
        } else {
            None
        },
        chronology_cases,
    })
}
