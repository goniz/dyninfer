//! Kernel candidate descriptors and cost-model policy.
//!
//! Does not generate device code; selection feeds the compiler pipeline.

#![forbid(unsafe_code)]

use dyninfer_core::TargetProfile;
use dyninfer_error::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelCandidateDescriptor {
    pub id: String,
    pub op: String,
    pub encoding: String,
    pub backends: Vec<String>,
    pub priority: u32,
    pub notes: String,
}

pub trait KernelCostModel: Send + Sync {
    fn score(
        &self,
        candidate: &KernelCandidateDescriptor,
        target: &TargetProfile,
    ) -> Result<i64>;
}

#[derive(Debug, Default)]
pub struct StaticPriorityCostModel;

impl KernelCostModel for StaticPriorityCostModel {
    fn score(
        &self,
        candidate: &KernelCandidateDescriptor,
        target: &TargetProfile,
    ) -> Result<i64> {
        let backend_bonus = if candidate.backends.iter().any(|b| b == &target.driver || b == "any")
        {
            1000
        } else {
            0
        };
        Ok(backend_bonus + candidate.priority as i64)
    }
}

#[derive(Debug, Default)]
pub struct KernelRegistry {
    candidates: Vec<KernelCandidateDescriptor>,
}

impl KernelRegistry {
    pub fn version_1() -> Self {
        let mut reg = Self::default();
        reg.register(KernelCandidateDescriptor {
            id: "dense.matmul.linalg".into(),
            op: "linear".into(),
            encoding: "plain".into(),
            backends: vec!["any".into()],
            priority: 100,
            notes: "Generic linalg matmul for dense weights".into(),
        });
        reg.register(KernelCandidateDescriptor {
            id: "q4_0.matmul.generated".into(),
            op: "linear".into(),
            encoding: "gguf.q4_0".into(),
            backends: vec!["any".into()],
            priority: 90,
            notes: "Portable generated Q4_0 linear kernel".into(),
        });
        reg.register(KernelCandidateDescriptor {
            id: "attention.decode.gqa".into(),
            op: "attention".into(),
            encoding: "plain".into(),
            backends: vec!["any".into()],
            priority: 80,
            notes: "Decode attention with GQA-aware KV access".into(),
        });
        reg
    }

    pub fn register(&mut self, candidate: KernelCandidateDescriptor) {
        self.candidates.push(candidate);
    }

    pub fn select(
        &self,
        op: &str,
        encoding: &str,
        target: &TargetProfile,
        cost: &dyn KernelCostModel,
    ) -> Result<Option<KernelCandidateDescriptor>> {
        let mut best: Option<(i64, &KernelCandidateDescriptor)> = None;
        for c in &self.candidates {
            if c.op != op || c.encoding != encoding {
                continue;
            }
            let score = cost.score(c, target)?;
            if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                best = Some((score, c));
            }
        }
        Ok(best.map(|(_, c)| c.clone()))
    }

    pub fn candidates(&self) -> &[KernelCandidateDescriptor] {
        &self.candidates
    }
}
