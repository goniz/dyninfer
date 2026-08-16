//! Numerically stable, per-row logit drift metrics.

use anyhow::{Context, Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TokenLogit {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenDelta {
    pub token_id: u32,
    pub reference: f32,
    pub dyninfer: f32,
    pub delta: f64,
    pub absolute_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RowMetrics {
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub rmse: f64,
    pub mean_signed_delta: f64,
    pub relative_l2_error: f64,
    pub cosine_similarity: f64,
    pub centered_rmse: f64,
    pub reference_argmax: u32,
    pub dyninfer_argmax: u32,
    pub argmax_match: bool,
    pub top_k: usize,
    pub top_k_overlap: usize,
    pub top_k_overlap_fraction: f64,
    pub softmax_kl_reference_dyninfer: f64,
    pub max_probability_delta: f64,
    pub largest_deltas: Vec<TokenDelta>,
    pub reference_top: Vec<TokenLogit>,
    pub dyninfer_top: Vec<TokenLogit>,
}

pub fn compare(reference: &[f32], dyninfer: &[f32], requested_top_k: usize) -> Result<RowMetrics> {
    if reference.is_empty() {
        bail!("cannot compare empty logit rows");
    }
    if reference.len() != dyninfer.len() {
        bail!(
            "logit row width mismatch: reference {}, dyninfer {}",
            reference.len(),
            dyninfer.len()
        );
    }
    if requested_top_k == 0 {
        bail!("top-k must be greater than zero");
    }
    for (engine, values) in [("reference", reference), ("dyninfer", dyninfer)] {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            bail!("{engine} logit {index} is non-finite: {value}");
        }
    }

    let n = reference.len() as f64;
    let mut max_absolute_error = 0.0f64;
    let mut sum_absolute_error = 0.0f64;
    let mut sum_squared_error = 0.0f64;
    let mut sum_delta = 0.0f64;
    let mut reference_squared_norm = 0.0f64;
    let mut dyninfer_squared_norm = 0.0f64;
    let mut dot = 0.0f64;

    for (&reference, &dyninfer) in reference.iter().zip(dyninfer) {
        let reference = f64::from(reference);
        let dyninfer = f64::from(dyninfer);
        let delta = dyninfer - reference;
        let absolute = delta.abs();
        max_absolute_error = max_absolute_error.max(absolute);
        sum_absolute_error += absolute;
        sum_squared_error += delta * delta;
        sum_delta += delta;
        reference_squared_norm += reference * reference;
        dyninfer_squared_norm += dyninfer * dyninfer;
        dot += reference * dyninfer;
    }

    let mean_signed_delta = sum_delta / n;
    let centered_squared_error = reference
        .iter()
        .zip(dyninfer)
        .map(|(&reference, &dyninfer)| {
            let centered_delta = f64::from(dyninfer) - f64::from(reference) - mean_signed_delta;
            centered_delta * centered_delta
        })
        .sum::<f64>();
    let reference_norm = reference_squared_norm.sqrt();
    let dyninfer_norm = dyninfer_squared_norm.sqrt();
    let cosine_similarity = if reference_norm == 0.0 || dyninfer_norm == 0.0 {
        if reference_norm == 0.0 && dyninfer_norm == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (dot / (reference_norm * dyninfer_norm)).clamp(-1.0, 1.0)
    };

    let top_k = requested_top_k.min(reference.len());
    let reference_indices = descending_indices(reference);
    let dyninfer_indices = descending_indices(dyninfer);
    let reference_argmax = u32::try_from(reference_indices[0]).context("vocabulary exceeds u32")?;
    let dyninfer_argmax = u32::try_from(dyninfer_indices[0]).context("vocabulary exceeds u32")?;
    let mut reference_in_top_k = vec![false; reference.len()];
    for &index in &reference_indices[..top_k] {
        reference_in_top_k[index] = true;
    }
    let top_k_overlap = dyninfer_indices[..top_k]
        .iter()
        .filter(|&&index| reference_in_top_k[index])
        .count();

    let reference_logsumexp = logsumexp(reference);
    let dyninfer_logsumexp = logsumexp(dyninfer);
    let mut kl = 0.0f64;
    let mut max_probability_delta = 0.0f64;
    for (&reference, &dyninfer) in reference.iter().zip(dyninfer) {
        let reference_log_probability = f64::from(reference) - reference_logsumexp;
        let dyninfer_log_probability = f64::from(dyninfer) - dyninfer_logsumexp;
        let reference_probability = reference_log_probability.exp();
        let dyninfer_probability = dyninfer_log_probability.exp();
        kl += reference_probability * (reference_log_probability - dyninfer_log_probability);
        max_probability_delta =
            max_probability_delta.max((reference_probability - dyninfer_probability).abs());
    }
    // Roundoff can make an otherwise-zero KL a few ulps negative.
    if kl < 0.0 && kl > -1.0e-12 {
        kl = 0.0;
    }

    let mut delta_indices = (0..reference.len()).collect::<Vec<_>>();
    delta_indices.sort_unstable_by(|&left, &right| {
        let left_delta = (f64::from(dyninfer[left]) - f64::from(reference[left])).abs();
        let right_delta = (f64::from(dyninfer[right]) - f64::from(reference[right])).abs();
        right_delta
            .total_cmp(&left_delta)
            .then_with(|| left.cmp(&right))
    });

    Ok(RowMetrics {
        max_absolute_error,
        mean_absolute_error: sum_absolute_error / n,
        rmse: (sum_squared_error / n).sqrt(),
        mean_signed_delta,
        relative_l2_error: sum_squared_error.sqrt() / reference_norm.max(f64::EPSILON),
        cosine_similarity,
        centered_rmse: (centered_squared_error / n).sqrt(),
        reference_argmax,
        dyninfer_argmax,
        argmax_match: reference_argmax == dyninfer_argmax,
        top_k,
        top_k_overlap,
        top_k_overlap_fraction: top_k_overlap as f64 / top_k as f64,
        softmax_kl_reference_dyninfer: kl,
        max_probability_delta,
        largest_deltas: delta_indices[..top_k]
            .iter()
            .map(|&index| {
                let delta = f64::from(dyninfer[index]) - f64::from(reference[index]);
                TokenDelta {
                    token_id: index as u32,
                    reference: reference[index],
                    dyninfer: dyninfer[index],
                    delta,
                    absolute_delta: delta.abs(),
                }
            })
            .collect(),
        reference_top: token_logits(reference, &reference_indices[..top_k]),
        dyninfer_top: token_logits(dyninfer, &dyninfer_indices[..top_k]),
    })
}

fn descending_indices(values: &[f32]) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        values[right]
            .total_cmp(&values[left])
            .then_with(|| left.cmp(&right))
    });
    indices
}

fn token_logits(values: &[f32], indices: &[usize]) -> Vec<TokenLogit> {
    indices
        .iter()
        .map(|&index| TokenLogit {
            token_id: index as u32,
            logit: values[index],
        })
        .collect()
}

fn logsumexp(values: &[f32]) -> f64 {
    let maximum = values
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    maximum
        + values
            .iter()
            .map(|&value| (f64::from(value) - maximum).exp())
            .sum::<f64>()
            .ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_zero_drift() {
        let metrics = compare(&[1.0, 3.0, -2.0], &[1.0, 3.0, -2.0], 2).unwrap();
        assert_eq!(metrics.max_absolute_error, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert_eq!(metrics.centered_rmse, 0.0);
        assert_eq!(metrics.cosine_similarity, 1.0);
        assert_eq!(metrics.softmax_kl_reference_dyninfer, 0.0);
        assert!(metrics.argmax_match);
        assert_eq!(metrics.top_k_overlap, 2);
    }

    #[test]
    fn constant_offset_is_removed_by_centering() {
        let metrics = compare(&[-1.0, 0.0, 2.0], &[4.0, 5.0, 7.0], 3).unwrap();
        assert!((metrics.mean_signed_delta - 5.0).abs() < 1.0e-12);
        assert!(metrics.centered_rmse < 1.0e-12);
        assert!(metrics.softmax_kl_reference_dyninfer < 1.0e-12);
        assert!(metrics.argmax_match);
    }

    #[test]
    fn detects_scale_and_argmax_changes() {
        let scaled = compare(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], 2).unwrap();
        assert!((scaled.relative_l2_error - 1.0).abs() < 1.0e-12);
        let changed = compare(&[3.0, 2.0], &[1.0, 4.0], 1).unwrap();
        assert!(!changed.argmax_match);
        assert_eq!(changed.reference_argmax, 0);
        assert_eq!(changed.dyninfer_argmax, 1);
    }

    #[test]
    fn stable_softmax_handles_large_logits() {
        let metrics = compare(&[3.0e38, 3.0e38 - 1.0e32], &[3.0e38, 3.0e38], 2).unwrap();
        assert!(metrics.softmax_kl_reference_dyninfer.is_finite());
        assert!(metrics.max_probability_delta.is_finite());
    }

    #[test]
    fn rejects_non_finite_values() {
        let error = compare(&[0.0, f32::NAN], &[0.0, 1.0], 1).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
    }
}
