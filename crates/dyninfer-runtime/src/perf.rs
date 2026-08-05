//! Synthetic-token prefill/decode benchmarking.

use crate::perf_sample::{PhaseSampleStats, PhaseSampler, percentile_summary};
use crate::{AllocatorMetrics, CausalLanguageModel, KvCacheMetrics};
use dyninfer_core::{SessionConfig, TokenId};
use dyninfer_error::{DynInferError, Result};
use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PerfConfig {
    /// Synthetic prompt length in tokens.
    pub prefill_tokens: usize,
    /// Decode steps to measure.
    pub tg: usize,
    /// Discarded iterations before measurement.
    pub warmup: usize,
    /// Measured iterations (latency averaged; peaks maxed).
    pub iters: usize,
    /// Checkpoint weight bytes for effective bandwidth (0 = omit).
    pub weight_bytes: u64,
    /// Token id used to fill the synthetic prompt.
    pub fill_token: TokenId,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            prefill_tokens: 128,
            tg: 32,
            warmup: 1,
            iters: 1,
            weight_bytes: 0,
            fill_token: 1,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PhaseMetrics {
    pub tokens: usize,
    pub secs: f64,
    pub tps: f64,
    pub peak_rss_bytes: u64,
    pub cpu_avg_pct: Option<f64>,
    pub cpu_p50_pct: Option<f64>,
    pub cpu_p90_pct: Option<f64>,
    pub gpu_busy_avg_pct: Option<f64>,
    pub gpu_busy_p50_pct: Option<f64>,
    pub gpu_busy_p90_pct: Option<f64>,
    pub mem_busy_avg_pct: Option<f64>,
    pub mem_busy_p50_pct: Option<f64>,
    pub mem_busy_p90_pct: Option<f64>,
    pub peak_vram_bytes: Option<u64>,
    pub iree_host_bytes_peak: u64,
    pub iree_device_bytes_peak: u64,
    pub kv_page_count: usize,
    pub kv_allocated_bytes: usize,
    pub eff_weight_bw_gbs: Option<f64>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerfReport {
    pub prefill: PhaseMetrics,
    pub decode: PhaseMetrics,
    pub warmup: usize,
    pub iters: usize,
    pub weight_bytes: u64,
}

/// Run synthetic-token prefill + decode, collecting per-phase metrics.
pub fn run_perf(
    model: &dyn CausalLanguageModel,
    config: &PerfConfig,
    session_cfg: SessionConfig,
) -> Result<PerfReport> {
    if config.prefill_tokens == 0 {
        return Err(DynInferError::io("perf --prefill must be > 0"));
    }
    if config.iters == 0 {
        return Err(DynInferError::io("perf --iters must be > 0"));
    }

    let prompt: Vec<TokenId> = vec![config.fill_token; config.prefill_tokens];
    let mut session = model.create_session(session_cfg)?;

    for _ in 0..config.warmup {
        session.reset()?;
        let mut next = session.prefill_argmax(&prompt)?;
        for _ in 0..config.tg {
            next = session.decode_argmax(next)?;
        }
    }

    let mut prefills = Vec::with_capacity(config.iters);
    let mut decodes = Vec::with_capacity(config.iters);

    for _ in 0..config.iters {
        session.reset()?;

        let prefill = measure_phase(|| {
            let next = session.prefill_argmax(&prompt)?;
            Ok(next)
        })?;
        let mut next = prefill.result;
        let prefill_kv = session.kv_cache_metrics().unwrap_or_default();
        let prefill_alloc = session.allocator_metrics().unwrap_or_default();
        prefills.push(RawPhase {
            tokens: config.prefill_tokens,
            secs: prefill.secs,
            samples: prefill.samples,
            kv: prefill_kv,
            alloc: prefill_alloc,
            weight_bytes: config.weight_bytes,
        });

        let decode = measure_phase(|| {
            let mut tok = next;
            for _ in 0..config.tg {
                tok = session.decode_argmax(tok)?;
            }
            next = tok;
            Ok(())
        })?;
        let decode_kv = session.kv_cache_metrics().unwrap_or_default();
        let decode_alloc = session.allocator_metrics().unwrap_or_default();
        decodes.push(RawPhase {
            tokens: config.tg,
            secs: decode.secs,
            samples: decode.samples,
            kv: decode_kv,
            alloc: decode_alloc,
            weight_bytes: config.weight_bytes,
        });
    }

    Ok(PerfReport {
        prefill: aggregate_phases(&prefills),
        decode: aggregate_phases(&decodes),
        warmup: config.warmup,
        iters: config.iters,
        weight_bytes: config.weight_bytes,
    })
}

struct Timed<T> {
    result: T,
    secs: f64,
    samples: PhaseSampleStats,
}

fn measure_phase<T, F>(f: F) -> Result<Timed<T>>
where
    F: FnOnce() -> Result<T>,
{
    let sampler = PhaseSampler::start();
    let t0 = Instant::now();
    let result = f()?;
    let secs = t0.elapsed().as_secs_f64();
    let samples = sampler.stop();
    Ok(Timed {
        result,
        secs,
        samples,
    })
}

struct RawPhase {
    tokens: usize,
    secs: f64,
    samples: PhaseSampleStats,
    kv: KvCacheMetrics,
    alloc: AllocatorMetrics,
    weight_bytes: u64,
}

fn aggregate_phases(phases: &[RawPhase]) -> PhaseMetrics {
    if phases.is_empty() {
        return PhaseMetrics::default();
    }
    let tokens = phases[0].tokens;
    let mean_secs = phases.iter().map(|p| p.secs).sum::<f64>() / phases.len() as f64;
    let tps = if mean_secs > 0.0 {
        tokens as f64 / mean_secs
    } else {
        0.0
    };

    let mut peak_rss = 0u64;
    let mut peak_vram: Option<u64> = None;
    let mut sample_count = 0usize;
    let mut host_peak = 0u64;
    let mut device_peak = 0u64;
    let mut kv_pages = 0usize;
    let mut kv_bytes = 0usize;

    for p in phases {
        peak_rss = peak_rss.max(p.samples.peak_rss_bytes);
        peak_vram = match (peak_vram, p.samples.peak_vram_bytes) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (None, Some(b)) => Some(b),
            (a, None) => a,
        };
        sample_count += p.samples.sample_count;
        host_peak = host_peak.max(p.alloc.host_bytes_peak);
        device_peak = device_peak.max(p.alloc.device_bytes_peak);
        kv_pages = kv_pages.max(p.kv.page_count);
        kv_bytes = kv_bytes.max(p.kv.allocated_bytes);
    }

    // Single iter: use sampler percentiles. Multi-iter: percentile of per-iter averages.
    let (cpu_avg, cpu_p50, cpu_p90, gpu_avg, gpu_p50, gpu_p90, mem_avg, mem_p50, mem_p90) =
        if phases.len() == 1 {
            let p = &phases[0].samples;
            (
                p.cpu_avg_pct,
                p.cpu_p50_pct,
                p.cpu_p90_pct,
                p.gpu_busy_avg_pct,
                p.gpu_busy_p50_pct,
                p.gpu_busy_p90_pct,
                p.mem_busy_avg_pct,
                p.mem_busy_p50_pct,
                p.mem_busy_p90_pct,
            )
        } else {
            let cpu_vals: Vec<f64> = phases.iter().filter_map(|p| p.samples.cpu_avg_pct).collect();
            let gpu_vals: Vec<f64> = phases
                .iter()
                .filter_map(|p| p.samples.gpu_busy_avg_pct)
                .collect();
            let mem_vals: Vec<f64> = phases
                .iter()
                .filter_map(|p| p.samples.mem_busy_avg_pct)
                .collect();
            let (ca, c50, c90) = percentile_summary(&cpu_vals);
            let (ga, g50, g90) = percentile_summary(&gpu_vals);
            let (ma, m50, m90) = percentile_summary(&mem_vals);
            (ca, c50, c90, ga, g50, g90, ma, m50, m90)
        };

    let weight_bytes = phases[0].weight_bytes;
    let eff_weight_bw_gbs = if weight_bytes > 0 && mean_secs > 0.0 {
        Some((weight_bytes as f64 / mean_secs) / 1e9)
    } else {
        None
    };

    PhaseMetrics {
        tokens,
        secs: mean_secs,
        tps,
        peak_rss_bytes: peak_rss,
        cpu_avg_pct: cpu_avg,
        cpu_p50_pct: cpu_p50,
        cpu_p90_pct: cpu_p90,
        gpu_busy_avg_pct: gpu_avg,
        gpu_busy_p50_pct: gpu_p50,
        gpu_busy_p90_pct: gpu_p90,
        mem_busy_avg_pct: mem_avg,
        mem_busy_p50_pct: mem_p50,
        mem_busy_p90_pct: mem_p90,
        peak_vram_bytes: peak_vram,
        iree_host_bytes_peak: host_peak,
        iree_device_bytes_peak: device_peak,
        kv_page_count: kv_pages,
        kv_allocated_bytes: kv_bytes,
        eff_weight_bw_gbs,
        sample_count,
    }
}

/// Format a human-readable perf report.
pub fn format_perf_report(report: &PerfReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "warmup={}  iters={}  weight_bytes={}\n",
        report.warmup, report.iters, report.weight_bytes
    ));
    out.push_str(&format_phase("prefill", &report.prefill));
    out.push_str(&format_phase("decode", &report.decode));
    out
}

fn format_phase(name: &str, m: &PhaseMetrics) -> String {
    let mut s = format!(
        "{name:<8} {} tok  {:.3}s  {:.1} tok/s\n",
        m.tokens, m.secs, m.tps
    );
    s.push_str(&format!(
        "  peak RSS {:.1} MiB | IREE host peak {:.1} MiB | IREE device peak {:.1} MiB | KV {} pages ({:.1} MiB)\n",
        bytes_mib(m.peak_rss_bytes),
        bytes_mib(m.iree_host_bytes_peak),
        bytes_mib(m.iree_device_bytes_peak),
        m.kv_page_count,
        bytes_mib(m.kv_allocated_bytes as u64),
    ));
    s.push_str(&format!(
        "  CPU avg/p50/p90  {}\n",
        fmt_triplet(m.cpu_avg_pct, m.cpu_p50_pct, m.cpu_p90_pct)
    ));
    let mut gpu_line = format!(
        "  GPU busy avg/p50/p90  {}",
        fmt_triplet(m.gpu_busy_avg_pct, m.gpu_busy_p50_pct, m.gpu_busy_p90_pct)
    );
    if m.mem_busy_avg_pct.is_some() {
        gpu_line.push_str(&format!(
            " | mem busy {}",
            fmt_triplet(m.mem_busy_avg_pct, m.mem_busy_p50_pct, m.mem_busy_p90_pct)
        ));
    }
    if let Some(vram) = m.peak_vram_bytes {
        gpu_line.push_str(&format!(" | VRAM peak {:.1} MiB", bytes_mib(vram)));
    }
    gpu_line.push('\n');
    s.push_str(&gpu_line);
    if let Some(bw) = m.eff_weight_bw_gbs {
        s.push_str(&format!("  eff. weight BW  {bw:.2} GB/s\n"));
    }
    s
}

fn bytes_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn fmt_triplet(avg: Option<f64>, p50: Option<f64>, p90: Option<f64>) -> String {
    match (avg, p50, p90) {
        (Some(a), Some(b), Some(c)) => format!("{a:.1}/{b:.1}/{c:.1} %"),
        _ => "n/a".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_report_includes_phases() {
        let report = PerfReport {
            prefill: PhaseMetrics {
                tokens: 128,
                secs: 1.0,
                tps: 128.0,
                ..Default::default()
            },
            decode: PhaseMetrics {
                tokens: 32,
                secs: 0.5,
                tps: 64.0,
                ..Default::default()
            },
            warmup: 1,
            iters: 1,
            weight_bytes: 0,
        };
        let text = format_perf_report(&report);
        assert!(text.contains("prefill"));
        assert!(text.contains("decode"));
        assert!(text.contains("128 tok"));
    }
}
