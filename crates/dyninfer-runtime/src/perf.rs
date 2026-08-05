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
    /// Device-wide dedicated VRAM peak from sysfs (not process-local / not IREE).
    pub peak_dedicated_vram_bytes: Option<u64>,
    /// Device-wide GTT / unified peak from sysfs (not process-local / not IREE).
    pub peak_gtt_bytes: Option<u64>,
    /// IREE HAL live host bytes at end of phase (`allocated - freed`).
    pub iree_host_live_bytes: u64,
    /// IREE HAL live device bytes at end of phase (`allocated - freed`).
    pub iree_device_live_bytes: u64,
    /// IREE HAL lifetime host peak (allocator creation → now; not phase-scoped).
    pub iree_host_bytes_peak: u64,
    /// IREE HAL lifetime device peak (allocator creation → now; not phase-scoped).
    pub iree_device_bytes_peak: u64,
    pub kv_page_count: usize,
    pub kv_allocated_bytes: usize,
    pub kv_capacity_bytes: u64,
    pub kv_used_bytes: u64,
    pub kv_filled_tokens: u64,
    pub kv_layers: u32,
    pub kv_heads: u32,
    pub kv_head_dim: u32,
    pub kv_max_seq: u32,
    pub kv_key_dtype: String,
    pub kv_value_dtype: String,
    pub kv_paged: bool,
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
            // Prefill streams weights once per measured phase.
            weight_passes: 1,
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
            // Each decode step streams the full weight tensor once.
            weight_passes: config.tg as u64,
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
    let result = f();
    let secs = t0.elapsed().as_secs_f64();
    // Always stop before propagating errors so the sampler thread cannot leak.
    let samples = sampler.stop();
    Ok(Timed {
        result: result?,
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
    /// How many full weight-tensor streams this phase performs.
    weight_passes: u64,
    weight_bytes: u64,
}

fn max_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (None, Some(y)) => Some(y),
        (x, None) => x,
    }
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
    let mut peak_dedicated: Option<u64> = None;
    let mut peak_gtt: Option<u64> = None;
    let mut sample_count = 0usize;
    let mut host_peak = 0u64;
    let mut device_peak = 0u64;
    let mut host_live = 0u64;
    let mut device_live = 0u64;
    let mut kv_pages = 0usize;
    let mut kv_alloc = 0usize;
    let mut kv_capacity = 0u64;
    let mut kv_used = 0u64;
    let mut kv_filled = 0u64;

    for p in phases {
        peak_rss = peak_rss.max(p.samples.peak_rss_bytes);
        peak_dedicated = max_opt(peak_dedicated, p.samples.peak_dedicated_vram_bytes);
        peak_gtt = max_opt(peak_gtt, p.samples.peak_gtt_bytes);
        sample_count += p.samples.sample_count;
        host_peak = host_peak.max(p.alloc.host_bytes_peak);
        device_peak = device_peak.max(p.alloc.device_bytes_peak);
        host_live = host_live.max(p.alloc.host_live_bytes());
        device_live = device_live.max(p.alloc.device_live_bytes());
        kv_pages = kv_pages.max(p.kv.page_count);
        kv_alloc = kv_alloc.max(p.kv.allocated_bytes);
        kv_capacity = kv_capacity.max(p.kv.capacity_bytes as u64);
        kv_used = kv_used.max(p.kv.used_bytes as u64);
        kv_filled = kv_filled.max(p.kv.filled_tokens);
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
    let weight_passes = phases[0].weight_passes;
    let eff_weight_bw_gbs = if weight_bytes > 0 && weight_passes > 0 && mean_secs > 0.0 {
        let traffic = (weight_bytes as f64) * (weight_passes as f64);
        Some((traffic / mean_secs) / 1e9)
    } else {
        None
    };

    let kv0 = &phases[0].kv;
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
        peak_dedicated_vram_bytes: peak_dedicated,
        peak_gtt_bytes: peak_gtt,
        iree_host_live_bytes: host_live,
        iree_device_live_bytes: device_live,
        iree_host_bytes_peak: host_peak,
        iree_device_bytes_peak: device_peak,
        kv_page_count: kv_pages,
        kv_allocated_bytes: kv_alloc,
        kv_capacity_bytes: kv_capacity,
        kv_used_bytes: kv_used,
        kv_filled_tokens: kv_filled,
        kv_layers: kv0.layers,
        kv_heads: kv0.kv_heads,
        kv_head_dim: kv0.head_dim,
        kv_max_seq: kv0.max_sequence_length,
        kv_key_dtype: kv0.key_dtype.to_string(),
        kv_value_dtype: kv0.value_dtype.to_string(),
        kv_paged: kv0.paged,
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
        "  RSS peak {:.1} MiB | IREE host live {:.1} MiB (lifetime peak {:.1}) | IREE device live {:.1} MiB (lifetime peak {:.1})\n",
        bytes_mib(m.peak_rss_bytes),
        bytes_mib(m.iree_host_live_bytes),
        bytes_mib(m.iree_host_bytes_peak),
        bytes_mib(m.iree_device_live_bytes),
        bytes_mib(m.iree_device_bytes_peak),
    ));
    s.push_str(&format_kv_line(m));
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
    gpu_line.push('\n');
    s.push_str(&gpu_line);
    if m.peak_dedicated_vram_bytes.is_some() || m.peak_gtt_bytes.is_some() {
        let mut mem = String::from("  GPU mem (device-wide sysfs)");
        if let Some(v) = m.peak_dedicated_vram_bytes {
            mem.push_str(&format!("  dedicated VRAM {:.1} MiB", bytes_mib(v)));
        }
        if let Some(g) = m.peak_gtt_bytes {
            mem.push_str(&format!("  GTT/unified {:.1} MiB", bytes_mib(g)));
        }
        mem.push_str("  [not process-local]\n");
        s.push_str(&mem);
    }
    if let Some(bw) = m.eff_weight_bw_gbs {
        s.push_str(&format!("  eff. weight BW  {bw:.2} GB/s\n"));
    }
    s
}

fn format_kv_line(m: &PhaseMetrics) -> String {
    if m.kv_layers == 0 && m.kv_capacity_bytes == 0 {
        return "  KV cache  n/a\n".into();
    }
    let storage = if m.kv_paged { "paged" } else { "static (preallocated)" };
    let mut line = format!(
        "  KV cache  capacity {:.2} MiB  used {:.2} MiB ({}/{} tok)  K={} V={}  {}×{} heads×{} dim×{} max_seq  [{storage}]",
        bytes_mib(m.kv_capacity_bytes),
        bytes_mib(m.kv_used_bytes),
        m.kv_filled_tokens,
        m.kv_max_seq,
        m.kv_key_dtype,
        m.kv_value_dtype,
        m.kv_layers,
        m.kv_heads,
        m.kv_head_dim,
        m.kv_max_seq,
    );
    if m.kv_paged {
        line.push_str(&format!(
            "  pages={} ({:.2} MiB alloc)",
            m.kv_page_count,
            bytes_mib(m.kv_allocated_bytes as u64)
        ));
    }
    line.push('\n');
    line
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
    use dyninfer_core::ScalarType;

    fn raw_phase(tokens: usize, secs: f64, weight_passes: u64, weight_bytes: u64) -> RawPhase {
        RawPhase {
            tokens,
            secs,
            samples: PhaseSampleStats::default(),
            kv: KvCacheMetrics::default(),
            alloc: AllocatorMetrics::default(),
            weight_passes,
            weight_bytes,
        }
    }

    #[test]
    fn format_report_includes_phases_and_kv() {
        let report = PerfReport {
            prefill: PhaseMetrics {
                tokens: 128,
                secs: 1.0,
                tps: 128.0,
                kv_capacity_bytes: 10 * 1024 * 1024,
                kv_used_bytes: 8 * 1024 * 1024,
                kv_filled_tokens: 128,
                kv_layers: 28,
                kv_heads: 8,
                kv_head_dim: 128,
                kv_max_seq: 160,
                kv_key_dtype: "f32".into(),
                kv_value_dtype: "f32".into(),
                kv_paged: false,
                iree_device_live_bytes: 500 * 1024 * 1024,
                iree_device_bytes_peak: 500 * 1024 * 1024,
                peak_dedicated_vram_bytes: Some(700 * 1024 * 1024),
                peak_gtt_bytes: Some(200 * 1024 * 1024),
                ..Default::default()
            },
            decode: PhaseMetrics {
                tokens: 32,
                secs: 0.5,
                tps: 64.0,
                kv_capacity_bytes: 10 * 1024 * 1024,
                kv_used_bytes: 10 * 1024 * 1024,
                kv_filled_tokens: 160,
                kv_layers: 28,
                kv_heads: 8,
                kv_head_dim: 128,
                kv_max_seq: 160,
                kv_key_dtype: "f32".into(),
                kv_value_dtype: "f32".into(),
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
        assert!(text.contains("KV cache"));
        assert!(text.contains("K=f32"));
        assert!(text.contains("V=f32"));
        assert!(text.contains("dedicated VRAM"));
        assert!(text.contains("GTT/unified"));
        assert!(text.contains("IREE device live"));
        assert!(text.contains("lifetime peak"));
    }

    #[test]
    fn eff_weight_bw_counts_one_pass_for_prefill() {
        // 2e9 bytes / 1s = 2 GB/s
        let m = aggregate_phases(&[raw_phase(128, 1.0, 1, 2_000_000_000)]);
        assert!((m.eff_weight_bw_gbs.unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn eff_weight_bw_counts_one_pass_per_decode_token() {
        // 1e9 bytes * 32 passes / 2s = 16 GB/s
        let m = aggregate_phases(&[raw_phase(32, 2.0, 32, 1_000_000_000)]);
        assert!((m.eff_weight_bw_gbs.unwrap() - 16.0).abs() < 1e-9);
    }

    #[test]
    fn kv_capacity_uses_key_and_value() {
        use dyninfer_core::{KvCacheDescriptor, KvCacheLayout, KvCacheStorage};
        let kv = KvCacheDescriptor {
            layer_count: 2,
            max_batch_size: 1,
            max_sequence_length: 10,
            kv_head_count: 4,
            head_dimension: 8,
            element_type: ScalarType::F32,
            layout: KvCacheLayout::LayersHeadsSeqDim,
            alignment: 64,
            storage: KvCacheStorage::StaticGlobals,
        };
        // 2 components * 2 layers * 10 seq * 4 heads * 8 dim * 4 bytes = 5120
        assert_eq!(kv.capacity_bytes(), 5120);
        assert_eq!(kv.bytes_for_tokens(5), 2560);
    }
}
