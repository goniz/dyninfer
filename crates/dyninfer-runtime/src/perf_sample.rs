//! Host/GPU resource sampling during prefill/decode phases.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
const TICKS_PER_SEC: f64 = 100.0; // Linux USER_HZ default

/// One sampled snapshot (raw counters / percentages).
#[derive(Debug, Clone, Default)]
struct Sample {
    rss_bytes: u64,
    cpu_pct: Option<f64>,
    gpu_busy_pct: Option<f64>,
    mem_busy_pct: Option<f64>,
    /// Dedicated device VRAM used (sysfs `mem_info_vram_used`, device-wide).
    dedicated_vram_bytes: Option<u64>,
    /// GTT / unified pool used (sysfs `mem_info_gtt_used`, device-wide).
    gtt_bytes: Option<u64>,
}

/// Aggregated samples collected during a timed phase.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct PhaseSampleStats {
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
    /// Peak dedicated VRAM observed (device-wide sysfs; not process-local).
    pub peak_dedicated_vram_bytes: Option<u64>,
    /// Peak GTT / unified memory observed (device-wide sysfs; not process-local).
    pub peak_gtt_bytes: Option<u64>,
    pub sample_count: usize,
}

/// Background sampler that polls `/proc` and drm sysfs while a phase runs.
pub struct PhaseSampler {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<Sample>>>,
    handle: Option<JoinHandle<()>>,
    started: Instant,
    start_cpu_ticks: Option<u64>,
}

impl PhaseSampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop_t = Arc::clone(&stop);
        let samples_t = Arc::clone(&samples);
        let gpu = discover_gpu_paths();
        let start_cpu_ticks = read_process_cpu_ticks();
        let started = Instant::now();
        let handle = thread::spawn(move || {
            let mut prev_cpu = start_cpu_ticks;
            let mut prev_wall = Instant::now();
            // Seed with an immediate sample so short phases still report RSS.
            {
                let mut guard = samples_t.lock().unwrap_or_else(|e| e.into_inner());
                guard.push(take_sample(&gpu, None));
            }
            while !stop_t.load(Ordering::Relaxed) {
                thread::sleep(SAMPLE_INTERVAL);
                if stop_t.load(Ordering::Relaxed) {
                    break;
                }
                let wall = prev_wall.elapsed().as_secs_f64().max(1e-9);
                let cpu_now = read_process_cpu_ticks();
                let cpu_pct = match (prev_cpu, cpu_now) {
                    (Some(prev), Some(now)) if now >= prev => {
                        Some(((now - prev) as f64 / TICKS_PER_SEC) / wall * 100.0)
                    }
                    _ => None,
                };
                prev_cpu = cpu_now;
                prev_wall = Instant::now();
                let sample = take_sample(&gpu, cpu_pct);
                if let Ok(mut guard) = samples_t.lock() {
                    guard.push(sample);
                }
            }
        });
        Self {
            stop,
            samples,
            handle: Some(handle),
            started,
            start_cpu_ticks,
        }
    }

    pub fn stop(mut self) -> PhaseSampleStats {
        self.join_worker();
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        // Whole-phase CPU% fallback when the phase was shorter than one sample interval.
        if samples.iter().all(|s| s.cpu_pct.is_none()) {
            let wall = self.started.elapsed().as_secs_f64().max(1e-9);
            if let (Some(start), Some(end)) = (self.start_cpu_ticks, read_process_cpu_ticks()) {
                if end >= start {
                    let cpu_pct = ((end - start) as f64 / TICKS_PER_SEC) / wall * 100.0;
                    if let Some(last) = samples.last_mut() {
                        last.cpu_pct = Some(cpu_pct);
                    } else {
                        samples.push(Sample {
                            rss_bytes: read_vm_rss_bytes().unwrap_or(0),
                            cpu_pct: Some(cpu_pct),
                            ..Default::default()
                        });
                    }
                }
            }
        }
        // Final RSS/GPU snapshot at stop.
        let gpu = discover_gpu_paths();
        samples.push(take_sample(&gpu, None));
        aggregate_samples(&samples)
    }

    fn join_worker(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PhaseSampler {
    fn drop(&mut self) {
        // Always join so an early return / panic cannot leak the sampler thread.
        self.join_worker();
    }
}

fn aggregate_samples(samples: &[Sample]) -> PhaseSampleStats {
    let mut stats = PhaseSampleStats {
        sample_count: samples.len(),
        ..Default::default()
    };
    if samples.is_empty() {
        return stats;
    }
    stats.peak_rss_bytes = samples.iter().map(|s| s.rss_bytes).max().unwrap_or(0);
    stats.peak_dedicated_vram_bytes = samples
        .iter()
        .filter_map(|s| s.dedicated_vram_bytes)
        .max();
    stats.peak_gtt_bytes = samples.iter().filter_map(|s| s.gtt_bytes).max();

    let cpu: Vec<f64> = samples.iter().filter_map(|s| s.cpu_pct).collect();
    let (avg, p50, p90) = percentile_summary(&cpu);
    stats.cpu_avg_pct = avg;
    stats.cpu_p50_pct = p50;
    stats.cpu_p90_pct = p90;

    let gpu: Vec<f64> = samples.iter().filter_map(|s| s.gpu_busy_pct).collect();
    let (avg, p50, p90) = percentile_summary(&gpu);
    stats.gpu_busy_avg_pct = avg;
    stats.gpu_busy_p50_pct = p50;
    stats.gpu_busy_p90_pct = p90;

    let mem: Vec<f64> = samples.iter().filter_map(|s| s.mem_busy_pct).collect();
    let (avg, p50, p90) = percentile_summary(&mem);
    stats.mem_busy_avg_pct = avg;
    stats.mem_busy_p50_pct = p50;
    stats.mem_busy_p90_pct = p90;

    stats
}

/// avg / p50 / p90 over a set of samples. Returns `None`s when empty.
pub fn percentile_summary(values: &[f64]) -> (Option<f64>, Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None, None);
    }
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = percentile_sorted(&sorted, 0.50);
    let p90 = percentile_sorted(&sorted, 0.90);
    (Some(avg), Some(p50), Some(p90))
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn take_sample(gpu: &Option<GpuPaths>, cpu_pct: Option<f64>) -> Sample {
    Sample {
        rss_bytes: read_vm_rss_bytes().unwrap_or(0),
        cpu_pct,
        gpu_busy_pct: gpu.as_ref().and_then(|g| read_percent_file(&g.gpu_busy)),
        mem_busy_pct: gpu.as_ref().and_then(|g| read_percent_file(&g.mem_busy)),
        dedicated_vram_bytes: gpu.as_ref().and_then(|g| read_u64_file(&g.vram_used)),
        gtt_bytes: gpu.as_ref().and_then(|g| read_u64_file(&g.gtt_used)),
    }
}

#[derive(Debug, Clone)]
struct GpuPaths {
    gpu_busy: PathBuf,
    mem_busy: PathBuf,
    vram_used: PathBuf,
    gtt_used: PathBuf,
}

fn discover_gpu_paths() -> Option<GpuPaths> {
    let drm = Path::new("/sys/class/drm");
    let entries = fs::read_dir(drm).ok()?;
    let mut cards: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("card") && !n.contains('-'))
        })
        .collect();
    cards.sort();
    for card in cards {
        let device = card.join("device");
        let gpu_busy = device.join("gpu_busy_percent");
        let mem_busy = device.join("mem_busy_percent");
        let vram_used = device.join("mem_info_vram_used");
        let gtt_used = device.join("mem_info_gtt_used");
        // Prefer cards that expose at least busy% or memory accounting.
        if gpu_busy.is_file() || vram_used.is_file() || gtt_used.is_file() {
            return Some(GpuPaths {
                gpu_busy,
                mem_busy,
                vram_used,
                gtt_used,
            });
        }
    }
    None
}

fn read_vm_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest
                .split_whitespace()
                .next()?
                .parse()
                .ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn read_process_cpu_ticks() -> Option<u64> {
    let stat = fs::read_to_string("/proc/self/stat").ok()?;
    // comm can contain spaces/parens — find the trailing ") " then split fields.
    let after_comm = stat.rsplit_once(") ")?.1;
    let mut fields = after_comm.split_whitespace();
    // After comm: state(1) … utime(12) stime(13) relative to after_comm start at 1.
    // Fields after ") ": 1=state … 12=utime, 13=stime (man proc_pid_stat).
    let utime: u64 = fields.nth(11)?.parse().ok()?;
    let stime: u64 = fields.next()?.parse().ok()?;
    Some(utime.saturating_add(stime))
}

fn read_percent_file(path: &Path) -> Option<f64> {
    let s = fs::read_to_string(path).ok()?;
    s.trim().parse::<f64>().ok()
}

fn read_u64_file(path: &Path) -> Option<u64> {
    let s = fs::read_to_string(path).ok()?;
    s.trim().parse::<u64>().ok()
}

/// Parse human size tokens: `128`, `4k`, `4K`, `1m`, `1M`.
pub fn parse_token_count(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty token count".into());
    }
    let (num, mult) = match s.as_bytes().last().copied() {
        Some(b'k' | b'K') => (&s[..s.len() - 1], 1024usize),
        Some(b'm' | b'M') => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1usize),
    };
    let n: usize = num
        .parse()
        .map_err(|_| format!("invalid token count '{s}'"))?;
    n.checked_mul(mult)
        .ok_or_else(|| format!("token count overflow '{s}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_token_count_plain_and_suffixes() {
        assert_eq!(parse_token_count("128").unwrap(), 128);
        assert_eq!(parse_token_count("4k").unwrap(), 4096);
        assert_eq!(parse_token_count("4K").unwrap(), 4096);
        assert_eq!(parse_token_count("1m").unwrap(), 1024 * 1024);
        assert!(parse_token_count("").is_err());
        assert!(parse_token_count("xk").is_err());
    }

    #[test]
    fn percentile_summary_basic() {
        let (avg, p50, p90) = percentile_summary(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        assert!((avg.unwrap() - 30.0).abs() < 1e-9);
        assert!((p50.unwrap() - 30.0).abs() < 1e-9);
        assert!((p90.unwrap() - 46.0).abs() < 1e-9);
        let empty = percentile_summary(&[]);
        assert!(empty.0.is_none());
    }

    #[test]
    fn sampler_reports_rss() {
        let sampler = PhaseSampler::start();
        thread::sleep(Duration::from_millis(80));
        let stats = sampler.stop();
        assert!(stats.sample_count >= 1);
        assert!(stats.peak_rss_bytes > 0);
    }

    #[test]
    fn sampler_drop_joins_worker() {
        let sampler = PhaseSampler::start();
        drop(sampler);
    }
}
