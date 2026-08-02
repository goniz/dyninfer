//! Hardware discovery, IREE target mapping, capability fingerprinting.
//!
//! Discovery uses `iree-run-module --dump_devices` so the same HAL drivers the
//! runtime can load are what we select for compile/run.

#![forbid(unsafe_code)]

use dyninfer_core::{Digest, TargetProfile};
use dyninfer_error::{ConfigError, DynInferError, Result};
use iree_runtime::IreeRunTools;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredDevice {
    pub driver: String,
    pub device_id: u32,
    pub name: String,
    /// IREE device URI (`hip://…`, `vulkan://…`, …).
    pub uri: String,
    /// GPU arch from dump (`gfx1151`, `sm_80`), when known.
    pub arch: Option<String>,
    pub profile: TargetProfile,
}

#[derive(Debug, Default)]
pub struct TargetDiscovery;

impl TargetDiscovery {
    /// Probe the host with IREE and return usable devices (best-first order).
    pub fn discover(&self) -> Result<Vec<DiscoveredDevice>> {
        match IreeRunTools::discover().and_then(|t| t.dump_devices()) {
            Ok(dump) => {
                let mut devices = parse_dump_devices(&dump);
                if devices.is_empty() {
                    debug!("IREE dump had no devices; falling back to CPU");
                    devices.push(cpu_device());
                }
                sort_by_preference(&mut devices);
                info!(
                    count = devices.len(),
                    best = devices.first().map(|d| d.profile.driver.as_str()),
                    "discovered IREE targets"
                );
                Ok(devices)
            }
            Err(e) => {
                debug!(error = %e, "IREE device dump unavailable; CPU only");
                Ok(vec![cpu_device()])
            }
        }
    }

    /// Best target for this machine: CUDA/HIP ≻ Vulkan ≻ CPU.
    pub fn select_best(&self) -> Result<TargetProfile> {
        let devices = self.discover()?;
        Ok(devices
            .into_iter()
            .next()
            .map(|d| d.profile)
            .unwrap_or_else(TargetProfile::llvm_cpu_host))
    }

    /// Parse a CLI / config target spec into a [`TargetProfile`].
    ///
    /// Supported:
    /// - `auto` → probe IREE HAL devices; pick CUDA/HIP over Vulkan/CPU
    /// - `cpu` / `llvm-cpu` / `local*` → host CPU
    /// - `vulkan` / `vulkan://…` → Vulkan (default chip `gfx1151`, or `DYNINFER_VULKAN_TARGET` / `DYNINFER_ROCM_TARGET`)
    /// - `vulkan:gfx1151` / `vulkan/rdna3` → Vulkan + arch for `--iree-vulkan-target`
    /// - `rocm` / `hip` → HIP (default chip `gfx1151`, or `DYNINFER_ROCM_TARGET`)
    /// - `rocm:gfx1151` / `hip:gfx1100` / `rocm/gfx1151` → HIP + chip
    /// - `cuda` / `cuda:sm_80` → CUDA (+ arch)
    /// - bare `gfx*` → HIP; bare `sm_*` → CUDA
    pub fn resolve(spec: &str) -> Result<TargetProfile> {
        let spec = spec.trim();
        if spec == "auto" {
            return TargetDiscovery.select_best();
        }
        if spec == "cpu" || spec == "llvm-cpu" || spec.starts_with("local") {
            return Ok(TargetProfile::llvm_cpu_host());
        }
        if let Some(chip) = parse_vulkan_spec(spec) {
            return Ok(TargetProfile::vulkan(&chip));
        }
        if let Some(arch) = parse_cuda_spec(spec) {
            return Ok(TargetProfile::cuda(&arch));
        }
        if let Some(chip) = parse_rocm_spec(spec) {
            return Ok(TargetProfile::hip_rocm(&chip));
        }
        Err(DynInferError::Config(ConfigError {
            message: format!(
                "unknown target specification: {spec} \
                 (expected auto|cpu|vulkan|vulkan:gfxXXXX|rocm|hip|cuda|rocm:gfxXXXX|cuda:sm_XX; \
                 default ROCm/Vulkan chip {}, default CUDA {})",
                TargetProfile::DEFAULT_ROCM_TARGET,
                TargetProfile::DEFAULT_CUDA_TARGET
            ),
        }))
    }

    pub fn capability_fingerprint(profile: &TargetProfile) -> Digest {
        profile.capability_fingerprint.clone()
    }
}

fn cpu_device() -> DiscoveredDevice {
    DiscoveredDevice {
        driver: "local-task".into(),
        device_id: 0,
        name: "host-cpu".into(),
        uri: "local-task://".into(),
        arch: None,
        profile: TargetProfile::llvm_cpu_host(),
    }
}

/// Preference: discrete GPU HAL (cuda, hip) ≻ vulkan ≻ CPU.
fn driver_rank(driver: &str) -> u8 {
    match driver {
        "cuda" => 0,
        "hip" | "rocm" => 1,
        "vulkan" => 2,
        "local-task" | "local-sync" | "local" => 3,
        _ => 4,
    }
}

fn sort_by_preference(devices: &mut [DiscoveredDevice]) {
    devices.sort_by(|a, b| {
        driver_rank(&a.driver)
            .cmp(&driver_rank(&b.driver))
            .then(a.device_id.cmp(&b.device_id))
    });
}

fn default_rocm_chip() -> String {
    std::env::var("DYNINFER_ROCM_TARGET")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| TargetProfile::DEFAULT_ROCM_TARGET.to_string())
}

fn default_vulkan_chip() -> String {
    std::env::var("DYNINFER_VULKAN_TARGET")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(default_rocm_chip)
}

fn default_cuda_arch() -> String {
    std::env::var("DYNINFER_CUDA_TARGET")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| TargetProfile::DEFAULT_CUDA_TARGET.to_string())
}

fn parse_vulkan_spec(spec: &str) -> Option<String> {
    let lower = spec.to_ascii_lowercase();
    if lower == "vulkan" || lower.starts_with("vulkan://") {
        return Some(default_vulkan_chip());
    }
    for sep in [':', '/'] {
        let p = format!("vulkan{sep}");
        if lower.strip_prefix(&p).is_some() {
            let raw = spec["vulkan".len() + 1..].trim();
            if raw.is_empty() {
                return Some(default_vulkan_chip());
            }
            return Some(raw.to_string());
        }
    }
    None
}

fn parse_rocm_spec(spec: &str) -> Option<String> {
    let lower = spec.to_ascii_lowercase();
    if lower == "rocm" || lower == "hip" {
        return Some(default_rocm_chip());
    }
    for (prefix, sep) in [("rocm", ':'), ("hip", ':'), ("rocm", '/'), ("hip", '/')] {
        let p = format!("{prefix}{sep}");
        if lower.strip_prefix(&p).is_some() {
            let raw = spec[prefix.len() + 1..].trim();
            if raw.is_empty() {
                return Some(default_rocm_chip());
            }
            return Some(raw.to_string());
        }
    }
    if lower.starts_with("gfx") || lower.starts_with("mi") || lower.starts_with("rdna") {
        return Some(spec.to_string());
    }
    None
}

fn parse_cuda_spec(spec: &str) -> Option<String> {
    let lower = spec.to_ascii_lowercase();
    if lower == "cuda" {
        return Some(default_cuda_arch());
    }
    for sep in [':', '/'] {
        let p = format!("cuda{sep}");
        if lower.strip_prefix(&p).is_some() {
            let raw = spec[4 + 1..].trim();
            if raw.is_empty() {
                return Some(default_cuda_arch());
            }
            return Some(raw.to_string());
        }
    }
    if lower.starts_with("sm_") {
        return Some(spec.to_string());
    }
    None
}

/// Parse `iree-run-module --dump_devices` stdout into discovered devices.
pub fn parse_dump_devices(dump: &str) -> Vec<DiscoveredDevice> {
    let mut devices = Vec::new();
    let mut current_driver: Option<String> = None;
    let mut pending: Option<PendingDevice> = None;
    let mut hip_index = 0u32;
    let mut cuda_index = 0u32;
    let mut vulkan_index = 0u32;

    let flush = |pending: &mut Option<PendingDevice>,
                 devices: &mut Vec<DiscoveredDevice>,
                 hip_index: &mut u32,
                 cuda_index: &mut u32,
                 vulkan_index: &mut u32| {
        if let Some(p) = pending.take() {
            if let Some(dev) = pending_to_device(p, hip_index, cuda_index, vulkan_index) {
                devices.push(dev);
            }
        }
    };

    for line in dump.lines() {
        let trimmed = line.trim();
        if let Some(driver) = trimmed
            .strip_prefix("# Enumerated devices for driver '")
            .and_then(|s| s.strip_suffix('\''))
        {
            flush(
                &mut pending,
                &mut devices,
                &mut hip_index,
                &mut cuda_index,
                &mut vulkan_index,
            );
            current_driver = Some(driver.to_string());
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("# --device=") {
            flush(
                &mut pending,
                &mut devices,
                &mut hip_index,
                &mut cuda_index,
                &mut vulkan_index,
            );
            let uri = rest.trim().to_string();
            let driver = current_driver
                .clone()
                .or_else(|| uri.split("://").next().map(|s| s.to_string()))
                .unwrap_or_default();
            pending = Some(PendingDevice {
                driver,
                uri,
                name: String::new(),
                arch: None,
            });
            continue;
        }

        if let Some(p) = pending.as_mut() {
            if trimmed.starts_with("#   ") && p.name.is_empty() {
                p.name = trimmed.trim_start_matches('#').trim().to_string();
                continue;
            }
            if let Some(arch) = trimmed.strip_prefix("- gpu-arch-name:") {
                p.arch = Some(arch.trim().to_string());
            }
        }
    }
    flush(
        &mut pending,
        &mut devices,
        &mut hip_index,
        &mut cuda_index,
        &mut vulkan_index,
    );

    // Always offer CPU as a fallback option in the inventory.
    if !devices.iter().any(|d| d.driver.starts_with("local")) {
        devices.push(cpu_device());
    }
    devices
}

struct PendingDevice {
    driver: String,
    uri: String,
    name: String,
    arch: Option<String>,
}

fn pending_to_device(
    p: PendingDevice,
    hip_index: &mut u32,
    cuda_index: &mut u32,
    vulkan_index: &mut u32,
) -> Option<DiscoveredDevice> {
    let driver = p.driver.as_str();
    // Skip empty local placeholders already covered by cpu_device; keep one local-task.
    match driver {
        "cuda" => {
            let arch = p
                .arch
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(default_cuda_arch);
            let id = *cuda_index;
            *cuda_index += 1;
            Some(DiscoveredDevice {
                driver: "cuda".into(),
                device_id: id,
                name: nonempty_name(&p.name, &format!("cuda-{arch}")),
                uri: p.uri,
                arch: Some(arch.clone()),
                profile: {
                    let mut profile = TargetProfile::cuda(&arch);
                    profile.device_id = Some(id);
                    profile
                },
            })
        }
        "hip" | "rocm" => {
            let arch = p
                .arch
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(default_rocm_chip);
            let id = *hip_index;
            *hip_index += 1;
            Some(DiscoveredDevice {
                driver: "hip".into(),
                device_id: id,
                name: nonempty_name(&p.name, &format!("rocm-{arch}")),
                uri: p.uri,
                arch: Some(arch.clone()),
                profile: {
                    let mut profile = TargetProfile::hip_rocm(&arch);
                    profile.device_id = Some(id);
                    profile
                },
            })
        }
        "vulkan" => {
            let id = *vulkan_index;
            *vulkan_index += 1;
            let mut profile = TargetProfile::vulkan(&default_vulkan_chip());
            profile.device_id = Some(id);
            Some(DiscoveredDevice {
                driver: "vulkan".into(),
                device_id: id,
                name: nonempty_name(&p.name, "vulkan"),
                uri: p.uri,
                arch: Some(default_vulkan_chip()),
                profile,
            })
        }
        "local-task" => Some(DiscoveredDevice {
            driver: "local-task".into(),
            device_id: 0,
            name: nonempty_name(&p.name, "host-cpu"),
            uri: p.uri,
            arch: None,
            profile: TargetProfile::llvm_cpu_host(),
        }),
        // Prefer local-task; ignore local-sync duplicates in inventory.
        "local-sync" => None,
        _ => None,
    }
}

fn nonempty_name(name: &str, fallback: &str) -> String {
    if name.trim().is_empty() {
        fallback.to_string()
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_DUMP: &str = r#"
# ============================================================================
# Enumerated devices for driver 'cuda'
# ============================================================================


# ============================================================================
# Enumerated devices for driver 'hip'
# ============================================================================

# ===----------------------------------------------------------------------===
# --device=hip://GPU-58580000-0000-0000-0000-000000000000
#   AMD Radeon 8060S Graphics
# ===----------------------------------------------------------------------===

- gpu-compute-capability: 11.5
- gpu-arch-name: gfx1151

# ============================================================================
# Enumerated devices for driver 'local-task'
# ============================================================================

# ===----------------------------------------------------------------------===
# --device=local-task://
#   default
# ===----------------------------------------------------------------------===

# ============================================================================
# Enumerated devices for driver 'vulkan'
# ============================================================================

# ===----------------------------------------------------------------------===
# --device=vulkan://00000000-f400-0000-0000-000000000000
#   AMD Radeon 8060S Graphics
# ===----------------------------------------------------------------------===
"#;

    #[test]
    fn parses_dump_and_ranks_hip_over_vulkan_cpu() {
        let mut devices = parse_dump_devices(SAMPLE_DUMP);
        sort_by_preference(&mut devices);
        assert_eq!(devices[0].driver, "hip");
        assert_eq!(devices[0].arch.as_deref(), Some("gfx1151"));
        assert_eq!(devices[0].profile.rocm_target(), Some("gfx1151"));
        assert!(devices.iter().any(|d| d.driver == "vulkan"));
        assert!(devices.iter().any(|d| d.driver == "local-task"));
    }

    #[test]
    fn resolves_explicit_specs() {
        let p = TargetDiscovery::resolve("rocm").unwrap();
        assert_eq!(p.driver, "hip");
        assert_eq!(p.rocm_target(), Some(TargetProfile::DEFAULT_ROCM_TARGET));

        let p = TargetDiscovery::resolve("hip:gfx1100").unwrap();
        assert_eq!(p.rocm_target(), Some("gfx1100"));

        let p = TargetDiscovery::resolve("cuda:sm_90").unwrap();
        assert_eq!(p.driver, "cuda");
        assert_eq!(p.cuda_target(), Some("sm_90"));

        let p = TargetDiscovery::resolve("cpu").unwrap();
        assert_eq!(p.driver, "local-task");
    }

    #[test]
    fn cuda_ranks_above_hip() {
        let dump = r#"
# Enumerated devices for driver 'cuda'
# --device=cuda://GPU-0
#   NVIDIA GeForce
- gpu-arch-name: sm_89

# Enumerated devices for driver 'hip'
# --device=hip://GPU-1
#   AMD
- gpu-arch-name: gfx1100
"#;
        let mut devices = parse_dump_devices(dump);
        sort_by_preference(&mut devices);
        assert_eq!(devices[0].driver, "cuda");
        assert_eq!(devices[0].arch.as_deref(), Some("sm_89"));
    }
}
