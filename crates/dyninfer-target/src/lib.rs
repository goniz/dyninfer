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
    /// IREE device URI (`hip://…`, `cuda://…`, …).
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

    /// Best target for this machine: CUDA/HIP ≻ CPU.
    pub fn select_best(&self) -> Result<TargetProfile> {
        let devices = self.discover()?;
        let device = devices.into_iter().next().unwrap_or_else(cpu_device);
        if !device.profile.is_compile_ready() {
            return Err(unverified_target_error(&device));
        }
        Ok(device.profile)
    }

    /// Parse a CLI / config target spec into a [`TargetProfile`].
    ///
    /// Supported:
    /// - `auto` → probe IREE HAL devices; pick CUDA/HIP over CPU
    /// - `cpu` / `llvm-cpu` / `local*` → host CPU
    /// - `rocm` / `hip` → a matching locally discovered HIP device
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
        let request = parse_gpu_request(spec).ok_or_else(|| {
            DynInferError::Config(ConfigError {
                message: format!(
                    "unknown target specification: {spec} \
                     (expected auto|cpu|rocm|hip|cuda|rocm:<gfx>|cuda:<sm>|\
                     hip://<device>|cuda://<device>)"
                ),
            })
        })?;
        let devices = TargetDiscovery.discover()?;
        let matching = devices.into_iter().find(|device| {
            canonical_driver(&device.driver) == request.driver
                && request
                    .architecture
                    .as_ref()
                    .is_none_or(|arch| device.arch.as_ref() == Some(arch))
                && request.uri.as_ref().is_none_or(|uri| &device.uri == uri)
        });
        let device = matching.ok_or_else(|| {
            DynInferError::Config(ConfigError {
                message: format!(
                    "target `{spec}` does not match a locally discovered device; \
                     cross-target or guessed GPU compilation is not supported"
                ),
            })
        })?;
        if !device.profile.is_compile_ready() {
            return Err(unverified_target_error(&device));
        }
        Ok(device.profile)
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

/// Preference: discrete GPU HAL (cuda, hip) ≻ CPU.
fn driver_rank(driver: &str) -> u8 {
    match driver {
        "cuda" => 0,
        "hip" | "rocm" => 1,
        "local-task" | "local-sync" | "local" => 2,
        _ => 3,
    }
}

fn sort_by_preference(devices: &mut [DiscoveredDevice]) {
    devices.sort_by(|a, b| {
        driver_rank(&a.driver)
            .cmp(&driver_rank(&b.driver))
            .then(a.device_id.cmp(&b.device_id))
    });
}

fn unverified_target_error(device: &DiscoveredDevice) -> DynInferError {
    DynInferError::Config(ConfigError {
        message: format!(
            "selected local device `{}` ({}) did not report an exact compile architecture; \
             refusing to guess target flags",
            device.name, device.uri
        ),
    })
}

fn canonical_driver(driver: &str) -> &str {
    match driver {
        "rocm" => "hip",
        other => other,
    }
}

struct GpuRequest {
    driver: &'static str,
    architecture: Option<String>,
    uri: Option<String>,
}

fn parse_gpu_request(spec: &str) -> Option<GpuRequest> {
    let lower = spec.to_ascii_lowercase();
    if lower.starts_with("hip://") || lower.starts_with("rocm://") {
        return Some(GpuRequest {
            driver: "hip",
            architecture: None,
            uri: Some(spec.to_string()),
        });
    }
    if lower.starts_with("cuda://") {
        return Some(GpuRequest {
            driver: "cuda",
            architecture: None,
            uri: Some(spec.to_string()),
        });
    }
    for (prefix, driver) in [("rocm", "hip"), ("hip", "hip"), ("cuda", "cuda")] {
        if lower == prefix {
            return Some(GpuRequest {
                driver,
                architecture: None,
                uri: None,
            });
        }
        for sep in [':', '/'] {
            let marker = format!("{prefix}{sep}");
            if lower.starts_with(&marker) {
                let raw = spec[marker.len()..].trim();
                return (!raw.is_empty()).then(|| GpuRequest {
                    driver,
                    architecture: Some(raw.to_string()),
                    uri: None,
                });
            }
        }
    }
    if lower.starts_with("gfx") || lower.starts_with("mi") || lower.starts_with("rdna") {
        return Some(GpuRequest {
            driver: "hip",
            architecture: Some(spec.to_string()),
            uri: None,
        });
    }
    if lower.starts_with("sm_") {
        return Some(GpuRequest {
            driver: "cuda",
            architecture: Some(spec.to_string()),
            uri: None,
        });
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

    let flush = |pending: &mut Option<PendingDevice>,
                 devices: &mut Vec<DiscoveredDevice>,
                 hip_index: &mut u32,
                 cuda_index: &mut u32| {
        if let Some(p) = pending.take() {
            if let Some(dev) = pending_to_device(p, hip_index, cuda_index) {
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
            flush(&mut pending, &mut devices, &mut hip_index, &mut cuda_index);
            current_driver = Some(driver.to_string());
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("# --device=") {
            flush(&mut pending, &mut devices, &mut hip_index, &mut cuda_index);
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
                features: Vec::new(),
                subgroup_size: None,
                max_workgroup_invocations: None,
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
                continue;
            }
            if let Some(capability) = trimmed.strip_prefix("- gpu-compute-capability:") {
                p.features
                    .push(format!("compute_capability={}", capability.trim()));
                continue;
            }
            if let Some(value) = trimmed.strip_prefix("- gpu-subgroup-size:") {
                p.subgroup_size = value.trim().parse().ok();
                continue;
            }
            if let Some(value) = trimmed.strip_prefix("- gpu-max-workgroup-invocations:") {
                p.max_workgroup_invocations = value.trim().parse().ok();
            }
        }
    }
    flush(&mut pending, &mut devices, &mut hip_index, &mut cuda_index);

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
    features: Vec<String>,
    subgroup_size: Option<u32>,
    max_workgroup_invocations: Option<u32>,
}

fn pending_to_device(
    p: PendingDevice,
    hip_index: &mut u32,
    cuda_index: &mut u32,
) -> Option<DiscoveredDevice> {
    let driver = p.driver.as_str();
    // Skip empty local placeholders already covered by cpu_device; keep one local-task.
    match driver {
        "cuda" => {
            let arch = p.arch.clone().filter(|s| !s.is_empty());
            let id = *cuda_index;
            *cuda_index += 1;
            let name = nonempty_name(
                &p.name,
                &arch
                    .as_ref()
                    .map(|arch| format!("cuda-{arch}"))
                    .unwrap_or_else(|| "cuda-device".into()),
            );
            let profile = TargetProfile::cuda(arch.as_deref().unwrap_or(""))
                .with_device_identity(id, p.uri.clone(), name.clone())
                .with_discovered_features(p.features)
                .with_execution_limits(p.subgroup_size, p.max_workgroup_invocations);
            Some(DiscoveredDevice {
                driver: "cuda".into(),
                device_id: id,
                name,
                uri: p.uri.clone(),
                arch,
                profile,
            })
        }
        "hip" | "rocm" => {
            let arch = p.arch.clone().filter(|s| !s.is_empty());
            let id = *hip_index;
            *hip_index += 1;
            let name = nonempty_name(
                &p.name,
                &arch
                    .as_ref()
                    .map(|arch| format!("rocm-{arch}"))
                    .unwrap_or_else(|| "rocm-device".into()),
            );
            let profile = TargetProfile::hip_rocm(arch.as_deref().unwrap_or(""))
                .with_device_identity(id, p.uri.clone(), name.clone())
                .with_discovered_features(p.features)
                .with_execution_limits(p.subgroup_size, p.max_workgroup_invocations);
            Some(DiscoveredDevice {
                driver: "hip".into(),
                device_id: id,
                name,
                uri: p.uri.clone(),
                arch,
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
    fn parses_dump_and_ranks_hip_over_cpu() {
        let mut devices = parse_dump_devices(SAMPLE_DUMP);
        sort_by_preference(&mut devices);
        assert_eq!(devices[0].driver, "hip");
        assert_eq!(devices[0].arch.as_deref(), Some("gfx1151"));
        assert_eq!(devices[0].profile.rocm_target(), Some("gfx1151"));
        assert!(devices.iter().any(|d| d.driver == "local-task"));
        // Unsupported HAL drivers reported by IREE are never offered.
        assert!(
            devices
                .iter()
                .all(|d| d.driver == "hip" || d.driver == "local-task")
        );
    }

    #[test]
    fn unsupported_driver_specs_are_rejected() {
        assert!(parse_gpu_request("vulkan").is_none());
        assert!(parse_gpu_request("vulkan://GPU-0").is_none());
        assert!(TargetDiscovery::resolve("vulkan").is_err());
    }

    #[test]
    fn parses_explicit_specs_without_inventing_architectures() {
        let request = parse_gpu_request("rocm").unwrap();
        assert_eq!(request.driver, "hip");
        assert!(request.architecture.is_none());

        let request = parse_gpu_request("hip:gfx1100").unwrap();
        assert_eq!(request.architecture.as_deref(), Some("gfx1100"));

        let request = parse_gpu_request("cuda:sm_90").unwrap();
        assert_eq!(request.driver, "cuda");
        assert_eq!(request.architecture.as_deref(), Some("sm_90"));

        let p = TargetDiscovery::resolve("cpu").unwrap();
        assert_eq!(p.driver, "local-task");
        assert!(p.is_compile_ready());
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
