//! Hardware discovery, IREE target mapping, capability fingerprinting.

#![forbid(unsafe_code)]

use dyninfer_core::{Digest, TargetProfile};
use dyninfer_error::{ConfigError, DynInferError, Result};
use serde::{Deserialize, Serialize};
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredDevice {
    pub driver: String,
    pub device_id: u32,
    pub name: String,
    pub profile: TargetProfile,
}

#[derive(Debug, Default)]
pub struct TargetDiscovery;

impl TargetDiscovery {
    /// Discover locally available targets without talking to IREE yet.
    ///
    /// Always includes a host CPU profile. Vulkan is advertised as available
    /// when the user requests it; actual device enumeration waits for IREE FFI.
    pub fn discover(&self) -> Result<Vec<DiscoveredDevice>> {
        let mut devices = vec![DiscoveredDevice {
            driver: "local-task".into(),
            device_id: 0,
            name: "host-cpu".into(),
            profile: TargetProfile::llvm_cpu_host(),
        }];
        // Placeholder Vulkan entry for compile-target selection in Milestone 0/1.
        devices.push(DiscoveredDevice {
            driver: "vulkan".into(),
            device_id: 0,
            name: "vulkan-generic".into(),
            profile: TargetProfile::vulkan_generic(),
        });
        debug!(count = devices.len(), "discovered targets (stub)");
        Ok(devices)
    }

    pub fn resolve(spec: &str) -> Result<TargetProfile> {
        let spec = spec.trim();
        if spec == "auto" || spec == "cpu" || spec == "llvm-cpu" || spec.starts_with("local") {
            return Ok(TargetProfile::llvm_cpu_host());
        }
        if spec == "vulkan" || spec.starts_with("vulkan://") {
            return Ok(TargetProfile::vulkan_generic());
        }
        Err(DynInferError::Config(ConfigError {
            message: format!("unknown target specification: {spec}"),
        }))
    }

    pub fn capability_fingerprint(profile: &TargetProfile) -> Digest {
        profile.capability_fingerprint.clone()
    }
}
