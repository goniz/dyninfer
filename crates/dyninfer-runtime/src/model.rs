use crate::session::IreeSession;
use crate::{CausalLanguageModel, ModelSession};
use dyninfer_architecture::{ArchitecturePackage, ArchitectureRegistry};
use dyninfer_binding::Binder;
use dyninfer_cache::{ArtifactCache, CacheKeyInputs, make_cache_key};
use dyninfer_checkpoint::{
    BuiltinCheckpointSupport, CheckpointCatalog, DecodeContext, InspectionLimits,
    build_runtime_provider_plan,
};
use dyninfer_compiler::{
    COMPILER_VERSION, CompileOptions, CompileRequest, IREE_REVISION, KERNEL_REGISTRY_VERSION,
    LocalCompiler, ModelCompiler, build_bound_model, default_shape_profile,
};
use dyninfer_core::{
    ArchitectureId, BindingPlan, ExecutableManifest, KvCacheStorage, ModelMetadata,
    PrecisionPolicy, SessionConfig, ShapeProfile, content_digest,
};
use dyninfer_error::{CacheError, DynInferError, Result};
use dyninfer_quantization::{CoverageReport, dry_run_coverage};
use dyninfer_target::TargetDiscovery;
use iree_runtime::{Context, FileParameterDescriptor, FileParameterStorage, Instance, Module};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info_span;

fn validate_executable_abi(manifest: &ExecutableManifest, bundle: &Path) -> Result<()> {
    if manifest.entrypoints == ["add"] {
        return Ok(());
    }
    let required: &[&str] = match (&manifest.kv_cache.storage, manifest.version) {
        (KvCacheStorage::StaticGlobals, 1 | 2) => &["prefill", "decode"],
        (
            KvCacheStorage::Paged {
                page_size,
                chunk_size,
            },
            3,
        ) if *page_size > 0 && *chunk_size > 0 => &["chunk_begin", "chunk_logits"],
        _ => {
            return Err(DynInferError::Cache(CacheError {
                message: format!(
                    "unsupported executable/KV ABI version {} ({:?})",
                    manifest.version, manifest.kv_cache.storage
                ),
                digest: None,
                path: Some(bundle.display().to_string()),
            }));
        }
    };
    if let Some(missing) = required.iter().find(|entrypoint| {
        !manifest
            .entrypoints
            .iter()
            .any(|entry| entry == **entrypoint)
    }) {
        return Err(DynInferError::Cache(CacheError {
            message: format!("bundle is missing required `{missing}` entrypoint"),
            digest: None,
            path: Some(bundle.display().to_string()),
        }));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundlePaths {
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub bindings: PathBuf,
    pub vmfb: PathBuf,
}

/// Immutable ingredients needed to open an independent IREE context (own KV).
enum RuntimeParameters {
    Direct(Arc<FileParameterStorage>),
}

struct ExecutableHandle {
    vmfb_path: PathBuf,
    parameters: RuntimeParameters,
    /// Driver name or full HAL device URI.
    device: Option<String>,
}

impl ExecutableHandle {
    fn open_context(&self) -> Result<Context> {
        let instance = Instance::new()?;
        let module = Module::from_path(&self.vmfb_path)?;
        let mut context = Context::create(instance, module)?;
        context = match &self.parameters {
            RuntimeParameters::Direct(parameters) => {
                context.with_file_parameters(Arc::clone(parameters))
            }
        };
        if let Some(device) = &self.device {
            context = context.with_device(device.clone());
        }
        Ok(context)
    }
}

pub struct LoadedModel {
    pub metadata: ModelMetadata,
    pub manifest: ExecutableManifest,
    pub binding: BindingPlan,
    pub catalog: CheckpointCatalog,
    pub bundle: BundlePaths,
    executable: ExecutableHandle,
}

impl LoadedModel {
    /// Open a fresh IREE context (own native session / KV cache).
    pub fn open_context(&self) -> Result<Context> {
        self.executable.open_context()
    }
}

impl CausalLanguageModel for LoadedModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn create_session(&self, config: SessionConfig) -> Result<Box<dyn ModelSession>> {
        // Each session must own its own context: the VMFB keeps mutable KV in
        // util.global state inside a single native IREE session.
        let context = Arc::new(self.executable.open_context()?);
        if let KvCacheStorage::Paged {
            page_size,
            chunk_size,
        } = self.manifest.kv_cache.storage
        {
            context.configure_paged_kv(
                self.manifest.kv_cache.layer_count as usize,
                page_size as usize,
                self.manifest.kv_cache.kv_head_count as usize,
                self.manifest.kv_cache.head_dimension as usize,
                chunk_size as usize,
            )?;
        }
        Ok(Box::new(IreeSession::new(
            self.metadata.clone(),
            config,
            self.manifest.kv_cache.clone(),
            self.manifest.prefill_window,
            context,
        )))
    }
}

pub struct ModelLoader {
    pub checkpoints: BuiltinCheckpointSupport,
    pub architectures: ArchitectureRegistry,
    pub cache: Option<ArtifactCache>,
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self {
            checkpoints: crate::default_checkpoint_support(),
            architectures: crate::default_architecture_registry(),
            cache: None,
        }
    }
}

impl ModelLoader {
    pub fn with_cache(mut self, cache: ArtifactCache) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn inspect(&self, checkpoint: impl AsRef<Path>) -> Result<CheckpointCatalog> {
        self.checkpoints.inspect_path(
            checkpoint,
            &InspectionLimits::default(),
            &DecodeContext::default(),
        )
    }

    /// Resolve architecture from override or checkpoint metadata.
    pub fn resolve_architecture(
        &self,
        override_id: Option<&str>,
        checkpoint: impl AsRef<Path>,
    ) -> Result<ArchitectureId> {
        let catalog = self.inspect(checkpoint)?;
        dyninfer_architecture::resolve_architecture(
            &self.architectures,
            override_id,
            &catalog.metadata,
        )
    }

    pub fn bind(
        &self,
        architecture_id: &ArchitectureId,
        checkpoint: impl AsRef<Path>,
        overrides: &dyninfer_core::MetadataMap,
    ) -> Result<(ArchitecturePackage, CheckpointCatalog, BindingPlan)> {
        let mut catalog = self.inspect(checkpoint)?;
        self.architectures
            .apply_naming(architecture_id, &mut catalog)?;
        let package =
            self.architectures
                .build_package(architecture_id, overrides, &catalog.metadata)?;
        let plan = Binder::default().bind(&package, &catalog)?;
        Ok((package, catalog, plan))
    }

    /// Resolve and validate production kernel coverage without emitting MLIR.
    pub fn kernel_coverage(
        &self,
        architecture_id: &ArchitectureId,
        checkpoint: impl AsRef<Path>,
        target_spec: &str,
        overrides: &dyninfer_core::MetadataMap,
    ) -> Result<CoverageReport> {
        let target = TargetDiscovery::resolve(target_spec)?;
        let (package, catalog, binding) =
            self.bind(architecture_id, checkpoint.as_ref(), overrides)?;
        let encodings = crate::default_quantization_registry()?;
        let kernels = crate::default_kernel_registry(&encodings)?;
        Ok(dry_run_coverage(
            &package.graph,
            &catalog.parameters,
            &binding,
            &encodings,
            &kernels,
            &target,
            &dyninfer_core::PrecisionPolicy::default(),
        ))
    }

    pub fn compile_to_bundle(
        &self,
        architecture_id: &ArchitectureId,
        checkpoint: impl AsRef<Path>,
        target_spec: &str,
        output: impl AsRef<Path>,
        options: &CompileOptions,
    ) -> Result<BundlePaths> {
        self.compile_to_bundle_with_overrides(
            architecture_id,
            checkpoint,
            target_spec,
            output,
            options,
            &Default::default(),
        )
    }

    pub fn compile_to_bundle_with_overrides(
        &self,
        architecture_id: &ArchitectureId,
        checkpoint: impl AsRef<Path>,
        target_spec: &str,
        output: impl AsRef<Path>,
        options: &CompileOptions,
        overrides: &dyninfer_core::MetadataMap,
    ) -> Result<BundlePaths> {
        let target = TargetDiscovery::resolve(target_spec)?;
        let (package, catalog, plan) =
            self.bind(architecture_id, checkpoint.as_ref(), overrides)?;
        let shape = default_shape_profile(&package);
        let precision_policy = PrecisionPolicy::default();
        let encodings = crate::default_quantization_registry()?;
        let kernels = crate::default_kernel_registry(&encodings)?;
        let coverage = dry_run_coverage(
            &package.graph,
            &catalog.parameters,
            &plan,
            &encodings,
            &kernels,
            &target,
            &precision_policy,
        );
        coverage.require_complete()?;
        let selected_kernels = coverage.selected_kernels();
        let bound_model = build_bound_model(
            &package,
            &plan,
            &target,
            &precision_policy,
            selected_kernels,
            &shape,
        )?;

        if let Some(cache) = &self.cache {
            let key = cache_key_for(
                &package,
                &catalog,
                &plan,
                &target,
                &precision_policy,
                &shape,
                options,
            )?;
            if let Some(hit) = cache.lookup(&key)? {
                return self.materialize_bundle_from_cache(
                    &hit.vmfb_path,
                    &hit.manifest_path,
                    &plan,
                    output,
                );
            }
        }

        let compiler = LocalCompiler::new(options)?;
        let output_compile = compiler.compile(&CompileRequest {
            bound_model: &bound_model,
            architecture_revision: &package.revision,
            checkpoint_schema: &catalog.schema_fingerprint,
            shape_profile: &shape,
            options,
        })?;

        if let Some(cache) = &self.cache {
            let key = cache_key_for(
                &package,
                &catalog,
                &plan,
                &target,
                &precision_policy,
                &shape,
                options,
            )?;
            cache.publish(
                &key,
                &output_compile.executable.bytes,
                &output_compile.manifest,
            )?;
        }

        self.write_bundle(
            output,
            &output_compile.executable.bytes,
            &output_compile.manifest,
            &plan,
            &catalog,
        )
    }

    pub fn load_bundle(
        &self,
        bundle: impl AsRef<Path>,
        checkpoint: impl AsRef<Path>,
    ) -> Result<LoadedModel> {
        let bundle = bundle.as_ref();
        let manifest_path = bundle.join("manifest.json");
        let bindings_path = bundle.join("bindings.json");
        let vmfb_path = bundle.join("executables").join("model.vmfb");
        let manifest: ExecutableManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        validate_executable_abi(&manifest, bundle)?;
        let binding: BindingPlan = serde_json::from_slice(&fs::read(&bindings_path)?)?;
        let checkpoint = checkpoint.as_ref();
        let mut catalog = self.inspect(checkpoint)?;
        self.architectures
            .apply_naming(&manifest.architecture_id, &mut catalog)?;
        if catalog.schema_fingerprint.digest != manifest.checkpoint_schema.digest {
            return Err(DynInferError::Cache(CacheError {
                message: "checkpoint schema does not match bundle".into(),
                digest: Some(manifest.checkpoint_schema.digest.to_string()),
                path: Some(bundle.display().to_string()),
            }));
        }
        let local_target = TargetDiscovery::resolve(manifest.target.runtime_device())?;
        if local_target.capability_fingerprint != manifest.target.capability_fingerprint {
            return Err(DynInferError::Cache(CacheError {
                message: format!(
                    "bundle target fingerprint does not match the currently discovered device `{}`",
                    manifest.target.runtime_device()
                ),
                digest: Some(manifest.target.capability_fingerprint.to_string()),
                path: Some(bundle.display().to_string()),
            }));
        }

        let _span = info_span!("parameters.open").entered();
        if manifest.derived_parameters_required {
            return Err(DynInferError::Cache(CacheError {
                message: "bundle requests forbidden derived parameters".into(),
                digest: Some(manifest.checkpoint_schema.digest.to_string()),
                path: Some(bundle.display().to_string()),
            }));
        }
        let provider_plan = build_runtime_provider_plan(&catalog, &binding)?;
        let expected_components: std::collections::BTreeMap<_, _> = manifest
            .parameter_components
            .iter()
            .map(|component| {
                (
                    (component.scope.as_str(), component.key.as_str()),
                    component.byte_length,
                )
            })
            .collect();
        let actual_components: std::collections::BTreeMap<_, _> = provider_plan
            .parameters
            .iter()
            .map(|parameter| {
                (
                    (
                        provider_plan.scope.as_str(),
                        parameter.external_key.as_str(),
                    ),
                    parameter.length,
                )
            })
            .collect();
        if expected_components != actual_components {
            return Err(DynInferError::Cache(CacheError {
                message: "checkpoint component keys or lengths do not match bundle manifest".into(),
                digest: Some(manifest.checkpoint_schema.digest.to_string()),
                path: Some(bundle.display().to_string()),
            }));
        }
        let mut descriptors = Vec::new();
        for parameter in &provider_plan.parameters {
            descriptors.push(FileParameterDescriptor {
                key: parameter.external_key.clone(),
                source_file_index: parameter.source_file_index as usize,
                offset: parameter.offset,
                length: parameter.length,
            });
            descriptors.extend(
                parameter
                    .aliases
                    .iter()
                    .map(|alias| FileParameterDescriptor {
                        key: alias.clone(),
                        source_file_index: parameter.source_file_index as usize,
                        offset: parameter.offset,
                        length: parameter.length,
                    }),
            );
        }
        let storage = FileParameterStorage::new(provider_plan.file_paths.clone(), descriptors)?;
        let parameters = RuntimeParameters::Direct(Arc::new(storage));
        let device = Some(manifest.target.runtime_device().to_string());

        // Eagerly validate that the VMFB + parameters can open; discard the
        // probe context so create_session still gets an independent KV.
        let executable = ExecutableHandle {
            vmfb_path: vmfb_path.clone(),
            parameters,
            device,
        };
        let _probe = executable.open_context()?;

        let vocabulary_size = catalog
            .metadata
            .get("vocab_size")
            .or_else(|| catalog.metadata.get("llama.vocab_size"))
            .or_else(|| catalog.metadata.get("qwen3.vocab_size"))
            .and_then(|v| v.as_u64())
            .or_else(|| {
                catalog
                    .parameters
                    .iter()
                    .find(|p| {
                        let n = p.canonical_name.as_str();
                        n == "token_embd.weight" || n == "output.weight"
                    })
                    .and_then(|p| p.logical_type.shape.dims().first().copied())
            })
            .unwrap_or(32000) as u32;

        let metadata = ModelMetadata {
            architecture_id: manifest.architecture_id.clone(),
            architecture_revision: manifest.architecture_revision.clone(),
            vocabulary_size,
            context_length: manifest.kv_cache.max_sequence_length,
            num_layers: manifest.kv_cache.layer_count,
            num_heads: catalog
                .metadata
                .get("num_heads")
                .or_else(|| catalog.metadata.get("llama.attention.head_count"))
                .and_then(|v| v.as_u64())
                .unwrap_or(manifest.kv_cache.kv_head_count as u64) as u32,
            num_kv_heads: manifest.kv_cache.kv_head_count,
            head_dim: manifest.kv_cache.head_dimension,
            hidden_size: catalog
                .metadata
                .get("hidden_size")
                .or_else(|| catalog.metadata.get("llama.embedding_length"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            extra: catalog.metadata.clone(),
        };

        Ok(LoadedModel {
            metadata,
            manifest,
            binding,
            catalog,
            bundle: BundlePaths {
                root: bundle.to_path_buf(),
                manifest: manifest_path,
                bindings: bindings_path,
                vmfb: vmfb_path,
            },
            executable,
        })
    }

    fn write_bundle(
        &self,
        output: impl AsRef<Path>,
        vmfb: &[u8],
        manifest: &ExecutableManifest,
        plan: &BindingPlan,
        catalog: &CheckpointCatalog,
    ) -> Result<BundlePaths> {
        let root = output.as_ref().to_path_buf();
        fs::create_dir_all(root.join("executables"))?;
        let vmfb_path = root.join("executables").join("model.vmfb");
        fs::write(&vmfb_path, vmfb)?;
        let manifest_path = root.join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec_pretty(manifest)?)?;
        let bindings_path = root.join("bindings.json");
        fs::write(&bindings_path, serde_json::to_vec_pretty(plan)?)?;
        fs::write(
            root.join("checkpoint-schema.json"),
            serde_json::to_vec_pretty(&catalog.schema_fingerprint)?,
        )?;
        Ok(BundlePaths {
            root,
            manifest: manifest_path,
            bindings: bindings_path,
            vmfb: vmfb_path,
        })
    }

    fn materialize_bundle_from_cache(
        &self,
        vmfb_src: &Path,
        manifest_src: &Path,
        plan: &BindingPlan,
        output: impl AsRef<Path>,
    ) -> Result<BundlePaths> {
        let manifest: ExecutableManifest = serde_json::from_slice(&fs::read(manifest_src)?)?;
        let vmfb = fs::read(vmfb_src)?;
        let root = output.as_ref().to_path_buf();
        fs::create_dir_all(root.join("executables"))?;
        let vmfb_path = root.join("executables").join("model.vmfb");
        fs::write(&vmfb_path, vmfb)?;
        let manifest_path = root.join("manifest.json");
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        let bindings_path = root.join("bindings.json");
        fs::write(&bindings_path, serde_json::to_vec_pretty(plan)?)?;
        Ok(BundlePaths {
            root,
            manifest: manifest_path,
            bindings: bindings_path,
            vmfb: vmfb_path,
        })
    }
}

fn cache_key_for(
    package: &ArchitecturePackage,
    catalog: &CheckpointCatalog,
    plan: &BindingPlan,
    target: &dyninfer_core::TargetProfile,
    precision_policy: &PrecisionPolicy,
    shape: &ShapeProfile,
    options: &CompileOptions,
) -> Result<dyninfer_cache::CacheKey> {
    // Typed Architecture IR digest. Checkpoint bindings and weight values are
    // deliberately separate cache-key inputs.
    let architecture_digest = content_digest(&(&package.id, &package.revision, &package.graph))?;
    let resolved_config_digest = content_digest(&package.resolved_config)?;
    let compile_options_digest = content_digest(options)?;
    make_cache_key(&CacheKeyInputs {
        architecture_id: package.id.as_str(),
        architecture_revision: &package.revision,
        architecture_digest,
        resolved_config_digest,
        binding: plan,
        checkpoint_schema: catalog.schema_fingerprint.digest.as_str(),
        target,
        precision_policy,
        shape_profile: shape,
        kernel_registry_version: KERNEL_REGISTRY_VERSION,
        compiler_version: COMPILER_VERSION,
        iree_revision: IREE_REVISION,
        compile_options_digest,
    })
}
