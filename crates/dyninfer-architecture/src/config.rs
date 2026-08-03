use dyninfer_core::MetadataMap;
use dyninfer_error::{ConfigError, DynInferError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigField {
    pub name: String,
    pub ty: String,
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    pub fields: Vec<ConfigField>,
}

impl ConfigSchema {
    pub fn resolve(
        &self,
        overrides: &MetadataMap,
        checkpoint_meta: &MetadataMap,
        defaults: &MetadataMap,
    ) -> Result<ResolvedModelConfig> {
        let mut values = BTreeMap::new();
        for field in &self.fields {
            if let Some(v) = overrides.get(&field.name) {
                values.insert(field.name.clone(), v.clone());
                continue;
            }
            if let Some(v) = checkpoint_meta.get(&field.name) {
                values.insert(field.name.clone(), v.clone());
                continue;
            }
            // Also accept dotted GGUF-style keys for common llama fields.
            let gguf_key = format!("llama.{}", field.name);
            if let Some(v) = checkpoint_meta.get(&gguf_key) {
                values.insert(field.name.clone(), v.clone());
                continue;
            }
            if let Some(v) = defaults.get(&field.name) {
                values.insert(field.name.clone(), v.clone());
                continue;
            }
            if let Some(v) = &field.default {
                values.insert(field.name.clone(), v.clone());
                continue;
            }
            if field.required {
                return Err(DynInferError::Config(ConfigError {
                    message: format!("missing required config field `{}`", field.name),
                }));
            }
        }
        Ok(ResolvedModelConfig { values })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedModelConfig {
    pub values: BTreeMap<String, Value>,
}

impl ResolvedModelConfig {
    pub fn get_u32(&self, key: &str) -> Result<u32> {
        self.values
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| {
                DynInferError::Config(ConfigError {
                    message: format!("config field `{key}` missing or not u32"),
                })
            })
    }

    pub fn get_u64(&self, key: &str) -> Result<u64> {
        self.values
            .get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                DynInferError::Config(ConfigError {
                    message: format!("config field `{key}` missing or not u64"),
                })
            })
    }

    pub fn get_f64(&self, key: &str) -> Result<f64> {
        self.values
            .get(key)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| {
                DynInferError::Config(ConfigError {
                    message: format!("config field `{key}` missing or not f64"),
                })
            })
    }

    pub fn num_layers(&self) -> Result<u32> {
        self.get_u32("num_layers")
            .or_else(|_| self.get_u32("n_layer"))
            .or_else(|_| self.get_u32("block_count"))
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(&self.values)?)
    }
}
