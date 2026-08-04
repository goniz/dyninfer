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

        // `head_dim` is commonly omitted because it can be derived from model
        // dimensions. Prefer that derivation over the schema's fixture default
        // whenever either dimension came from real configuration data.
        let explicit_head_dim = has_config_value("head_dim", overrides, checkpoint_meta, defaults);
        let configured_dimensions =
            has_config_value("hidden_size", overrides, checkpoint_meta, defaults)
                || has_config_value("num_heads", overrides, checkpoint_meta, defaults);
        if !explicit_head_dim && configured_dimensions && values.contains_key("head_dim") {
            let hidden_size = config_u64(&values, "hidden_size")?;
            let num_heads = config_u64(&values, "num_heads")?;
            if num_heads == 0 || hidden_size % num_heads != 0 {
                return Err(DynInferError::Config(ConfigError {
                    message: format!(
                        "cannot derive `head_dim`: hidden_size ({hidden_size}) must be divisible by non-zero num_heads ({num_heads})"
                    ),
                }));
            }
            values.insert("head_dim".into(), Value::from(hidden_size / num_heads));
        }

        Ok(ResolvedModelConfig { values })
    }
}

fn has_config_value(
    key: &str,
    overrides: &MetadataMap,
    checkpoint_meta: &MetadataMap,
    defaults: &MetadataMap,
) -> bool {
    overrides.contains_key(key)
        || checkpoint_meta.contains_key(key)
        || checkpoint_meta.contains_key(&format!("llama.{key}"))
        || defaults.contains_key(key)
}

fn config_u64(values: &BTreeMap<String, Value>, key: &str) -> Result<u64> {
    values.get(key).and_then(Value::as_u64).ok_or_else(|| {
        DynInferError::Config(ConfigError {
            message: format!("config field `{key}` missing or not u32"),
        })
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> ConfigSchema {
        ConfigSchema {
            fields: vec![
                ConfigField {
                    name: "head_dim".into(),
                    ty: "u32".into(),
                    required: true,
                    default: Some(Value::from(64)),
                    description: None,
                },
                ConfigField {
                    name: "hidden_size".into(),
                    ty: "u32".into(),
                    required: true,
                    default: Some(Value::from(256)),
                    description: None,
                },
                ConfigField {
                    name: "num_heads".into(),
                    ty: "u32".into(),
                    required: true,
                    default: Some(Value::from(4)),
                    description: None,
                },
            ],
        }
    }

    #[test]
    fn derives_head_dim_when_dimensions_are_configured() {
        let overrides = MetadataMap::from([
            ("hidden_size".into(), Value::from(1024)),
            ("num_heads".into(), Value::from(8)),
        ]);

        let config = schema()
            .resolve(&overrides, &MetadataMap::new(), &MetadataMap::new())
            .unwrap();

        assert_eq!(config.get_u32("head_dim").unwrap(), 128);
    }

    #[test]
    fn explicit_head_dim_takes_precedence_over_derivation() {
        let overrides = MetadataMap::from([
            ("hidden_size".into(), Value::from(1024)),
            ("num_heads".into(), Value::from(16)),
            ("head_dim".into(), Value::from(128)),
        ]);

        let config = schema()
            .resolve(&overrides, &MetadataMap::new(), &MetadataMap::new())
            .unwrap();

        assert_eq!(config.get_u32("head_dim").unwrap(), 128);
    }

    #[test]
    fn schema_defaults_remain_coherent_without_external_dimensions() {
        let config = schema()
            .resolve(
                &MetadataMap::new(),
                &MetadataMap::new(),
                &MetadataMap::new(),
            )
            .unwrap();

        assert_eq!(config.get_u32("head_dim").unwrap(), 64);
    }
}
