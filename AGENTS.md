Use bazel.
Make sure your decisions are in line with docs in ./docs/plans/
when running validations, include generate coherency as well.
When validating engine or model-architecture changes/additions, run `bazel run //tools/llama-logits:drift -- --checkpoint "$GGUF" --target "$TARGET" --prompt "$PROMPT"` whenever a suitable GGUF checkpoint is available.
