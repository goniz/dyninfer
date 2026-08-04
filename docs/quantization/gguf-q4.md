# GGUF Qwen3 mixed-quant qualification

The executable GGUF baseline is deliberately per tensor. It supports direct
Q4_0, Q4_1, Q8_0, and Q6_K embedding, linear, and output-projection operations in
both prefill and decode on IREE CPU and HIP targets. There is no
model-wide quantization selection and no host dequantization fallback.

## Layouts

- Q4_0: 32 logical values in an 18-byte GGML block. The F16 block scale is
  followed by 16 bytes whose low nibbles encode lanes 0–15 and high nibbles
  encode lanes 16–31.
- Q4_1: 32 logical values in a 20-byte block with F16 scale and minimum,
  followed by the same 16-byte nibble arrangement. Dequantization is
  `value = scale * lane + minimum`.
- Q8_0: 32 signed byte values following one F16 block scale in a 34-byte
  block. Dequantization is `value = scale * signed_quant`.
- Q6_K: 256 logical values in a 210-byte block: 128 low-nibble bytes, 64
  high-two-bit bytes, 16 signed sub-block scales, and one F16 block scale.
  The effective signed quant is `(low4 | high2 << 4) - 32`.

The lowering preserves these original interleaved bytes in the file-backed
parameter provider. Structured bitcasts expose packed lanes inside the
consuming device helper; no repacked checkpoint or dense external shadow is
created.

## Real-model evidence

`unsloth/Qwen3-0.6B-GGUF`'s `Qwen3-0.6B-Q4_0.gguf` is a mixed checkpoint using
all three layouts. Its 1,240 operation-mode requests have complete strict
coverage. Full-model Bazel tests compile it, bind original GGUF byte ranges,
run a one-token prefill, select the next token, and run decode with finite
151,936-element logits on CPU and HIP (`gfx1151` on the qualification host).

The tiny Q4_0/Q4_1/Q8_0 fixture compares the executable against an independent
dequantized Rust reference with `max_abs_err < 1e-3`. Provider tests also prove
that the original packed ranges are retained.

## Unsloth dynamic variants

The UD variants are not treated as a single quantization family:

- UD-Q4_K_XL mixes supported Q6_K with currently unsupported IQ4_XS, Q4_K,
  and Q5_K tensors.
- UD-IQ1_S mixes IQ1_M, IQ1_S, IQ2_S, IQ2_XXS, IQ3_S, IQ3_XXS, Q2_K, and
  Q5_K tensors that remain schema-valid but lack complete kernels.

Compilation rejects these models with per-operation coverage diagnostics. It
does not silently substitute Q4_0, dense weights, CPU execution, or a scalar
reference decoder.
