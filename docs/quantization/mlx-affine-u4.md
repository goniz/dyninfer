# MLX affine U4 qualification

`mlx.affine.u4` is production-enabled for local IREE CPU, HIP, and Vulkan
backends. CPU selection requires AVX2; GPU selection requires an exactly
discovered local device and compile target. The checkpoint contract is unsigned
four-bit lanes packed least-significant lane first in U32 words, plus one scale
and one bias per group. Dequantization is `value = lane * scale + bias`.

## Correctness and storage evidence

- `tiny_mlx_u4_matches_independent_dequantized_reference` exercises direct
  packed embedding gather, all transformer linears, and output projection
  through IREE and compares final logits with an independent CPU
  quantize/dequantize reference (`max_abs_err < 1e-3`).
- The encoding-owned MLIR verification test checks unsigned I4 extension and
  grouped reduction shapes. GPU lowering also checks the explicit on-device
  dequantization boundary that keeps I4 out of IREE's contraction distributor.
- The real `mlx-community/Qwen3-0.6B-4bit` checkpoint selects direct kernels for
  all 396 quantized operation-mode requests and compiles/runs without a derived
  parameter directory or dense external shadow tensors. Its manifest contains
  707 original components and sets `derived_parameters_required=false`.
- The same real checkpoint completes short Qwen3 prefill/decode tests through
  CPU, HIP (`gfx1151` on the qualification host), and Vulkan. No backend retry
  or CPU fallback is permitted.

GPU dequantization is a device dispatch separated from the F32 contraction.
This is the initial reliable IREE path: checkpoint storage remains packed and
file-backed, and the temporary is device-local and non-persistent. Fusing the
sub-byte unpack with GPU contractions is a performance follow-up because the
current IREE distributor rejects or compiles pathologically on those shapes.

## CPU performance qualification

Measured 2026-08-03 on an AMD Ryzen AI MAX+ 395, IREE local-task, host target,
Qwen3-0.6B, prompt `Hello`, 32 greedy decode tokens:

| Checkpoint | File bytes | Prefill latency | Decode rate |
|---|---:|---:|---:|
| Qwen BF16 | 1,503,300,328 | 0.283 s | 25.3 tok/s |
| MLX affine U4 | 335,450,584 | 0.521 s | 36.7 tok/s |

The initial production bar for the fused grouped CPU lowering is: at least
1.25x the BF16 decode rate, no more than 2.5x the BF16 prefill latency, direct
packed storage through the consuming helper, and no persistent derived weight
artifact. This run achieved 1.45x decode throughput, 1.84x prefill latency, and
4.48x smaller checkpoint storage. Re-qualification is required before adding a
different bit width, group formula, or serialized layout.

## GPU qualification

The HIP and Vulkan registrations currently have functional qualification on
the same Qwen3 model: complete strict coverage, direct parameter binding,
successful compile, finite prefill/decode logits, and no host or persistent
dense materialization. Throughput and temporary-memory benchmarks remain
required before claiming parity with a native fused U4 GPU kernel.

With the repository's current IREE revision, Vulkan validation layers may
report that emitted `NoSignedWrap` decorations omit
`SPV_KHR_no_integer_wrap_decoration`. Execution succeeds on the qualification
device, but this compiler-side validation issue should be resolved before a
warning-free Vulkan release gate is claimed.
