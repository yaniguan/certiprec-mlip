# Borrowed components and precision

GPU results: **PENDING**. CPU reference checks use the pure Python and torch
packages from [Yani Guan's cuEquivariance fork](https://github.com/yaniguan/cuEquivariance),
revision `f5c9ada7ca688ad062f3b7d5125a1fbd0b71cd40` (version `0.13.0`).
The fork derives from [NVIDIA/cuEquivariance](https://github.com/NVIDIA/cuEquivariance).
The reference at `/workspace/ref_cueq` was read only. Both packages were built
and installed into `/workspace/venv` from disposable copies under `/tmp`.
The first copy preserved relative symlinks and failed packaging; resolving the
VERSION, README, LICENSE and attribution symlinks in the disposable copy fixed
it. No files in the reference clone were changed.

## What is used

All paths below are relative to the reference repository. These are runtime
dependencies and inspected reference implementations; no upstream Python
source has been vendored into CertiPrec.

| Upstream file | Functions or data used | CertiPrec use |
| --- | --- | --- |
| `cuequivariance/cuequivariance/group_theory/descriptors/irreps_tp.py` | `channelwise_tensor_product`, `linear`; their normalization and CG tables | Message coupling, scalar/vector channel mixing, quadratic scalar invariants in `equivariant.py` |
| `cuequivariance/cuequivariance/group_theory/descriptors/spherical_harmonics_.py` | `spherical_harmonics` | l=0 and l=1 angular features |
| `cuequivariance/cuequivariance/segmented_polynomials/segmented_tensor_product.py` | `SegmentedTensorProduct`, segment shapes, subscripts and path dimensions | Path execution and FLOP estimates in `segmented.py` |
| `cuequivariance/cuequivariance/segmented_polynomials/{segmented_operand,path,operation,segmented_polynomial}.py` | Segment slices, path indices/coefficients, operation buffer mappings, `SegmentedPolynomial` | The descriptor format interpreted by our torch executor |
| `cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial.py` | `SegmentedPolynomial(method='naive')` | Independent CPU FP32 reference; GPU runner requests `fused_tp` and rejects fallback |
| `cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial_naive.py` | Inspected `_tensor_product_fx` and dtype behavior | Confirms the segment/path einsum interpretation and CPU reference semantics |
| `cuequivariance_torch/cuequivariance_torch/primitives/{segmented_polynomial_fused_tp,segmented_polynomial_uniform_1d}.py` | Inspected math dtype limits and dispatch to ops | Defines the limits of the planned GPU integration |
| `cuequivariance_jax/cuequivariance_jax/segmented_polynomials/{utils,segmented_polynomial_naive,segmented_polynomial_uniform_1d}.py` | Inspected `math_dtype_for_naive_method` and compute dtype handling | JAX/TPU port design only; JAX was not installed or run |

`segmented.py` is an original, small interpreter of that public format. Its
binary-einsum-to-GEMM lowering, BF16 split arithmetic, custom backward, force
certificate, allocation and MD experiments are CertiPrec code. Descriptor
mathematics and numerical coefficient values are provided by cuEquivariance.
The general `descriptors.symmetric_contraction` and MACE U-matrix machinery are
not used: v0 uses a channelwise self product restricted to scalar outputs to
keep correlation order two small.

## License and attribution

cuEquivariance's source headers state:

    Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

The upstream license is in `LICENSES/LICENSE`, with third-party notices in
`LICENSES/Third_party_attr.txt` and `LICENSES/Third_party_attr.json`.
See the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
CertiPrec's original code keeps the repository's MIT license; dependent code
and upstream coefficient generation keep their own licenses. The spherical
harmonics source identifies an adaptation from e3nn-jax; its attribution stays
with the installed upstream package. No claim is made that these components
or cuEquivariance's CUDA kernels were written for CertiPrec.

The BF16x3 formula is inspired by FACTO, arXiv:2611.03276, Appendix Note 2,
equations A3–A4, supplied as `/workspace/mlip_paper.txt`, and its cited Henry,
Tang and Heinecke paper, [arXiv:1904.06376](https://arxiv.org/abs/1904.06376).
No FACTO code, weights, dataset, or reported numerical results are copied into
our evidence files. Our LJ results come from the scripts in this repository.

## How cuEquivariance handles precision

The inspected torch `SegmentedPolynomial` accepts `math_dtype` independently
of the input/output dtype. Its `naive` implementation converts path operands
and coefficients to that dtype, evaluates `torch.einsum`, and converts outputs
according to the output dtype mapping. If unspecified it follows input dtype;
explicit math dtype is retained across module dtype conversions. Our CPU
reference requests `math_dtype=torch.float32`.

Torch `fused_tp` accepts only FP32 or FP64 math and defaults to FP32. Torch
`uniform_1d` also accepts only FP32/FP64 math; its default follows supported
input dtype and otherwise uses FP32. `indexed_linear` does not support an
explicit math dtype. These are backend-specific rules, not a universal BF16
switch. The [official API documentation](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SegmentedPolynomial.html)
records the same distinctions.

In JAX, `math_dtype_for_naive_method` returns compute dtype and lax precision:
unspecified math dtype uses the I/O dtype with `Precision.HIGHEST`; a JAX dtype
name chooses that dtype with `HIGHEST`; `tensor_float32` chooses FP32 with
`Precision.HIGH`. The JAX `uniform_1d` path accepts explicit float32, float64,
float16 or bfloat16 compute dtypes and otherwise selects FP64 only for FP64
buffers, FP32 otherwise. None of these dtype choices is our adaptive BF16xk
policy. Actual arithmetic depends on the selected backend and lowering.

## Where BF16xk would enter on GPU — PENDING

For dense channel mixing, replace a GEMM with retained BF16 slice GEMMs,
FP32 outputs and FP32 accumulation, including in backward. For segmented TP,
keep cuEq descriptors and layout/index logic; replace the binary contractions
inside a kernel or provide a new backend implementation. Bilinear kernels can
also be called k times on slice pairs to validate numerics. Trilinear kernels
need a consistent contraction decomposition; naively applying three bilinear
pairs to three inputs is incorrect. Policy dispatch, coefficient handling,
FP32 neighbor reductions and memory traffic must all be accounted for.

`colab_a100_bench.py` has two explicit measurements, both still **PENDING**:

- Dense torch CUDA GEMMs compare strict FP32, TF32, and BF16x1/x2/x3/x6. BF16
  operands are passed to `torch.mm(..., out_dtype=torch.float32)`; the script
  refuses to replace this with FP32 input GEMMs. This is the hardware BF16
  tensor-core experiment. See the [torch.mm dtype contract](https://docs.pytorch.org/docs/2.15/generated/torch.mm.html).
- A cuEq bilinear equivariant linear descriptor runs real `fused_tp` CUDA
  kernels. BF16xk here means k FP32 kernel calls on BF16-rounded slices, with
  FP32 sums. This validates numerical integration and launch/splitting cost;
  it does not implement BF16 tensor-core cuEq kernels. TF32 is unsupported for
  this API and is measured only in the separate dense benchmark. The runner
  rejects a fallback to `naive`.

A fused BF16xk cuEq tensor-core backend, GPU model forces/MD and measured
speedups remain **PENDING**. The supplied CUDA ops are separately installed
runtime dependencies; their distribution terms are not relicensed by this
repository. The open-source frontend's Apache license does not replace the
terms attached to binary ops packages.
