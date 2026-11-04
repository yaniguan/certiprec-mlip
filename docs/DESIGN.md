# Design

CertiPrec v0 asks whether each layer can use fewer BF16 products while keeping a
specified force error small. It provides a local arithmetic bound and a force
certificate on a finite set of configurations. A global energy-drift certificate
is future work. GPU results: **PENDING**.

## BF16 splitting

Let B round to BF16 with round-to-nearest-even, and let u = 2^-8. Starting with
r0 = x, compute xi = B(ri), then r(i+1) = ri - xi in FP32. Subtraction is exact
under the normal-range assumptions used here. Three slices reconstruct an FP32
number exactly in the tested normal range; arbitrary subnormals are excluded.
The slices are stored as FP32 tensors whose values are representable in BF16.
CPU products use FP32 arithmetic. This emulates the input rounding and FP32
accumulation contract, but hardware accumulation order can differ.

For X and W, choose a set P of slice pairs and calculate

    Y_P = sum_(i,j in P) Xi @ Wj.

Products are added from the smaller slice orders to the larger ones. There is
no claim of bitwise agreement with a particular tensor-core instruction.

| Scheme | Slices | Retained pairs (i,j) |
| --- | --- | --- |
| BF16x1 | 1 | (0,0) |
| BF16x2 | 2 | (0,0), (1,0) |
| BF16x3 | 2 | (0,0), (0,1), (1,0) |
| BF16x6 | 3 | (0,0), (0,1), (1,0), (1,1), (0,2), (2,0) |
| BF16x9 | 3 | all nine pairs; available as a diagnostic |

BF16x2 is an explicitly asymmetric experiment: only the left operand is
refined. It has the same worst-case error order as BF16x1 and is not generally
better for every layer or its backward pass. BF16x3 is the three-product scheme
in FACTO Appendix Note 2, equations A3–A4, following Henry et al. (2019).

TF32 on CPU rounds each FP32 operand to 11 significand bits before FP32 matmul.
It is a numerical emulation, with no CPU speed claim.

## Local arithmetic certificate

Let A = |X| @ |W|, K be the reduction length, and g(t) = t*u32/(1-t*u32), with
u32 = 2^-24. Assume finite operands, round-to-nearest arithmetic, no overflow,
and no subnormal effects in slices/products. Require t*u32 < 1. These are
relative-error assumptions: small subnormal values need extra absolute terms.

For n slices,

    |Xi| <= (1+u) u^i |X|,
    |Rn| <= u^n |X|.

For the omitted pairs within the n-by-n expansion, sum
(1+u)^2 u^(i+j). Bound the two residual tails and their product by
2 u^n (1+u^n) + u^(2n). Call the sum of these terms T. If q pairs are retained,
let M = sum_(i,j in P) (1+u)^2 u^(i+j). Then a conservative elementwise bound is

    |Y_P - XW| <= [T + (a + b*(1+a))*M] A,
    a = g(K+2), b = g(q).

The implementation uses FP64 to evaluate A. This bound includes splitting
truncation, each FP32 dot product and the FP32 sum of products. It is loose,
especially for cancellation. `error_bound` returns this bound; it does not
turn it into an interval-arithmetic proof. The experiment checks every output
entry against the same FP64 reference. FP32 and TF32 use their corresponding
roundoff constants, including the operand-rounding factor on accumulation.

## Equivariant backbone

The default model is `EquivariantMLIP` in `equivariant.py`:

1. Form directed displacement vectors for all distinct atom pairs. A free
   eight-atom cluster is used for experiments. The optional scalar periodic
   box is axis-aligned; rotating a periodic cell requires rotating the cell too.
2. Calculate smooth Gaussian radial features and cuEq spherical harmonics with
   l=0 and l=1, in `ir_mul` layout.
3. Use a cuEq channelwise tensor product to couple radial scalars and harmonics.
   Sum messages into atoms.
4. Use a cuEq equivariant linear descriptor to mix scalar and vector channels.
5. Couple atom features with themselves using a channelwise product restricted
   to even scalars. This supplies degree-two invariants, including vector dot
   products. It is a small correlation-two analogue, not the general MACE
   symmetric-contraction basis.
6. Concatenate radial sums and quadratic invariants, then use a scalar linear
   readout to give total energy.

The backbone carries nonzero vector features that affect the scalar energy.
Its exact-arithmetic energy is O(3) invariant; FP32 forces transform as vectors
up to rounding. The experiment rotates one held-out free-cluster configuration;
the tests also check a nontrivial orthogonal transform and translations.
Low precision can break numerical equivariance, and that error is measured.
The earlier radial-only `RadialMLIP` remains as a baseline.

`segmented.py` reads actual SegmentedTensorProduct operands, segment shapes,
path indices, and Clebsch–Gordan coefficient tensors. Each path is evaluated
as binary einsums, each lowered to a batched GEMM with selectable arithmetic.
Reshapes, FP32 segment accumulation, radial functions, geometry, and neighbor
sums remain native arithmetic. Constants with no dynamic input need no GEMM.
All binary path contractions, including coefficient contractions, use the
selected scheme. The interpreter supports the descriptors in this model; it
is not a general indexed or multiple-output cuEq runtime.

FP32/FP64 use native differentiable torch operations. Low-precision matmuls
have an explicit backward that uses the same scheme for both input gradients.
This is a surrogate derivative of quantized arithmetic: quantized forward
energy is not a smooth function whose exact derivative is that backward.
Consequently low-precision forces need not be exactly conservative. The
certificate concerns the force values actually evaluated by that backward.
Low-precision higher-order training derivatives are not certified or tested.

Only the final 45 readout parameters are fitted, with fixed seeded backbone
features. Scaled, ridge-regularized least squares uses energy/atom and force
components with equal weight on 48 configurations; 12 are withheld. This is
a quick local LJ fit, not a broadly trained or validated interatomic model.
No checkpoint is needed. Training does not use low-precision surrogate forces.

## Allocation and finite-set force certificate

`allocate` enumerates a small candidate space (four choices, five layers).
Its cost is estimated forward FLOPs times retained product count. It proposes
a policy from individual layer sensitivities and a conservative additive score.
This proposal score is not a proof for combined layer changes.

`certify_policy` then constructs prefix policies p0,...,pL: p0 is FP32 and each
successive prefix changes one layer. On a fixed set S, stack all force
components into a vector. With the RMS norm,

    ||F_pL - F_64||_S
      <= ||F_p0 - F_64||_S + sum_l ||F_pl - F_p(l-1)||_S.

This telescopes exactly, includes cross-layer interactions through the actual
prefix evaluations, and requires no independence assumption. The summed
measured norms form the finite-set triangle bound. The reference is the same
model evaluated in FP64, not the physical LJ force. Numerical norm evaluation
is in FP64; this is not a formal outward-rounded interval proof.

The default budget is max(2 times the all-BF16x3 proposal score, 0.03 times the
all-BF16x1 score). It is a transparent calibration choice, not a physical drift
tolerance. `--force-budget` overrides it. A proposed policy is used only if its
checked prefix bound meets that budget. Otherwise the runner checks uniform
BF16x3, BF16x6, then FP32. An infeasible budget raises an error. Separate
held-out audits are reported even when they fail. No policy is changed using
those audit results. No generalization or per-atom guarantee follows.

## Link to drift

For continuous dynamics with reference potential U and force error deltaF,

    dH/dt = v dot deltaF,
    |H(T)-H(0)| <= integral_0^T ||v||_2 ||deltaF||_2 dt.

A trajectory-uniform force norm bound would give a deterministic work bound.
The finite calibration certificate is not such a bound. RMS across sampled
configurations is also not a uniform force bound. Velocity Verlet introduces
its own integration error; this must be accounted for separately. We do not
claim to certify the total discrete NVE drift.

The old helpers `predict_energy_walk` and `force_budget` remain for comparison.
Their random-walk estimate is

    RMS(deltaH/N) ~ sigmaF * dt * sqrt(3*N*kT/m) * sqrt(T/dt) / N.

It requires zero-mean, decorrelated force noise. The current allocator does not
use this assumption and the figures do not call this estimate a certificate.

NVE uses FP64 positions and velocities, reduced LJ units and unit masses. Every
policy sees the same initial positions and velocities for each seed. Sampling
always measures the Hamiltonian with the FP64 fitted model, separating force
error from energy measurement rounding. Drift is the least-squares slope of
(H(t)-H(0))/N; RMS is after subtracting that line. Short trajectories cannot
establish long-time stability, and timestep error can dominate FP32/BF16x3.

## Profiling and reproducibility

`CERTIPREC_PROFILE=1` or `--profile` enables inclusive per-layer CPU timers and
estimated forward FLOP counters. Counts include selected slice-product counts
for segmented and linear layers; native radial/scatter work and backward FLOPs are
excluded. Timers are CPU wall time, with no claim of GPU kernel timing.
`--torch-profiler` records one BF16x3 force call and saves a small text table,
not a large trace. The GPU runner uses synchronized CUDA events separately.

Each experiment and seed/policy pair is saved atomically. A hash of source
files, arguments and torch version identifies reusable stages. Fitting is
always recreated deterministically. Changing source or experiment flags
invalidates stage caches. Thread count defaults to one to keep tiny CPU
contractions fast. Figure generation consumes only completed JSON results.

## JAX/TPU plan — PENDING

Reuse the same cuEq descriptors and segment/path tables. Port the binary
contractions to `jax.lax.dot_general`, with BF16 slice inputs and explicitly
FP32 preferred accumulation. Use custom VJP rules for the same backward
policy. Keep FP32 geometry and reductions, and compare energy, force and
rotation errors against the torch CPU reference before timing anything.

Compile each precision policy statically with `jit`; bucket atoms if a later
per-atom policy is worthwhile. Avoid dynamic branches within tensor-core tiles.
Test XLA precision and lowering to ensure BF16 products do not round the output
to BF16. Then repeat finite-set certificate audits and multi-seed NVE checks.
An online controller would need to re-certify encountered configurations or
establish a uniform force/Jacobian bound, plus account for timestep error.
No JAX/TPU implementation, measurements, GPU speedup, or global drift guarantee
is part of v0. All GPU results remain **PENDING**.
