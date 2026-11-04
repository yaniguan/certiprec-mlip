"""Reproduce the v0 evidence chain (CPU). Writes results/v0/*.json and figures."""
import argparse, json, math, time, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from certiprec import precision as P
from certiprec.lj import sample_configs, lj_energy_forces
from certiprec.train import train
from certiprec.layers import set_schemes
from certiprec import certify as C
from certiprec.md import run_nve, drift_and_rms

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/v0")
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--dt", type=float, default=0.004)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--target_rms", type=float, default=None,
                help="energy-walk tolerance per atom over the horizon (reduced units)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(4)
R = {"meta": {"torch": torch.__version__, "device": "cpu", "date": time.strftime("%Y-%m-%d %H:%M %Z")}}
SCH = ["fp32", "tf32", "bf16x1", "bf16x3", "bf16x6", "bf16x9"]

# (a) matmul error vs fp64 ------------------------------------------------------
g = torch.Generator().manual_seed(0)
mm = {}
for K in (16, 64, 256, 1024):
    X = torch.randn(256, K, generator=g); W = torch.randn(K, 256, generator=g) / math.sqrt(K)
    ref = X.double() @ W.double()
    mm[K] = {}
    for s in SCH:
        y = P.emu_matmul_raw(X, W, s).double()
        rel = ((y - ref).norm() / ref.norm()).item()
        bnd = P.error_bound(X, W, s)
        mm[K][s] = {"rel_fro_err": rel, "max_abs_err": (y - ref).abs().max().item(),
                    "bound_holds": bool(((y - ref).abs() <= bnd + 1e-30).all()),
                    "bound_tightness_median": float(((y - ref).abs() / bnd).median())}
R["a_matmul_error"] = mm
print("(a) done", {s: mm[256][s]["rel_fro_err"] for s in SCH})

# train MLIP on LJ ----------------------------------------------------------------
t0 = time.time()
Xc, Ec, Fc, box = sample_configs()
model, tr = train(Xc, Ec, Fc, box, epochs=a.epochs)
tr["train_seconds"] = time.time() - t0; tr["n_configs"] = len(Xc); tr["n_atoms"] = Xc.shape[1]
R["mlip_fit"] = tr; print("fit", tr)
torch.save(model.state_dict(), os.path.join(a.out, "mlip_lj.pt"))
N = Xc.shape[1]

# (b) force error per uniform scheme + sensitivity table ---------------------------
test = [Xc[i] for i in range(0, len(Xc), 20)]
ref = C.reference_forces(model, test, box)
fe = {s: C.force_rms_error(model, test, box, s, ref) for s in SCH}
R["b_force_rms_error_vs_fp64"] = fe; print("(b)", fe)
floor, table, _ = C.sensitivity_table(model, test, box)
R["sensitivity"] = {"fp32_floor": floor, "per_layer": {f"L{l}:{s}": v for (l, s), v in table.items()}}

# certificate -> budget -> allocation ----------------------------------------------
horizon = a.steps * a.dt; kT = 0.9
flops = C.layer_flops(model, N)
target = a.target_rms
if target is None:
    # Default tolerance: allow the energy walk of a force noise 1.5x the FP32 floor.
    target = C.predict_energy_walk(1.5 * floor, horizon, a.dt, N, kT)
budget = C.force_budget(target, horizon, a.dt, N, kT)
alloc = C.allocate(floor, table, flops, budget)
R["allocation"] = {"target_rms_per_atom": target, "force_budget": budget, **alloc,
                   "layer_flops": flops}
print("alloc", R["allocation"])
pol = alloc["policy"]
fe["certiprec"] = C.force_rms_error(model, test, box, pol, ref)

# (c) NVE ---------------------------------------------------------------------------
pos0, _ = (Xc[len(Xc) // 2], None)
policies = {"fp64": "fp64", **{s: s for s in SCH}, "certiprec": pol}
nve = {k: {"drift": [], "rms": [], "final_abs_dH": []} for k in policies}
traces = {}
for seed in range(a.seeds):
    rng = np.random.default_rng(100 + seed)
    v0 = rng.standard_normal(pos0.shape) * math.sqrt(kT); v0 -= v0.mean(0)
    for k, p in policies.items():
        t1 = time.time()
        ts, hs, _, _ = run_nve(model, pos0, v0, box, p, a.dt, a.steps)
        d, r = drift_and_rms(ts, hs, N)
        nve[k]["drift"].append(d); nve[k]["rms"].append(r)
        nve[k]["final_abs_dH"].append(float(abs(hs[-1] - hs[0]) / N))
        if seed == 0: traces[k] = (ts.tolist(), ((hs - hs[0]) / N).tolist())
        print(f"seed {seed} {k:10s} drift={d:.3e} rms={r:.3e} ({time.time()-t1:.0f}s)", flush=True)
for k in nve:
    nve[k]["mean_abs_drift"] = float(np.mean(np.abs(nve[k]["drift"])))
    nve[k]["geomean_rms"] = float(np.exp(np.mean(np.log(nve[k]["rms"]))))
    nve[k]["rms_final_dH"] = float(np.sqrt(np.mean(np.square(nve[k]["final_abs_dH"]))))
    s = fe.get(k)
    nve[k]["predicted_walk"] = C.predict_energy_walk(s, horizon, a.dt, N, kT) if s else None
R["c_nve"] = {"dt": a.dt, "steps": a.steps, "seeds": a.seeds, "kT_init": kT, "N": N, "units": "LJ reduced (eps, sigma, m)", "runs": nve}
R["c_traces_seed0"] = traces

# (d) cost model ------------------------------------------------------------------
cost = {s: C.policy_cost(flops, [s] * len(flops)) for s in ["bf16x1", "bf16x3", "bf16x6", "bf16x9"]}
cost["certiprec"] = alloc["cost"]
# A100 dense peak (NVIDIA datasheet): BF16 312, TF32 156, FP32 (non-tensor) 19.5 TFLOPS
peak = {"bf16": 312e12, "tf32": 156e12, "fp32": 19.5e12}
tot = sum(flops)
R["d_cost_model"] = {"tensor_core_products_relative_to_bf16x3": {k: v / cost["bf16x3"] for k, v in cost.items()},
                     "ideal_A100_time_ratio_vs_fp32_simt": {**{k: (v / peak["bf16"]) / (tot / peak["fp32"]) for k, v in cost.items()},
                                                            "tf32": (tot / peak["tf32"]) / (tot / peak["fp32"])},
                     "note": "Peak-FLOP model only; ignores splitting/memory overhead. NOT a measured speedup."}
json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=str)
print("saved")
