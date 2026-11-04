import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1] if len(sys.argv) > 1 else "results/v0"
R = json.load(open(os.path.join(out, "results.json")))
mm = R["a_matmul_error"]; Ks = sorted(mm, key=int)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for s in mm[Ks[0]]:
    ax[0].loglog([int(k) for k in Ks], [mm[k][s]["rel_fro_err"] for k in Ks], "o-", label=s)
ax[0].set_xlabel("inner dim K"); ax[0].set_ylabel("rel. Frobenius error vs FP64"); ax[0].legend(); ax[0].set_title("(a) matmul error")
fe = R["b_force_rms_error_vs_fp64"]
ax[1].bar(list(fe), list(fe.values())); ax[1].set_yscale("log"); ax[1].set_title("(b) force RMS error vs FP64 model"); ax[1].tick_params(axis="x", rotation=45)
for k, (t, e) in R["c_traces_seed0"].items():
    ax[2].plot(t, e, label=k, lw=0.8)
ax[2].set_xlabel("t (LJ units)"); ax[2].set_ylabel("[H(t)-H(0)]/N"); ax[2].set_yscale("symlog", linthresh=1e-7); ax[2].legend(fontsize=7); ax[2].set_title("(c) NVE energy, seed 0")
plt.tight_layout(); plt.savefig(os.path.join(out, "evidence_v0.png"), dpi=130)
r = R["c_nve"]["runs"]
fig, ax = plt.subplots(figsize=(5, 4))
for k, v in r.items():
    if v["predicted_walk"]:
        ax.loglog(v["predicted_walk"], v["rms_final_dH"], "o"); ax.annotate(k, (v["predicted_walk"], v["rms_final_dH"]))
lim = [1e-9, 1e-2]; ax.plot(lim, lim, "k--", lw=0.7)
ax.set_xlabel("certificate-predicted RMS |dH|/N"); ax.set_ylabel("observed RMS |dH(T)|/N"); ax.set_title("NVE certificate vs observation")
plt.tight_layout(); plt.savefig(os.path.join(out, "certificate_v0.png"), dpi=130)
print("figs saved")
