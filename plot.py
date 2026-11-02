import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    R = json.load(f)

plt.rcParams.update({"font.size": 12, "figure.dpi": 140})

# ---------------- Figure 1: detection sensitivity vs eps ----------------
eps = np.array([p["eps"] for p in R["sweep1"]])
dist_est = np.array([p["dist_est_vs_ideal"] for p in R["sweep1"]])
dist_est_std = np.array([p["dist_est_std"] for p in R["sweep1"]])
dist_true = np.array([p["dist_true_vs_ideal"] for p in R["sweep1"]])
noise_floor = np.mean([p["mean_stderr"] for p in R["sweep1"]]) * np.sqrt(len(R["term_names"]))

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.errorbar(eps, dist_est, yerr=dist_est_std, marker="o", capsize=3,
            label="Estimated profile distance\n(restricted-control measurement)", color="#1f77b4")
ax.plot(eps, dist_true, marker="s", linestyle="--",
        label="True profile distance\n(exact ground state)", color="#d62728")
ax.axhline(noise_floor, color="gray", linestyle=":", label="Expected shot-noise floor")
ax.set_xlabel(r"Relative coupling-error strength $\epsilon$")
ax.set_ylabel(r"$\|\,\langle h\rangle_{\rm est} - \langle h\rangle_{\rm ideal}\,\|_2$")
ax.set_title("Detecting a miscalibrated Hamiltonian\nvia restricted (global-rotation) randomized measurements")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig1_detection_vs_eps.png")
print("Saved fig1_detection_vs_eps.png")

# ---------------- Figure 2: RMSE vs shot count ----------------
Ns = np.array([p["N"] for p in R["sweep2"]])
rmse_r = np.array([p["rmse_restricted"] for p in R["sweep2"]])
rmse_f = np.array([p["rmse_full"] for p in R["sweep2"]])

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.loglog(Ns, rmse_r, marker="o", label="Restricted control\n(global rotation, this work)", color="#1f77b4")
ax.loglog(Ns, rmse_f, marker="s", label="Full control\n(per-qubit classical shadows)", color="#2ca02c")
ref = rmse_r[0] * np.sqrt(Ns[0] / Ns)
ax.loglog(Ns, ref, linestyle=":", color="gray", label=r"$1/\sqrt{N}$ reference")
ax.set_xlabel("Number of measurement shots $N$")
ax.set_ylabel("RMSE of estimated Hamiltonian-term profile")
ax.set_title("Sample-efficiency: restricted vs. full control")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig2_rmse_vs_shots.png")
print("Saved fig2_rmse_vs_shots.png")

# ---------------- Figure 3: visible-space rank ----------------
fig, ax = plt.subplots(figsize=(5, 4))
rank, dim = R["frame_rank"], R["frame_dim"]
ax.bar(["Visible", "Invisible"], [rank, dim - rank], color=["#1f77b4", "#cccccc"])
ax.set_ylabel("Number of Pauli-basis directions")
ax.set_title(f"Visible space under global SU(2) control\n({rank} / {dim} operator directions, n=4 qubits)")
fig.tight_layout()
fig.savefig("fig3_visible_space.png")
print("Saved fig3_visible_space.png")
