"""
Main experiment: can restricted-control (global-rotation) randomized
measurements detect that a device is running a *noisy, miscalibrated*
realization of an intended Hamiltonian, rather than the Hamiltonian as
designed?

Protocol
--------
1. Intended Hamiltonian H0 (4-qubit Heisenberg-type chain, uniform bonds).
2. "Realized" (device) Hamiltonian H_eps = H0 + random per-bond coupling
   errors of relative strength eps (fixed random pattern, scaled by eps).
3. The physical state actually produced by the device is the ground state
   of H_eps (what you get if you run the intended state-prep protocol on
   miscalibrated hardware).
4. We are only allowed global (identical, all-qubit) random single-qubit
   rotations + computational-basis measurement (Van Kirk et al.'s
   restricted-control model) -- no per-qubit addressing, no full tomography.
5. From N such randomized-measurement shots on the device state, we estimate
   every local Hamiltonian term's expectation value using the numerically
   constructed shadow/kernel estimator (shadows.RestrictedFrame), and compare
   the inferred energy-term profile against the profile predicted for the
   *intended* H0's own ground state.
6. We quantify detectability (profile distance vs eps, vs N) and compare
   against a standard unrestricted per-qubit classical-shadow baseline.
"""

import json
import numpy as np
from numpy.random import default_rng

from core import xxz_hamiltonian, hamiltonian_terms, ground_state
from shadows import (
    RestrictedFrame, random_su2, global_rotation,
    measure_full_control, full_shadow_single_qubit_snapshot, kron_list, I2,
)

N_QUBITS = 4
MC_FRAME_SAMPLES = 6000
SEED = 7

rng = default_rng(SEED)

# -----------------------------------------------------------------
# Intended Hamiltonian and its term basis
# -----------------------------------------------------------------
H0 = xxz_hamiltonian(N_QUBITS, J=1.0, Delta=1.0, h=0.1)
e0, psi0 = ground_state(H0)
terms = hamiltonian_terms(N_QUBITS)
term_names = list(terms.keys())

ideal_profile = np.array([
    np.real(psi0.conj() @ (terms[name] @ psi0)) for name in term_names
])

# fixed random per-bond miscalibration pattern (so eps alone controls magnitude)
bond_errors = rng.uniform(-1, 1, size=N_QUBITS - 1)


def noisy_hamiltonian(eps):
    J = 1.0 + eps * bond_errors
    return xxz_hamiltonian(N_QUBITS, J=J, Delta=1.0, h=0.1)


# -----------------------------------------------------------------
# Restricted-control frame (built once, reused for every eps / N)
# -----------------------------------------------------------------
print("Building restricted-control measurement frame...")
frame = RestrictedFrame(N_QUBITS, mc_samples=MC_FRAME_SAMPLES, seed=SEED + 1)
print(f"  visible-space rank: {np.linalg.matrix_rank(frame.M_mat, tol=1e-6)} / {frame.D}")

print("Precomputing dual operators M^{-1}(O) for each Hamiltonian term...")
duals = {name: frame.dual_operator(terms[name]) for name in term_names}


def vectorized_restricted_estimates(psi, n_shots, rng_local):
    """
    Draw n_shots restricted (global-rotation) measurement shots from state psi,
    and return (estimates, stderrs) arrays over all tracked Hamiltonian terms,
    using fully vectorized numpy evaluation (fast enough for large shot counts).
    """
    d = 2 ** N_QUBITS
    U_batch = np.empty((n_shots, d, d), dtype=complex)
    b_arr = np.empty(n_shots, dtype=int)
    for s in range(n_shots):
        u = random_su2(rng_local)
        U = global_rotation(N_QUBITS, u)
        U_batch[s] = U
        probs = np.abs(U @ psi) ** 2
        probs = probs / probs.sum()
        b_arr[s] = rng_local.choice(d, p=probs)

    # v_s = U_s^dagger |b_s>  = conj( row b_s of U_s )
    v_batch = np.conj(U_batch[np.arange(n_shots), b_arr, :])  # (n_shots, d)

    ests, errs = [], []
    for name in term_names:
        X = duals[name]
        vals = np.real(np.einsum("si,ij,sj->s", np.conj(v_batch), X, v_batch))
        ests.append(vals.mean())
        errs.append(vals.std(ddof=1) / np.sqrt(n_shots))
    return np.array(ests), np.array(errs)


def full_control_estimates(psi, n_shots, rng_local):
    """Standard unrestricted per-qubit classical-shadow baseline."""
    n = N_QUBITS
    shots_data = [measure_full_control(rng_local, psi, n) for _ in range(n_shots)]
    ests, errs = [], []
    for name in term_names:
        O = terms[name]
        vals = []
        for us, bits in shots_data:
            snaps = [full_shadow_single_qubit_snapshot(us[i], bits[i]) for i in range(n)]
            rho_hat = kron_list(snaps)
            vals.append(np.real(np.trace(rho_hat @ O)))
        vals = np.array(vals)
        ests.append(vals.mean())
        errs.append(vals.std(ddof=1) / np.sqrt(n_shots))
    return np.array(ests), np.array(errs)


# -----------------------------------------------------------------
# Sweep 1: detection sensitivity vs. noise strength eps (fixed N)
# -----------------------------------------------------------------
print("\nSweep 1: profile distance vs. noise strength eps (restricted control)")
eps_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
N_fixed = 8000
N_REPEATS = 6

sweep1 = []
for eps in eps_values:
    Heps = noisy_hamiltonian(eps)
    _, psi_eps = ground_state(Heps)
    true_profile_eps = np.array([
        np.real(psi_eps.conj() @ (terms[name] @ psi_eps)) for name in term_names
    ])
    dist_true_vs_ideal = np.linalg.norm(true_profile_eps - ideal_profile)

    trial_dists = []
    mean_stderrs = []
    for _ in range(N_REPEATS):
        est, err = vectorized_restricted_estimates(psi_eps, N_fixed, rng)
        trial_dists.append(np.linalg.norm(est - ideal_profile))
        mean_stderrs.append(np.mean(err))
    dist_mean = float(np.mean(trial_dists))
    dist_std = float(np.std(trial_dists, ddof=1))

    sweep1.append(dict(eps=eps, dist_est_vs_ideal=dist_mean, dist_est_std=dist_std,
                        dist_true_vs_ideal=dist_true_vs_ideal,
                        mean_stderr=float(np.mean(mean_stderrs))))
    print(f"  eps={eps:5.2f}  ||est-ideal||={dist_mean:.4f}±{dist_std:.4f}  "
          f"||true-ideal||={dist_true_vs_ideal:.4f}  mean_stderr={np.mean(mean_stderrs):.4f}")

# -----------------------------------------------------------------
# Sweep 2: estimation error vs. shot count N (fixed, detectable eps)
#          restricted control vs. full (unrestricted) control baseline
# -----------------------------------------------------------------
print("\nSweep 2: estimation error vs. shot count (eps=0.2)")
eps_demo = 0.2
Heps_demo = noisy_hamiltonian(eps_demo)
_, psi_demo = ground_state(Heps_demo)
true_profile_demo = np.array([
    np.real(psi_demo.conj() @ (terms[name] @ psi_demo)) for name in term_names
])

shot_counts = [200, 500, 1000, 2000, 5000, 10000]
sweep2 = []
for N in shot_counts:
    est_r, err_r = vectorized_restricted_estimates(psi_demo, N, rng)
    rmse_r = np.sqrt(np.mean((est_r - true_profile_demo) ** 2))

    est_f, err_f = full_control_estimates(psi_demo, min(N, 3000), rng)  # cap for speed
    rmse_f = np.sqrt(np.mean((est_f - true_profile_demo) ** 2))

    sweep2.append(dict(N=N, rmse_restricted=float(rmse_r), rmse_full=float(rmse_f)))
    print(f"  N={N:6d}  RMSE restricted={rmse_r:.4f}  RMSE full-control(N={min(N,3000)})={rmse_f:.4f}")

# -----------------------------------------------------------------
# Save everything
# -----------------------------------------------------------------
results = dict(
    term_names=term_names,
    ideal_profile=ideal_profile.tolist(),
    bond_errors=bond_errors.tolist(),
    sweep1=sweep1,
    sweep2=sweep2,
    frame_rank=int(np.linalg.matrix_rank(frame.M_mat, tol=1e-6)),
    frame_dim=int(frame.D),
)
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results.json")
