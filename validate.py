import numpy as np
from numpy.random import default_rng
from core import xxz_hamiltonian, hamiltonian_terms, ground_state
from shadows import RestrictedFrame, measure_restricted, full_shadow_estimate, measure_full_control

n = 4
rng = default_rng(42)

H0 = xxz_hamiltonian(n, J=1.0, Delta=1.0, h=0.1)
e0, psi0 = ground_state(H0)
terms = hamiltonian_terms(n)

print("Building restricted-control frame (Monte Carlo over SU(2))...")
frame = RestrictedFrame(n, mc_samples=4000, seed=1)
print("Frame matrix rank:", np.linalg.matrix_rank(frame.M_mat, tol=1e-6), "/", frame.D)

# ground truth expectation values
truth = {name: np.real(psi0.conj() @ (op @ psi0)) for name, op in terms.items()}

n_shots = 20000
shots = [measure_restricted(rng, psi0, n) for _ in range(n_shots)]

print(f"\n{'term':10s} {'truth':>10s} {'restricted-est':>16s} {'stderr':>10s}")
for name, op in terms.items():
    est, err = frame.kernel_estimate(op, shots)
    print(f"{name:10s} {truth[name]:10.4f} {est:16.4f} {err:10.4f}")
