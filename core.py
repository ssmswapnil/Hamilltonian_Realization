"""
Hamiltonian-learning mini-project core library.

Implements:
  - A 4-qubit XXZ spin-chain Hamiltonian (intended vs. noisy "realized" version)
  - The restricted-control randomized-measurement scheme of Van Kirk, Cotler,
    Huang & Lukin (arXiv:2212.06084): identical random single-qubit rotation
    applied to every qubit ("global SU(2) control"), followed by computational
    basis measurement.
  - A numerically constructed classical-shadow / kernel estimator for this
    restricted ensemble (Monte Carlo estimate of the measurement channel M,
    pseudo-inverted to obtain the dual frame operators M^{-1}(O) for target
    Hamiltonian terms O), used to estimate local energy-density terms from
    simulated measurement data.
  - A standard (unrestricted) per-qubit random-Clifford classical-shadow
    baseline, for comparison.

All computations are classical/exact-diagonalization based (n=4 qubits,
Hilbert space dimension 16) -- no external quantum SDK required.
"""

import numpy as np
from numpy.random import default_rng
from scipy.linalg import eigh, expm

# ----------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS_1Q = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def op_on_sites(n, ops):
    """ops: dict {site_index: 2x2 matrix}. Missing sites get identity."""
    mats = [ops.get(i, I2) for i in range(n)]
    return kron_list(mats)


def all_pauli_strings(n):
    """Return list of (label, matrix) for all 4^n n-qubit Pauli strings."""
    labels = ["I", "X", "Y", "Z"]
    strings = [""]
    for _ in range(n):
        strings = [s + l for s in strings for l in labels]
    out = []
    for s in strings:
        mats = [PAULIS_1Q[c] for c in s]
        out.append((s, kron_list(mats)))
    return out


# ----------------------------------------------------------------------
# Hamiltonian construction
# ----------------------------------------------------------------------

def xxz_hamiltonian(n, J, Delta, h, open_bc=True):
    """
    H = sum_<i,i+1> J_i (XX + YY + Delta_i ZZ) + sum_i h_i Z_i

    J, Delta can be scalars or length-(n-1) arrays (per-bond).
    h can be scalar or length-n array (per-site).
    """
    if np.isscalar(J):
        J = np.full(n - 1, J)
    if np.isscalar(Delta):
        Delta = np.full(n - 1, Delta)
    if np.isscalar(h):
        h = np.full(n, h)

    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n - 1):
        H += J[i] * op_on_sites(n, {i: X, i + 1: X})
        H += J[i] * op_on_sites(n, {i: Y, i + 1: Y})
        H += J[i] * Delta[i] * op_on_sites(n, {i: Z, i + 1: Z})
    for i in range(n):
        H += h[i] * op_on_sites(n, {i: Z})
    return H


def hamiltonian_terms(n):
    """
    Returns a dict {term_name: matrix} for every local term that appears
    in the XXZ chain Hamiltonian above -- these are exactly the operators
    we will try to "learn" (estimate expectation values of) from restricted
    randomized measurements.
    """
    terms = {}
    for i in range(n - 1):
        terms[f"XX_{i}{i+1}"] = op_on_sites(n, {i: X, i + 1: X})
        terms[f"YY_{i}{i+1}"] = op_on_sites(n, {i: Y, i + 1: Y})
        terms[f"ZZ_{i}{i+1}"] = op_on_sites(n, {i: Z, i + 1: Z})
    for i in range(n):
        terms[f"Z_{i}"] = op_on_sites(n, {i: Z})
    return terms


def ground_state(H):
    evals, evecs = eigh(H)
    return evals[0], evecs[:, 0]


def thermal_state(H, beta):
    """Gibbs state rho = exp(-beta H)/Z."""
    evals, evecs = eigh(H)
    w = np.exp(-beta * (evals - evals.min()))
    w /= w.sum()
    rho = (evecs * w) @ evecs.conj().T
    return rho


if __name__ == "__main__":
    n = 4
    H0 = xxz_hamiltonian(n, J=1.0, Delta=1.0, h=0.1)
    e0, psi0 = ground_state(H0)
    print("n =", n, " dim =", H0.shape[0])
    print("Ground energy of intended H0:", e0)
    terms = hamiltonian_terms(n)
    print("Number of Hamiltonian terms tracked:", len(terms))
