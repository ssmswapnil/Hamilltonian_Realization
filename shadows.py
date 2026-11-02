"""
Restricted-control (global single-qubit rotation) randomized-measurement
estimator, built numerically, plus a standard per-qubit classical-shadow
baseline for comparison.
"""

import numpy as np
from numpy.random import default_rng
from core import all_pauli_strings, kron_list, I2, X, Y, Z

# ----------------------------------------------------------------------
# Random single-qubit unitaries (Haar on SU(2))
# ----------------------------------------------------------------------

def random_su2(rng):
    """Haar-random single-qubit unitary via QR decomposition of a Ginibre matrix."""
    z = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = q * ph
    return q


def global_rotation(n, u):
    """U = u (x) u (x) ... (x) u, n copies."""
    return kron_list([u] * n)


def sample_bitstring(rng, probs):
    dim = len(probs)
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    return rng.choice(dim, p=probs)


def measure_restricted(rng, rho, n):
    """
    One shot of the restricted (global-rotation) measurement scheme on state rho
    (density matrix OR state vector).
    Returns (u, b) where u is the 2x2 single-qubit unitary applied identically
    to every qubit, and b is the measured computational-basis index (0..2^n-1).
    """
    u = random_su2(rng)
    U = global_rotation(n, u)
    if rho.ndim == 1:
        psi_rot = U @ rho
        probs = np.abs(psi_rot) ** 2
    else:
        rho_rot = U @ rho @ U.conj().T
        probs = np.real(np.diag(rho_rot))
    b = sample_bitstring(rng, probs)
    return u, b


# ----------------------------------------------------------------------
# Numerically constructed frame operator M for the restricted ensemble
# ----------------------------------------------------------------------

class RestrictedFrame:
    """
    Numerically estimates (via Monte Carlo over the Haar measure on SU(2),
    applied identically to every qubit) the measurement-response superoperator

        M(O) = E_u[ U^dag * Diag(diag(U O U^dag)) * U ],   U = u^{(x)n}

    expressed in the full n-qubit Pauli-string basis, then provides the
    pseudo-inverse dual operators M^{-1}(O) needed to build unbiased
    estimators  tr(rho O) ~ <b| U M^{-1}(O) U^dag |b>  (Van Kirk et al.
    kernel construction), for a chosen set of target observables O.

    This is a direct numerical (rather than representation-theoretic/
    closed-form) implementation of the general shadow-tomography machinery
    for a restricted measurement ensemble.
    """

    def __init__(self, n, mc_samples=4000, seed=0, rcond=1e-6):
        self.n = n
        self.dim = 2 ** n
        self.rng = default_rng(seed)
        self.labels, self.basis = zip(*all_pauli_strings(n))
        self.basis = np.array(self.basis)  # (D, dim, dim)
        self.D = len(self.labels)
        self.mc_samples = mc_samples
        self.rcond = rcond
        self._build_frame_matrix()

    def _build_frame_matrix(self):
        # Using the identity Tr(M(P_a) P_b) = E_u[ sum_i (U P_a U^dag)_ii (U P_b U^dag)_ii ],
        # we only need the *diagonals* of U P_a U^dag for every basis element a
        # (shape (D, d)), avoiding the need to ever form the full rotated/dephased
        # (D, d, d) operators. This is the same physics, computed far more cheaply.
        D, d = self.D, self.dim
        n = self.n
        M_accum = np.zeros((D, D), dtype=float)
        batch = 25
        done = 0
        while done < self.mc_samples:
            b = min(batch, self.mc_samples - done)
            for _ in range(b):
                u = random_su2(self.rng)
                U = global_rotation(n, u)
                diag_vals = np.real(
                    np.einsum("ij,ajk,ik->ai", U, self.basis, U.conj(), optimize=True)
                )  # (D, d)
                M_accum += diag_vals @ diag_vals.T / d
            done += b
        self.M_mat = M_accum / self.mc_samples

    def dual_operator(self, O):
        """
        Given a target Hermitian operator O (dim x dim matrix), returns the
        dual operator M^{-1}(O) as a (dim x dim) matrix, using a pseudo-inverse
        (least-norm) solve -- this automatically restricts the solution to the
        "visible space" (row space of M), exactly as in Van Kirk et al.
        """
        d = self.dim
        # coefficients of O in the Pauli basis: o_b = Tr(O P_b)/d
        o_vec = np.real(np.einsum("ij,bji->b", O, self.basis) / d)
        # Solve M_mat @ x = o_vec  for x  (pseudo-inverse / least-norm solution)
        x_vec = np.linalg.pinv(self.M_mat, rcond=self.rcond) @ o_vec
        # reconstruct operator: X = sum_a x_a * P_a
        X = np.einsum("a,aij->ij", x_vec.astype(complex), self.basis)
        return X

    def kernel_estimate(self, O, shots_data):
        """
        shots_data: list of (u, b) tuples from measure_restricted().
        Returns the empirical mean of <b| U Xdual U^dag |b> over shots,
        where Xdual = M^{-1}(O). This is an unbiased estimator of tr(rho O).
        """
        Xdual = self.dual_operator(O)
        n = self.n
        vals = []
        for u, b in shots_data:
            U = global_rotation(n, u)
            v = U.conj().T[:, b]  # column of U^dag corresponding to |b>, i.e. U^dag|b>
            val = np.real(v.conj() @ Xdual @ v)
            vals.append(val)
        vals = np.array(vals)
        return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


# ----------------------------------------------------------------------
# Baseline: standard per-qubit random-Clifford/Haar classical shadows
# (Huang, Kueng & Preskill 2020) -- unrestricted, independent rotation
# on every qubit. Closed-form dual channel: per qubit, 3 U^dag|b><b|U - I.
# ----------------------------------------------------------------------

def measure_full_control(rng, rho, n):
    """One shot: independent Haar-random single-qubit rotation on every qubit."""
    us = [random_su2(rng) for _ in range(n)]
    U = kron_list(us)
    if rho.ndim == 1:
        psi_rot = U @ rho
        probs = np.abs(psi_rot) ** 2
    else:
        rho_rot = U @ rho @ U.conj().T
        probs = np.real(np.diag(rho_rot))
    b = sample_bitstring(rng, probs)
    bits = [(b >> (n - 1 - i)) & 1 for i in range(n)]
    return us, bits


def full_shadow_single_qubit_snapshot(u, bit):
    """Per-qubit classical-shadow inverse channel: 3 U^dag|b><b|U - I."""
    b_vec = np.array([1, 0]) if bit == 0 else np.array([0, 1])
    proj = np.outer(b_vec, b_vec.conj())
    Udag = u.conj().T
    rot_proj = Udag @ proj @ u
    return 3 * rot_proj - I2


def full_shadow_estimate(O, shots_data_full, n):
    """
    shots_data_full: list of (us, bits) from measure_full_control().
    Estimates tr(rho O) for an arbitrary Hermitian O using standard
    (unrestricted) classical shadows.
    """
    vals = []
    for us, bits in shots_data_full:
        snaps = [full_shadow_single_qubit_snapshot(us[i], bits[i]) for i in range(n)]
        rho_hat = kron_list(snaps)
        vals.append(np.real(np.trace(rho_hat @ O)))
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))
