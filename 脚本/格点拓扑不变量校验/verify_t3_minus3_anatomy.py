# verify_t3_minus3_anatomy.py
# Anatomy of the flat-band eigenvalue -3: how does it decompose?
# V's question: "-1 + -1 + -1" (each axis contributes -1) vs "3 x (-1)"
# (one mechanism with multiplicity) may be different narratives.
# Tests, all gauge-invariant per k (k is a genuine BZ label here):
#   1. per-axis compression C_a = V^dag H_a(k) V on the 6D flat space:
#      is C_a = -I for each axis a?
#   2. onsite compression trace Tr(P_k T0) listed per k (pattern? Gamma clean?)
#   3. trivial sum check: C_x+C_y+C_z+V^dag T0 V = -3 I.

import itertools
import numpy as np
from verify_pending_4b_rank19_shell import (
    build_bcc_lattice_periodic, build_all_faces, build_adjacency_matrix,
)

def main():
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A = build_adjacency_matrix(face_to_idx)
    perm = np.array([f for bc in range(27) for f in bc_face_indices[bc]])
    A_r = A[np.ix_(perm, perm)]
    T = {}
    for bc_j, bc_ijk in enumerate(body_centers):
        R = tuple(c % 3 for c in bc_ijk)
        T[R] = A_r[0:20, bc_j*20:(bc_j+1)*20].copy()
    T0 = T[(0, 0, 0)]

    axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    def axis_block(k, ax):
        a = np.array(ax, float)
        Rp, Rm = tuple(np.array(ax) % 3), tuple((-np.array(ax)) % 3)
        return (np.exp(1j*np.dot(k, a)) * T[Rp]
                + np.exp(-1j*np.dot(k, a)) * T[Rm])

    print(f"{'k':12s} {'Tr(P T0)':>9s}  {'|C_x+I|':>8s} {'|C_y+I|':>8s} {'|C_z+I|':>8s}  {'sum=-3I':>8s}")
    per_axis_max = 0.0
    for n in itertools.product(range(3), repeat=3):
        k = np.array([2*np.pi*x/3.0 for x in n])
        Hk = T0 + sum(axis_block(k, ax) for ax in axes)
        evals, evecs = np.linalg.eigh(Hk)
        assert np.allclose(evals[:6], -3.0, atol=1e-10)
        V = evecs[:, :6]
        tr0 = np.real(np.trace(V.conj().T @ T0 @ V))
        devs = []
        Csum = np.zeros((6, 6), dtype=complex)
        for ax in axes:
            C = V.conj().T @ axis_block(k, ax) @ V
            Csum += C
            devs.append(np.max(np.abs(C + np.eye(6))))
        per_axis_max = max(per_axis_max, max(devs))
        total = Csum + V.conj().T @ T0 @ V
        sum_dev = np.max(np.abs(total + 3*np.eye(6)))
        print(f"{str(n):12s} {tr0:9.4f}  {devs[0]:8.4f} {devs[1]:8.4f} {devs[2]:8.4f}  {sum_dev:8.1e}")

    print(f"\nper-axis '-1 each' hypothesis: max dev {per_axis_max:.4f} -> "
          f"{'PASS' if per_axis_max < 1e-9 else 'FAIL'}")

if __name__ == "__main__":
    main()
