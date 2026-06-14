# verify_t3_spectral_pinning_weak.py
# The strong form (T(0)v = -2v per flat eigenvector) FAILED (residual ~1.6).
# Test the weaker subspace forms of the v352 spectral-pinning claim:
#   (a) trace form:    Tr(P_k T(0)) = 6 * (-2)  (basis-invariant average)
#   (b) compression:   P_k T(0) P_k = -2 P_k    (all expectation values exactly -2)
# where P_k projects onto the 6D flat-band subspace of H(k).

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

    max_trace_dev = 0.0
    max_compress_dev = 0.0
    for n in itertools.product(range(3), repeat=3):
        k = np.array([2*np.pi*x/3.0 for x in n])
        Hk = sum(np.exp(1j*np.dot(k, np.array(R, float))) * M for R, M in T.items())
        evals, evecs = np.linalg.eigh(Hk)
        assert np.allclose(evals[:6], -3.0, atol=1e-10)
        V = evecs[:, :6]                      # 20x6 flat-band basis
        P = V @ V.conj().T                    # projector
        tr = np.real(np.trace(P @ T0))
        max_trace_dev = max(max_trace_dev, abs(tr - 6*(-2.0)))
        comp = V.conj().T @ T0 @ V            # 6x6 compression of T(0)
        max_compress_dev = max(max_compress_dev, np.max(np.abs(comp - (-2.0)*np.eye(6))))

    print(f"(a) trace form  max |Tr(P T0) + 12|        = {max_trace_dev:.3e}")
    print(f"(b) compression max |P T0 P + 2P| (as 6x6) = {max_compress_dev:.3e}")
    print(f"(a) {'PASS' if max_trace_dev < 1e-9 else 'FAIL'}  (b) {'PASS' if max_compress_dev < 1e-9 else 'FAIL'}")

if __name__ == "__main__":
    main()
