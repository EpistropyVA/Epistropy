# verify_pending_4c_corner_shadow.py
# Quick check: is corner shell's rank-7 contribution fully contained in
# the span of edge+face+center (the "shadow/hologram" hypothesis)?
# Test: overlap submatrix WITHOUT corner simplices should still have rank 19.
# If rank drops below 19, corner carries independent information (hypothesis fails).

import numpy as np
from verify_pending_4b_rank19_shell import (
    build_bcc_lattice_periodic, build_all_faces,
    classify_bc_shells, compute_pi_modes,
)

def main():
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    shells = classify_bc_shells(body_centers)

    # Simplex indicator vectors (54 x 540), same as 4b
    simplex_vectors = np.zeros((54, 540))
    for bc_idx in range(27):
        global_faces = bc_face_indices[bc_idx]
        for local_idx in range(10):
            simplex_vectors[2 * bc_idx, global_faces[local_idx]] = 1.0
        for local_idx in range(10, 20):
            simplex_vectors[2 * bc_idx + 1, global_faces[local_idx]] = 1.0

    # Pi-modes projected to 540-dim, same as 4b
    wilson_pi_modes, vecs_all = compute_pi_modes(body_centers, bc_face_indices)
    assert len(wilson_pi_modes) == 54
    pi_modes_540 = np.zeros((54, 540), dtype=complex)
    for m_idx, (loop_key, k_start, Psi_bloch) in enumerate(wilson_pi_modes):
        k_vec = np.array([2 * np.pi * n / 3.0 for n in k_start])
        for bc_idx, bc_ijk in enumerate(body_centers):
            phase = np.exp(1j * np.dot(k_vec, np.array(bc_ijk, dtype=float)))
            for local_idx, g_face in enumerate(bc_face_indices[bc_idx]):
                pi_modes_540[m_idx, g_face] = Psi_bloch[local_idx] * phase

    overlap = simplex_vectors @ pi_modes_540.T  # (54, 54)
    rank_full = int(np.sum(np.linalg.svd(overlap, compute_uv=False) > 1e-5))

    corner_simplices = set()
    for bc_idx in shells['corner']:
        corner_simplices.update((2 * bc_idx, 2 * bc_idx + 1))
    keep = [s for s in range(54) if s not in corner_simplices]

    sub = overlap[keep, :]
    rank_no_corner = int(np.sum(np.linalg.svd(sub, compute_uv=False) > 1e-5))

    print(f"Full overlap rank (54 simplices):      {rank_full}")
    print(f"Overlap rank without corner (38 rows): {rank_no_corner}")
    if rank_no_corner == rank_full:
        print("PASS: corner rows add no new dimensions -> corner = shadow of inner shells")
    else:
        print(f"FAIL: corner carries {rank_full - rank_no_corner} independent dimension(s)")

if __name__ == "__main__":
    main()
