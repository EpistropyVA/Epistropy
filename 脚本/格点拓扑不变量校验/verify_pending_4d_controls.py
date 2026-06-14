# verify_pending_4d_controls.py
# Adversarial controls for the rank-19 shell decomposition (4b/4c).
# On the periodic 3-torus there is NO boundary: "corner" = {0,2}^3 is just a
# 2x2x2 contiguous block (mod 3, coords 0 and 2 are adjacent).
#
# Controls:
#   C1. Translation covariance: all 27 translates of the 2x2x2 block should
#       give the same rank (7) -> "corner" specialness is origin choice.
#   C2. Random 8-BC subsets: generic rank should be 8. If most random subsets
#       give 8 and the contiguous block gives 7, the block has one real
#       internal dependency. If random subsets also give 7, nothing special.
#   C3. Removal tautology: removing 8 random BCs (38 rows left) should
#       generically keep rank 19 -> 4c's "corner=shadow" PASS is generic,
#       not corner-specific.
#   C4. k-orbit reading: decompose overlap rank by the k-label of pi-modes.
#       27 k-points under cubic symmetry split 1+6+12+8; hypothesis: the
#       SVD multiplicities 1+6+12 are k-orbits and the missing 8 dims are
#       a k-sector, not a position shell.

import itertools
import numpy as np
from verify_pending_4b_rank19_shell import (
    build_bcc_lattice_periodic, build_all_faces,
    classify_bc_shells, compute_pi_modes,
)

rng = np.random.default_rng(20260612)

def rank_of(M, tol=1e-5):
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol))

def main():
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    shells = classify_bc_shells(body_centers)
    bc_pos = {tuple(bc): idx for idx, bc in enumerate(body_centers)}

    simplex_vectors = np.zeros((54, 540))
    for bc_idx in range(27):
        gf = bc_face_indices[bc_idx]
        for li in range(10):
            simplex_vectors[2 * bc_idx, gf[li]] = 1.0
        for li in range(10, 20):
            simplex_vectors[2 * bc_idx + 1, gf[li]] = 1.0

    wilson_pi_modes, vecs_all = compute_pi_modes(body_centers, bc_face_indices)
    assert len(wilson_pi_modes) == 54
    pi_modes_540 = np.zeros((54, 540), dtype=complex)
    mode_kstart = []
    for m_idx, (loop_key, k_start, Psi_bloch) in enumerate(wilson_pi_modes):
        mode_kstart.append(k_start)
        k_vec = np.array([2 * np.pi * n / 3.0 for n in k_start])
        for bc_idx, bc_ijk in enumerate(body_centers):
            phase = np.exp(1j * np.dot(k_vec, np.array(bc_ijk, dtype=float)))
            for li, g_face in enumerate(bc_face_indices[bc_idx]):
                pi_modes_540[m_idx, g_face] = Psi_bloch[li] * phase

    overlap = simplex_vectors @ pi_modes_540.T  # (54 simplex-rows, 54 mode-cols)
    print(f"Full overlap rank: {rank_of(overlap)} (expect 19)")

    def rows_of_bcs(bc_indices):
        rows = []
        for b in bc_indices:
            rows.extend((2 * b, 2 * b + 1))
        return sorted(rows)

    # --- C1: translation covariance of the 2x2x2 block ---
    print("\n[C1] Rank of all 27 translates of the {0,2}^3 block:")
    base_block = [(i, j, k) for i, j, k in itertools.product((0, 2), repeat=3)]
    ranks_c1 = []
    for s in itertools.product(range(3), repeat=3):
        shifted = [bc_pos[((i+s[0]) % 3, (j+s[1]) % 3, (k+s[2]) % 3)]
                   for (i, j, k) in base_block]
        ranks_c1.append(rank_of(overlap[rows_of_bcs(shifted), :]))
    vals, counts = np.unique(ranks_c1, return_counts=True)
    print(f"  rank distribution over 27 translates: {dict(zip(vals.tolist(), counts.tolist()))}")

    # --- C2: random 8-BC subsets ---
    print("\n[C2] Rank of 200 random 8-BC subsets (16 rows each):")
    ranks_c2 = []
    for _ in range(200):
        sub = rng.choice(27, size=8, replace=False)
        ranks_c2.append(rank_of(overlap[rows_of_bcs(sub), :]))
    vals, counts = np.unique(ranks_c2, return_counts=True)
    print(f"  rank distribution: {dict(zip(vals.tolist(), counts.tolist()))}")

    # --- C3: removal tautology ---
    print("\n[C3] Rank after removing 8 random BCs (38 rows left), 200 trials:")
    ranks_c3 = []
    for _ in range(200):
        sub = set(rng.choice(27, size=8, replace=False).tolist())
        keep = [r for b in range(27) if b not in sub for r in (2*b, 2*b+1)]
        ranks_c3.append(rank_of(overlap[keep, :]))
    vals, counts = np.unique(ranks_c3, return_counts=True)
    print(f"  rank distribution: {dict(zip(vals.tolist(), counts.tolist()))}")

    # --- C4: k-orbit decomposition of mode columns ---
    print("\n[C4] Overlap rank restricted to pi-modes by k_start orbit type:")
    def k_type(kt):
        nz = sum(1 for c in kt if c != 0)
        return nz  # 0:Gamma(1pt) 1:axis(6) 2:plane-diag(12) 3:body-diag(8)
    type_names = {0: "Gamma(1)", 1: "axis-type(6)", 2: "plane-diag(12)", 3: "body-diag(8)"}
    for t in range(4):
        cols = [m for m in range(54) if k_type(mode_kstart[m]) == t]
        if not cols:
            print(f"  {type_names[t]:16s}: 0 modes")
            continue
        r = rank_of(overlap[:, cols])
        print(f"  {type_names[t]:16s}: {len(cols)} modes, overlap rank {r}")

    # Which mode columns are in the null space of the overlap?
    U, S, Vh = np.linalg.svd(overlap)
    null_cols = Vh[19:, :]  # (35, 54) right-null basis
    # weight of each mode in the null space
    w = np.sum(np.abs(null_cols) ** 2, axis=0)  # length 54
    print("\n  Null-space weight by k_start type (sum of |v|^2 over null basis):")
    for t in range(4):
        cols = [m for m in range(54) if k_type(mode_kstart[m]) == t]
        if cols:
            print(f"  {type_names[t]:16s}: total weight {np.sum(w[cols]):.4f} over {len(cols)} modes")

if __name__ == "__main__":
    main()
