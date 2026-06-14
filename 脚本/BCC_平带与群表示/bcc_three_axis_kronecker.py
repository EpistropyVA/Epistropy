"""
bcc_three_axis_kronecker.py

Tests the three-axis Kronecker decomposition of the 540x540 BCC face-adjacency matrix:

    A = I_27 (x) L_local + L_x (x) M_x + L_y (x) M_y + L_z (x) M_z

where L_x, L_y, L_z are 27x27 axis-specific coupling matrices on the 3x3x3 BC grid,
and M_x, M_y, M_z are 20x20 inter-BC coupling patterns for each axis direction.

Steps:
  1. Build and reorder the 540x540 matrix into 27 BC-patch blocks
  2. Build axis adjacency matrices L_x, L_y, L_z
  3. Extract axis-specific coupling matrices M_x, M_y, M_z
  4. Verify three-axis Kronecker sum
  5. Symmetry analysis of M_x, M_y, M_z
  6. Predict 540 eigenvalues analytically (commutativity check)
"""

import sys
import os
import importlib.util
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Import construction helpers from bcc_540_spectrum.py
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec_path = os.path.join(_here, "bcc_540_spectrum.py")
_spec_mod = importlib.util.spec_from_file_location("bcc_540_spectrum", _spec_path)
_bcc = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(_bcc)

build_bcc_lattice      = _bcc.build_bcc_lattice
build_simplices        = _bcc.build_simplices
build_faces            = _bcc.build_faces
build_adjacency_matrix = _bcc.build_adjacency_matrix
bc_grid_index          = _bcc.bc_grid_index

OUT_PATH = os.path.join(_here, "bcc_three_axis_results.txt")

_LOG_LINES = []


def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))


def frob(M):
    return float(np.linalg.norm(M, "fro"))


def rel_frob(M, ref):
    r = frob(ref)
    return frob(M) / r if r > 1e-15 else float("inf")


def commutator_frob(A, B):
    return frob(A @ B - B @ A)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 78)
    log("BCC 540x540 Three-Axis Kronecker Decomposition")
    log("A = I_27(x)L_local + L_x(x)M_x + L_y(x)M_y + L_z(x)M_z")
    log("=" * 78)

    # -----------------------------------------------------------------------
    # Step 1: Build and reorder
    # -----------------------------------------------------------------------
    log("\n[Step 1] Building BCC lattice and adjacency matrix ...")
    body_centers, cube_corners = build_bcc_lattice()
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    _W, A, _sigmas, _mc, _np_count = build_adjacency_matrix(faces)
    log(f"  A shape: {A.shape}  nnz: {int(np.count_nonzero(A))}")

    n_bc  = len(body_centers)   # 27
    n_loc = 20                  # faces per BC

    # Reorder into block form
    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0] for bc_i in range(n_bc)]
    bc_patch_sizes = [len(g) for g in bc_face_idxs]
    log(f"  Patch sizes: min={min(bc_patch_sizes)}, max={max(bc_patch_sizes)}, "
        f"all==20? {all(s == n_loc for s in bc_patch_sizes)}")

    perm = np.concatenate([bc_face_idxs[a] for a in range(n_bc)])
    A_block = A[np.ix_(perm, perm)]
    log(f"  A_block shape: {A_block.shape} (reordered, same spectrum as A)")

    # Extract L_local (mean of diagonal blocks)
    diag_blocks = []
    for a in range(n_bc):
        ra = slice(a * n_loc, (a + 1) * n_loc)
        diag_blocks.append(A_block[ra, ra].copy())
    L_local = np.mean(diag_blocks, axis=0)
    ev_local = np.sort(np.linalg.eigvalsh(L_local))
    log(f"\n  L_local eigenvalues (sorted): " + " ".join(f"{v:.4f}" for v in ev_local))

    # BC grid indices (0-indexed i,j,k in {0,1,2})
    bc_grids = [bc_grid_index(bc) for bc in body_centers]
    log(f"\n  BC grid indices (first 5): {bc_grids[:5]}")

    # -----------------------------------------------------------------------
    # Step 2: Build axis adjacency matrices L_x, L_y, L_z
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 2] Build axis adjacency matrices L_x, L_y, L_z")
    log("  L_x[a,b]=1 iff BCs differ by 1 in x only; similarly for y,z")
    log("=" * 78)

    L_x = np.zeros((n_bc, n_bc))
    L_y = np.zeros((n_bc, n_bc))
    L_z = np.zeros((n_bc, n_bc))

    for a in range(n_bc):
        ia, ja, ka = bc_grids[a]
        for b in range(n_bc):
            ib, jb, kb = bc_grids[b]
            dx = abs(ia - ib)
            dy = abs(ja - jb)
            dz = abs(ka - kb)
            if dx == 1 and dy == 0 and dz == 0:
                L_x[a, b] = 1.0
            elif dx == 0 and dy == 1 and dz == 0:
                L_y[a, b] = 1.0
            elif dx == 0 and dy == 0 and dz == 1:
                L_z[a, b] = 1.0

    # Verify: L_x + L_y + L_z should equal the full grid adjacency (L_grid)
    L_grid = np.zeros((n_bc, n_bc))
    for a in range(n_bc):
        ia, ja, ka = bc_grids[a]
        for b in range(n_bc):
            ib, jb, kb = bc_grids[b]
            if abs(ia-ib) + abs(ja-jb) + abs(ka-kb) == 1:
                L_grid[a, b] = 1.0

    L_sum = L_x + L_y + L_z
    log(f"\n  ||L_x + L_y + L_z - L_grid||_F = {frob(L_sum - L_grid):.6e}")
    log(f"  L_x + L_y + L_z == L_grid? {np.allclose(L_sum, L_grid, atol=1e-9)}")

    log(f"\n  Non-zero entries: L_x={int(np.sum(L_x>0))}, L_y={int(np.sum(L_y>0))}, "
        f"L_z={int(np.sum(L_z>0))}")

    for name, Lax in [("L_x", L_x), ("L_y", L_y), ("L_z", L_z)]:
        ev = np.sort(np.linalg.eigvalsh(Lax))
        log(f"  Eigenvalues of {name}: " + " ".join(f"{v:.4f}" for v in ev))

    # Check symmetry: L_x, L_y, L_z eigenvalue spectra should be identical
    ev_x = np.sort(np.linalg.eigvalsh(L_x))
    ev_y = np.sort(np.linalg.eigvalsh(L_y))
    ev_z = np.sort(np.linalg.eigvalsh(L_z))
    log(f"\n  ||spec(L_x) - spec(L_y)||_2 = {np.linalg.norm(ev_x - ev_y):.6e}")
    log(f"  ||spec(L_x) - spec(L_z)||_2 = {np.linalg.norm(ev_x - ev_z):.6e}")
    log(f"  L_x, L_y, L_z are isospectal? {np.allclose(ev_x, ev_y) and np.allclose(ev_x, ev_z)}")

    # -----------------------------------------------------------------------
    # Step 3: Extract axis-specific coupling matrices M_x, M_y, M_z
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 3] Extract M_x, M_y, M_z (inter-BC coupling patterns per axis)")
    log("=" * 78)

    def extract_axis_coupling(L_ax, axis_name):
        """Collect all off-diagonal blocks A_block[a,b] where L_ax[a,b]=1."""
        pairs = [(a, b) for a in range(n_bc) for b in range(n_bc) if L_ax[a, b] > 0.5]
        log(f"\n  {axis_name}: {len(pairs)} adjacent BC pairs")

        blocks = []
        for a, b in pairs:
            ra = slice(a * n_loc, (a + 1) * n_loc)
            rb = slice(b * n_loc, (b + 1) * n_loc)
            blocks.append(A_block[ra, rb].copy())

        # Check: are all blocks identical?
        max_diff = 0.0
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                d = float(np.max(np.abs(blocks[i] - blocks[j])))
                if d > max_diff:
                    max_diff = d
        log(f"  Max element-wise difference across all {len(blocks)} blocks: {max_diff:.6e}")
        all_same = (max_diff < 1e-9)
        log(f"  All blocks identical? {all_same}")

        M_mean = np.mean(blocks, axis=0)

        if not all_same:
            # Check if blocks cluster by pair type (e.g., which (i,j,k) row the BC is in)
            # Classify each pair by the fixed coordinates
            log(f"\n  Investigating block variation by pair type ...")
            pair_type_blocks = defaultdict(list)
            for idx, (a, b) in enumerate(pairs):
                ia, ja, ka = bc_grids[a]
                ib, jb, kb = bc_grids[b]
                # For x-axis: (j, k) are fixed
                if axis_name == "L_x":
                    key = (min(ja, jb), min(ka, kb))  # fixed coords
                elif axis_name == "L_y":
                    key = (min(ia, ib), min(ka, kb))
                else:  # L_z
                    key = (min(ia, ib), min(ja, jb))
                pair_type_blocks[key].append(blocks[idx])

            log(f"  {len(pair_type_blocks)} distinct pair types (by fixed coords):")
            for key, blk_list in sorted(pair_type_blocks.items()):
                # Are blocks in this group identical?
                max_d = 0.0
                for i in range(len(blk_list)):
                    for j in range(i + 1, len(blk_list)):
                        d = float(np.max(np.abs(blk_list[i] - blk_list[j])))
                        if d > max_d:
                            max_d = d
                log(f"    type {key}: {len(blk_list)} blocks, max diff = {max_d:.6e}")

        ev_M = np.sort(np.linalg.eigvalsh(M_mean))
        log(f"  Eigenvalues of M_{axis_name[-1]} (mean, sorted): " +
            " ".join(f"{v:.4f}" for v in ev_M))
        log(f"  ||M_{axis_name[-1]}||_F = {frob(M_mean):.6e}")

        return M_mean, blocks, pairs

    M_x, blocks_x, pairs_x = extract_axis_coupling(L_x, "L_x")
    M_y, blocks_y, pairs_y = extract_axis_coupling(L_y, "L_y")
    M_z, blocks_z, pairs_z = extract_axis_coupling(L_z, "L_z")

    ev_Mx = np.sort(np.linalg.eigvalsh(M_x))
    ev_My = np.sort(np.linalg.eigvalsh(M_y))
    ev_Mz = np.sort(np.linalg.eigvalsh(M_z))
    log(f"\n  Eigenvalue comparison of M_x, M_y, M_z:")
    log(f"  ||spec(M_x) - spec(M_y)||_2 = {np.linalg.norm(ev_Mx - ev_My):.6e}")
    log(f"  ||spec(M_x) - spec(M_z)||_2 = {np.linalg.norm(ev_Mx - ev_Mz):.6e}")
    log(f"  M_x, M_y, M_z isospectral? {np.allclose(ev_Mx, ev_My) and np.allclose(ev_Mx, ev_Mz)}")

    # -----------------------------------------------------------------------
    # Step 4: Verify three-axis Kronecker sum
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 4] Verify three-axis Kronecker sum")
    log("  A_kron3 = I_27(x)L_local + L_x(x)M_x + L_y(x)M_y + L_z(x)M_z")
    log("=" * 78)

    A_kron3 = (np.kron(np.eye(n_bc), L_local) +
               np.kron(L_x, M_x) +
               np.kron(L_y, M_y) +
               np.kron(L_z, M_z))

    resid_abs = frob(A_block - A_kron3)
    resid_rel = rel_frob(A_block - A_kron3, A_block)
    log(f"\n  ||A_block - A_kron3||_F          = {resid_abs:.6e}")
    log(f"  ||A_block - A_kron3||_F / ||A||_F = {resid_rel:.6e}")
    log(f"  Exact three-axis decomposition (rel < 1e-9)? {resid_rel < 1e-9}")

    # Residual analysis
    R = A_block - A_kron3
    log(f"\n  Residual matrix R = A_block - A_kron3:")
    log(f"  ||R||_F = {frob(R):.6e}")

    rank_R = np.linalg.matrix_rank(R, tol=1e-6)
    log(f"  rank(R) approx (tol=1e-6) = {rank_R}")

    sv_R = np.linalg.svd(R, compute_uv=False)
    log(f"  Top 10 singular values of R: " + " ".join(f"{s:.4f}" for s in sv_R[:10]))

    nnz_R = int(np.count_nonzero(np.abs(R) > 1e-9))
    sparsity_R = 1.0 - nnz_R / (R.shape[0] * R.shape[1])
    log(f"  Nonzero entries (|r|>1e-9): {nnz_R}  sparsity: {sparsity_R:.4f}")

    # Block-wise residual norms
    R_reshaped = R.reshape(n_bc, n_loc, n_bc, n_loc)
    r_block_norms = np.zeros((n_bc, n_bc))
    for a in range(n_bc):
        for b in range(n_bc):
            r_block_norms[a, b] = frob(R_reshaped[a, :, b, :])
    nz_blocks = [(a, b, r_block_norms[a, b])
                 for a in range(n_bc) for b in range(n_bc)
                 if r_block_norms[a, b] > 1e-9]
    log(f"\n  Non-trivial residual blocks (||R_block||>1e-9): {len(nz_blocks)} of {n_bc*n_bc}")
    if nz_blocks:
        norms_only = [v for _, _, v in nz_blocks]
        diag_nz = [(a, b, v) for a, b, v in nz_blocks if a == b]
        offdiag_nz = [(a, b, v) for a, b, v in nz_blocks if a != b]
        log(f"  Max block ||R_block||_F: {max(norms_only):.6e}")
        log(f"  Mean block ||R_block||_F (non-zero): {np.mean(norms_only):.6e}")
        log(f"  Diagonal residual blocks: {len(diag_nz)}, off-diagonal: {len(offdiag_nz)}")

        # Show which off-diagonal blocks have residuals
        if offdiag_nz:
            log(f"\n  Off-diagonal residual blocks (a, b, norm):")
            for a, b, v in sorted(offdiag_nz, key=lambda x: -x[2])[:20]:
                ia, ja, ka = bc_grids[a]
                ib, jb, kb = bc_grids[b]
                dx, dy, dz = abs(ia-ib), abs(ja-jb), abs(ka-kb)
                dist = dx + dy + dz
                log(f"    ({a:2d},{b:2d}) dist={dist} delta=({dx},{dy},{dz}) ||R||={v:.4e}")

    # Eigenvalue comparison
    log(f"\n  Eigenvalue comparison: A_kron3 vs full A")
    ev_A = np.sort(np.linalg.eigvalsh(A))
    ev_kron3 = np.sort(np.linalg.eigvalsh(A_kron3))
    max_eig_dev = float(np.max(np.abs(ev_kron3 - ev_A)))
    log(f"  Max eigenvalue deviation: {max_eig_dev:.6e}")
    log(f"  Eigenvalues match (max dev < 1e-4)? {max_eig_dev < 1e-4}")

    # Show first 30 eigenvalue pairs
    log(f"\n  First 30 eigenvalue pairs (A vs A_kron3):")
    log(f"  {'idx':>4} | {'ev(A)':>10} | {'ev(kron3)':>10} | {'diff':>10}")
    log(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i in range(min(30, len(ev_A))):
        log(f"  {i:4d} | {ev_A[i]:10.4f} | {ev_kron3[i]:10.4f} | {ev_kron3[i]-ev_A[i]:10.6f}")

    # -----------------------------------------------------------------------
    # Step 5: Symmetry analysis of M_x, M_y, M_z
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 5] Symmetry analysis of M_x, M_y, M_z")
    log("=" * 78)

    log(f"\n  M_x eigenvalues: " + " ".join(f"{v:.4f}" for v in ev_Mx))
    log(f"  M_y eigenvalues: " + " ".join(f"{v:.4f}" for v in ev_My))
    log(f"  M_z eigenvalues: " + " ".join(f"{v:.4f}" for v in ev_Mz))

    # Are M_x, M_y, M_z related by permutation? Find P such that M_y ≈ P M_x P^T
    # Use eigendecomposition: if M_y = P M_x P^T, then P maps eigenvectors of M_x to M_y
    def find_permutation_relation(M1, M2, name1, name2):
        """Check if M2 ≈ P M1 P^T for some permutation matrix P."""
        # Compute eigenvectors
        w1, V1 = np.linalg.eigh(M1)
        w2, V2 = np.linalg.eigh(M2)

        # If related by permutation, eigenvalues must be identical
        if not np.allclose(w1, w2, atol=1e-6):
            log(f"  {name1} and {name2}: eigenvalues differ — not permutation-related")
            return None

        # Transition matrix: V2 = P V1, so P = V2 @ V1.T (approx for degenerate eigs)
        # Actually for permutation P: M2 = P M1 P^T => we look at the direct comparison
        # Check ||M2 - P M1 P^T||_F for each row-permutation candidate via Hungarian-style match
        # Simpler: check if M1 and M2 have the same set of rows (up to permutation)
        # Find the permutation P as integer mapping
        n = M1.shape[0]
        # Attempt: for each row of M2, find the matching row in M1
        from scipy.optimize import linear_sum_assignment
        # Cost matrix: ||row_i(M2) - row_j(M1)||_2
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cost[i, j] = np.linalg.norm(M2[i, :] - M1[j, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        perm_vec = col_ind  # P[i] = j means row i of M2 corresponds to row j of M1

        P = np.zeros((n, n))
        for i in range(n):
            P[i, perm_vec[i]] = 1.0

        M2_reconstructed = P @ M1 @ P.T
        err = frob(M2 - M2_reconstructed)
        log(f"  {name2} ≈ P {name1} P^T: ||err||_F = {err:.6e}  "
            f"(is permutation? {err < 1e-6})")
        return P, perm_vec

    P_yx, perm_yx = find_permutation_relation(M_x, M_y, "M_x", "M_y") or (None, None)
    P_zx, perm_zx = find_permutation_relation(M_x, M_z, "M_x", "M_z") or (None, None)

    if perm_yx is not None:
        log(f"  Permutation P (M_x->M_y): {perm_yx}")
    if perm_zx is not None:
        log(f"  Permutation Q (M_x->M_z): {perm_zx}")

    # -----------------------------------------------------------------------
    # Step 6: Commutativity and eigenvalue prediction
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 6] Commutativity analysis and eigenvalue prediction")
    log("=" * 78)

    log(f"\n  Commutator norms (Frobenius):")
    log(f"  ||[L_local, M_x]||_F = {commutator_frob(L_local, M_x):.6e}")
    log(f"  ||[L_local, M_y]||_F = {commutator_frob(L_local, M_y):.6e}")
    log(f"  ||[L_local, M_z]||_F = {commutator_frob(L_local, M_z):.6e}")
    log(f"  ||[M_x, M_y]||_F     = {commutator_frob(M_x, M_y):.6e}")
    log(f"  ||[M_x, M_z]||_F     = {commutator_frob(M_x, M_z):.6e}")
    log(f"  ||[M_y, M_z]||_F     = {commutator_frob(M_y, M_z):.6e}")
    log(f"  ||[L_x, L_y]||_F     = {commutator_frob(L_x, L_y):.6e}")
    log(f"  ||[L_x, L_z]||_F     = {commutator_frob(L_x, L_z):.6e}")
    log(f"  ||[L_y, L_z]||_F     = {commutator_frob(L_y, L_z):.6e}")

    commutes_local_Mx = commutator_frob(L_local, M_x) < 1e-9
    commutes_local_My = commutator_frob(L_local, M_y) < 1e-9
    commutes_local_Mz = commutator_frob(L_local, M_z) < 1e-9
    commutes_M = (commutator_frob(M_x, M_y) < 1e-9 and
                  commutator_frob(M_x, M_z) < 1e-9 and
                  commutator_frob(M_y, M_z) < 1e-9)
    commutes_L = (commutator_frob(L_x, L_y) < 1e-9 and
                  commutator_frob(L_x, L_z) < 1e-9 and
                  commutator_frob(L_y, L_z) < 1e-9)

    log(f"\n  L_local commutes with M_x,M_y,M_z? "
        f"{commutes_local_Mx}, {commutes_local_My}, {commutes_local_Mz}")
    log(f"  M_x,M_y,M_z mutually commute? {commutes_M}")
    log(f"  L_x,L_y,L_z mutually commute? {commutes_L}")

    all_commute = (commutes_local_Mx and commutes_local_My and commutes_local_Mz
                   and commutes_M and commutes_L)
    log(f"\n  All matrices commute (necessary for tensor-product diagonalization)? {all_commute}")

    # Attempt analytical eigenvalue prediction regardless
    # For the simplified case: assume M_x = M_y = M_z = M_mean
    M_all = (M_x + M_y + M_z) / 3.0
    ev_M_all = np.sort(np.linalg.eigvalsh(M_all))
    log(f"\n  Mean M = (M_x+M_y+M_z)/3 eigenvalues: " +
        " ".join(f"{v:.4f}" for v in ev_M_all))

    # The L_grid = L_x + L_y + L_z eigenvalues:
    ev_grid = np.sort(np.linalg.eigvalsh(L_grid))
    log(f"\n  L_grid eigenvalues: " + " ".join(f"{v:.4f}" for v in ev_grid))

    # If all three M_axis are equal to M_mean and L_x=L_y=L_z spectrally:
    # A ≈ I(x)L_local + (L_x+L_y+L_z)(x)M_mean = I(x)L_local + L_grid(x)M_mean
    # Check this simplified Kronecker sum
    A_kron_simplified = (np.kron(np.eye(n_bc), L_local) +
                         np.kron(L_grid, M_all))
    resid_simplified = rel_frob(A_block - A_kron_simplified, A_block)
    log(f"\n  Simplified sum A ≈ I(x)L_local + L_grid(x)M_mean:")
    log(f"  ||A_block - A_kron_simplified||_F / ||A||_F = {resid_simplified:.6e}")

    # Full analytical prediction attempt via simultaneous diagonalization
    # Even if they don't commute, compute the "would-be" spectrum
    log(f"\n  Analytical spectrum prediction (tensor product structure):")
    log(f"  λ_A = λ_local(m) + λ_x(kx)·μ_x + λ_y(ky)·μ_y + λ_z(kz)·μ_z")
    log(f"  (valid only if all matrices commute)")

    # Eigenvalues of local and axis matrices
    ev_Lx = np.sort(np.linalg.eigvalsh(L_x))
    ev_Ly = np.sort(np.linalg.eigvalsh(L_y))
    ev_Lz = np.sort(np.linalg.eigvalsh(L_z))
    # Eigenvalues of M matrices
    ev_Mx_s = np.sort(np.linalg.eigvalsh(M_x))
    ev_My_s = np.sort(np.linalg.eigvalsh(M_y))
    ev_Mz_s = np.sort(np.linalg.eigvalsh(M_z))

    log(f"\n  L_x unique eigenvalues (sorted): " + " ".join(f"{v:.4f}" for v in np.unique(np.round(ev_Lx, 4))))
    log(f"  M_x eigenvalues (sorted):        " + " ".join(f"{v:.4f}" for v in ev_Mx_s))

    # If the decomposition is A = I(x)L_local + L_x(x)M_x + L_y(x)M_y + L_z(x)M_z
    # and all pieces commute in the Kronecker sense, then we can diagonalize jointly.
    # The Kronecker product structure gives:
    # A acts on C^27 ⊗ C^20, with:
    #   I_27 ⊗ L_local : acts on the local mode index
    #   L_x ⊗ M_x : couples BC momentum (x) with local mode
    # If L_x, L_y, L_z and M_x, M_y, M_z all commute among themselves AND
    # with L_local, then we can find a joint eigenbasis.

    # For now: compute the "product formula" eigenvalues using pairs
    # λ(a,m) = ev_local[m] + sum_axis ev_Lax[a_axis] * ev_Max[m_axis]
    # This only works if the above commutation holds.

    # Check: do L_x, L_y, L_z share eigenvectors? (They should if they commute)
    if commutes_L:
        # Diagonalize L_x
        w_Lx, V_Lx = np.linalg.eigh(L_x)
        log(f"\n  L_x, L_y, L_z commute — shared eigenbasis exists")
        # Verify L_y and L_z diagonal in this basis
        Ly_diag = V_Lx.T @ L_y @ V_Lx
        Lz_diag = V_Lx.T @ L_z @ V_Lx
        off_Ly = frob(Ly_diag - np.diag(np.diag(Ly_diag)))
        off_Lz = frob(Lz_diag - np.diag(np.diag(Lz_diag)))
        log(f"  Off-diagonal ||L_y in L_x eigenbasis||_F = {off_Ly:.6e}")
        log(f"  Off-diagonal ||L_z in L_x eigenbasis||_F = {off_Lz:.6e}")
    else:
        log(f"\n  L_x, L_y, L_z do NOT commute — no shared eigenbasis")

    if commutes_M:
        w_Mx, V_Mx = np.linalg.eigh(M_x)
        log(f"\n  M_x, M_y, M_z commute — shared eigenbasis exists")
        My_diag = V_Mx.T @ M_y @ V_Mx
        Mz_diag = V_Mx.T @ M_z @ V_Mx
        off_My = frob(My_diag - np.diag(np.diag(My_diag)))
        off_Mz = frob(Mz_diag - np.diag(np.diag(Mz_diag)))
        log(f"  Off-diagonal ||M_y in M_x eigenbasis||_F = {off_My:.6e}")
        log(f"  Off-diagonal ||M_z in M_x eigenbasis||_F = {off_Mz:.6e}")
    else:
        log(f"\n  M_x, M_y, M_z do NOT commute — no shared eigenbasis")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Summary]")
    log("=" * 78)
    log(f"\n  Three-axis Kronecker sum relative residual: {resid_rel:.6e}")
    log(f"  Exact decomposition (rel < 1e-9)? {resid_rel < 1e-9}")
    log(f"  Max eigenvalue deviation (A vs A_kron3): {max_eig_dev:.6e}")
    log(f"\n  L_local eigenvalues: " + " ".join(f"{v:.4f}" for v in ev_local))
    log(f"  M_x eigenvalues:     " + " ".join(f"{v:.4f}" for v in ev_Mx))
    log(f"  M_y eigenvalues:     " + " ".join(f"{v:.4f}" for v in ev_My))
    log(f"  M_z eigenvalues:     " + " ".join(f"{v:.4f}" for v in ev_Mz))
    log(f"\n  All matrices commute? {all_commute}")
    log(f"  M_x isospectral with M_y, M_z? "
        f"{np.allclose(ev_Mx, ev_My) and np.allclose(ev_Mx, ev_Mz)}")

    log("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(_LOG_LINES) + "\n")
            print(f"\n[results written to {OUT_PATH}]")
        except Exception as e:
            print(f"WARNING: failed to write results file: {e}", file=sys.stderr)
