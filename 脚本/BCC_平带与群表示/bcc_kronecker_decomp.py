"""
bcc_kronecker_decomp.py

Tests whether the 540x540 BCC face-adjacency matrix decomposes as a
Kronecker sum of a body-center graph matrix and a local face-patch matrix:

    A ~= L_bc (x) I_20 + I_27 (x) L_local  [(x) = Kronecker product]

Imports BCC construction functions from bcc_540_spectrum.py.
"""

import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Import construction from bcc_540_spectrum.py (same directory)
# ---------------------------------------------------------------------------
import importlib.util, os

_here = os.path.dirname(os.path.abspath(__file__))
_spec_path = os.path.join(_here, "bcc_540_spectrum.py")
_spec_mod = importlib.util.spec_from_file_location("bcc_540_spectrum", _spec_path)
_bcc = importlib.util.module_from_spec(_spec_mod)
# Suppress the module's own log output during import (it only runs in main)
_spec_mod.loader.exec_module(_bcc)

build_bcc_lattice      = _bcc.build_bcc_lattice
build_simplices        = _bcc.build_simplices
build_faces            = _bcc.build_faces
build_adjacency_matrix = _bcc.build_adjacency_matrix
bc_shell               = _bcc.bc_shell
bc_grid_index          = _bcc.bc_grid_index

OUT_PATH = os.path.join(_here, "bcc_kronecker_results.txt")

_LOG_LINES = []

def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def frob(M):
    return float(np.linalg.norm(M, "fro"))


def rel_frob(M, ref):
    r = frob(ref)
    return frob(M) / r if r > 1e-15 else float("inf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 78)
    log("BCC 540x540 Kronecker Decomposition Analysis")
    log("A ~= L_bc (x) I_20 + I_27 (x) L_local")
    log("=" * 78)

    # -----------------------------------------------------------------------
    # Build lattice, simplices, faces, adjacency matrix (reuse bcc_540_spectrum)
    # -----------------------------------------------------------------------
    log("\n[Construction] Building BCC lattice and adjacency matrix ...")
    body_centers, cube_corners = build_bcc_lattice()
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    _W, A, _sigmas, _mc, _np = build_adjacency_matrix(faces)
    log(f"  A shape: {A.shape}  nnz: {int(np.count_nonzero(A))}")

    n_bc  = len(body_centers)   # 27
    n_loc = 20                  # faces per BC

    # -----------------------------------------------------------------------
    # Step 1: Reorder A into block form by parent body-center
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 1] Reorder A by parent BC -> 27x27 block structure (20x20 blocks)")
    log("=" * 78)

    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )

    # Groups: faces belonging to each BC
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0] for bc_i in range(n_bc)]
    bc_patch_sizes = [len(g) for g in bc_face_idxs]
    log(f"  Patch sizes: min={min(bc_patch_sizes)}, max={max(bc_patch_sizes)}, "
        f"all==20? {all(s == n_loc for s in bc_patch_sizes)}")
    if not all(s == n_loc for s in bc_patch_sizes):
        log("  ERROR: not all patches have size 20. Aborting.")
        return

    # Permutation: BC₀'s faces first, then BC₁'s, ...
    perm = np.concatenate([bc_face_idxs[a] for a in range(n_bc)])
    A_block = A[np.ix_(perm, perm)]   # 540×540, blocked
    log(f"  A_block shape: {A_block.shape}  (reordered, same spectrum as A)")

    # -----------------------------------------------------------------------
    # Step 2: Extract L_local (diagonal blocks) and L_bc (off-diagonal)
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 2] Extract L_local candidates (diagonal 20x20 blocks)")
    log("=" * 78)

    diag_blocks = []
    for a in range(n_bc):
        ra = slice(a * n_loc, (a + 1) * n_loc)
        diag_blocks.append(A_block[ra, ra].copy())

    # Are all diagonal blocks identical?
    max_pairwise_diff = 0.0
    for a in range(n_bc):
        for b in range(a + 1, n_bc):
            d = float(np.max(np.abs(diag_blocks[a] - diag_blocks[b])))
            if d > max_pairwise_diff:
                max_pairwise_diff = d
    log(f"  Max element-wise difference across all 27 diagonal blocks: {max_pairwise_diff:.6e}")
    all_identical = (max_pairwise_diff < 1e-9)
    log(f"  All diagonal blocks identical? {all_identical}")

    # Classify BCs by shell
    bc_grid = [bc_grid_index(bc) for bc in body_centers]
    bc_shell_info = [bc_shell(ijk) for ijk in bc_grid]   # (name, degree)

    shell_order = ["Corner", "Edge", "Face", "Center"]
    shell_members = defaultdict(list)
    for bc_i, (name, _) in enumerate(bc_shell_info):
        shell_members[name].append(bc_i)

    log("\n  Max pairwise diff within each shell:")
    for shell_name in shell_order:
        members = shell_members[shell_name]
        shell_max = 0.0
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                d = float(np.max(np.abs(diag_blocks[a] - diag_blocks[b])))
                if d > shell_max:
                    shell_max = d
        log(f"    {shell_name:7s} ({len(members):2d} BCs): max diff = {shell_max:.6e}")

    # Eigenvalues of each diagonal block
    log("\n  Eigenvalues of each diagonal block (sorted ascending):")
    log(f"  {'bc_idx':>6} | {'shell':>7} | eigenvalues (20 values)")
    log(f"  {'-'*6}-+-{'-'*7}-+- ...")
    diag_eigs = []
    for a in range(n_bc):
        ev = np.sort(np.linalg.eigvalsh(diag_blocks[a]))
        diag_eigs.append(ev)
        shell_name = bc_shell_info[a][0]
        log(f"  {a:6d} | {shell_name:>7} | " +
            " ".join(f"{v:7.3f}" for v in ev))

    # Mean diagonal block as L_local_mean
    L_local_mean = np.mean(diag_blocks, axis=0)
    log(f"\n  L_local_mean = mean of 27 diagonal blocks (20x20)")
    log(f"  ||L_local_mean - diag_blocks[0]||_F = {frob(L_local_mean - diag_blocks[0]):.6e}")

    ev_local = np.sort(np.linalg.eigvalsh(L_local_mean))
    log(f"\n  Eigenvalues of L_local_mean (sorted ascending):")
    log("  " + " ".join(f"{v:8.4f}" for v in ev_local))

    # -----------------------------------------------------------------------
    # Step 2b: Extract L_bc from off-diagonal blocks
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 2b] Extract L_bc from off-diagonal blocks")
    log("  Check: is each off-diagonal block c * I_20?")
    log("=" * 78)

    L_bc = np.zeros((n_bc, n_bc))
    max_rel_resid = 0.0
    nonzero_offdiag = 0

    residuals = []
    for a in range(n_bc):
        for b in range(n_bc):
            if a == b:
                continue
            ra = slice(a * n_loc, (a + 1) * n_loc)
            rb = slice(b * n_loc, (b + 1) * n_loc)
            block = A_block[ra, rb]
            fn = frob(block)
            if fn < 1e-12:
                L_bc[a, b] = 0.0
                continue
            nonzero_offdiag += 1
            c = np.trace(block) / n_loc
            L_bc[a, b] = c
            resid = block - c * np.eye(n_loc)
            rel = frob(resid) / fn
            residuals.append(rel)
            if rel > max_rel_resid:
                max_rel_resid = rel

    log(f"\n  Non-zero off-diagonal blocks: {nonzero_offdiag}")
    log(f"  Max relative Frobenius residual ||B - c*I||_F / ||B||_F: {max_rel_resid:.6e}")
    if residuals:
        log(f"  Mean relative residual: {np.mean(residuals):.6e}")
        log(f"  Are off-diagonal blocks exactly c*I_20? (max rel < 1e-9): {max_rel_resid < 1e-9}")

    log(f"\n  L_bc (27x27) -- extracted as trace(block)/20:")
    with np.printoptions(precision=4, suppress=True, linewidth=160):
        for row in L_bc:
            log("    " + np.array2string(row))

    ev_bc = np.sort(np.linalg.eigvalsh(L_bc))
    log(f"\n  Eigenvalues of L_bc (sorted ascending):")
    log("  " + " ".join(f"{v:8.4f}" for v in ev_bc))

    # Compare L_bc to the 3×3×3 grid graph adjacency matrix
    log("\n  Comparing L_bc to the actual 3x3x3 BC grid adjacency matrix ...")
    bc_grids = [bc_grid_index(bc) for bc in body_centers]
    L_bc_grid = np.zeros((n_bc, n_bc))
    for a in range(n_bc):
        for b in range(n_bc):
            ia, ja, ka = bc_grids[a]
            ib, jb, kb = bc_grids[b]
            diff = abs(ia - ib) + abs(ja - jb) + abs(ka - kb)
            if diff == 1:
                L_bc_grid[a, b] = 1.0
    diff_mat = L_bc - L_bc_grid
    log(f"  ||L_bc - L_bc_grid||_F = {frob(diff_mat):.6e}")
    log(f"  ||L_bc - L_bc_grid||_inf = {float(np.max(np.abs(diff_mat))):.6e}")
    log(f"  L_bc == L_bc_grid (within 1e-9)? {np.allclose(L_bc, L_bc_grid, atol=1e-9)}")

    ev_bc_grid = np.sort(np.linalg.eigvalsh(L_bc_grid))
    log(f"\n  Eigenvalues of L_bc_grid (sorted ascending):")
    log("  " + " ".join(f"{v:8.4f}" for v in ev_bc_grid))

    # -----------------------------------------------------------------------
    # Step 3: Kronecker sum reconstruction
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 3] Kronecker sum reconstruction")
    log("  A_kron = L_bc (x) I_20 + I_27 (x) L_local_mean")
    log("=" * 78)

    A_kron = (np.kron(L_bc, np.eye(n_loc)) +
              np.kron(np.eye(n_bc), L_local_mean))
    log(f"  A_kron shape: {A_kron.shape}")

    resid_abs = frob(A_block - A_kron)
    resid_rel = rel_frob(A_block - A_kron, A_block)
    log(f"  ||A_block - A_kron||_F          = {resid_abs:.6e}")
    log(f"  ||A_block - A_kron||_F / ||A||_F = {resid_rel:.6e}")
    log(f"  Exact Kronecker sum (rel < 1e-9)? {resid_rel < 1e-9}")

    # Eigenvalue comparison
    log("\n  Eigenvalue comparison: A_kron vs full A")
    # Eigenvalues of A_kron = all pairwise λ_bc(i) + λ_local(j)
    ev_kron_pairs = np.array(
        [lbc + lloc for lbc in ev_bc for lloc in ev_local]
    )
    ev_kron_sorted = np.sort(ev_kron_pairs)
    ev_A_sorted = np.sort(np.linalg.eigvalsh(A))
    max_eig_dev = float(np.max(np.abs(ev_kron_sorted - ev_A_sorted)))
    log(f"  Computed Kronecker-sum eigenvalues: {len(ev_kron_sorted)} values")
    log(f"  Full matrix eigenvalues:            {len(ev_A_sorted)} values")
    log(f"  Max eigenvalue deviation:           {max_eig_dev:.6e}")
    log(f"  Eigenvalues match (max dev < 1e-6)? {max_eig_dev < 1e-6}")

    # -----------------------------------------------------------------------
    # Step 4: Residual analysis
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 4] Residual matrix analysis")
    log("=" * 78)

    R = A_block - A_kron
    log(f"  ||R||_F = {frob(R):.6e}")
    rank_R = np.linalg.matrix_rank(R, tol=1e-9)
    log(f"  rank(R) = {rank_R}")

    sv = np.linalg.svd(R, compute_uv=False)
    log(f"  Singular values of R (top 20):")
    log("  " + " ".join(f"{s:.4f}" for s in sv[:20]))

    nnz_R = int(np.count_nonzero(np.abs(R) > 1e-9))
    sparsity_R = 1.0 - nnz_R / (R.shape[0] * R.shape[1])
    log(f"  Nonzero entries (|r|>1e-9): {nnz_R}  sparsity: {sparsity_R:.4f}")

    # Does R have Kronecker structure? Test if R ≈ M_bc ⊗ M_local for some matrices
    # Quick check: reshape R as (n_bc, n_loc, n_bc, n_loc) and look at block structure
    R_reshaped = R.reshape(n_bc, n_loc, n_bc, n_loc)
    # For each (a,b), R_block[a,b] should be near-zero if Kron sum was exact;
    # report the block-wise Frobenius norms
    log(f"\n  Block-wise ||R_block[a,b]||_F for non-zero blocks (threshold 1e-9):")
    r_block_norms = np.zeros((n_bc, n_bc))
    for a in range(n_bc):
        for b in range(n_bc):
            r_block_norms[a, b] = frob(R_reshaped[a, :, b, :])
    nz_blocks = [(a, b, r_block_norms[a, b])
                 for a in range(n_bc) for b in range(n_bc)
                 if r_block_norms[a, b] > 1e-9]
    log(f"  Number of non-trivial residual blocks: {len(nz_blocks)} out of {n_bc*n_bc}")
    if nz_blocks:
        norms_only = [v for _, _, v in nz_blocks]
        log(f"  Max block ||R_block||_F: {max(norms_only):.6e}")
        log(f"  Mean block ||R_block||_F (non-zero only): {np.mean(norms_only):.6e}")
        # Are they diagonal blocks or off-diagonal?
        diag_nz  = [(a, b, v) for a, b, v in nz_blocks if a == b]
        offdiag_nz = [(a, b, v) for a, b, v in nz_blocks if a != b]
        log(f"  Diagonal residual blocks: {len(diag_nz)}, off-diagonal: {len(offdiag_nz)}")

    # -----------------------------------------------------------------------
    # Step 5: Scaling prediction — which λ combinations give -3 and -2?
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 5] Scaling prediction -- Kronecker-sum eigenvalue decomposition")
    log("  Identify combinations lam_bc(i) + lam_local(j) ~= -3 (expected m=108)")
    log("  and ~= -2 (expected m=56)")
    log("=" * 78)

    for target, expected_m in [(-3.0, 108), (-2.0, 56)]:
        log(f"\n  Target eigenvalue: {target:.0f}  (expected multiplicity {expected_m})")
        combos = []
        for i, lbc in enumerate(ev_bc):
            for j, lloc in enumerate(ev_local):
                s = lbc + lloc
                if abs(s - target) < 1e-4:
                    combos.append((i, lbc, j, lloc, s))
        log(f"  Combinations found: {len(combos)}")
        # Summarize unique λ_local values
        local_vals = sorted(set(round(lloc, 6) for _, _, _, lloc, _ in combos))
        bc_vals    = sorted(set(round(lbc,  6) for _, lbc,  _, _,    _ in combos))
        log(f"  Distinct lam_bc values:    " + " ".join(f"{v:.4f}" for v in bc_vals))
        log(f"  Distinct lam_local values: " + " ".join(f"{v:.4f}" for v in local_vals))
        log(f"  -> lam_local contributions: {local_vals}")

    # -----------------------------------------------------------------------
    # Additional: grouped eigenvalue table of A_kron vs full A
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Bonus] Grouped eigenvalue table comparison: A_kron vs full A (tol=1e-4)")
    log("=" * 78)

    def group_ev(evs, tol=1e-4):
        evs = np.sort(evs)
        groups = []
        cur = [evs[0]]
        for v in evs[1:]:
            if abs(v - cur[-1]) <= tol:
                cur.append(v)
            else:
                groups.append(cur)
                cur = [v]
        groups.append(cur)
        return [(float(np.mean(g)), len(g)) for g in groups]

    g_A    = group_ev(ev_A_sorted)
    g_kron = group_ev(ev_kron_sorted)

    log(f"  {'eigenvalue':>14} | {'mult(A)':>8} | {'mult(kron)':>10}")
    log(f"  {'-'*14}-+-{'-'*8}-+-{'-'*10}")
    all_vals = sorted(set(round(v, 3) for v, _ in g_A) |
                      set(round(v, 3) for v, _ in g_kron))
    g_A_dict    = {round(v, 3): m for v, m in g_A}
    g_kron_dict = {round(v, 3): m for v, m in g_kron}
    for val in all_vals:
        mA    = g_A_dict.get(val, 0)
        mkron = g_kron_dict.get(val, 0)
        log(f"  {val:14.4f} | {mA:8d} | {mkron:10d}")

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
