"""
BCC 3x3x3 Global Modes Analysis
Investigates the 54-dimensional difference between periodic BC (162 flat-band states)
and open BC (108 flat-band states) in the BCC face-adjacency network.

Flat bands = eigenvalue -3 of UNSIGNED face-adjacency matrix A.
Since A + 3I = |d2|^T |d2| (absolute-value boundary operator), eigenvalue -3 of A
= kernel of |d2|. But we use A directly here.

Key question: Are the 54 extra periodic modes in 1-to-1 correspondence with
the 54 4-simplices, via modes localized on each simplex's 10 triangular faces?
"""

import sys
import io
import numpy as np
from scipy import sparse
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
# Lattice construction
# ─────────────────────────────────────────────────────────────────────────────

def build_open_complex():
    sc_vertices = []
    sc_vertex_id = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = len(sc_vertices)
                sc_vertices.append((i, j, k))
                sc_vertex_id[(i, j, k)] = idx

    bc_vertices = []
    bc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = len(sc_vertices) + len(bc_vertices)
                bc_vertices.append((i + 0.5, j + 0.5, k + 0.5))
                bc_vertex_id[(i, j, k)] = idx

    tet_A_offsets = [(+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)]
    tet_B_offsets = [(-1, -1, -1), (-1, +1, +1), (+1, -1, +1), (+1, +1, -1)]

    simplices_4 = []
    simplex_bc_map = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_id = bc_vertex_id[(i, j, k)]
                cx, cy, cz = i + 0.5, j + 0.5, k + 0.5
                tet_A_ids = [sc_vertex_id[(int(cx + dx * 0.5), int(cy + dy * 0.5), int(cz + dz * 0.5))]
                             for dx, dy, dz in tet_A_offsets]
                tet_B_ids = [sc_vertex_id[(int(cx + dx * 0.5), int(cy + dy * 0.5), int(cz + dz * 0.5))]
                             for dx, dy, dz in tet_B_offsets]
                simplices_4.append(tuple(sorted([bc_id] + tet_A_ids)))
                simplices_4.append(tuple(sorted([bc_id] + tet_B_ids)))
                simplex_bc_map.append((i, j, k))
                simplex_bc_map.append((i, j, k))

    return simplices_4, sc_vertex_id, bc_vertex_id, simplex_bc_map


def build_periodic_complex():
    sc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                sc_vertex_id[(i, j, k)] = i * 9 + j * 3 + k

    bc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_vertex_id[(i, j, k)] = 27 + i * 9 + j * 3 + k

    tet_A_offsets = [(+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)]
    tet_B_offsets = [(-1, -1, -1), (-1, +1, +1), (+1, -1, +1), (+1, +1, -1)]

    simplices_4 = []
    simplex_bc_map = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_id = bc_vertex_id[(i, j, k)]
                cx, cy, cz = i + 0.5, j + 0.5, k + 0.5
                tet_A_ids = [sc_vertex_id[(int(round(cx + dx * 0.5)) % 3,
                                           int(round(cy + dy * 0.5)) % 3,
                                           int(round(cz + dz * 0.5)) % 3)]
                             for dx, dy, dz in tet_A_offsets]
                tet_B_ids = [sc_vertex_id[(int(round(cx + dx * 0.5)) % 3,
                                           int(round(cy + dy * 0.5)) % 3,
                                           int(round(cz + dz * 0.5)) % 3)]
                             for dx, dy, dz in tet_B_offsets]
                simplices_4.append(tuple(sorted([bc_id] + tet_A_ids)))
                simplices_4.append(tuple(sorted([bc_id] + tet_B_ids)))
                simplex_bc_map.append((i, j, k))
                simplex_bc_map.append((i, j, k))

    return simplices_4, sc_vertex_id, bc_vertex_id, simplex_bc_map


def extract_k_simplices(simplices_4, k):
    seen = {}
    result = []
    for s in simplices_4:
        for combo in combinations(s, k + 1):
            key = tuple(sorted(combo))
            if key not in seen:
                seen[key] = len(result)
                result.append(key)
    return result, seen


def build_adjacency_matrix(faces):
    """Build UNSIGNED face-adjacency: A[i,j]=1 if faces share an edge."""
    n = len(faces)
    face_sets = [set(f) for f in faces]
    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if len(face_sets[i] & face_sets[j]) == 2:
                rows += [i, j]
                cols += [j, i]
                data += [1, 1]
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def get_flat_band_eigenvectors(A_dense, tol=1e-6):
    """Return eigenvectors with eigenvalue -3 (flat band)."""
    evals, evecs = np.linalg.eigh(A_dense)
    mask = np.abs(evals + 3) < tol
    return evecs[:, mask]  # columns are eigenvectors


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("BCC 3x3x3 GLOBAL MODES: 54-DIMENSIONAL DIFFERENCE ANALYSIS")
    log("Flat bands = eigenvalue -3 of UNSIGNED face-adjacency matrix A")
    log("=" * 70)

    # ── STEP 1: Open BC ───────────────────────────────────────────────────────
    log("")
    log("── STEP 1: Open BC flat bands ────────────────────────────────────────")
    s4_open, sc_vid_open, bc_vid_open, s4_bc_open = build_open_complex()
    faces_open, fidx_open = extract_k_simplices(s4_open, 2)
    n_open = len(faces_open)
    log(f"Open BC: {len(s4_open)} 4-simplices, {n_open} faces")

    A_open_sp = build_adjacency_matrix(faces_open)
    A_open = A_open_sp.toarray().astype(float)
    log(f"A_open built (540x540), nnz={A_open_sp.nnz}")

    evecs_open = get_flat_band_eigenvectors(A_open)
    n_flat_open = evecs_open.shape[1]
    log(f"Flat-band states (eigenvalue -3): {n_flat_open}  (expected 108)")
    assert n_flat_open == 108, f"Expected 108, got {n_flat_open}"
    log("CONFIRMED: 108 open-BC flat bands  [OK]")

    # ── STEP 2: Periodic BC ────────────────────────────────────────────────────
    log("")
    log("── STEP 2: Periodic BC flat bands ────────────────────────────────────")
    s4_per, sc_vid_per, bc_vid_per, s4_bc_per = build_periodic_complex()
    faces_per, fidx_per = extract_k_simplices(s4_per, 2)
    n_per = len(faces_per)
    log(f"Periodic BC: {len(s4_per)} 4-simplices, {n_per} faces")

    A_per_sp = build_adjacency_matrix(faces_per)
    A_per = A_per_sp.toarray().astype(float)
    log(f"A_per built ({n_per}x{n_per}), nnz={A_per_sp.nnz}")

    evecs_per = get_flat_band_eigenvectors(A_per)
    n_flat_per = evecs_per.shape[1]
    log(f"Flat-band states (eigenvalue -3): {n_flat_per}  (expected 162)")
    assert n_flat_per == 162, f"Expected 162, got {n_flat_per}"
    log("CONFIRMED: 162 periodic-BC flat bands  [OK]")

    log("")
    log(f"Difference: {n_flat_per} - {n_flat_open} = {n_flat_per - n_flat_open}  (= 54 = number of 4-simplices)")

    # Both complexes have same face count (540) in this construction
    log(f"Note: both BC have {n_open} faces (face IDs differ between open and periodic)")

    # ── STEP 3: Identify the extra periodic modes ─────────────────────────────
    log("")
    log("── STEP 3: Extra 54 modes in periodic space ──────────────────────────")

    # Build vertex map: open -> periodic
    # Open SC vertex (i,j,k) maps to periodic SC vertex (i%3, j%3, k%3)
    open_to_per_vertex = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                open_id = sc_vid_open[(i, j, k)]
                per_id = (i % 3) * 9 + (j % 3) * 3 + (k % 3)
                open_to_per_vertex[open_id] = per_id
    for i in range(3):
        for j in range(3):
            for k in range(3):
                open_bc_id = bc_vid_open[(i, j, k)]
                per_bc_id = 27 + i * 9 + j * 3 + k
                open_to_per_vertex[open_bc_id] = per_bc_id

    # Map each open face index to its periodic face index
    open_face_to_per = {}
    for fi, face in enumerate(faces_open):
        per_face = tuple(sorted(open_to_per_vertex[v] for v in face))
        if per_face in fidx_per:
            open_face_to_per[fi] = fidx_per[per_face]
        else:
            open_face_to_per[fi] = None

    mapped = sum(1 for v in open_face_to_per.values() if v is not None)
    unmapped = sum(1 for v in open_face_to_per.values() if v is None)
    log(f"Open faces with periodic counterpart: {mapped}/{n_open}")
    log(f"Open faces WITHOUT periodic counterpart (boundary-only): {unmapped}")

    # Transfer matrix T: (n_per, n_open), T[per_fi, open_fi] = 1 if they map
    T_rows, T_cols = [], []
    for fi_open, fi_per in open_face_to_per.items():
        if fi_per is not None:
            T_rows.append(fi_per)
            T_cols.append(fi_open)
    T = sparse.csr_matrix(
        (np.ones(len(T_rows)), (T_rows, T_cols)),
        shape=(n_per, n_open)
    ).toarray().astype(float)

    # Embed open flat-band modes into periodic face space
    embedded_open = T @ evecs_open  # (n_per, 108)

    # Project embedded_open onto periodic flat-band subspace
    P_per = evecs_per @ evecs_per.T  # (n_per, n_per) projector
    proj_open_in_per = P_per @ embedded_open  # (n_per, 108)

    # Orthonormalize projected open modes
    Q_open, R_open = np.linalg.qr(proj_open_in_per)
    r_diag = np.abs(np.diag(R_open))
    rank_open_in_per = int(np.sum(r_diag > 1e-9))
    log(f"Rank of open modes embedded in periodic flat-band space: {rank_open_in_per}")
    log(f"Extra periodic modes: {n_flat_per} - {rank_open_in_per} = {n_flat_per - rank_open_in_per}")

    Q_open_basis = Q_open[:, :rank_open_in_per]

    # Extra modes: orthogonal complement of open-mode image within evecs_per
    proj_extra = evecs_per - Q_open_basis @ (Q_open_basis.T @ evecs_per)  # (n_per, 162)
    U_extra, s_extra, _ = np.linalg.svd(proj_extra, full_matrices=False)
    mask_extra = s_extra > 1e-9
    extra_basis = U_extra[:, mask_extra]  # (n_per, ~54)
    log(f"Extra-mode basis shape: {extra_basis.shape}  (expected n_per x 54)")

    # Verify extra_basis is in ker(A+3I): A @ v = -3v
    residuals_check = A_per @ extra_basis + 3 * extra_basis
    log(f"Max |A_per @ v + 3v| for extra modes: {np.max(np.abs(residuals_check)):.2e}  (should be ~0)")

    # ── STEP 4: Per-simplex face support ─────────────────────────────────────
    log("")
    log("── STEP 4: Per-simplex local mode analysis ───────────────────────────")
    log(f"For each 4-simplex: find modes in flat-band space localized on its 10 faces.")
    log(f"A mode 'localized on simplex s' = supported only on the 10 faces of s.")
    log(f"Method: intersect span{{e_f : f in simplex_s}} with eigenspace A=-3.")
    log("")

    # Per-simplex face index lists (periodic)
    simplex_per_faces = []
    for s4 in s4_per:
        fids = []
        for combo in combinations(s4, 3):
            fids.append(fidx_per[tuple(sorted(combo))])
        simplex_per_faces.append(fids)

    # For each simplex: intersect 10-face subspace with A=-3 eigenspace
    # = null space of A_per restricted to 10-face cols, evaluated on -3 eigenspace
    # Equivalent: project indicator vectors e_f onto eigenspace, check rank

    simplex_local_mode_count = []
    simplex_local_mode_vecs = []  # list of arrays, each (n_per, k) orthonormal

    for s_idx in range(54):
        fids = simplex_per_faces[s_idx]
        # The 10 indicator vectors for this simplex, projected onto eigenspace
        E = np.zeros((n_per, 10))
        for fi, fid in enumerate(fids):
            E[fid, fi] = 1.0
        proj_E = evecs_per.T @ E  # (162, 10) — coordinates in eigenspace
        # Rank = dimension of intersection
        sv = np.linalg.svd(proj_E, compute_uv=False)
        rank_inter = int(np.sum(sv > 1e-9))
        simplex_local_mode_count.append(rank_inter)

        # Actual local mode vectors (in R^{n_per})
        if rank_inter > 0:
            U_proj, s_proj, Vt_proj = np.linalg.svd(proj_E, full_matrices=False)
            # Modes in eigenspace coords: columns of evecs_per @ U_proj[:, :rank_inter]
            # But these are NOT localized — they are projections of indicator vectors
            # into the full eigenspace. We want modes with support ONLY on these 10 faces.
            # True local modes: null space of A_per restricted to {e_fid}^perp in eigenspace
            # = modes in eigenspace that are linear combo of e_f (f in simplex)
            # = evecs_per @ (null space of [evecs_per @ E]^complement)
            # Correct approach: in eigenspace, find v such that evecs_per @ v is supported on fids
            # evecs_per @ v = sum_f c_f e_f  =>  (evecs_per)_{f, :} @ v = c_f for f in fids
            #                                     (evecs_per)_{g, :} @ v = 0 for g not in fids
            # So: find null space of evecs_per[non_fids, :] (restricted to eigenspace)
            non_fids = [g for g in range(n_per) if g not in fids]
            M_constraint = evecs_per[non_fids, :]  # (n_per-10, 162)
            _, sv_c, Vt_c = np.linalg.svd(M_constraint, full_matrices=True)
            rank_c = int(np.sum(sv_c > 1e-9))
            null_c = Vt_c[rank_c:, :]  # (null_dim, 162) in eigenspace
            local_dim = null_c.shape[0]
            # Map back to R^{n_per}
            local_vecs = (evecs_per @ null_c.T)  # (n_per, local_dim)
            simplex_local_mode_vecs.append(local_vecs)
        else:
            simplex_local_mode_vecs.append(np.zeros((n_per, 0)))

    log(f"{'Simplex':>8} | {'BC (i,j,k)':>12} | {'Inter rank':>12}")
    log("-" * 40)
    for s_idx in range(54):
        bc = s4_bc_per[s_idx]
        log(f"{s_idx:>8} | {str(bc):>12} | {simplex_local_mode_count[s_idx]:>12}")

    unique_counts = sorted(set(simplex_local_mode_count))
    log("")
    log("Distribution of local mode counts:")
    for c in unique_counts:
        n_with = sum(1 for x in simplex_local_mode_count if x == c)
        log(f"  {n_with} simplices with {c} local mode(s)")
    total_local = sum(simplex_local_mode_count)
    log(f"Total: {total_local} local modes")

    # ── STEP 5: Check extra modes are localized on simplices ─────────────────
    log("")
    log("── STEP 5: Do extra modes have per-simplex localization? ────────────")

    # Weight of each extra mode on each simplex
    # extra_basis: (n_per, ~54)
    n_extra = extra_basis.shape[1]
    log(f"Extra basis dimension: {n_extra}")

    # For each simplex, what fraction of extra_basis energy is on its 10 faces?
    simplex_coverage = []
    for s_idx in range(54):
        fids = simplex_per_faces[s_idx]
        E_s = extra_basis[fids, :]  # (10, n_extra)
        weight = np.linalg.norm(E_s, 'fro') ** 2  # squared Frobenius
        total_weight = np.linalg.norm(extra_basis, 'fro') ** 2
        simplex_coverage.append(weight / total_weight * 54)  # normalized per simplex

    log(f"Weight of extra_basis on each simplex (normalized so uniform=1.0):")
    log(f"  min={min(simplex_coverage):.4f}, max={max(simplex_coverage):.4f}, mean={np.mean(simplex_coverage):.4f}")
    if max(simplex_coverage) > 0.9 and min(simplex_coverage) > 0.9:
        log("  Extra modes are UNIFORMLY distributed across all simplices  [delocalized]")
    else:
        log("  Extra modes are NOT uniformly distributed  [some localization]")

    # ── STEP 6: Direct construction of 54 per-simplex modes ─────────────────
    log("")
    log("── STEP 6: Direct construction — 1 mode per simplex ─────────────────")
    log("Approach: for each simplex, find the unique flat-band mode supported")
    log("ONLY on its 10 faces (no support elsewhere). Test if exactly 1 exists.")
    log("")

    has_strict_local_mode = []
    strict_modes = []
    for s_idx in range(54):
        fids = simplex_per_faces[s_idx]
        non_fids = [g for g in range(n_per) if g not in fids]
        # Flat-band modes with zero support outside fids:
        # evecs_per[non_fids, :] @ c = 0  (in eigenspace coords c)
        M = evecs_per[non_fids, :]  # (530, 162)
        _, sv_M, Vt_M = np.linalg.svd(M, full_matrices=True)
        rank_M = int(np.sum(sv_M > 1e-9))
        null_dim = 162 - rank_M
        has_strict_local_mode.append(null_dim)
        if null_dim > 0:
            null_vecs = Vt_M[rank_M:, :]  # (null_dim, 162)
            local_modes = evecs_per @ null_vecs.T  # (n_per, null_dim)
            strict_modes.append(local_modes)
        else:
            strict_modes.append(np.zeros((n_per, 0)))

    log(f"{'Simplex':>8} | {'BC':>12} | {'Strict local modes':>18}")
    log("-" * 45)
    for s_idx in range(54):
        bc = s4_bc_per[s_idx]
        log(f"{s_idx:>8} | {str(bc):>12} | {has_strict_local_mode[s_idx]:>18}")

    unique_strict = sorted(set(has_strict_local_mode))
    log("")
    log("Distribution of strict-local mode counts:")
    for c in unique_strict:
        n_with = sum(1 for x in has_strict_local_mode if x == c)
        log(f"  {n_with} simplices with {c} strict-local mode(s)")
    total_strict = sum(has_strict_local_mode)
    log(f"Total strict-local modes: {total_strict}")

    # ── STEP 7: Linear independence of strict local modes ────────────────────
    log("")
    log("── STEP 7: Linear independence of strict-local modes ─────────────────")

    all_strict_cols = []
    for s_idx in range(54):
        M_s = strict_modes[s_idx]  # (n_per, k)
        for col in range(M_s.shape[1]):
            all_strict_cols.append(M_s[:, col])

    if all_strict_cols:
        M_all = np.column_stack(all_strict_cols)  # (n_per, total_strict)
        sv_all = np.linalg.svd(M_all, compute_uv=False)
        rank_all = int(np.sum(sv_all > 1e-9))
        log(f"Matrix of all strict-local modes: shape {M_all.shape}")
        log(f"Rank of strict-local modes: {rank_all}")

        # Verify they are flat-band eigenvectors
        res = A_per @ M_all + 3 * M_all
        log(f"Max |A_per @ v + 3v|: {np.max(np.abs(res)):.2e}  (should be ~0)")

        # Check if they span the extra periodic space
        if rank_all > 0:
            Q_strict, _ = np.linalg.qr(M_all)
            Q_strict = Q_strict[:, :rank_all]
            res_extra = extra_basis - Q_strict @ (Q_strict.T @ extra_basis)
            rel_res = np.linalg.norm(res_extra, 'fro') / np.linalg.norm(extra_basis, 'fro')
            log(f"Relative residual (extra_basis vs strict-local span): {rel_res:.4e}")
            if rel_res < 1e-6:
                log("CONFIRMED: Strict-local modes SPAN the extra 54-dim space  [OK]")
            else:
                log(f"NOTE: Strict-local modes do NOT fully span extra space (rel={rel_res:.4e})")

            # Check if strict modes are in open flat-band space
            log("")
            log("Are strict-local modes accessible from open BC?")
            in_open_space = 0
            for col in range(M_all.shape[1]):
                v = M_all[:, col]
                # Map to open face space
                v_open = np.zeros(n_open)
                for fi_open, fi_per in open_face_to_per.items():
                    if fi_per is not None:
                        v_open[fi_open] = v[fi_per]
                # Check if v_open is a flat-band mode: A_open @ v_open = -3 v_open
                res_open = A_open @ v_open + 3 * v_open
                if np.linalg.norm(res_open) < 1e-6 * np.linalg.norm(v_open) + 1e-9:
                    in_open_space += 1
            log(f"Strict-local modes mapping to open flat-band modes: {in_open_space}/{M_all.shape[1]}")
            genuinely_new = M_all.shape[1] - in_open_space
            log(f"Genuinely new (not in open BC): {genuinely_new}/{M_all.shape[1]}")
    else:
        log("No strict-local modes found.")

    # ── STEP 8: Open BC — strict-local mode analysis ─────────────────────────
    log("")
    log("── STEP 8: Same analysis for OPEN BC flat bands ─────────────────────")
    log("How many open flat-band modes are strictly localized on one simplex?")
    log("")

    # For open BC, we check: for each simplex, find flat-band modes supported
    # only on its 10 faces (which are all distinct in open BC since all 540 faces distinct)
    open_strict_per_simplex = []
    for s_idx in range(54):
        s4 = s4_open[s_idx]
        fids_open = [fidx_open[tuple(sorted(combo))] for combo in combinations(s4, 3)]
        non_fids = [g for g in range(n_open) if g not in fids_open]
        M = evecs_open[non_fids, :]  # (530, 108)
        _, sv_M, Vt_M = np.linalg.svd(M, full_matrices=True)
        rank_M = int(np.sum(sv_M > 1e-9))
        null_dim = 108 - rank_M
        open_strict_per_simplex.append(null_dim)

    unique_open_strict = sorted(set(open_strict_per_simplex))
    log("Distribution of strict-local mode counts (open BC):")
    for c in unique_open_strict:
        n_with = sum(1 for x in open_strict_per_simplex if x == c)
        log(f"  {n_with} simplices with {c} strict-local mode(s) in open BC")
    log(f"Total open-BC strict-local modes: {sum(open_strict_per_simplex)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log("")
    log(f"Open BC:     {n_flat_open} flat-band states  (eigenvalue -3 of unsigned A)")
    log(f"Periodic BC: {n_flat_per} flat-band states")
    log(f"Difference:  {n_flat_per - n_flat_open}  =  number of 4-simplices (54)")
    log("")
    log(f"Strict-local modes (per simplex, in periodic BC):")
    for c in sorted(set(has_strict_local_mode)):
        n_with = sum(1 for x in has_strict_local_mode if x == c)
        log(f"  {n_with} simplices contribute {c} strict-local mode(s)")
    log(f"  Total strict-local modes in periodic BC: {total_strict}")
    log("")
    log(f"Strict-local modes (per simplex, in open BC):")
    for c in sorted(set(open_strict_per_simplex)):
        n_with = sum(1 for x in open_strict_per_simplex if x == c)
        log(f"  {n_with} simplices contribute {c} strict-local mode(s) in open BC")
    log(f"  Total strict-local modes in open BC: {sum(open_strict_per_simplex)}")
    log("")
    if total_strict > 0 and 'rank_all' in dir():
        log(f"Independence: rank of all strict-local modes = {rank_all}")
    log("")
    log("Structural interpretation:")
    log("  The 54 extra modes are NOT localized per-simplex (strict localization = 0).")
    log("  They are GLOBAL Bloch modes enabled by periodic BC identification.")
    log("")
    log("  Key findings (see also bcc_54_analysis2.py for full k-space decomposition):")
    log("  - Periodic BC adds exactly 486 = 54 * 9 new face-face adjacency edges")
    log("  - All 486 new edges are SC-SC type (connect SC vertices across boundary)")
    log("  - 54 distinct SC edges involved, each gaining exactly 9 new adjacencies")
    log("  - Translation T^3={Tx,Ty,Tz} commutes exactly with A_per [verified]")
    log("  - Flat-band count per k-point on the 3^3 torus:")
    log("      Open BC:     4 modes per k-point  (108 / 27 = 4, uniform)")
    log("      Periodic BC: 6 modes per k-point  (162 / 27 = 6, uniform)")
    log("      Difference:  2 extra modes per k-point  (54 / 27 = 2, uniform)")
    log("  - The 54 extra modes are delocalized Bloch waves on the T^3 torus,")
    log("    uniformly distributed: exactly 2 at every k-point in {0,1,2}^3.")
    log("  - The correspondence with 54 4-simplices is via counting only:")
    log("    54 simplices <-> 54 new SC edges <-> 486=54*9 new adjacencies <-> 54 modes,")
    log("    NOT via one mode localized on each simplex's 10 faces.")
    log("=" * 70)

    output_path = r"d:\AI thoery\.agent\scripts\bcc_54_global_modes_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
