"""
bcc_L5_corner_deviation.py

BCC 5x5x5 lattice corner-deviation analysis for the lambda=-3 eigenspace.

Corner deviation = avg_proj - graph_degree for corner-shell BCs, where
  avg_proj = sum_{f in BC's 20 faces} sum_{i in lambda=-3 eigvecs} v_i(f)^2

This matches the metric from bcc_finite_size_scaling.py:
  L=3: corner deviation = +0.01378
  L=4: corner deviation = +0.01386
  Testing: does delta -> 1/72 = 0.013889 as L->inf?

Includes 3-point (L=3,4,5) extrapolation: delta(L) = delta_inf + c/L^alpha, alpha=1,2,3.

Only numpy used.
"""

import io
import itertools
import sys
from collections import defaultdict

import numpy as np

# Force stdout to UTF-8 on Windows (avoids GBK codec errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_L5_corner_results.txt"

_LOG_LINES = []


def log(msg=""):
    """Print to stdout AND buffer for the results file."""
    line = str(msg)
    print(line)
    _LOG_LINES.append(line)


# ---------------------------------------------------------------------------
# Lattice construction
# ---------------------------------------------------------------------------

def build_bcc_lattice(L):
    body_centers = []
    cube_corners = {}
    for i, j, k in itertools.product(range(L), repeat=3):
        bc = (i + 0.5, j + 0.5, k + 0.5)
        body_centers.append(bc)
        corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            corners.append((i + dx, j + dy, k + dz))
        cube_corners[bc] = tuple(corners)
    return body_centers, cube_corners


def build_simplices(body_centers, cube_corners):
    simplices = []
    simplex_owner_bc_idx = []
    for bc_idx, bc in enumerate(body_centers):
        corners = cube_corners[bc]
        even_tet = [c for c in corners if (c[0] + c[1] + c[2]) % 2 == 0]
        odd_tet  = [c for c in corners if (c[0] + c[1] + c[2]) % 2 == 1]
        assert len(even_tet) == 4 and len(odd_tet) == 4
        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))
        for tet in (even_tet, odd_tet):
            tet_f = [(float(c[0]), float(c[1]), float(c[2])) for c in tet]
            simplex = frozenset([bc_f] + tet_f)
            assert len(simplex) == 5
            simplices.append(simplex)
            simplex_owner_bc_idx.append(bc_idx)
    return simplices, simplex_owner_bc_idx


def build_faces(simplices):
    face_to_simplices = defaultdict(list)
    faces_in_order = []
    face_owner_simplex_idx = []
    for s_idx, simplex in enumerate(simplices):
        verts = sorted(simplex)
        for combo in itertools.combinations(verts, 3):
            face = frozenset(combo)
            faces_in_order.append(face)
            face_to_simplices[face].append(s_idx)
            face_owner_simplex_idx.append(s_idx)

    n_total  = len(faces_in_order)
    n_unique = len(face_to_simplices)
    duplicated = {f: o for f, o in face_to_simplices.items() if len(o) > 1}
    if duplicated:
        log(f"  WARNING: {len(duplicated)} faces shared by multiple simplices "
            f"(n_total={n_total}, n_unique={n_unique})")
    else:
        log(f"  VERIFY OK: all {n_total} faces are distinct (unique={n_unique})")
    return faces_in_order, n_total, n_unique, face_owner_simplex_idx


def build_adjacency_matrix(faces):
    ordered_faces = [tuple(sorted(f)) for f in faces]
    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(ordered_faces):
        for v in f:
            vertex_to_faces[v].add(idx)

    n = len(faces)
    A = np.zeros((n, n), dtype=float)
    face_sets = [frozenset(f) for f in ordered_faces]

    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a in range(len(flist)):
            for b in range(a + 1, len(flist)):
                candidate_pairs.add((flist[a], flist[b]))

    n_adj = 0
    for i, j in candidate_pairs:
        if len(face_sets[i] & face_sets[j]) == 2:
            A[i, j] = 1.0
            A[j, i] = 1.0
            n_adj += 1
    return A, n_adj


def group_eigenvalues(eigvals, tol=1e-6):
    groups = []
    cur = [eigvals[0]]
    for ev in eigvals[1:]:
        if abs(ev - cur[-1]) <= tol:
            cur.append(ev)
        else:
            groups.append(cur)
            cur = [ev]
    groups.append(cur)
    out = []
    cum = 0
    for g in groups:
        rep = float(np.mean(g))
        deg = len(g)
        cum += deg
        out.append((rep, deg, cum))
    return out


# ---------------------------------------------------------------------------
# BC classification helpers
# ---------------------------------------------------------------------------

def bc_grid_index(bc):
    return tuple(int(round(c - 0.5)) for c in bc)


def bc_interior_axes(ijk, L):
    """Number of axes where coord is strictly interior: 1 <= c <= L-2."""
    return sum(1 for c in ijk if 1 <= c <= L - 2)


def bc_graph_degree(ijk, L):
    """
    Number of existing BC neighbors in the simple-cubic BC graph.
    Corner BCs (all coords in {0, L-1}): degree = 3.
    """
    deg = 0
    for c in ijk:
        if c > 0:
            deg += 1
        if c < L - 1:
            deg += 1
    return deg


def bc_shell_name(n_interior):
    names = {0: "Corner", 1: "Edge", 2: "Face", 3: "Center"}
    return names.get(n_interior, f"Interior-{n_interior}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse_L5():
    L = 5
    log("=" * 78)
    log(f"BCC {L}x{L}x{L} corner-deviation analysis (lambda=-3 eigenspace)")
    log("Open boundary conditions")
    log("=" * 78)

    expected_bcs       = L**3
    expected_simplices = 2 * L**3
    expected_faces     = 20 * L**3

    # [1] Build lattice
    log("\n[1] Building lattice...")
    body_centers, cube_corners = build_bcc_lattice(L)
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    log(f"  Body centers : {len(body_centers)} (expect {expected_bcs})")
    log(f"  Simplices    : {len(simplices)}  (expect {expected_simplices})")

    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    log(f"  Faces (total): {n_total}  (expect {expected_faces})")

    if not (len(body_centers) == expected_bcs and
            len(simplices) == expected_simplices and
            n_total == expected_faces and
            n_unique == expected_faces):
        log("  *** WARNING: counts do not match expected! ***")

    # [2] Adjacency matrix
    log("\n[2] Building face-adjacency matrix...")
    A, n_adj = build_adjacency_matrix(faces)
    log(f"  Matrix size    : {n_total} x {n_total}")
    log(f"  Adjacent pairs : {n_adj}")

    # [3] Spectrum
    log(f"\n[3] Diagonalising {n_total}x{n_total} adjacency matrix...")
    eigvals, eigvecs = np.linalg.eigh(A)
    order   = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    grouped = group_eigenvalues(eigvals, tol=1e-6)
    log("\n  Full eigenvalue table (grouped, tol=1e-6):")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for rep, deg, cum in grouped:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")

    int_eigs = {}
    for rep, deg, _ in grouped:
        nearest_int = round(rep)
        if abs(rep - nearest_int) < 1e-4:
            int_eigs[nearest_int] = deg

    lam_neg3_deg = int_eigs.get(-3, 0)
    lam_neg2_deg = int_eigs.get(-2, 0)
    log(f"\n  lambda=-3 degeneracy : {lam_neg3_deg}")
    log(f"  lambda=-2 degeneracy : {lam_neg2_deg}")

    # [4] BC classification
    log("\n[4] BC shell classification...")
    bc_grid       = [bc_grid_index(bc) for bc in body_centers]
    bc_n_interior = [bc_interior_axes(ijk, L) for ijk in bc_grid]
    bc_graph_deg  = [bc_graph_degree(ijk, L) for ijk in bc_grid]

    shell_members = defaultdict(list)
    for bc_i, n_int in enumerate(bc_n_interior):
        shell_members[n_int].append(bc_i)

    log(f"  {'n_interior':>10} | {'shell':>7} | {'count':>6} | "
        f"{'graph_deg (first)':>18}")
    log(f"  {'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*18}")
    for n_int in sorted(shell_members.keys()):
        members = shell_members[n_int]
        name = bc_shell_name(n_int)
        gd   = bc_graph_deg[members[0]]
        log(f"  {n_int:10d} | {name:>7} | {len(members):6d} | {gd:18d}")

    # face -> parent BC
    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0]
                    for bc_i in range(len(body_centers))]
    patch_sizes = [len(x) for x in bc_face_idxs]
    log(f"\n  Per-BC face patches: min={min(patch_sizes)}, max={max(patch_sizes)}, "
        f"all==20? {all(s == 20 for s in patch_sizes)}")

    # [5] lambda=-3 eigenspace projection (matching reference script metric)
    log(f"\n[5] lambda=-3 eigenspace projection analysis...")
    neg3_mask = np.abs(eigvals - (-3.0)) <= 1e-6
    neg3_idx  = np.where(neg3_mask)[0]
    m_neg3    = len(neg3_idx)
    log(f"  Eigenvectors for lambda=-3: {m_neg3}")

    if m_neg3 == 0:
        log("  ERROR: no lambda=-3 eigenvectors found!")
        return None

    vecs = eigvecs[:, neg3_idx]          # (n_faces, m_neg3)
    sq   = np.sum(vecs**2, axis=1)       # per-face squared weight, summed over eigenspace

    # Per-BC: projection = sum_{f in BC's 20 faces} sq[f]
    # deviation = |projection - graph_degree|  (reference script metric)
    # Corner deviation = projection - graph_degree for corner BCs
    log(f"\n  Per-BC projections (corner BCs only, for verification):")
    log(f"  {'bc_idx':>6} | {'grid':>13} | {'n_int':>5} | {'graph_deg':>9} | "
        f"{'projection':>12} | {'deviation':>10}")
    log(f"  {'-'*6}-+-{'-'*13}-+-{'-'*5}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}")

    bc_rows = []
    for bc_i in range(len(body_centers)):
        idxs = bc_face_idxs[bc_i]
        proj = float(np.sum(sq[idxs]))
        d    = bc_graph_deg[bc_i]
        dev  = proj - d   # signed deviation (reference: abs, but sign tells direction)
        bc_rows.append((bc_i, bc_grid[bc_i], bc_n_interior[bc_i], d, proj, dev))

    # Print corner BCs
    corner_rows = [r for r in bc_rows if r[2] == 0]
    for bc_i, ijk, n_int, d, proj, dev in corner_rows:
        log(f"  {bc_i:6d} | {str(ijk):>13} | {n_int:5d} | {d:9d} | "
            f"{proj:12.6f} | {dev:10.6f}")

    total_proj = sum(r[4] for r in bc_rows)
    log(f"\n  Sum of all BC projections = {total_proj:.6f}  (expect ~{m_neg3})")

    # Per-shell summary (matching reference script)
    log(f"\n  Per-shell average projection:")
    log(f"  {'n_int':>5} | {'shell':>7} | {'count':>5} | {'avg_proj':>10} | "
        f"{'avg_deg':>8} | {'avg_dev':>10} | {'max_dev':>10}")
    log(f"  {'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    shell_results = {}
    for n_int in sorted(shell_members.keys()):
        members    = shell_members[n_int]
        name       = bc_shell_name(n_int)
        rows       = [r for r in bc_rows if r[2] == n_int]
        avg_proj   = float(np.mean([r[4] for r in rows]))
        avg_deg    = float(np.mean([r[3] for r in rows]))
        avg_dev    = float(np.mean([abs(r[5]) for r in rows]))
        max_dev    = float(np.max ([abs(r[5]) for r in rows]))
        signed_dev = float(np.mean([r[5] for r in rows]))  # signed, for corner
        shell_results[n_int] = {
            "name": name, "count": len(members),
            "avg_proj": avg_proj, "avg_deg": avg_deg,
            "avg_dev": avg_dev, "max_dev": max_dev,
            "signed_dev": signed_dev,
        }
        log(f"  {n_int:5d} | {name:>7} | {len(members):5d} | {avg_proj:10.6f} | "
            f"{avg_deg:8.2f} | {avg_dev:10.6f} | {max_dev:10.6f}")

    # Corner deviation = avg_proj - graph_degree for corner BCs
    # (graph_degree for corners at L=5 is 3, same as L=3,4)
    corner_info   = shell_results[0]
    corner_dev    = corner_info["signed_dev"]   # avg_proj - graph_degree
    corner_dev_abs = corner_info["avg_dev"]

    log(f"\n  Corner BC graph_degree         : {shell_results[0]['avg_deg']:.1f}")
    log(f"  Corner BC avg_proj             : {shell_results[0]['avg_proj']:.6f}")
    log(f"  Corner deviation (avg_proj - graph_deg): {corner_dev:.6f}")
    log(f"  Compare: L=3: 0.01378, L=4: 0.01386, 1/72 = {1/72:.6f}")

    return corner_dev, shell_results, m_neg3, n_total


def extrapolate_to_infinity(corner_dev_L5):
    """
    3-point (L=3,4,5) extrapolation: delta(L) = delta_inf + c/L^alpha
    for alpha=1,2,3.
    """
    log("\n" + "=" * 78)
    log("3-POINT EXTRAPOLATION TO L -> infinity")
    log("=" * 78)

    Ls   = np.array([3.0, 4.0, 5.0])
    devs = np.array([0.01378, 0.01386, corner_dev_L5])

    log(f"\n  Data points:")
    for Li, di in zip(Ls, devs):
        log(f"    L={int(Li)}: delta = {di:.6f}")
    log(f"  1/72 = {1/72:.6f}")

    log(f"\n  Fitting delta(L) = delta_inf + c/L^alpha (least-squares, 3 pts, 2 params):")
    log(f"  {'alpha':>6} | {'delta_inf':>12} | {'c':>12} | {'residual_rms':>14} | "
        f"{'diff from 1/72':>15}")
    log(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}-+-{'-'*15}")

    for alpha in [1, 2, 3]:
        X      = np.column_stack([np.ones(3), 1.0 / Ls**alpha])
        params = np.linalg.lstsq(X, devs, rcond=None)[0]
        delta_inf, c = params[0], params[1]
        fitted    = X @ params
        rms       = float(np.sqrt(np.mean((devs - fitted)**2)))
        diff      = delta_inf - 1.0/72
        log(f"  {alpha:6d} | {delta_inf:12.6f} | {c:12.6f} | {rms:14.8f} | {diff:+15.6f}")

    # Also: exact 2-point (L=4,5) extrapolation for each alpha
    log(f"\n  2-point exact extrapolation from (L=4, L=5):")
    log(f"  {'alpha':>6} | {'delta_inf':>12} | {'c':>12} | {'diff from 1/72':>15}")
    log(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*15}")
    d4 = 0.01386
    d5 = corner_dev_L5
    for alpha in [1, 2, 3]:
        inv4 = 1.0 / 4**alpha
        inv5 = 1.0 / 5**alpha
        c         = (d5 - d4) / (inv5 - inv4)
        delta_inf = d4 - c * inv4
        diff      = delta_inf - 1.0/72
        log(f"  {alpha:6d} | {delta_inf:12.6f} | {c:12.6f} | {diff:+15.6f}")


def main():
    log("=" * 78)
    log("BCC L=5 corner-deviation analysis")
    log("Metric: avg_proj - graph_degree for Corner shell BCs (lambda=-3 eigenspace)")
    log("Open boundary conditions")
    log("=" * 78)

    result = analyse_L5()
    if result is None:
        log("Analysis failed.")
        return

    corner_dev, shell_results, m_neg3, n_total = result

    log("\n" + "=" * 78)
    log("CORNER DEVIATION SUMMARY")
    log("=" * 78)
    log(f"  L=3 : delta_corner = 0.013780")
    log(f"  L=4 : delta_corner = 0.013860")
    log(f"  L=5 : delta_corner = {corner_dev:.6f}")
    log(f"  1/72 = {1/72:.6f}")
    log(f"  L=5 vs 1/72: diff = {corner_dev - 1/72:+.6f}")
    if 0.01386 < corner_dev < 1/72:
        trend = "monotonically converging toward 1/72 from below"
    elif corner_dev > 1/72:
        trend = "above 1/72 (overshooting)"
    else:
        trend = "non-monotone or below L=4 value"
    log(f"  Trend: {trend}")

    extrapolate_to_infinity(corner_dev)

    log("")
    log("=== DONE ===")


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
