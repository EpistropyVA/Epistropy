"""
bcc_L6_corner_deviation.py

Compute corner deviation for BCC 6x6x6 lattice (open BC).
4th data point for L->infinity extrapolation of delta(L).

Previous results:
  L=3: corner deviation = 0.013782
  L=4: corner deviation = 0.013860
  L=5: corner deviation = 0.013861
  Target: test if delta -> 1/72 = 0.013889 or some other rational

Construction: same as bcc_finite_size_scaling.py
  - BCC 6x6x6: 216 BCs, 432 simplices, 4320 faces
  - Face-adjacency matrix: 4320x4320
  - Use scipy.linalg.eigh (dense, ~30-60s) for full spectrum

Additional: pure-corner vs BC-containing face projection analysis.
"""

import io
import itertools
import sys
import time
from collections import defaultdict

import numpy as np
import scipy.linalg
import scipy.optimize

# Force stdout to UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_L6_corner_results.txt"

_LOG_LINES = []


def log(msg=""):
    line = str(msg)
    print(line, flush=True)
    _LOG_LINES.append(line)


# ---------------------------------------------------------------------------
# Lattice construction (same as bcc_finite_size_scaling.py)
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


def build_adjacency_matrix_sparse(faces):
    """Build adjacency matrix using sparse intermediate, return dense for eigh."""
    n = len(faces)
    ordered_faces = [tuple(sorted(f)) for f in faces]

    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(ordered_faces):
        for v in f:
            vertex_to_faces[v].add(idx)

    face_sets = [frozenset(f) for f in ordered_faces]

    # Collect adjacency pairs
    rows = []
    cols = []

    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a_pos in range(len(flist)):
            for b_pos in range(a_pos + 1, len(flist)):
                candidate_pairs.add((flist[a_pos], flist[b_pos]))

    n_adj = 0
    for i, j in candidate_pairs:
        if len(face_sets[i] & face_sets[j]) == 2:
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
            n_adj += 1

    # Build dense matrix
    A = np.zeros((n, n), dtype=np.float64)
    for r, c in zip(rows, cols):
        A[r, c] = 1.0

    return A, n_adj


def bc_grid_index(bc):
    return tuple(int(round(c - 0.5)) for c in bc)


def bc_interior_axes(ijk, L):
    return sum(1 for c in ijk if 1 <= c <= L - 2)


def bc_shell_name(n_interior):
    names = {0: "Corner", 1: "Edge", 2: "Face", 3: "Center"}
    return names.get(n_interior, f"Shell-{n_interior}")


def bc_degree_general(ijk, L):
    deg = 0
    for c in ijk:
        if c > 0:
            deg += 1
        if c < L - 1:
            deg += 1
    return deg


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
# Face type classification: pure-corner vs BC-containing
# ---------------------------------------------------------------------------

def classify_face_type(face, bc_set):
    """
    Classify a face (frozenset of 3 vertices) as:
      - 'pure_corner': no vertex is a BC (all 3 are integer-coord corner vertices)
      - 'bc_containing': at least one vertex is a BC (half-integer coords)
    A BC vertex has .5 in all coordinates.
    """
    for v in face:
        # BC vertices have half-integer coords: v = (i+0.5, j+0.5, k+0.5)
        if v[0] != round(v[0]):  # non-integer x => BC vertex
            return 'bc_containing'
    return 'pure_corner'


# ---------------------------------------------------------------------------
# Main L=6 analysis
# ---------------------------------------------------------------------------

def analyse_L6():
    L = 6
    log("=" * 78)
    log(f"  BCC {L}x{L}x{L} lattice (L={L}, open BC)")
    log("=" * 78)

    expected_bcs       = L**3      # 216
    expected_simplices = 2 * L**3  # 432
    expected_faces     = 20 * L**3 # 4320

    log(f"\n  Expected: {expected_bcs} BCs, {expected_simplices} simplices, {expected_faces} faces")

    t0 = time.time()

    # Build lattice
    log("\n  [1] Building BCC lattice...")
    body_centers, cube_corners = build_bcc_lattice(L)
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    log(f"  Body centers : {len(body_centers)} (expect {expected_bcs})")
    log(f"  Simplices    : {len(simplices)}  (expect {expected_simplices})")

    # Build faces
    log("\n  [2] Extracting faces...")
    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    log(f"  Faces (total): {n_total}  (expect {expected_faces})")
    log(f"  Elapsed: {time.time()-t0:.1f}s")

    # Build adjacency matrix
    log("\n  [3] Building adjacency matrix ({0}x{0})...".format(n_total))
    t1 = time.time()
    A, n_adj = build_adjacency_matrix_sparse(faces)
    log(f"  Adjacent face-pairs: {n_adj}")
    log(f"  Matrix build time: {time.time()-t1:.1f}s")

    # Full eigendecomposition
    log(f"\n  [4] Full eigendecomposition of {n_total}x{n_total} matrix (scipy.linalg.eigh)...")
    t2 = time.time()
    eigvals, eigvecs = scipy.linalg.eigh(A)
    log(f"  Eigendecomposition time: {time.time()-t2:.1f}s")
    log(f"  Total eigenvalues: {len(eigvals)}")

    # Group eigenvalues
    grouped = group_eigenvalues(eigvals, tol=1e-6)
    log(f"\n  Full eigenvalue table (grouped, tol=1e-6):")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for rep, deg, cum in grouped:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")

    # Find lambda=-3
    neg3_mask = np.abs(eigvals - (-3.0)) <= 1e-6
    neg3_idx  = np.where(neg3_mask)[0]
    m_neg3    = len(neg3_idx)
    log(f"\n  lambda=-3 eigenvalues found: {m_neg3}")

    # BC classification
    bc_grid      = [bc_grid_index(bc) for bc in body_centers]
    bc_n_interior = [bc_interior_axes(ijk, L) for ijk in bc_grid]
    bc_graph_deg  = [bc_degree_general(ijk, L) for ijk in bc_grid]

    shell_members = defaultdict(list)
    for bc_i, n_int in enumerate(bc_n_interior):
        shell_members[n_int].append(bc_i)

    log(f"\n  BC shell classification (L={L}):")
    log(f"  {'n_interior':>10} | {'shell':>7} | {'count':>6} | {'graph_deg':>9}")
    log(f"  {'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*9}")
    for n_int in sorted(shell_members.keys()):
        members = shell_members[n_int]
        name = bc_shell_name(n_int)
        gd = bc_graph_deg[members[0]]
        log(f"  {n_int:10d} | {name:>7} | {len(members):6d} | {gd:9d}")

    # face -> parent BC
    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0]
                    for bc_i in range(len(body_centers))]

    # --- Lambda=-3 projection analysis ---
    log(f"\n  [5] lambda=-3 eigenspace projection analysis")

    corner_deviation = None
    if m_neg3 > 0:
        vecs = eigvecs[:, neg3_idx]   # (n_faces, m_neg3)
        sq   = np.sum(vecs**2, axis=1)  # squared weight per face

        bc_rows = []
        for bc_i in range(len(body_centers)):
            idxs = bc_face_idxs[bc_i]
            proj = float(np.sum(sq[idxs]))
            d    = bc_graph_deg[bc_i]
            dev  = proj - d  # signed deviation
            bc_rows.append((bc_i, bc_grid[bc_i], bc_n_interior[bc_i], d, proj, dev))

        total_proj = sum(r[4] for r in bc_rows)
        log(f"  Sum of projections = {total_proj:.6f}  (expect {m_neg3})")

        # Per-shell average
        log(f"\n  Per-shell average projection:")
        log(f"  {'n_int':>5} | {'shell':>7} | {'count':>5} | {'avg_proj':>10} | "
            f"{'avg_deg':>8} | {'avg_dev(signed)':>15} | {'avg|dev|':>10}")
        log(f"  {'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*15}-+-{'-'*10}")

        shell_corner_dev = None
        for n_int in sorted(shell_members.keys()):
            members    = shell_members[n_int]
            name       = bc_shell_name(n_int)
            shell_rows = [r for r in bc_rows if r[2] == n_int]
            avg_proj   = float(np.mean([r[4] for r in shell_rows]))
            avg_deg    = float(np.mean([r[3] for r in shell_rows]))
            avg_dev_signed = float(np.mean([r[5] for r in shell_rows]))
            avg_abs_dev = float(np.mean([abs(r[5]) for r in shell_rows]))
            log(f"  {n_int:5d} | {name:>7} | {len(members):5d} | {avg_proj:10.6f} | "
                f"{avg_deg:8.2f} | {avg_dev_signed:15.6f} | {avg_abs_dev:10.6f}")
            if n_int == 0:  # Corner shell
                # Corner deviation = (avg_proj / avg_deg) - 1
                corner_deviation = avg_dev_signed / avg_deg  # = proj/deg - 1
                shell_corner_dev = avg_dev_signed

        # Correct definition: corner deviation = avg_proj - avg_deg (absolute, not normalized)
        # This matches L=3: 0.013782, L=4: 0.013860, L=5: 0.013861 convention
        corner_deviation_normalized = corner_deviation  # = avg_dev / avg_deg (was computed above)
        corner_deviation = shell_corner_dev  # absolute: avg_proj - avg_deg

        log(f"\n  Corner deviation (abs: avg_proj - graph_deg for corner BCs) = {corner_deviation:.8f}")
        log(f"  Corner deviation (normalized: /deg) = {corner_deviation_normalized:.8f}")
        log(f"  Corner deviation * 72 = {corner_deviation * 72:.6f}")
        log(f"  Corner deviation * 50 = {corner_deviation * 50:.6f}")
        log(f"  Corner deviation * 720 = {corner_deviation * 720:.6f}")

    # --- Pure-corner vs BC-containing face projection ---
    log(f"\n  [6] Pure-corner vs BC-containing face projection analysis")
    log("  (Tests 'perpendicular to 1D connectivity' hypothesis)")

    # Classify each face
    bc_set = set(body_centers)
    face_types = []
    for face in faces:
        ft = classify_face_type(face, bc_set)
        face_types.append(ft)

    face_types = np.array(face_types)
    pure_corner_mask = (face_types == 'pure_corner')
    bc_containing_mask = (face_types == 'bc_containing')
    n_pure = int(np.sum(pure_corner_mask))
    n_bc_c = int(np.sum(bc_containing_mask))
    log(f"  Pure-corner faces: {n_pure}")
    log(f"  BC-containing faces: {n_bc_c}")
    log(f"  Total: {n_pure + n_bc_c} (expect {n_total})")

    # Each BC has 20 faces total. How many are pure-corner?
    # A face is pure-corner if all 3 vertices are integer coords (no BC vertex)
    # In each simplex {BC, v1, v2, v3, v4}: 10 faces = C(5,3)
    #   - 1 face with all 3 from {v1,v2,v3,v4} and not BC: C(4,3)=4 pure-corner faces per simplex
    #   - C(4,2)=6 faces containing BC: each is {BC, vi, vj}
    # So per BC: 2 simplices * 4 pure = 8 pure-corner faces, 2*6=12 BC-containing faces
    log(f"  Expected: 8 pure-corner + 12 BC-containing per BC = {8*len(body_centers)} + {12*len(body_centers)}")

    if m_neg3 > 0:
        sq = np.sum(eigvecs[:, neg3_idx]**2, axis=1)

        log(f"\n  lambda=-3 projection weight split by face type and BC shell:")
        log(f"  {'n_int':>5} | {'shell':>7} | {'count':>5} | "
            f"{'pure_proj/BC':>12} | {'bc_c_proj/BC':>12} | "
            f"{'pure_frac':>9} | {'bc_c_frac':>9} | "
            f"{'pure_excess':>11} | {'uniform_pure':>12}")
        log(f"  {'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-"
            f"{'-'*9}-+-{'-'*9}-+-{'-'*11}-+-{'-'*12}")

        # Expected uniform: 8/20 = 0.4 pure, 12/20 = 0.6 bc_containing per BC
        uniform_pure_frac = 8.0 / 20.0
        uniform_bc_frac   = 12.0 / 20.0

        for n_int in sorted(shell_members.keys()):
            members = shell_members[n_int]
            name    = bc_shell_name(n_int)

            total_pure_proj = 0.0
            total_bc_c_proj = 0.0
            for bc_i in members:
                idxs = bc_face_idxs[bc_i]
                for fi in idxs:
                    w = sq[fi]
                    if pure_corner_mask[fi]:
                        total_pure_proj += w
                    else:
                        total_bc_c_proj += w

            n_bc = len(members)
            avg_pure = total_pure_proj / n_bc
            avg_bc_c = total_bc_c_proj / n_bc
            total_per_bc = avg_pure + avg_bc_c

            pure_frac = avg_pure / total_per_bc if total_per_bc > 0 else 0
            bc_c_frac = avg_bc_c / total_per_bc if total_per_bc > 0 else 0

            # Excess of pure-corner fraction over uniform expectation
            pure_excess = pure_frac - uniform_pure_frac

            # Uniform prediction: avg_proj * 8/20
            uniform_pure_pred = total_per_bc * uniform_pure_frac

            log(f"  {n_int:5d} | {name:>7} | {n_bc:5d} | "
                f"{avg_pure:12.6f} | {avg_bc_c:12.6f} | "
                f"{pure_frac:9.6f} | {bc_c_frac:9.6f} | "
                f"{pure_excess:11.6f} | {uniform_pure_pred:12.6f}")

    return corner_deviation


# ---------------------------------------------------------------------------
# 4-point extrapolation
# ---------------------------------------------------------------------------

def extrapolation_analysis(L_vals, delta_vals):
    log("")
    log("=" * 78)
    log("4-POINT EXTRAPOLATION: delta(L) = delta_inf + c/L^alpha")
    log("=" * 78)

    L_arr = np.array(L_vals, dtype=float)
    d_arr = np.array(delta_vals, dtype=float)

    log(f"\n  Data points:")
    for L, d in zip(L_vals, delta_vals):
        log(f"    L={L}: delta = {d:.8f}")

    log(f"\n  1/72 = {1.0/72:.8f}")
    log(f"  1/50 = {1.0/50:.8f}")

    # Try various alpha values
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0]

    log(f"\n  Fit results for delta_inf + c/L^alpha:")
    log(f"  {'alpha':>6} | {'delta_inf':>12} | {'c':>10} | "
        f"{'delta_inf*72':>12} | {'delta_inf*50':>12} | {'delta_inf*720':>13} | {'residual':>10}")
    log(f"  {'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}")

    for alpha in alphas:
        # Linear fit: delta = delta_inf + c * (1/L^alpha)
        # y = a + b*x where x = 1/L^alpha, y = delta, a = delta_inf, b = c
        x = 1.0 / L_arr**alpha
        # Least squares: [1, x] @ [a, b]^T = y
        X = np.column_stack([np.ones_like(x), x])
        coeffs, res, rank, sv = np.linalg.lstsq(X, d_arr, rcond=None)
        delta_inf = coeffs[0]
        c_coeff   = coeffs[1]

        y_pred = X @ coeffs
        residual = float(np.sum((d_arr - y_pred)**2))

        log(f"  {alpha:6.1f} | {delta_inf:12.8f} | {c_coeff:10.6f} | "
            f"{delta_inf*72:12.6f} | {delta_inf*50:12.6f} | {delta_inf*720:13.6f} | "
            f"{residual:10.2e}")

    # Best rational approximation for delta_inf from alpha=2.0 fit
    log(f"\n  Rational approximation search for delta_inf (alpha=2.0 fit):")
    alpha = 2.0
    x = 1.0 / L_arr**alpha
    X = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, d_arr, rcond=None)
    delta_inf_2 = coeffs[0]

    log(f"  delta_inf (alpha=2) = {delta_inf_2:.10f}")

    best_fracs = []
    for q in range(1, 1001):
        p = round(delta_inf_2 * q)
        frac = p / q
        err = abs(frac - delta_inf_2)
        best_fracs.append((err, p, q, frac))

    best_fracs.sort()
    log(f"\n  Top 15 rational approximations (q <= 1000):")
    log(f"  {'p':>6} / {'q':>6} = {'fraction':>12} | {'error':>12}")
    for err, p, q, frac in best_fracs[:15]:
        log(f"  {p:6d} / {q:6d} = {frac:12.8f} | {err:12.2e}")

    # Also check specific targets
    targets = {
        "1/72": 1.0/72,
        "1/70": 1.0/70,
        "1/68": 1.0/68,
        "1/75": 1.0/75,
        "1/80": 1.0/80,
        "13/936": 13.0/936,
    }
    log(f"\n  Distance from delta_inf (alpha=2) to specific targets:")
    for name, val in targets.items():
        log(f"    {name} = {val:.8f}, error = {abs(val - delta_inf_2):.2e}")

    # Also do alpha=1 (log-style) and alpha=1 separately
    log(f"\n  Summary: delta_inf for each alpha:")
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x = 1.0 / L_arr**alpha
        X = np.column_stack([np.ones_like(x), x])
        coeffs, _, _, _ = np.linalg.lstsq(X, d_arr, rcond=None)
        di = coeffs[0]
        log(f"    alpha={alpha:.1f}: delta_inf={di:.8f}  (*72={di*72:.6f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 78)
    log("BCC L=6 corner deviation + 4-point extrapolation")
    log("=" * 78)

    t_start = time.time()

    corner_dev_L6 = analyse_L6()

    t_total = time.time() - t_start
    log(f"\n  Total time: {t_total:.1f}s")

    log(f"\n  *** L=6 CORNER DEVIATION (absolute: avg_proj - graph_deg) = {corner_dev_L6:.8f} ***")
    log(f"  Previous: L=3: 0.013782, L=4: 0.013860, L=5: 0.013861")
    log(f"  1/72 = {1.0/72:.8f}")

    # 4-point extrapolation using absolute corner deviation
    L_vals     = [3, 4, 5, 6]
    delta_vals = [0.013782, 0.013860, 0.013861, corner_dev_L6]

    extrapolation_analysis(L_vals, delta_vals)

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
