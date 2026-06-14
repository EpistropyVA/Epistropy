"""
bcc_sympy_exact_deviation.py

Compute the EXACT rational corner deviation for the BCC 3x3x3 (L=3)
4-simplex face-adjacency matrix using the spectral projector formula.

Approach:
1. Build A (540x540 integer numpy array, open BC)
2. Get distinct eigenvalues via numpy, round to integers, verify
3. Build spectral projector P_{-3} = prod_{lambda != -3} (A - lambda*I) / (-3 - lambda)
   using numpy float64 matrix products (only ~5 multiplications of 540x540)
4. Extract corner-BC face diagonal entries
5. Sum to get corner deviation
6. Use fractions.Fraction.limit_denominator() for rational identification
7. Check delta * 72, delta * 720, delta * 24, delta * 360, etc.
"""

import io
import itertools
import sys
from collections import defaultdict

import numpy as np
from fractions import Fraction

# Force stdout to UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_sympy_exact_results.txt"

_LOG_LINES = []


def log(msg=""):
    """Print to stdout AND buffer for the results file."""
    line = str(msg)
    print(line)
    _LOG_LINES.append(line)


# ---------------------------------------------------------------------------
# Lattice construction (same as bcc_finite_size_scaling.py, L=3)
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
        log(f"  WARNING: {len(duplicated)} faces shared by multiple simplices")
    else:
        log(f"  VERIFY OK: all {n_total} faces distinct (unique={n_unique})")

    return faces_in_order, n_total, n_unique, face_owner_simplex_idx


def build_adjacency_matrix_int(faces):
    """Build adjacency matrix as numpy int32 array."""
    n = len(faces)
    ordered_faces = [tuple(sorted(f)) for f in faces]

    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(ordered_faces):
        for v in f:
            vertex_to_faces[v].add(idx)

    A = np.zeros((n, n), dtype=np.int32)
    face_sets = [frozenset(f) for f in ordered_faces]

    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a_pos in range(len(flist)):
            for b_pos in range(a_pos + 1, len(flist)):
                candidate_pairs.add((flist[a_pos], flist[b_pos]))

    n_adj = 0
    for i, j in candidate_pairs:
        if len(face_sets[i] & face_sets[j]) == 2:
            A[i, j] = 1
            A[j, i] = 1
            n_adj += 1

    return A, n_adj


def bc_grid_index(bc):
    return tuple(int(round(c - 0.5)) for c in bc)


def bc_degree_general(ijk, L):
    deg = 0
    for c in ijk:
        if c > 0:
            deg += 1
        if c < L - 1:
            deg += 1
    return deg


def bc_interior_axes(ijk, L):
    return sum(1 for c in ijk if 1 <= c <= L - 2)


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
# Main
# ---------------------------------------------------------------------------

def main():
    L = 3
    log("=" * 70)
    log(f"BCC {L}x{L}x{L} -- Exact Rational Corner Deviation")
    log(f"Spectral projector method, numpy float64")
    log("=" * 70)

    # --- Build lattice ---
    log(f"\n[1] Building BCC lattice L={L}...")
    body_centers, cube_corners = build_bcc_lattice(L)
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    log(f"  Body centers: {len(body_centers)}")
    log(f"  Simplices:    {len(simplices)}")
    log(f"  Faces total:  {n_total}")

    # --- Build integer adjacency matrix ---
    log(f"\n[2] Building integer adjacency matrix ({n_total}x{n_total})...")
    A_int, n_adj = build_adjacency_matrix_int(faces)
    log(f"  Adjacent pairs: {n_adj}")
    log(f"  A dtype: {A_int.dtype}, symmetric: {np.array_equal(A_int, A_int.T)}")

    # --- Get eigenvalues ---
    log(f"\n[3] Computing eigenvalues (numpy float64)...")
    A_float = A_int.astype(np.float64)
    eigvals_raw = np.linalg.eigvalsh(A_float)
    eigvals_sorted = np.sort(eigvals_raw)

    grouped = group_eigenvalues(eigvals_sorted, tol=1e-4)
    log(f"\n  Eigenvalue table:")
    log(f"  {'eigenvalue':>14} | {'rounded':>7} | {'deg':>6} | {'within 1e-4?':>12}")
    log(f"  {'-'*14}-+-{'-'*7}-+-{'-'*6}-+-{'-'*12}")

    distinct_int_eigs = []
    for rep, deg, cum in grouped:
        nr = round(rep)
        is_int = abs(rep - nr) < 1e-4
        log(f"  {rep:14.6f} | {nr:7d} | {deg:6d} | {'YES' if is_int else 'NO':>12}")
        if is_int:
            distinct_int_eigs.append(nr)

    log(f"\n  Distinct integer eigenvalues: {distinct_int_eigs}")
    log(f"  Total eigenvalues: {sum(d for _, d, _ in grouped)} (expect {n_total})")

    # Note: eigenvalues are NOT all integers for this matrix.
    # Use numpy eigenvectors for lambda=-3 subspace directly.
    log(f"\n  NOTE: Eigenvalues are NOT all integers (continuous spectrum).")
    log(f"  Using eigenvector projection method for lambda=-3 subspace.")

    # --- Get full eigenvectors and build P_{-3} from eigenvectors ---
    log(f"\n[4] Computing full eigenvectors and building P_{{-3}} projector...")
    eigvals_full, eigvecs_full = np.linalg.eigh(A_float)

    # Find lambda=-3 eigenvectors (tolerance 1e-6)
    neg3_mask = np.abs(eigvals_full - (-3.0)) <= 1e-6
    neg3_idx  = np.where(neg3_mask)[0]
    m_neg3    = len(neg3_idx)
    log(f"  lambda=-3 eigenvectors: {m_neg3}")

    # Build projector P_{-3} = V V^T where V columns are lambda=-3 eigenvectors
    V = eigvecs_full[:, neg3_idx]   # (n, m_neg3)
    P = V @ V.T                      # (n, n)

    log(f"  P shape: {P.shape}")

    # Verify P is a projector: P^2 should equal P
    log(f"\n[5] Verifying projector property P^2 = P ...")
    P2 = P @ P
    proj_err = np.max(np.abs(P2 - P))
    log(f"  max|P^2 - P| = {proj_err:.4e}  ({'OK' if proj_err < 1e-6 else 'FAIL - may have float64 precision issue'})")

    # Verify trace = degeneracy of lambda=-3
    trace_P = np.trace(P)
    neg3_deg = sum(deg for rep, deg, cum in grouped if abs(rep - (-3)) < 1e-4)
    log(f"  trace(P) = {trace_P:.6f}  (expect {neg3_deg})")
    log(f"  |trace(P) - {neg3_deg}| = {abs(trace_P - neg3_deg):.4e}")

    # --- BC/face classification ---
    log(f"\n[6] Classifying faces by BC type...")
    bc_grid = [bc_grid_index(bc) for bc in body_centers]
    bc_n_interior = [bc_interior_axes(ijk, L) for ijk in bc_grid]
    bc_graph_deg  = [bc_degree_general(ijk, L) for ijk in bc_grid]

    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0]
                    for bc_i in range(len(body_centers))]

    # Identify corner BCs (n_interior = 0, for L=3: coords in {0,2})
    corner_bc_indices = [i for i, n_int in enumerate(bc_n_interior) if n_int == 0]
    log(f"  Corner BCs (n_interior=0): {len(corner_bc_indices)}")
    log(f"  Corner BCs grid positions: {[bc_grid[i] for i in corner_bc_indices]}")

    # --- Compute projections per BC ---
    log(f"\n[7] Computing P_{{-3}} diagonal projections per BC...")

    diag_P = np.diag(P)

    # Projection for bc_i = sum of P[f,f] for f in faces owned by bc_i
    bc_proj = []
    for bc_i in range(len(body_centers)):
        idxs = bc_face_idxs[bc_i]
        proj = float(np.sum(diag_P[idxs]))
        bc_proj.append(proj)

    # Per-BC detail for corner BCs
    log(f"\n  Corner BC projections (graph degree = 3):")
    log(f"  {'bc_idx':>6} | {'grid':>13} | {'proj':>14} | {'graph_deg':>9} | {'deviation':>14}")
    log(f"  {'-'*6}-+-{'-'*13}-+-{'-'*14}-+-{'-'*9}-+-{'-'*14}")

    corner_deviations = []
    for bc_i in corner_bc_indices:
        proj = bc_proj[bc_i]
        deg  = bc_graph_deg[bc_i]
        dev  = proj - deg  # signed
        corner_deviations.append(dev)
        log(f"  {bc_i:6d} | {str(bc_grid[bc_i]):>13} | {proj:14.10f} | {deg:9d} | {dev:+14.10f}")

    # --- Summary statistics for corner deviation ---
    log(f"\n[8] Corner deviation summary...")
    corner_devs_arr = np.array(corner_deviations)
    log(f"  All equal?  {np.allclose(corner_devs_arr, corner_devs_arr[0], atol=1e-8)}")
    log(f"  Mean signed deviation: {np.mean(corner_devs_arr):.12f}")
    log(f"  Std  signed deviation: {np.std(corner_devs_arr):.4e}")

    delta = float(np.mean(corner_devs_arr))
    delta_abs = abs(delta)
    log(f"\n  Corner deviation delta        = {delta:.12f}")
    log(f"  Corner deviation |delta|      = {delta_abs:.12f}")

    # --- Rational identification ---
    log(f"\n[9] Rational identification...")

    # Continued fraction / limit_denominator approach
    for max_denom in [100, 1000, 10000, 100000]:
        f_rat = Fraction(delta_abs).limit_denominator(max_denom)
        log(f"  limit_denominator({max_denom:6d}): {f_rat} = {float(f_rat):.12f}  "
            f"(error = {abs(float(f_rat) - delta_abs):.4e})")

    log(f"\n  Checking delta * N for small N (integer proximity):")
    check_multiples = [6, 8, 12, 18, 24, 36, 48, 60, 72, 80, 90, 100,
                       120, 144, 180, 216, 240, 360, 432, 540, 720,
                       1080, 1440, 2160]
    for N in check_multiples:
        val = delta_abs * N
        nearest = round(val)
        err = abs(val - nearest)
        flag = "  <-- INTEGER" if err < 1e-6 else (
               "  <-- near"   if err < 1e-4 else "")
        log(f"  delta * {N:5d} = {val:.10f}  (nearest int = {nearest}, err = {err:.4e}){flag}")

    # Best rational guess
    best_frac = Fraction(delta_abs).limit_denominator(10000)
    log(f"\n  Best rational (denom <= 10000): {best_frac}")
    log(f"    Numerator  p = {best_frac.numerator}")
    log(f"    Denominator q = {best_frac.denominator}")
    log(f"    float(p/q)  = {float(best_frac):.12f}")
    log(f"    error       = {abs(float(best_frac) - delta_abs):.4e}")
    log(f"    q mod 72    = {best_frac.denominator % 72}")
    log(f"    q mod 24    = {best_frac.denominator % 24}")
    log(f"    q mod 360   = {best_frac.denominator % 360}")

    # Is 1/72 a good fit?
    log(f"\n  Specific fraction checks:")
    for num, den in [(1, 72), (1, 60), (1, 80), (1, 90), (1, 100),
                     (1, 108), (1, 120), (1, 144), (1, 180), (1, 216),
                     (5, 360), (7, 540), (1, 540), (1, 720)]:
        val = num / den
        err = abs(val - delta_abs)
        flag = "  <-- EXACT MATCH" if err < 1e-10 else (
               "  <-- very close"  if err < 1e-6  else (
               "  <-- close"       if err < 1e-4  else ""))
        log(f"  {num}/{den} = {val:.12f}  (err = {err:.4e}){flag}")

    # --- Also compute per-shell projections for full picture ---
    log(f"\n[10] Per-shell projection summary (lambda=-3)...")
    shell_members = defaultdict(list)
    for bc_i, n_int in enumerate(bc_n_interior):
        shell_members[n_int].append(bc_i)

    shell_names = {0: "Corner", 1: "Edge", 2: "Face", 3: "Center"}
    log(f"  {'n_int':>5} | {'shell':>7} | {'count':>5} | {'avg_proj':>14} | "
        f"{'graph_deg':>9} | {'avg_dev':>14} | {'std_dev':>14}")
    log(f"  {'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*14}-+-{'-'*9}-+-{'-'*14}-+-{'-'*14}")

    for n_int in sorted(shell_members.keys()):
        members = shell_members[n_int]
        projs = [bc_proj[bc_i] for bc_i in members]
        degs  = [bc_graph_deg[bc_i] for bc_i in members]
        devs  = [bc_proj[bc_i] - bc_graph_deg[bc_i] for bc_i in members]
        name = shell_names.get(n_int, f"Int-{n_int}")
        log(f"  {n_int:5d} | {name:>7} | {len(members):5d} | "
            f"{np.mean(projs):14.10f} | {int(round(np.mean(degs))):9d} | "
            f"{np.mean(devs):+14.10f} | {np.std(devs):14.4e}")

    # --- Final verdict ---
    log(f"\n{'='*70}")
    log(f"FINAL VERDICT")
    log(f"{'='*70}")
    log(f"  Corner deviation (mean) : {delta:.12f}")
    log(f"  Corner deviation |delta|: {delta_abs:.12f}")
    log(f"  Best rational (<=10000) : {best_frac}")
    log(f"  Is 1/72 exact (1e-10)?  : {abs(delta_abs - 1/72) < 1e-10}")
    log(f"  Is 1/72 close (1e-4)?   : {abs(delta_abs - 1/72) < 1e-4}")
    log(f"  delta * 72              : {delta_abs * 72:.12f}")
    log(f"  delta * 720             : {delta_abs * 720:.12f}")
    log(f"  Projector error P^2-P   : {proj_err:.4e}")
    log(f"  Trace error             : {abs(trace_P - neg3_deg):.4e}")


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
