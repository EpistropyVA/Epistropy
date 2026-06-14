"""
bcc_finite_size_scaling.py

Finite-size scaling analysis of the BCC L×L×L 4-simplex face-adjacency spectrum.

Runs the full spectral pipeline for L=2, 3, 4 (open boundary conditions) and
compares how key spectral features (eigenvalue degeneracies, λ=-3 projection
deviation) scale with L.

Key question: does the λ=-3 projection deviation scale as O(1/L²)?

Only numpy and scipy are used.
"""

import io
import itertools
import sys
from collections import defaultdict

import numpy as np
import scipy  # noqa: F401

# Force stdout to UTF-8 on Windows (avoids GBK codec errors for Greek/math chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_finite_size_results.txt"

_LOG_LINES = []


def log(msg=""):
    """Print to stdout AND buffer for the results file."""
    line = str(msg)
    print(line)
    _LOG_LINES.append(line)


# ---------------------------------------------------------------------------
# Lattice construction (generalised to L)
# ---------------------------------------------------------------------------

def build_bcc_lattice(L):
    """
    Build an L×L×L BCC lattice.

    Body-center atoms sit at (i+0.5, j+0.5, k+0.5) for i,j,k in {0,...,L-1}.
    Each body center's 8 nearest-neighbor corners are the integer vertices of
    its enclosing unit cube (no periodic wrapping — open BCs).

    Returns:
        body_centers: list of L³ tuples (float coords)
        cube_corners: dict body_center -> tuple of 8 corner tuples (int coords)
    """
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
    """
    Split each cube's 8 corners into 2 tetrahedra by parity of (x+y+z).
    Each 4-simplex = {body_center} ∪ {4 corners of one tetrahedron}.

    Returns:
        simplices: list of 2L³ frozensets, each with 5 vertex tuples
        simplex_owner_bc_idx: list of ints, one per simplex
    """
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
    """
    Extract all C(5,3)=10 triangular faces per simplex.
    Verifies all faces are distinct (no shared faces between simplices for open BC).

    Returns:
        faces: list of frozensets (3 vertices each)
        n_total, n_unique: counts
        face_owner_simplex_idx: parent simplex index per face
    """
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
    """
    Build unsigned adjacency matrix A: A[i,j]=1 iff faces i,j share exactly 2 vertices.

    Returns A, n_adjacent_pairs.
    """
    n = len(faces)
    ordered_faces = [tuple(sorted(f)) for f in faces]

    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(ordered_faces):
        for v in f:
            vertex_to_faces[v].add(idx)

    A = np.zeros((n, n), dtype=float)
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
            A[i, j] = 1.0
            A[j, i] = 1.0
            n_adj += 1

    return A, n_adj


def group_eigenvalues(eigvals, tol=1e-6):
    """
    Group sorted eigenvalues within tol.
    Returns list of (representative_value, degeneracy, cumulative_count).
    """
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
# BC classification helpers (generalised to L)
# ---------------------------------------------------------------------------

def bc_grid_index(bc):
    """Recover integer grid coords from bc=(i+.5,j+.5,k+.5)."""
    return tuple(int(round(c - 0.5)) for c in bc)


def bc_degree_general(ijk, L):
    """
    Degree of a BC in the L×L×L BC graph (simple-cubic adjacency):
    number of axes where coord is in {1,...,L-2} (has a neighbour on both sides).
    Equivalent to counting axes where coord is strictly interior.

    For L=2: {1,...,0} is empty → all BCs have degree 0 "interior axes",
             but the actual graph degree counts direct neighbours:
             coord 0 has 1 neighbour (coord 1), coord 1 has 1 neighbour (coord 0).
    So the graph-degree formula (counting existing neighbours) is:
        deg = sum over axes of (1 if coord > 0) + (1 if coord < L-1)
    """
    deg = 0
    for c in ijk:
        if c > 0:
            deg += 1
        if c < L - 1:
            deg += 1
    return deg


def bc_interior_axes(ijk, L):
    """
    Number of axes where coord is strictly interior (has neighbours on BOTH sides).
    = number of axes where 1 <= coord <= L-2.
    This is the "degree" notion from the spec's shell classification for general L:
      L=3: 0→Corner(deg 3), 1→Edge(deg 4), 2→Face(deg 5), 3→Center(deg 6)
    For L=2: all coords are in {0,1}, none are strictly interior → all = 0
    For L=4: interior = {1,2}; corner = {0,3}
    """
    return sum(1 for c in ijk if 1 <= c <= L - 2)


def bc_shell_name(n_interior):
    """Map number of interior axes to a shell name."""
    names = {0: "Corner", 1: "Edge", 2: "Face", 3: "Center"}
    return names.get(n_interior, f"Interior-{n_interior}")


# ---------------------------------------------------------------------------
# Per-L analysis
# ---------------------------------------------------------------------------

def analyse_L(L):
    """
    Run the full pipeline for an L×L×L BCC lattice.
    Returns a dict of summary statistics for the scaling table.
    """
    log("")
    log("=" * 78)
    log(f"  BCC {L}×{L}×{L} lattice  (L={L})")
    log("=" * 78)

    expected_bcs       = L**3
    expected_simplices = 2 * L**3
    expected_faces     = 20 * L**3

    # --- Build ---
    body_centers, cube_corners = build_bcc_lattice(L)
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    log(f"\n  Body centers : {len(body_centers)} (expect {expected_bcs})")
    log(f"  Simplices    : {len(simplices)}  (expect {expected_simplices})")

    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    log(f"  Faces (total): {n_total}  (expect {expected_faces})")

    if not (len(body_centers) == expected_bcs and
            len(simplices) == expected_simplices and
            n_total == expected_faces and
            n_unique == expected_faces):
        log("  *** WARNING: counts do not match expected! ***")

    # --- Adjacency ---
    A, n_adj = build_adjacency_matrix(faces)
    log(f"  Adjacent face-pairs: {n_adj}")

    # --- Spectrum ---
    log(f"\n  Diagonalising {n_total}×{n_total} adjacency matrix ...")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)
    eigvals  = eigvals[order]
    eigvecs  = eigvecs[:, order]

    grouped = group_eigenvalues(eigvals, tol=1e-6)
    log(f"\n  Full eigenvalue table (grouped, tol=1e-6):")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for rep, deg, cum in grouped:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")
    log(f"  Total eigenvalues: {sum(d for _, d, _ in grouped)}  (expect {n_total})")

    # --- Integer eigenvalue check ---
    log(f"\n  Exact integer eigenvalue check (within 1e-4):")
    int_eigs = {}
    for rep, deg, _ in grouped:
        nearest_int = round(rep)
        if abs(rep - nearest_int) < 1e-4:
            int_eigs[nearest_int] = deg
            log(f"    λ = {nearest_int:+d}  (rep={rep:.6f})  degeneracy = {deg}")

    lam_neg3_deg = int_eigs.get(-3, 0)
    lam_neg2_deg = int_eigs.get(-2, 0)
    log(f"\n  λ=-3 degeneracy : {lam_neg3_deg}")
    log(f"  λ=-2 degeneracy : {lam_neg2_deg}")

    # --- BC shell classification ---
    bc_grid = [bc_grid_index(bc) for bc in body_centers]

    # For general L, classify by number of interior axes
    bc_n_interior = [bc_interior_axes(ijk, L) for ijk in bc_grid]
    bc_graph_deg  = [bc_degree_general(ijk, L) for ijk in bc_grid]

    # Group by n_interior
    shell_members = defaultdict(list)
    for bc_i, n_int in enumerate(bc_n_interior):
        shell_members[n_int].append(bc_i)

    log(f"\n  BC shell classification (by number of interior axes, L={L}):")
    log(f"  {'n_interior':>10} | {'shell':>7} | {'count':>6} | "
        f"{'graph_deg (first member)':>25}")
    log(f"  {'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*25}")
    for n_int in sorted(shell_members.keys()):
        members = shell_members[n_int]
        name = bc_shell_name(n_int)
        gd = bc_graph_deg[members[0]]
        log(f"  {n_int:10d} | {name:>7} | {len(members):6d} | {gd:25d}")

    # face -> parent BC index
    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0]
                    for bc_i in range(len(body_centers))]
    patch_sizes = [len(x) for x in bc_face_idxs]
    log(f"\n  Per-BC face-patch sizes: min={min(patch_sizes)}, max={max(patch_sizes)}, "
        f"all==20? {all(s == 20 for s in patch_sizes)}")

    # --- λ=-3 eigenspace projection per shell ---
    log(f"\n  [λ=-3 eigenspace projection analysis]")
    result_neg3 = {}
    avg_dev_neg3 = None
    max_dev_neg3 = None

    neg3_mask = np.abs(eigvals - (-3.0)) <= 1e-6
    neg3_idx  = np.where(neg3_mask)[0]
    m_neg3    = len(neg3_idx)
    log(f"    Eigenvectors found for λ=-3: {m_neg3}")

    if m_neg3 > 0:
        vecs = eigvecs[:, neg3_idx]   # (n_faces, m)
        sq   = np.sum(vecs**2, axis=1)  # squared weight per face, summed over eigenspace

        log(f"\n    Per-BC projections:")
        log(f"    {'bc_idx':>6} | {'grid':>13} | {'n_int':>5} | {'graph_deg':>9} | "
            f"{'projection':>12} | {'deviation':>10}")
        log(f"    {'-'*6}-+-{'-'*13}-+-{'-'*5}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}")

        bc_rows = []
        for bc_i in range(len(body_centers)):
            idxs = bc_face_idxs[bc_i]
            proj = float(np.sum(sq[idxs]))
            d    = bc_graph_deg[bc_i]
            dev  = abs(proj - d)
            bc_rows.append((bc_i, bc_grid[bc_i], bc_n_interior[bc_i], d, proj, dev))

        for bc_i, ijk, n_int, d, proj, dev in bc_rows:
            log(f"    {bc_i:6d} | {str(ijk):>13} | {n_int:5d} | {d:9d} | "
                f"{proj:12.6f} | {dev:10.6f}")

        total_proj = sum(r[4] for r in bc_rows)
        log(f"\n    Sum of projections = {total_proj:.6f}  (expect ~{m_neg3})")

        # Average projection and deviation per shell
        log(f"\n    Per-shell average projection (predict: avg_proj ≈ graph_degree):")
        log(f"    {'n_int':>5} | {'shell':>7} | {'count':>5} | {'avg_proj':>10} | "
            f"{'avg_deg':>8} | {'avg_dev':>10} | {'max_dev':>10}")
        log(f"    {'-'*5}-+-{'-'*7}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

        all_devs = []
        for n_int in sorted(shell_members.keys()):
            members   = shell_members[n_int]
            name      = bc_shell_name(n_int)
            shell_rows = [r for r in bc_rows if r[2] == n_int]
            avg_proj   = float(np.mean([r[4] for r in shell_rows]))
            avg_deg    = float(np.mean([r[3] for r in shell_rows]))
            avg_dev    = float(np.mean([r[5] for r in shell_rows]))
            max_dev    = float(np.max ([r[5] for r in shell_rows]))
            result_neg3[n_int] = {
                "name": name, "count": len(members),
                "avg_proj": avg_proj, "avg_deg": avg_deg,
                "avg_dev": avg_dev, "max_dev": max_dev,
            }
            all_devs.extend([r[5] for r in shell_rows])
            log(f"    {n_int:5d} | {name:>7} | {len(members):5d} | {avg_proj:10.6f} | "
                f"{avg_deg:8.2f} | {avg_dev:10.6f} | {max_dev:10.6f}")

        avg_dev_neg3 = float(np.mean(all_devs))
        max_dev_neg3 = float(np.max (all_devs))
        log(f"\n    OVERALL avg |projection - degree| = {avg_dev_neg3:.6f}")
        log(f"    OVERALL max |projection - degree| = {max_dev_neg3:.6f}")

    # --- λ=-2 eigenspace: degeneracy + boundary confinement ---
    log(f"\n  [λ=-2 eigenspace analysis]")
    neg2_mask = np.abs(eigvals - (-2.0)) <= 1e-6
    neg2_idx  = np.where(neg2_mask)[0]
    m_neg2    = len(neg2_idx)
    log(f"    Eigenvectors found for λ=-2: {m_neg2}")

    boundary_frac_neg2 = None
    if m_neg2 > 0:
        vecs2 = eigvecs[:, neg2_idx]
        sq2   = np.sum(vecs2**2, axis=1)

        # Classify each BC as boundary (has at least one coord == 0 or == L-1)
        def is_boundary(ijk):
            return any(c == 0 or c == L - 1 for c in ijk)

        bc_is_boundary  = [is_boundary(ijk) for ijk in bc_grid]
        n_boundary_bcs  = sum(bc_is_boundary)
        log(f"    Boundary BCs: {n_boundary_bcs} / {len(body_centers)}")

        # Projection weight on boundary BCs vs total
        boundary_weight = 0.0
        interior_weight = 0.0
        for bc_i, is_b in enumerate(bc_is_boundary):
            idxs = bc_face_idxs[bc_i]
            w = float(np.sum(sq2[idxs]))
            if is_b:
                boundary_weight += w
            else:
                interior_weight += w
        total_w2 = boundary_weight + interior_weight
        boundary_frac_neg2 = boundary_weight / total_w2 if total_w2 > 0 else 0.0
        log(f"    Boundary weight fraction: {boundary_frac_neg2:.6f}  "
            f"(boundary={boundary_weight:.4f}, interior={interior_weight:.4f}, "
            f"total={total_w2:.4f} ≈ {m_neg2})")

    return {
        "L"                 : L,
        "n_bcs"             : len(body_centers),
        "n_simplices"       : len(simplices),
        "n_faces"           : n_total,
        "n_adj"             : n_adj,
        "lam_neg3_deg"      : lam_neg3_deg,
        "lam_neg2_deg"      : lam_neg2_deg,
        "avg_dev_neg3"      : avg_dev_neg3,
        "max_dev_neg3"      : max_dev_neg3,
        "boundary_frac_neg2": boundary_frac_neg2,
        "shell_results_neg3": result_neg3,
        "m_neg3_actual"     : m_neg3,
        "m_neg2_actual"     : m_neg2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=" * 78)
    log("BCC finite-size scaling: L×L×L 4-simplex face-adjacency spectrum")
    log("Open boundary conditions, L = 2, 3, 4")
    log("=" * 78)

    results = {}
    for L in (2, 3, 4):
        results[L] = analyse_L(L)

    # -------------------------------------------------------------------------
    # Summary scaling table
    # -------------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("SUMMARY SCALING TABLE")
    log("=" * 78)
    log(f"  {'L':>4} | {'n_bcs':>6} | {'n_simp':>7} | {'n_faces':>8} | "
        f"{'λ=-3 deg':>8} | {'λ=-2 deg':>8} | "
        f"{'avg|dev|λ=-3':>13} | {'max|dev|λ=-3':>13} | "
        f"{'bdry frac λ=-2':>15}")
    log(f"  {'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-"
        f"{'-'*8}-+-{'-'*8}-+-"
        f"{'-'*13}-+-{'-'*13}-+-{'-'*15}")
    for L in (2, 3, 4):
        r = results[L]
        avg_s = f"{r['avg_dev_neg3']:.6f}" if r['avg_dev_neg3'] is not None else "  N/A"
        max_s = f"{r['max_dev_neg3']:.6f}" if r['max_dev_neg3'] is not None else "  N/A"
        bdf_s = f"{r['boundary_frac_neg2']:.6f}" if r['boundary_frac_neg2'] is not None else "  N/A"
        log(f"  {L:>4} | {r['n_bcs']:>6} | {r['n_simplices']:>7} | "
            f"{r['n_faces']:>8} | {r['lam_neg3_deg']:>8} | {r['lam_neg2_deg']:>8} | "
            f"{avg_s:>13} | {max_s:>13} | {bdf_s:>15}")

    # -------------------------------------------------------------------------
    # O(1/L²) scaling check for λ=-3 deviation
    # -------------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("O(1/L²) SCALING CHECK FOR λ=-3 PROJECTION DEVIATION")
    log("=" * 78)
    log("  If deviation ~ C/L², then deviation(L) * L² should be approximately constant.")
    log("  Ratios: dev(L=2)/dev(L=3) should be ≈ (3/2)² = 2.25,")
    log("          dev(L=2)/dev(L=4) should be ≈ (4/2)² = 4.00,")
    log("          dev(L=3)/dev(L=4) should be ≈ (4/3)² ≈ 1.78.")
    log("")

    devs = {L: results[L]['avg_dev_neg3'] for L in (2, 3, 4)}

    log(f"  {'L':>4} | {'avg|dev|':>12} | {'dev × L²':>12} | "
        f"{'expected dev × L² (from L=2)':>30}")
    ref_L2 = devs[2]
    if ref_L2 is not None and ref_L2 > 0:
        log(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*30}")
        for L in (2, 3, 4):
            d = devs[L]
            if d is not None:
                dl2   = d * L**2
                exp   = ref_L2 * 4  # C = dev(2) * 4, so expected = C / L²
                exp_v = exp / L**2
                log(f"  {L:>4} | {d:12.6f} | {dl2:12.6f} | {exp_v:30.6f}")
    else:
        log("  (λ=-3 not found for L=2, cannot compute ratios)")

    log("")
    log("  Deviation ratios between L values:")
    for La, Lb in [(2, 3), (2, 4), (3, 4)]:
        da, db = devs[La], devs[Lb]
        if da is not None and db is not None and db > 0:
            ratio    = da / db
            expected = (Lb / La)**2
            log(f"    dev(L={La})/dev(L={Lb}) = {ratio:.4f}  "
                f"(O(1/L²) prediction: ({Lb}/{La})² = {expected:.4f}, "
                f"relative error = {abs(ratio - expected)/expected*100:.1f}%)")

    log("")
    log("  Max deviation ratios:")
    max_devs = {L: results[L]['max_dev_neg3'] for L in (2, 3, 4)}
    for La, Lb in [(2, 3), (2, 4), (3, 4)]:
        da, db = max_devs[La], max_devs[Lb]
        if da is not None and db is not None and db > 0:
            ratio    = da / db
            expected = (Lb / La)**2
            log(f"    max_dev(L={La})/max_dev(L={Lb}) = {ratio:.4f}  "
                f"(O(1/L²) prediction: {expected:.4f})")

    # -------------------------------------------------------------------------
    # λ=-2 boundary confinement scaling
    # -------------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("λ=-2 BOUNDARY CONFINEMENT SCALING")
    log("=" * 78)
    log("  Boundary weight fraction for λ=-2 eigenspace:")
    for L in (2, 3, 4):
        r = results[L]
        f = r['boundary_frac_neg2']
        n_b = sum(1 for ijk in [bc_grid_index(bc) for bc in build_bcc_lattice(L)[0]]
                  if any(c == 0 or c == L - 1 for c in ijk))
        n_tot = r['n_bcs']
        surf_frac = n_b / n_tot
        fs = f"{f:.6f}" if f is not None else "N/A"
        log(f"  L={L}: boundary_frac={fs}  "
            f"(surface BCs = {n_b}/{n_tot} = {surf_frac:.4f})")

    # -------------------------------------------------------------------------
    # λ=-3 degeneracy scaling (predicted: related to BC-BC edge count)
    # -------------------------------------------------------------------------
    log("")
    log("=" * 78)
    log("λ=-3 DEGENERACY SCALING")
    log("=" * 78)
    log("  L³ × degeneracy per BC (should show L scaling pattern):")
    for L in (2, 3, 4):
        r = results[L]
        deg = r['lam_neg3_deg']
        n   = r['n_bcs']
        log(f"  L={L}: n_bcs={n}, λ=-3 deg={deg}, deg/n_bcs={deg/n:.4f}  "
            f"(× L³ = {deg/n * L**3:.2f})")

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
