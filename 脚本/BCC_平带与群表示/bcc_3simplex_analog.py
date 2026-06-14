"""
bcc_3simplex_analog.py

Builds a 3-simplex (tetrahedron) network on the BCC 3x3x3 lattice and
computes the edge-adjacency spectrum (edges of 3-simplices as "faces"),
to compare with the 4-simplex (d=4) case.

Adams hypothesis: In the 4-simplex BCC network, the lambda=-3 eigenspace
projects onto Corner BCs with a fixed deviation of +0.0138 from their
graph degree. This script tests whether the analogous deviation for d=3
vanishes (Adams dimensions: d=1,3,7 where division algebras R,C,H,O exist).

Construction:
  - BCC 3x3x3: 27 body centers at (i+.5, j+.5, k+.5)
  - Each BC: 8 corners split into 2 tetrahedra by parity (stella octangula)
  - Each tetrahedron has C(4,3)=4 triangular faces
  - A 3-simplex = {BC, 3 corners of one triangular face of a tet}
  - 27 BCs x 2 tets x 4 faces = 216 three-simplices
  - Each 3-simplex has 4 vertices -> C(4,2)=6 edges
  - Two edges are adjacent iff they share exactly 1 vertex
  - Total edge-instances: 216 x 6 = 1296

Also runs a PERIODIC BC version: corners mod 3.

Only numpy and scipy are used.
"""

import itertools
import sys
from collections import defaultdict

import numpy as np
import scipy  # noqa: F401

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_3simplex_results.txt"

_LOG_LINES = []


def log(msg=""):
    """Print to stdout AND buffer for the results file."""
    print(msg)
    _LOG_LINES.append(str(msg))


# ---------------------------------------------------------------------------
# Step 1: Build the 3x3x3 BCC lattice
# ---------------------------------------------------------------------------
def build_bcc_lattice():
    """
    Body-center atoms at (i+0.5, j+0.5, k+0.5) for i,j,k in {0,1,2}.
    Each BC's 8 nearest-neighbor corners: (i+dx, j+dy, k+dz) with dx,dy,dz in {0,1}.
    """
    body_centers = []
    cube_corners = {}
    for i, j, k in itertools.product(range(3), repeat=3):
        bc = (i + 0.5, j + 0.5, k + 0.5)
        body_centers.append(bc)
        corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            corners.append((i + dx, j + dy, k + dz))
        cube_corners[bc] = tuple(corners)
    return body_centers, cube_corners


# ---------------------------------------------------------------------------
# Step 2: Enumerate the 216 three-simplices
# ---------------------------------------------------------------------------
def build_3simplices(body_centers, cube_corners):
    """
    Each cube: 8 corners split by parity of (x+y+z) into 2 tetrahedra (stella octangula).
    Each tetrahedron has 4 corners -> C(4,3)=4 triangular faces.
    A 3-simplex = {BC} union {3 corners of one triangular face}.
    So: 27 BCs x 2 tets x 4 tri-faces = 216 three-simplices.

    Returns:
        simplices: list of 216 frozensets, each containing 4 vertex tuples
        simplex_owner_bc_idx: list of 216 ints (which BC owns each simplex)
        simplex_parent_tet_idx: list of 216 ints (0=even-tet, 1=odd-tet, within BC)
    """
    simplices = []
    simplex_owner_bc_idx = []
    simplex_parent_tet_idx = []

    for bc_idx, bc in enumerate(body_centers):
        corners = cube_corners[bc]
        even_tet = [c for c in corners if (c[0] + c[1] + c[2]) % 2 == 0]
        odd_tet  = [c for c in corners if (c[0] + c[1] + c[2]) % 2 == 1]
        assert len(even_tet) == 4 and len(odd_tet) == 4

        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))

        for tet_idx, tet in enumerate((even_tet, odd_tet)):
            tet_f = [(float(c[0]), float(c[1]), float(c[2])) for c in tet]
            # C(4,3)=4 triangular faces of this tetrahedron
            for tri in itertools.combinations(tet_f, 3):
                simplex = frozenset([bc_f] + list(tri))
                assert len(simplex) == 4, "3-simplex must have 4 distinct vertices"
                simplices.append(simplex)
                simplex_owner_bc_idx.append(bc_idx)
                simplex_parent_tet_idx.append(tet_idx)

    return simplices, simplex_owner_bc_idx, simplex_parent_tet_idx


# ---------------------------------------------------------------------------
# Step 3: Extract edges from 3-simplices
# ---------------------------------------------------------------------------
def build_edges(simplices):
    """
    Each 3-simplex (4 vertices) has C(4,2)=6 edges (1-faces = pairs of vertices).
    216 simplices x 6 = 1296 edge-instances.

    Track which simplices share each edge (for verification).

    Returns:
        edges: list of 1296 frozensets of 2 vertex tuples
        edge_owner_simplex_idx: list of 1296 ints
        n_total: total instances
        n_unique: unique distinct edges
        edge_sharing: dict edge_frozenset -> list of simplex indices (for edges shared by >1 simplex)
    """
    edge_to_simplices = defaultdict(list)
    edges = []
    edge_owner_simplex_idx = []

    for s_idx, simplex in enumerate(simplices):
        verts = sorted(simplex)
        for pair in itertools.combinations(verts, 2):
            edge = frozenset(pair)
            edges.append(edge)
            edge_to_simplices[edge].append(s_idx)
            edge_owner_simplex_idx.append(s_idx)

    n_total = len(edges)
    n_unique = len(edge_to_simplices)

    shared = {e: owners for e, owners in edge_to_simplices.items() if len(owners) > 1}

    return edges, edge_owner_simplex_idx, n_total, n_unique, edge_to_simplices, shared


# ---------------------------------------------------------------------------
# Step 4: Build the edge-adjacency matrix
# Two edges adjacent iff they share exactly 1 vertex (share an endpoint)
# ---------------------------------------------------------------------------
def build_edge_adjacency(edges):
    """
    Two edges (pairs of vertices) are adjacent iff they share exactly 1 vertex.
    This is the line graph of the simplex vertex-edge incidence.

    Returns:
        A: (N, N) numpy array, unsigned adjacency (entries in {0,1}, zero diagonal)
        n_adj_pairs: number of adjacent edge-pairs
    """
    n = len(edges)
    ordered_edges = [tuple(sorted(e)) for e in edges]

    # vertex -> set of edge indices
    vertex_to_edges = defaultdict(set)
    for idx, e in enumerate(ordered_edges):
        for v in e:
            vertex_to_edges[v].add(idx)

    A = np.zeros((n, n), dtype=float)
    edge_sets = [frozenset(e) for e in ordered_edges]

    candidate_pairs = set()
    for v, eset in vertex_to_edges.items():
        elist = sorted(eset)
        for a_pos in range(len(elist)):
            for b_pos in range(a_pos + 1, len(elist)):
                candidate_pairs.add((elist[a_pos], elist[b_pos]))

    n_adjacent_pairs = 0
    for i, j in candidate_pairs:
        shared = len(edge_sets[i] & edge_sets[j])
        if shared == 1:
            A[i, j] = 1.0
            A[j, i] = 1.0
            n_adjacent_pairs += 1

    return A, n_adjacent_pairs


# ---------------------------------------------------------------------------
# Step 5: Spectrum grouping helper
# ---------------------------------------------------------------------------
def group_eigenvalues(eigvals, tol=1e-6):
    """Group sorted eigenvalues within `tol` of each other."""
    groups = []
    cur_vals = [eigvals[0]]
    for ev in eigvals[1:]:
        if abs(ev - cur_vals[-1]) <= tol:
            cur_vals.append(ev)
        else:
            groups.append(cur_vals)
            cur_vals = [ev]
    groups.append(cur_vals)

    out = []
    cumulative = 0
    for g in groups:
        rep = float(np.mean(g))
        deg = len(g)
        cumulative += deg
        out.append((rep, deg, cumulative))
    return out


# ---------------------------------------------------------------------------
# BC classification helpers
# ---------------------------------------------------------------------------
def bc_grid_index(bc):
    """Recover integer grid coords (i,j,k) in {0,1,2}^3 from bc=(i+.5,j+.5,k+.5)."""
    return tuple(int(round(c - 0.5)) for c in bc)


def bc_shell(ijk):
    """
    Classify BC by shell in 3x3x3 grid:
      Corner (0 coords ==1): 8 BCs, graph-degree 3
      Edge   (1 coord  ==1): 12 BCs, graph-degree 4
      Face   (2 coords ==1): 6 BCs,  graph-degree 5
      Center (3 coords ==1): 1 BC,   graph-degree 6
    """
    n_mid = sum(1 for c in ijk if c == 1)
    return {
        0: ("Corner", 3),
        1: ("Edge",   4),
        2: ("Face",   5),
        3: ("Center", 6),
    }[n_mid]


def bc_graph_degree(ijk):
    """Degree in the 3x3x3 BC grid graph (face-adjacent neighbors)."""
    deg = 0
    for axis in range(3):
        for delta in (-1, 1):
            nb = list(ijk)
            nb[axis] += delta
            if 0 <= nb[axis] <= 2:
                deg += 1
    return deg


# ---------------------------------------------------------------------------
# OPEN BC main analysis
# ---------------------------------------------------------------------------
def run_open_bc():
    log("=" * 78)
    log("BCC 3x3x3 lattice -> 3-simplex network -> edge-adjacency spectrum (OPEN BC)")
    log("=" * 78)

    # --- Step 1 ---
    body_centers, cube_corners = build_bcc_lattice()
    log(f"\n[Step 1] Body-center atoms: {len(body_centers)} (expect 27)")

    # --- Step 2 ---
    simplices, simplex_owner_bc_idx, simplex_parent_tet_idx = build_3simplices(body_centers, cube_corners)
    log(f"[Step 2] 3-simplices enumerated: {len(simplices)} (expect 216)")
    log(f"         27 BCs x 2 tets x C(4,3)=4 tri-faces = 216")

    # --- Step 3 ---
    edges, edge_owner_simplex_idx, n_total, n_unique, edge_to_simplices, shared = build_edges(simplices)
    log(f"\n[Step 3] Edge instances: {n_total} (expect 1296 = 216 x 6)")
    log(f"         Unique edges:    {n_unique}")
    if shared:
        log(f"         Edges shared by multiple simplices: {len(shared)}")
        for e, owners in list(shared.items())[:5]:
            log(f"           {set(e)} -> simplices {owners}")
    else:
        log(f"         All {n_unique} edges are distinct (no sharing between simplices)")

    # --- Step 4 ---
    log(f"\n[Step 4] Building {n_total}x{n_total} edge-adjacency matrix ...")
    log(f"         Two edges adjacent iff they share exactly 1 vertex")
    A, n_adj_pairs = build_edge_adjacency(edges)
    log(f"         Shape: {A.shape}")
    log(f"         Adjacent edge-pairs: {n_adj_pairs}")
    log(f"         A symmetric: {np.allclose(A, A.T)}")
    log(f"         trace(A): {np.trace(A):.1f}")

    # --- Step 5: eigendecomposition ---
    log(f"\n[Step 5] Diagonalizing A ({n_total}x{n_total}) with numpy.linalg.eigh ...")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    grouped = group_eigenvalues(eigvals, tol=1e-6)
    log(f"\n  Full eigenvalue table (grouped within tol=1e-6):")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for rep, deg, cum in grouped:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")
    deg_sum = sum(g[1] for g in grouped)
    log(f"\n  degeneracy sum = {deg_sum}  (expect {n_total}) -> {'OK' if deg_sum == n_total else 'MISMATCH'}")

    # Identify integer eigenvalues
    log(f"\n  Integer (or near-integer) eigenvalues:")
    for rep, deg, cum in grouped:
        if abs(rep - round(rep)) < 1e-4:
            log(f"    lambda = {rep:+.1f}  (deg = {deg})")

    # --- Step 6: largest-degeneracy eigenvalue analysis ---
    max_deg = max(g[1] for g in grouped)
    largest_deg_groups = [(rep, deg, cum) for rep, deg, cum in grouped if deg == max_deg]
    log(f"\n[Step 6] Largest degeneracy eigenvalue analysis")
    log(f"  Largest degeneracy: {max_deg}")
    for rep, deg, cum in largest_deg_groups:
        log(f"  lambda = {rep:.6f}  (deg = {deg})")

    # Use the most-negative large-degeneracy eigenvalue for BC projection
    # (analogous to lambda=-3 in d=4 case)
    log(f"\n  Selecting eigenvalue for BC projection analysis:")
    log(f"  (In d=4, lambda=-3 with m=108 was used. We select the most-negative")
    log(f"  eigenvalue with the largest degeneracy.)")

    # Find most negative eigenvalue with largest degeneracy
    most_neg_large = None
    for rep, deg, cum in grouped:
        if deg == max_deg:
            if most_neg_large is None or rep < most_neg_large[0]:
                most_neg_large = (rep, deg, cum)

    # Also look for integer eigenvalues with high degeneracy
    log(f"\n  All groups sorted by degeneracy (descending):")
    sorted_groups = sorted(grouped, key=lambda x: -x[1])
    for rep, deg, cum in sorted_groups[:10]:
        log(f"    lambda={rep:10.6f}  deg={deg}")

    # --- Step 7: BC eigenspace projection ---
    log(f"\n[Step 7] Body-center eigenspace projection analysis")
    log(f"  Per-BC edge-patch: each BC owns 2 tets x 4 tri-faces x 6 edges = 48 edges")
    log(f"  But edges may be shared between simplices! Checking...")

    # Map edge -> owner BC (via edge -> owner simplex -> owner BC)
    edge_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[edge_owner_simplex_idx[f]] for f in range(n_total)],
        dtype=int,
    )
    bc_edge_idxs = [np.where(edge_owner_bc_idx == bc_i)[0] for bc_i in range(len(body_centers))]
    bc_patch_sizes = [len(idxs) for idxs in bc_edge_idxs]
    log(f"  Per-BC edge-patch sizes: min={min(bc_patch_sizes)}, max={max(bc_patch_sizes)}")
    unique_sizes = set(bc_patch_sizes)
    log(f"  Unique patch sizes: {unique_sizes}")
    log(f"  Note: if edges are shared between simplices, some BCs may have overlapping")
    log(f"  edge-patches in the owner-simplex mapping (each edge counted once per owner simplex)")

    # Classify BCs
    bc_grid = [bc_grid_index(bc) for bc in body_centers]
    bc_shell_info = [bc_shell(ijk) for ijk in bc_grid]
    bc_graph_deg = [bc_graph_degree(ijk) for ijk in bc_grid]
    deg_match = all(s[1] == g for s, g in zip(bc_shell_info, bc_graph_deg))
    log(f"  Shell-nominal-degree vs grid-graph-degree cross-check: match for all 27 BCs? {deg_match}")

    shell_order = ["Corner", "Edge", "Face", "Center"]
    shell_members = defaultdict(list)
    for bc_i, (name, _deg) in enumerate(bc_shell_info):
        shell_members[name].append(bc_i)
    log("  Shell membership (expect Corner=8, Edge=12, Face=6, Center=1):")
    for name in shell_order:
        members = shell_members[name]
        log(f"    {name:7s}: count={len(members):2d}, graph-degree={bc_shell_info[members[0]][1]}")

    # Project eigenspaces onto BC patches for the top-5 by degeneracy
    log(f"\n  Eigenspace projection for top groups by degeneracy:")

    corner_deviations = {}  # lambda -> deviation from graph-degree for Corner shell

    for rep, deg, cum in sorted_groups[:8]:
        member_mask = np.abs(eigvals - rep) <= 1e-4
        member_idx = np.where(member_mask)[0]
        m_actual = len(member_idx)
        if m_actual == 0:
            continue

        vecs = eigvecs[:, member_idx]
        sq = np.sum(vecs ** 2, axis=1)

        # Per-shell average projection
        shell_avgs = {}
        for shell_name in shell_order:
            members = shell_members[shell_name]
            projs = [float(np.sum(sq[bc_edge_idxs[bc_i]])) for bc_i in members]
            shell_avgs[shell_name] = float(np.mean(projs))

        corner_avg = shell_avgs["Corner"]
        corner_gdeg = bc_shell_info[shell_members["Corner"][0]][1]  # = 3
        deviation = corner_avg - corner_gdeg
        corner_deviations[rep] = deviation

        log(f"\n  lambda={rep:+.4f}  (deg={m_actual}):")
        for shell_name in shell_order:
            gdeg = bc_shell_info[shell_members[shell_name][0]][1]
            avg = shell_avgs[shell_name]
            dev = avg - gdeg
            log(f"    {shell_name:7s} (graph-deg={gdeg}): avg_proj={avg:.6f}  deviation={dev:+.6f}")

    # KEY COMPARISON
    log(f"\n{'='*78}")
    log(f"KEY COMPARISON: d=3 (3-simplex) vs d=4 (4-simplex)")
    log(f"{'='*78}")
    log(f"  d=4 result: lambda=-3 eigenspace -> Corner deviation = +0.0138")
    log(f"  (deviation = avg_projection - graph_degree for Corner shell BCs)")
    log(f"")
    log(f"  d=3 results (all groups, Corner deviation):")
    for lam, dev in sorted(corner_deviations.items()):
        flag = "  <-- ZERO (Adams dim?)" if abs(dev) < 0.005 else ""
        log(f"    lambda={lam:+.4f}: Corner deviation = {dev:+.8f}{flag}")

    return eigvals, eigvecs, A, body_centers, bc_edge_idxs, bc_shell_info, shell_members, shell_order, grouped


# ---------------------------------------------------------------------------
# PERIODIC BC version
# ---------------------------------------------------------------------------
def build_bcc_lattice_periodic():
    """Build BCC lattice with corners mod 3."""
    body_centers = []
    cube_corners_periodic = {}
    for i, j, k in itertools.product(range(3), repeat=3):
        bc = (i + 0.5, j + 0.5, k + 0.5)
        body_centers.append(bc)
        corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            cx = (i + dx) % 3
            cy = (j + dy) % 3
            cz = (k + dz) % 3
            corners.append((cx, cy, cz))
        cube_corners_periodic[bc] = tuple(corners)
    return body_centers, cube_corners_periodic


def build_3simplices_periodic(body_centers, cube_corners_periodic):
    """
    Same as build_3simplices but using wrapped (mod-3) corner coordinates.
    Parity split must use the ORIGINAL (pre-mod-3) coordinates.
    """
    simplices = []
    simplex_owner_bc_idx = []

    for bc_idx, bc in enumerate(body_centers):
        i, j, k = int(bc[0] - 0.5), int(bc[1] - 0.5), int(bc[2] - 0.5)
        corners_periodic = cube_corners_periodic[bc]  # mod-3 coords
        # Original corners for parity splitting
        corners_orig = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            corners_orig.append((i + dx, j + dy, k + dz))
        # Pair original coords with periodic coords
        orig_to_periodic = {}
        for corig, cper in zip(corners_orig, corners_periodic):
            orig_to_periodic[corig] = cper

        even_tet_orig = [c for c in corners_orig if (c[0] + c[1] + c[2]) % 2 == 0]
        odd_tet_orig  = [c for c in corners_orig if (c[0] + c[1] + c[2]) % 2 == 1]

        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))

        for tet_orig in (even_tet_orig, odd_tet_orig):
            # Convert tet corners to periodic (mod-3) float tuples for vertex identity
            tet_per = [(float(orig_to_periodic[c][0]),
                        float(orig_to_periodic[c][1]),
                        float(orig_to_periodic[c][2])) for c in tet_orig]
            for tri in itertools.combinations(tet_per, 3):
                simplex = frozenset([bc_f] + list(tri))
                if len(simplex) < 4:
                    # Vertex collision from wrapping - still valid, just degenerate
                    pass
                simplices.append(simplex)
                simplex_owner_bc_idx.append(bc_idx)

    return simplices, simplex_owner_bc_idx


def run_periodic_bc():
    log("\n" + "=" * 78)
    log("BCC 3x3x3 lattice -> 3-simplex network -> edge-adjacency spectrum (PERIODIC BC)")
    log("=" * 78)

    body_centers, cube_corners_periodic = build_bcc_lattice_periodic()

    simplices, simplex_owner_bc_idx = build_3simplices_periodic(body_centers, cube_corners_periodic)
    log(f"\n[P-Step 2] 3-simplices (periodic): {len(simplices)} (expect 216)")

    edges, edge_owner_simplex_idx, n_total, n_unique, edge_to_simplices, shared = build_edges(simplices)
    log(f"[P-Step 3] Edge instances: {n_total} (= 216 x 6)")
    log(f"           Unique edges:   {n_unique}")
    if shared:
        log(f"           Edges shared by >1 simplex: {len(shared)}")
    else:
        log(f"           All edges distinct")

    log(f"\n[P-Step 4] Building {n_total}x{n_total} edge-adjacency matrix ...")
    A_per, n_adj_pairs = build_edge_adjacency(edges)
    log(f"           Adjacent edge-pairs: {n_adj_pairs}")

    log(f"\n[P-Step 5] Diagonalizing ({n_total}x{n_total}) ...")
    eigvals_per, eigvecs_per = np.linalg.eigh(A_per)
    order = np.argsort(eigvals_per)
    eigvals_per = eigvals_per[order]

    grouped_per = group_eigenvalues(eigvals_per, tol=1e-6)
    log(f"\n  Full eigenvalue table (periodic BC):")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for rep, deg, cum in grouped_per:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")
    deg_sum = sum(g[1] for g in grouped_per)
    log(f"\n  degeneracy sum = {deg_sum}  (expect {n_total})")

    # Integer eigenvalues
    log(f"\n  Integer eigenvalues (periodic BC):")
    for rep, deg, cum in grouped_per:
        if abs(rep - round(rep)) < 1e-4:
            log(f"    lambda = {rep:+.1f}  (deg = {deg})")

    # Flat bands (eigenvalues with large degeneracy)
    max_deg_per = max(g[1] for g in grouped_per)
    log(f"\n  Largest degeneracy (periodic): {max_deg_per}")
    log(f"  Flat band count (periodic): bands with deg > {n_total//10}:")
    for rep, deg, cum in sorted(grouped_per, key=lambda x: -x[1])[:5]:
        log(f"    lambda={rep:.4f}  deg={deg}")

    # All-BC equality check
    edge_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[edge_owner_simplex_idx[f]] for f in range(n_total)],
        dtype=int,
    )
    bc_edge_idxs_per = [np.where(edge_owner_bc_idx == bc_i)[0] for bc_i in range(len(body_centers))]
    bc_patch_sizes = [len(idxs) for idxs in bc_edge_idxs_per]
    log(f"\n  Per-BC edge-patch sizes (periodic): min={min(bc_patch_sizes)}, max={max(bc_patch_sizes)}")

    # Check if all per-BC local 6x6 blocks are identical
    # (Translation symmetry -> all BCs should be equivalent)
    # Check by comparing projection of largest-deg eigenspace
    sorted_groups_per = sorted(grouped_per, key=lambda x: -x[1])
    rep0, deg0, _ = sorted_groups_per[0]
    member_mask = np.abs(eigvals_per - rep0) <= 1e-4
    member_idx = np.where(member_mask)[0]
    vecs0 = eigvecs_per[:, member_idx]
    sq0 = np.sum(vecs0 ** 2, axis=1)

    bc_projs = [float(np.sum(sq0[bc_edge_idxs_per[bc_i]])) for bc_i in range(len(body_centers))]
    log(f"\n  Largest-deg eigenspace (lambda={rep0:.4f}, deg={deg0}) per-BC projections:")
    log(f"  min={min(bc_projs):.6f}, max={max(bc_projs):.6f}, std={np.std(bc_projs):.6f}")
    all_equal = (max(bc_projs) - min(bc_projs)) < 1e-6
    log(f"  All per-BC projections equal? {all_equal}  (expected True for periodic BC with translation symmetry)")

    return grouped_per


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("BCC 3-SIMPLEX ANALOG: testing Adams dimension hypothesis at d=3")
    log("Adams dimensions: d=1,3,7 (division algebras R,C,H,O)")
    log("Hypothesis: at Adams dimensions, Corner BC deviation vanishes")
    log("")

    # Open BC
    (eigvals, eigvecs, A, body_centers, bc_edge_idxs,
     bc_shell_info, shell_members, shell_order, grouped) = run_open_bc()

    # Periodic BC
    grouped_per = run_periodic_bc()

    # Final summary
    log(f"\n{'='*78}")
    log("FINAL SUMMARY")
    log(f"{'='*78}")
    log("")
    log(f"d=4 (4-simplex) reference:")
    log(f"  - 54 simplices, 540 triangular faces")
    log(f"  - lambda=-3 eigenspace (deg=108): Corner deviation = +0.0138")
    log(f"  - 'deviation' = avg_projection_onto_BC_patch - BC_graph_degree")
    log("")
    log(f"d=3 (3-simplex) this run:")
    n_simplices = 216
    n_edges = n_simplices * 6
    log(f"  - {n_simplices} simplices, {n_edges} edges (= {n_simplices} x 6)")

    # Report spectrum shape
    log(f"  - Eigenvalue spectrum: {len(grouped)} distinct groups")
    log(f"  - Degeneracy spectrum (top 5):")
    sorted_groups = sorted(grouped, key=lambda x: -x[1])
    for rep, deg, cum in sorted_groups[:5]:
        log(f"      lambda={rep:+.4f}  deg={deg}")

    log("")
    log("See above for full per-shell deviation table.")
    log("")
    log("Adams hypothesis verdict for d=3:")
    log("  If Corner deviation is ~0 at d=3 but ~0.0138 at d=4:")
    log("  -> Supports hypothesis (d=3 is Adams dimension, d=4 is not)")
    log("  If both nonzero: -> Refuted or not applicable at d=3")
    log("  If both ~0: -> Uninformative (need more dimensions)")
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
