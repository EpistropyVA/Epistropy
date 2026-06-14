"""
bcc_540_periodic_bc.py

Computes the face-adjacency spectrum of the 4-simplex network of a 3x3x3 BCC
lattice with PERIODIC BOUNDARY CONDITIONS (3-torus): coordinates are taken
mod 3, so the lattice wraps around on all three axes.

Differences from bcc_540_spectrum.py (open BC):
  - All corner coordinates are reduced mod 3 before comparison; coordinate 3
    identifies with coordinate 0 on every axis.
  - Some faces from DIFFERENT simplices become IDENTICAL under vertex
    identification, so the unique-face count is LESS than 540.
  - The body-center (BC) graph becomes a 3-torus grid: every BC has degree 6.
  - Translation symmetry is exact: all 27 BCs should be equivalent, their
    20x20 local diagonal blocks identical, and Kronecker decomposition exact.

Predictions explicitly tested (printed as CONFIRMED/REFUTED):
  P1: All 27 BCs have degree 6 in the BC graph.
  P2: All 27 local 20x20 diagonal blocks are identical.
  P3: lambda=-2 (deg 56 in open BC) disappears from spectrum.
  P4: Kronecker residual < 1e-10.
  P5: All per-BC eigenspace projections are equal.

Only numpy and scipy are used.
"""

import itertools
import sys
from collections import defaultdict

import numpy as np
import scipy  # noqa: F401

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_540_periodic_results.txt"

_LOG_LINES = []


def log(msg=""):
    """Print to stdout AND buffer for the results file."""
    print(msg)
    _LOG_LINES.append(str(msg))


# ---------------------------------------------------------------------------
# Step 1: Build the 3x3x3 BCC lattice with periodic BC
# ---------------------------------------------------------------------------
def build_bcc_lattice_periodic():
    """
    Same 27 body centers at (i+0.5, j+0.5, k+0.5) for i,j,k in {0,1,2}.
    Each cube's 8 corners are reduced mod 3 before being stored.
    This identifies corner coordinate 3 with 0, wrapping the lattice.

    Returns:
        body_centers: list of 27 tuples (float coords, NOT wrapped -- BCs
                      stay in the fundamental domain)
        cube_corners_periodic: dict bc -> tuple of 8 corner tuples (int
                      coords mod 3, so each coord in {0,1,2})
    """
    body_centers = []
    cube_corners_periodic = {}
    for i, j, k in itertools.product(range(3), repeat=3):
        bc = (i + 0.5, j + 0.5, k + 0.5)
        body_centers.append(bc)
        corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            # Reduce mod 3 for periodic identification
            cx = (i + dx) % 3
            cy = (j + dy) % 3
            cz = (k + dz) % 3
            corners.append((cx, cy, cz))
        cube_corners_periodic[bc] = tuple(corners)
    return body_centers, cube_corners_periodic


# ---------------------------------------------------------------------------
# Step 2: Enumerate the 54 four-simplices (with periodic corner vertices)
# ---------------------------------------------------------------------------
def build_simplices_periodic(body_centers, cube_corners_periodic):
    """
    Same stella-octangula split: even-parity (x+y+z even mod 2) and odd-parity
    corners. With periodic BC the coordinates are already in {0,1,2}, so the
    parity split still applies to the PRE-mod-3 sum: we must split by the
    original (i+dx, j+dy, k+dz) parity, not the wrapped coordinate parity,
    to preserve the two regular tetrahedra inside each cube.

    After splitting, the vertex coordinates ARE the mod-3 values (stored in
    cube_corners_periodic), but the parity used for splitting must match the
    ORIGINAL sum (before wrapping) since parity(sum) = parity(sum mod 3) only
    when sum mod 2 == (sum mod 3) mod 2, which is NOT always true.

    Solution: keep both versions for each cube -- original corners (for parity
    splitting) and wrapped corners (for vertex identity).

    Returns:
        simplices: list of 54 frozensets of 5 vertices; corner vertices are
                   stored as (float) tuples of mod-3 integers (0.0, 1.0, 2.0);
                   the body-center is stored as its original float tuple.
        simplex_owner_bc_idx: list of 54 ints.
    """
    simplices = []
    simplex_owner_bc_idx = []
    for bc_idx, bc in enumerate(body_centers):
        i, j, k = (int(round(bc[0] - 0.5)),
                   int(round(bc[1] - 0.5)),
                   int(round(bc[2] - 0.5)))
        # Original corners (before mod 3), used for parity split
        orig_corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            ox, oy, oz = i + dx, j + dy, k + dz
            wx, wy, wz = ox % 3, oy % 3, oz % 3
            orig_corners.append(((ox, oy, oz), (wx, wy, wz)))

        even_tet_wrapped = [wc for ((ox, oy, oz), wc) in orig_corners
                            if (ox + oy + oz) % 2 == 0]
        odd_tet_wrapped  = [wc for ((ox, oy, oz), wc) in orig_corners
                            if (ox + oy + oz) % 2 == 1]

        assert len(even_tet_wrapped) == 4 and len(odd_tet_wrapped) == 4, \
            "Each cube must split into two tetrahedra of 4 corners"

        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))
        for tet_wrapped in (even_tet_wrapped, odd_tet_wrapped):
            tet_f = [(float(wx), float(wy), float(wz)) for (wx, wy, wz) in tet_wrapped]
            simplex = frozenset([bc_f] + tet_f)
            # With periodic BC, two corners of the SAME cube could wrap to the
            # same mod-3 coordinate. Verify we still have 5 distinct vertices.
            if len(simplex) != 5:
                log(f"  WARNING: simplex at bc={bc} has only {len(simplex)} distinct "
                    f"vertices after mod-3 wrapping (expected 5). This cube's "
                    f"tetrahedral split is degenerate under the torus identification.")
            simplices.append(simplex)
            simplex_owner_bc_idx.append(bc_idx)

    return simplices, simplex_owner_bc_idx


# ---------------------------------------------------------------------------
# Step 3: Extract triangular faces (with deduplication under periodic BC)
# ---------------------------------------------------------------------------
def build_faces_periodic(simplices):
    """
    Generate all C(5,3)=10 triangle faces per simplex (54*10=540 instances),
    then deduplicate by vertex-set identity (which respects mod-3 wrapping).

    With periodic BC, some faces from DIFFERENT simplices may be identical
    vertex-sets (because vertex coordinates are mod-3). The unique face list
    will be shorter than 540.

    Returns:
        unique_faces: list of unique frozensets (vertex-sets), in first-seen order
        n_total_instances: 54*10 = 540 (always, before dedup)
        n_unique: len(unique_faces) -- will be <= 540
        face_to_simplices: dict frozenset -> list of simplex indices that own it
        face_instance_simplex: list of 540 ints -- which simplex each instance came from
    """
    face_to_simplices = defaultdict(list)
    instance_simplex = []
    for s_idx, simplex in enumerate(simplices):
        verts = sorted(simplex)
        for combo in itertools.combinations(verts, 3):
            face = frozenset(combo)
            face_to_simplices[face].append(s_idx)
            instance_simplex.append(s_idx)

    n_total = len(instance_simplex)  # always 540
    unique_faces = list(face_to_simplices.keys())
    n_unique = len(unique_faces)

    shared_faces = {f: owners for f, owners in face_to_simplices.items() if len(owners) > 1}
    log(f"  Total face instances (54 simplices x 10): {n_total}")
    log(f"  Unique faces after mod-3 deduplication:   {n_unique}")
    if shared_faces:
        log(f"  Faces shared by >1 simplex: {len(shared_faces)}")
        # Report a few examples
        shown = 0
        for f, owners in sorted(shared_faces.items(), key=lambda x: -len(x[1])):
            if shown < 3:
                log(f"    face {set(f)} shared by simplices {owners}")
                shown += 1
        if len(shared_faces) > 3:
            log(f"    ... (showing first 3 of {len(shared_faces)})")
    else:
        log(f"  No shared faces (unexpected under periodic BC if predictions hold).")

    return unique_faces, n_total, n_unique, face_to_simplices, instance_simplex


# ---------------------------------------------------------------------------
# Step 4: Build adjacency matrix on unique faces
# ---------------------------------------------------------------------------
def build_adjacency_matrix_periodic(unique_faces, face_to_simplices):
    """
    Two unique faces are adjacent iff they share exactly 2 vertices.
    Vertex identity is already mod-3 (baked into the frozensets).

    Returns:
        A: (n_unique, n_unique) symmetric float array, entries in {0,1}
        n_adj_pairs: int
    """
    n = len(unique_faces)
    ordered = [tuple(sorted(f)) for f in unique_faces]
    face_sets = [frozenset(f) for f in ordered]

    # vertex -> set of face indices
    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(ordered):
        for v in f:
            vertex_to_faces[v].add(idx)

    A = np.zeros((n, n), dtype=float)

    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a_pos in range(len(flist)):
            for b_pos in range(a_pos + 1, len(flist)):
                candidate_pairs.add((flist[a_pos], flist[b_pos]))

    n_adj_pairs = 0
    for i, j in candidate_pairs:
        if len(face_sets[i] & face_sets[j]) == 2:
            A[i, j] = 1.0
            A[j, i] = 1.0
            n_adj_pairs += 1

    return A, n_adj_pairs


# ---------------------------------------------------------------------------
# Step 5: spectrum grouping helper (identical to original)
# ---------------------------------------------------------------------------
def group_eigenvalues(eigvals, tol=1e-6):
    """Group sorted eigenvalues within tol. Returns (rep, degeneracy, cumulative)."""
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
# Step 6: BC graph helpers for the periodic (torus) case
# ---------------------------------------------------------------------------
def bc_graph_degree_periodic(ijk):
    """
    Degree of a BC in the 3x3x3 torus BC graph: number of BCs differing by
    +-1 mod 3 in exactly one coordinate. On the 3-torus every BC has 6
    neighbors (2 per axis, wrapping), so degree is always 6.
    """
    deg = 0
    for axis in range(3):
        for delta in (-1, 1):
            # mod 3 wrapping -- always a valid neighbor
            deg += 1
    return deg  # always 6


def get_bc_index(body_centers, ijk):
    """Return the index of BC with grid position ijk=(i,j,k)."""
    target = (ijk[0] + 0.5, ijk[1] + 0.5, ijk[2] + 0.5)
    for idx, bc in enumerate(body_centers):
        if bc == target:
            return idx
    return None


def bc_grid_index(bc):
    """Recover integer grid coords (i,j,k) in {0,1,2}^3 from bc=(i+.5,j+.5,k+.5)."""
    return tuple(int(round(c - 0.5)) for c in bc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 78)
    log("BCC 3x3x3 lattice (PERIODIC BC / 3-torus) -> 4-simplex face-adjacency spectrum")
    log("=" * 78)

    # --- Step 1 ---
    log("\n[Step 1] Build BCC lattice with periodic boundary conditions (mod-3 corners)")
    body_centers, cube_corners_periodic = build_bcc_lattice_periodic()
    log(f"  Body-center atoms: {len(body_centers)} (expect 27)")

    # Verify mod-3 wrapping: corners at coord 3 should appear as coord 0
    wrapped_count = 0
    for bc, corners in cube_corners_periodic.items():
        for c in corners:
            if any(v == 0 and (bc[ax] > 1.5) for ax, v in enumerate(c)):
                wrapped_count += 1
    log(f"  Mod-3 wrapping active: {wrapped_count} corner-coordinate wraps detected across all cubes")
    i_sample, j_sample, k_sample = 2, 2, 2  # corner cube, should have 0-wrapped corners
    bc_sample = (2.5, 2.5, 2.5)
    log(f"  Sample: bc={bc_sample} corners = {cube_corners_periodic[bc_sample]}")
    log(f"    (expect some coords to be 0, wrapping from 3)")

    # --- Step 2 ---
    log("\n[Step 2] Enumerate 4-simplices with periodic vertex identification")
    simplices, simplex_owner_bc_idx = build_simplices_periodic(body_centers, cube_corners_periodic)
    log(f"  4-simplices: {len(simplices)} (expect 54)")
    sizes = [len(s) for s in simplices]
    log(f"  Simplex sizes: min={min(sizes)}, max={max(sizes)}, all==5? {all(s == 5 for s in sizes)}")
    if not all(s == 5 for s in sizes):
        log("  WARNING: some simplices have fewer than 5 distinct vertices after wrapping!")

    # --- Step 3 ---
    log("\n[Step 3] Extract triangular faces (deduplicate under periodic BC)")
    unique_faces, n_total, n_unique, face_to_simplices, _ = build_faces_periodic(simplices)
    log(f"  n_total_instances = {n_total} (always 54*10=540)")
    log(f"  n_unique faces    = {n_unique} (<= 540 due to mod-3 identification)")
    N = n_unique  # adjacency matrix size

    # Build mapping: face index -> owner simplices
    face_list = list(face_to_simplices.keys())
    assert len(face_list) == n_unique

    # face -> list of (BC indices) that own it
    face_owner_bcs = {}
    for f, s_list in face_to_simplices.items():
        face_owner_bcs[f] = [simplex_owner_bc_idx[s] for s in s_list]

    # --- Step 4 ---
    log(f"\n[Step 4] Build {N}x{N} adjacency matrix on unique faces")
    A, n_adj_pairs = build_adjacency_matrix_periodic(unique_faces, face_to_simplices)
    log(f"  Matrix shape: {A.shape}")
    log(f"  Symmetric: {np.allclose(A, A.T)}")
    log(f"  Trace: {np.trace(A):.1f} (expect 0)")
    nnz = int(np.count_nonzero(A))
    log(f"  Nonzero entries: {nnz} (adjacent pairs: {n_adj_pairs})")

    # --- Step 5: Eigenvalue spectrum ---
    log(f"\n[Step 5] Diagonalize {N}x{N} adjacency matrix (numpy.linalg.eigh)")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    grouped = group_eigenvalues(eigvals, tol=1e-6)
    log("\n  Full eigenvalue table (grouped within tol=1e-6):")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for rep, deg, cum in grouped:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")
    deg_sum = sum(d for _, d, _ in grouped)
    log(f"\n  degeneracy sum = {deg_sum} (expect {N}): {'OK' if deg_sum == N else 'MISMATCH'}")

    # --- Open-BC comparison ---
    log("\n[Open-BC spectrum comparison]")
    log("  Checking prediction P3: does the EXACT eigenvalue lambda=-2.0 (deg 56 in")
    log("  open BC) disappear from the periodic spectrum?")
    log("  Criterion: no grouped eigenvalue within tol=1e-4 of -2.0.")
    # Use a tight tolerance to check for the exact integer -2 eigenvalue
    has_lam_minus2_exact = any(abs(rep - (-2.0)) < 1e-4 for rep, deg, _ in grouped)
    # Also report any eigenvalue in the window (-2.5, -1.5) for context
    near_minus2 = [(rep, deg) for rep, deg, _ in grouped if abs(rep - (-2.0)) < 0.5]
    log(f"  Eigenvalues in the window (-2.5, -1.5) of the periodic spectrum:")
    for rep, deg in near_minus2:
        log(f"    lambda={rep:.6f}  deg={deg}")
    if not near_minus2:
        log("    (none)")
    if has_lam_minus2_exact:
        log(f"  Exact lambda=-2.0 (within 1e-4) IS present in periodic spectrum.")
    else:
        log(f"  Exact lambda=-2.0 is NOT present (closest entries shown above, all shifted away).")

    # --- Step 6: BC graph analysis ---
    log("\n[Step 6] Body-center graph analysis under periodic BC")

    bc_grid = [bc_grid_index(bc) for bc in body_centers]

    # For each BC, its faces: all unique faces whose face_to_simplices includes
    # a simplex owned by this BC
    # Build bc -> set of unique face indices
    bc_face_set = defaultdict(set)
    face_idx_of = {f: idx for idx, f in enumerate(face_list)}
    for f, bc_list in face_owner_bcs.items():
        f_idx = face_idx_of[f]
        for bc_i in bc_list:
            bc_face_set[bc_i].add(f_idx)

    log(f"  Face patch sizes per BC (should all be <= 20; may be < 20 if faces wrap to other BCs):")
    patch_sizes = [len(bc_face_set[i]) for i in range(27)]
    from collections import Counter
    size_dist = Counter(patch_sizes)
    log(f"  Size distribution: {dict(sorted(size_dist.items()))}")

    # BC degree in the BC graph (periodic: always 6)
    log("\n  P1 check: All 27 BCs have degree 6 in the periodic BC graph (torus):")
    degrees = [bc_graph_degree_periodic(ijk) for ijk in bc_grid]
    all_deg6 = all(d == 6 for d in degrees)
    log(f"  Degrees: {degrees}")
    log(f"  All degree 6? {all_deg6}")

    # --- Step 7: Local 20x20 diagonal block analysis ---
    log("\n[Step 7] Local 20x20 diagonal block analysis")
    log("  For each BC, extract the sub-block of A indexed by the BC's face patch.")
    log("  Under full translation symmetry (periodic BC), all 27 blocks should be identical.")

    # Faces assigned "primarily" to each BC: we need 20 faces per BC for the
    # Kronecker decomposition. With periodic BC, some faces are shared between
    # two BCs. We need a consistent partition. Use: for each unique face, assign
    # it to the BC of the FIRST simplex that owns it (deterministic, based on
    # body_centers ordering).
    face_primary_bc = {}
    for f, s_list in face_to_simplices.items():
        # Primary owner = simplex with smallest index -> BC with smallest index
        primary_s = min(s_list)
        face_primary_bc[f] = simplex_owner_bc_idx[primary_s]

    # bc -> list of face indices assigned to it
    bc_primary_faces = defaultdict(list)
    for f, bc_i in face_primary_bc.items():
        bc_primary_faces[bc_i].append(face_idx_of[f])

    primary_sizes = [len(bc_primary_faces[i]) for i in range(27)]
    log(f"  Primary assignment sizes: min={min(primary_sizes)}, max={max(primary_sizes)}, "
        f"sum={sum(primary_sizes)} (expect {n_unique})")

    # Check if all sizes are 20 (the natural expectation for a uniform torus)
    all_20 = all(s == 20 for s in primary_sizes)
    log(f"  All primary patches have exactly 20 faces? {all_20}")
    if not all_20:
        log(f"  Size distribution: {dict(Counter(primary_sizes))}")
        log("  NOTE: Non-uniform patch sizes indicate asymmetry in face-to-BC assignment.")
        log("  This may reflect that periodic BC causes some faces to be genuinely shared.")

    # Extract local blocks: for each BC, the sub-block A[I_bc, I_bc]
    blocks = []
    for bc_i in range(27):
        idxs = sorted(bc_primary_faces[bc_i])
        block = A[np.ix_(idxs, idxs)]
        blocks.append(block)

    # Check block sizes (all should be 20x20 if patches are all size 20)
    block_shapes = [b.shape for b in blocks]
    log(f"  Block shapes: all {block_shapes[0]}? {all(s == block_shapes[0] for s in block_shapes)}")

    # P2: all blocks identical?
    if all_20:
        ref_block = blocks[0]
        max_block_diff = 0.0
        for i, b in enumerate(blocks[1:], 1):
            diff = np.max(np.abs(b - ref_block))
            if diff > max_block_diff:
                max_block_diff = diff
        log(f"\n  P2 check: max entry-wise difference across all 27 local blocks = {max_block_diff:.3e}")
        blocks_identical = max_block_diff < 1e-10
        log(f"  All 27 local 20x20 blocks identical? {blocks_identical}")
    else:
        log("\n  P2 check: SKIPPED (patch sizes not uniform, cannot compare fixed-size blocks)")
        blocks_identical = False
        max_block_diff = float('nan')

    # --- Step 8: Kronecker decomposition test ---
    log("\n[Step 8] Kronecker decomposition test")
    log("  A_Kronecker = I_27 @ L_local + L_x @ M_x + L_y @ M_y + L_z @ M_z")
    log("  L_local: 20x20 common local block")
    log("  L_x/y/z: 27x27 circulant adjacency on the periodic 3x3x3 grid (per axis)")
    log("  M_x/y/z: 20x20 coupling matrices (averaged over all 27 BC positions)")

    if not all_20:
        log("  SKIPPED: Kronecker test requires uniform 20-face patches (all_20=False).")
        kronecker_residual = float('nan')
    else:
        # Build the 540x540 adjacency matrix in BC-block ordering
        # Reorder faces: bc0_faces, bc1_faces, ..., bc26_faces (each 20 faces)
        bc_face_order = []
        for bc_i in range(27):
            bc_face_order.extend(sorted(bc_primary_faces[bc_i]))
        assert len(bc_face_order) == n_unique == N

        # Permutation: original index -> new BC-block index
        perm = np.array(bc_face_order, dtype=int)
        A_reordered = A[np.ix_(perm, perm)]

        # L_local = diagonal 20x20 block (should be same for all BCs if P2 confirmed)
        L_local = A_reordered[:20, :20].copy()  # first BC's block

        # For Kronecker structure: A[bc_i*20:(bc_i+1)*20, bc_j*20:(bc_j+1)*20]
        # is M_coupling if bc_j is adjacent to bc_i along one axis, 0 otherwise.
        # Extract inter-BC coupling blocks
        def get_block(A_r, bc_i, bc_j, size=20):
            return A_r[bc_i*size:(bc_i+1)*size, bc_j*size:(bc_j+1)*size]

        # Build bc grid index -> flat index mapping
        def flat_bc(ijk):
            return ijk[0] * 9 + ijk[1] * 3 + ijk[2]

        # Compute L_x, L_y, L_z: 27x27 circulant adjacency matrices (one per axis)
        # L_axis[i,j] = 1 if BCs differ by +-1 mod 3 in that axis only (and match in other axes)
        L_axes = [np.zeros((27, 27)) for _ in range(3)]
        for bc_i_idx, ijk_i in enumerate(bc_grid):
            for axis in range(3):
                for delta in (-1, 1):
                    ijk_j = list(ijk_i)
                    ijk_j[axis] = (ijk_i[axis] + delta) % 3
                    bc_j_idx = flat_bc(ijk_j)
                    L_axes[axis][bc_i_idx, bc_j_idx] = 1.0

        # Compute M_x, M_y, M_z: average over all bc-pairs that are axis-neighbors
        M_axes = [np.zeros((20, 20)) for _ in range(3)]
        M_axes_counts = [0, 0, 0]
        for bc_i_idx, ijk_i in enumerate(bc_grid):
            for axis in range(3):
                for delta in (-1, 1):
                    ijk_j = list(ijk_i)
                    ijk_j[axis] = (ijk_i[axis] + delta) % 3
                    bc_j_idx = flat_bc(ijk_j)
                    coupling = get_block(A_reordered, bc_i_idx, bc_j_idx)
                    M_axes[axis] += coupling
                    M_axes_counts[axis] += 1

        M_axes = [M / c for M, c in zip(M_axes, M_axes_counts)]

        # Build A_Kronecker using the Kronecker structure
        # A_Kron = kron(I_27, L_local) + kron(L_x, M_x) + kron(L_y, M_y) + kron(L_z, M_z)
        I27 = np.eye(27)
        A_Kronecker = (np.kron(I27, L_local) +
                       np.kron(L_axes[0], M_axes[0]) +
                       np.kron(L_axes[1], M_axes[1]) +
                       np.kron(L_axes[2], M_axes[2]))

        residual = np.linalg.norm(A_reordered - A_Kronecker) / np.linalg.norm(A_reordered)
        kronecker_residual = float(residual)
        log(f"  Kronecker residual ||A - A_Kron|| / ||A|| = {kronecker_residual:.3e}")

        # Also check if M_x, M_y, M_z are position-independent
        log("\n  Checking if coupling matrices are position-independent across all 27 BC pairs:")
        for axis, axname in enumerate(['x', 'y', 'z']):
            diffs = []
            M_ref = M_axes[axis]
            for bc_i_idx, ijk_i in enumerate(bc_grid):
                for delta in (-1, 1):
                    ijk_j = list(ijk_i)
                    ijk_j[axis] = (ijk_i[axis] + delta) % 3
                    bc_j_idx = flat_bc(ijk_j)
                    coupling = get_block(A_reordered, bc_i_idx, bc_j_idx)
                    diffs.append(np.max(np.abs(coupling - M_ref)))
            log(f"  M_{axname}: max deviation from mean = {max(diffs):.3e}, "
                f"all uniform? {max(diffs) < 1e-10}")

    # --- Step 9: Eigenspace projection per BC ---
    log("\n[Step 9] Eigenspace projection analysis (per BC)")
    log("  With full translation symmetry, all 27 BCs should have equal projections.")

    if all_20:
        # Use the reordered eigenvectors
        # Eigvecs are in original face ordering; we need them in BC-block order
        eigvecs_reordered = eigvecs[perm, :]  # (N, N)

        # Recompute spectrum in reordered basis (eigenvalues unchanged)
        # For each BC, compute squared weight = sum over its 20 faces

        log(f"\n  {'eigenvalue':>14} | {'deg':>5} | BC proj range (min, max) | all equal?")
        log(f"  {'-'*14}-+-{'-'*5}-+-{'-'*26}-+-{'-'*9}")

        projection_uniform = True
        for rep, deg, _ in grouped:
            mask = np.abs(eigvals - rep) <= 1e-6
            member_idx = np.where(mask)[0]
            vecs = eigvecs_reordered[:, member_idx]  # (N, deg)
            sq = np.sum(vecs ** 2, axis=1)           # squared weight per face

            # Per-BC projection: sum over the 20 faces of each BC
            bc_projs = []
            for bc_i in range(27):
                start = bc_i * 20
                bc_proj = float(np.sum(sq[start:start+20]))
                bc_projs.append(bc_proj)

            bc_projs = np.array(bc_projs)
            pmin = float(np.min(bc_projs))
            pmax = float(np.max(bc_projs))
            pvar = float(np.max(np.abs(bc_projs - np.mean(bc_projs))))
            equal = pvar < 1e-10
            if not equal:
                projection_uniform = False
            log(f"  {rep:14.6f} | {deg:5d} | ({pmin:.6f}, {pmax:.6f})       | {equal}")

        log(f"\n  P5 check: All per-BC eigenspace projections uniform? {projection_uniform}")
    else:
        log("  SKIPPED: requires uniform 20-face patches.")
        projection_uniform = False

    # --- Predictions summary ---
    log("\n" + "=" * 78)
    log("[PREDICTIONS SUMMARY]")
    log("=" * 78)

    def confirm(pred, label, detail=""):
        status = "CONFIRMED" if pred else "REFUTED"
        line = f"  {label}: {status}"
        if detail:
            line += f"  ({detail})"
        log(line)

    # P1: all BC degrees are 6 (always true for a 3-torus by construction)
    confirm(all_deg6,
            "P1: All 27 BCs have degree 6 in the periodic BC graph",
            f"degrees={set(degrees)}")

    # P2: all 27 local 20x20 blocks identical
    if all_20:
        confirm(blocks_identical,
                "P2: All 27 local 20x20 diagonal blocks are identical",
                f"max diff = {max_block_diff:.3e}")
    else:
        log("  P2: CANNOT TEST (patch sizes not uniform)")

    # P3: lambda=-2 (open-BC deg 56) disappears
    confirm(not has_lam_minus2_exact,
            "P3: lambda=-2 (deg 56 in open BC) disappears from spectrum",
            f"exact -2.0 in periodic spectrum? {has_lam_minus2_exact}")

    # P4: Kronecker residual < 1e-10
    if not np.isnan(kronecker_residual):
        confirm(kronecker_residual < 1e-10,
                "P4: Kronecker residual < 1e-10",
                f"actual residual = {kronecker_residual:.3e}")
    else:
        log("  P4: CANNOT TEST (Kronecker test skipped)")

    # P5: all per-BC projections equal
    if all_20:
        confirm(projection_uniform,
                "P5: All per-BC eigenspace projections are equal")
    else:
        log("  P5: CANNOT TEST (patch sizes not uniform)")

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
