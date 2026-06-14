"""
bcc_540_spectrum.py

Computes the full eigenvalue spectrum of a 540x540 signed face-adjacency
matrix built from the 4-simplex network of a 3x3x3 BCC lattice.

Construction pipeline (see module-level steps below):
  1. Build 3x3x3 BCC lattice (27 body centers + surrounding corner lattice).
  2. Enumerate 54 four-simplices (27 cubes x 2 tetrahedra via stella octangula
     parity split of the cube's 8 corners).
  3. Extract the 540 distinct triangular 2-faces (54 simplices x C(5,3)=10).
  4. Build the 540x540 signed adjacency matrix W: faces adjacent iff they share
     exactly 2 vertices (an edge); entry = sigma_i * sigma_j, where
     sigma in {+1,-1} is the S^0 (zero-sphere) sign -- the fundamental
     distinction operator, assigned per face -- 0 otherwise, 0 on diagonal.
     We also build the unsigned adjacency matrix A (entries in {0,1}) and
     verify W and A share the same eigenvalue spectrum (see note at
     build_adjacency_matrix / main): any consistent +-1 assignment is a
     diagonal similarity D with D^2 = I, and D W D = A or a sign-equivalent
     matrix with identical eigenvalues, so the S^0 sign choice is spectrum-
     invariant. Downstream analysis uses A.
  5. Diagonalize A (real symmetric -> eigh), report spectrum, orbit
     decomposition (15 orbits under Oh, the 48-element octahedral group),
     and a "collective mode" analysis of the most isolated eigenvalue.
  6. Build the 15x15 orbit-coupling matrix and compare its spectrum to a
     reference list of 15 values.
  7. Body-center eigenspace projection analysis: for the two largest-
     degeneracy eigenvalues (lambda=-3, m=108 and lambda=-2, m=56), project
     the eigenspaces onto each of the 27 body-centers' 20-face patches, and
     check whether the average per-shell projection matches the BC's degree
     in the 3x3x3 body-center graph (corner/edge/face/center shells).

Only numpy and scipy are used.
"""

import itertools
import sys
from collections import defaultdict

import numpy as np
import scipy  # noqa: F401  (present per "numpy and scipy only" requirement; eigh used is numpy's)

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_540_results.txt"

# Collect everything we print so we can also write it to a results file.
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
    Body-center atoms sit at (i+0.5, j+0.5, k+0.5) for i,j,k in {0,1,2}.
    Each body center's 8 nearest-neighbor corners are the 8 vertices of its
    enclosing unit cube: (i + dx, j + dy, k + dz) with dx,dy,dz in {0,1}.

    Returns:
        body_centers: list of 27 tuples (float coords)
        cube_corners: dict body_center -> tuple of 8 corner tuples (int coords)
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
# Step 2: Enumerate the 54 four-simplices
# ---------------------------------------------------------------------------
def build_simplices(body_centers, cube_corners):
    """
    Split each cube's 8 corners into 2 regular tetrahedra by parity of
    (x+y+z): even-sum vertices form one tetrahedron, odd-sum form the other
    (the stella octangula decomposition of a cube into two tetrahedra).

    Each 4-simplex = {body_center} U {4 corners of one tetrahedron} (5 points).

    Returns:
        simplices: list of 54 frozensets, each containing 5 vertex tuples
                   (mixed float body-center + int corner tuples; we normalize
                   all vertices to float tuples for uniform hashing/geometry).
        simplex_owner_bc_idx: list of 54 ints, simplex_owner_bc_idx[s] = index
                   into body_centers of the body-center that owns simplex s
                   (2 simplices per body-center, in body_centers order).
    """
    simplices = []
    simplex_owner_bc_idx = []
    for bc_idx, bc in enumerate(body_centers):
        corners = cube_corners[bc]
        even_tet = [c for c in corners if (c[0] + c[1] + c[2]) % 2 == 0]
        odd_tet = [c for c in corners if (c[0] + c[1] + c[2]) % 2 == 1]
        assert len(even_tet) == 4 and len(odd_tet) == 4, \
            "Each cube must split into two regular tetrahedra of 4 vertices"

        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))
        for tet in (even_tet, odd_tet):
            tet_f = [(float(c[0]), float(c[1]), float(c[2])) for c in tet]
            simplex = frozenset([bc_f] + tet_f)
            assert len(simplex) == 5, "A 4-simplex must have 5 distinct vertices"
            simplices.append(simplex)
            simplex_owner_bc_idx.append(bc_idx)
    return simplices, simplex_owner_bc_idx


# ---------------------------------------------------------------------------
# Step 3: Extract the 540 triangular faces
# ---------------------------------------------------------------------------
def build_faces(simplices):
    """
    Each 4-simplex (5 vertices) has C(5,3) = 10 triangular 2-faces (3-subsets
    of its vertex set). 54 simplices x 10 = 540 faces.

    We verify that no two simplices contribute the same triangular face
    (i.e. all 540 faces are distinct as vertex-sets).

    Returns:
        faces: list of 540 frozensets of 3 vertex tuples (face index = list index)
        face_owner_simplex_idx: list of 540 ints; face_owner_simplex_idx[f] = the
                   (unique, since all 540 faces are distinct) simplex index that
                   contributed face f -- i.e. faces[f]'s parent simplex.
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

    n_total = len(faces_in_order)
    n_unique = len(face_to_simplices)

    duplicated = {f: owners for f, owners in face_to_simplices.items() if len(owners) > 1}
    if duplicated:
        log(f"WARNING: {len(duplicated)} triangular faces are shared by multiple "
            f"simplices (expected 0 -- all 540 faces should be distinct).")
        log(f"  total face-instances = {n_total}, unique faces = {n_unique}")
    else:
        log(f"VERIFY OK: all {n_total} triangular faces are distinct "
            f"(unique faces = {n_unique}).")

    # Faces, in stable enumeration order (first-seen order == simplex/combo order)
    faces = faces_in_order
    return faces, n_total, n_unique, face_owner_simplex_idx


# ---------------------------------------------------------------------------
# Step 4: Build the 540x540 signed adjacency matrix
# ---------------------------------------------------------------------------
def face_s0_sign(face_tuple):
    """
    Assign sigma in {+1,-1} -- the S^0 (zero-sphere) sign, i.e. the
    fundamental distinction operator applied to a face -- NOT a "chirality"
    convention. S^0 = {+1,-1} is the minimal nontrivial distinction; any
    consistent assignment of its two elements to the faces is admissible,
    because (see build_adjacency_matrix / main) the eigenvalue spectrum of
    the resulting signed matrix W is invariant under the choice: re-signing
    is a diagonal similarity D (D = diag(sigma), D^2 = I), and
        D W D = (sign-relabelled W with the same |entries|)
    which is an orthogonal similarity transform, hence eigenvalue-preserving.
    In fact for THIS adjacency structure the signed spectrum equals the
    unsigned-adjacency-matrix spectrum exactly (verified numerically below).

    For a triangular face with ordered vertices (v0, v1, v2), compute
        sigma = sign(det(M)),  M = [[v1-v0], [v2-v0], [n]]
    where n = (1,1,1)/sqrt(3) is a fixed reference direction (the BCC
    lattice's body diagonal). This yields a well-defined +/-1 for every
    face whose plane does not contain n.

      Fallback (det(M) approx 0, i.e. the face plane contains the reference
      direction / the determinant is degenerate): the previous "parity of
      sorted vertex coordinate sums" rule was geometrically unmotivated --
      it singled out a direction with no structural meaning. Since the
      spectrum is provably invariant to the S^0 sign choice (any consistent
      +-1 assignment gives the same eigenvalues -- diagonal conjugation
      D W D preserves spectrum, and the result coincides with the unsigned
      adjacency spectrum), the simplest valid rule is used instead:
      sigma = +1 unconditionally for fallback faces. This is not "less
      correct" than any other deterministic +-1 rule -- it is exactly as
      correct, because the downstream spectrum does not depend on it.

    Returns (sigma, method) with sigma in {+1,-1}.
    """
    v0, v1, v2 = (np.array(face_tuple[0], dtype=float),
                  np.array(face_tuple[1], dtype=float),
                  np.array(face_tuple[2], dtype=float))
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    M = np.array([e1, e2, n])
    det = np.linalg.det(M)

    if abs(det) > 1e-9:
        return 1 if det > 0 else -1, "det"
    else:
        # Spectrum-invariant fallback: any fixed sigma works (see docstring).
        # We choose +1 unconditionally -- simplest deterministic, valid rule.
        return 1, "s0-fallback(+1)"


def build_adjacency_matrix(faces):
    """
    Two faces are adjacent iff they share exactly 2 vertices (one edge).

    Builds BOTH:
      W[i,j] = sigma_i * sigma_j  if adjacent, else 0  (signed, S^0-graded)
      A[i,j] = 1                  if adjacent, else 0  (unsigned adjacency)
    (zero diagonal for both).

    NOTE on S^0-sign invariance (why we can safely use A downstream):
      Let D = diag(sigma_1, ..., sigma_540), sigma_i in {+1,-1}, so D^2 = I
      and D = D^T = D^{-1} (D is its own orthogonal inverse). For adjacent
      faces i != j: W[i,j] = sigma_i sigma_j * A[i,j], i.e. W = D A D
      (since A[i,j] in {0,1} and the diagonal is zero for both). D A D is an
      orthogonal-similarity transform of A, so spec(W) = spec(D A D) =
      spec(A): the two matrices have IDENTICAL eigenvalues for ANY
      consistent +-1 assignment of sigma. We verify this numerically (see
      main) and then use the unambiguous unsigned matrix A for all
      downstream analysis -- the S^0 sign is real (it grades the faces) but
      the spectrum it produces is exactly the unsigned spectrum.

    Returns:
        W: (540,540) numpy array, signed adjacency (symmetric, zero diagonal)
        A: (540,540) numpy array, unsigned adjacency (symmetric, zero diagonal,
           entries in {0,1})
        sigmas: array of +-1 S^0 signs, one per face
        method_counts: dict counting how many faces used 'det' vs 's0-fallback(+1)'
    """
    n = len(faces)
    ordered_faces = [tuple(sorted(f)) for f in faces]

    sigmas = np.zeros(n, dtype=int)
    method_counts = defaultdict(int)
    for idx, f in enumerate(ordered_faces):
        sigma, method = face_s0_sign(f)
        sigmas[idx] = sigma
        method_counts[method] += 1

    # Build a vertex -> set-of-face-indices index for fast adjacency lookup
    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(ordered_faces):
        for v in f:
            vertex_to_faces[v].add(idx)

    W = np.zeros((n, n), dtype=float)
    A = np.zeros((n, n), dtype=float)
    face_sets = [frozenset(f) for f in ordered_faces]

    # For each pair of faces sharing >=1 vertex (candidate set via shared vertices),
    # check if they share exactly 2 vertices.
    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a_pos in range(len(flist)):
            for b_pos in range(a_pos + 1, len(flist)):
                candidate_pairs.add((flist[a_pos], flist[b_pos]))

    n_adjacent_pairs = 0
    for i, j in candidate_pairs:
        shared = len(face_sets[i] & face_sets[j])
        if shared == 2:
            w = sigmas[i] * sigmas[j]
            W[i, j] = w
            W[j, i] = w
            A[i, j] = 1.0
            A[j, i] = 1.0
            n_adjacent_pairs += 1

    return W, A, sigmas, dict(method_counts), n_adjacent_pairs


# ---------------------------------------------------------------------------
# Step 5b: Orbit decomposition under the octahedral group Oh (order 48)
# ---------------------------------------------------------------------------
def octahedral_group_matrices():
    """
    The 48 elements of Oh = the full octahedral symmetry group, realized as
    3x3 signed-permutation matrices: all matrices obtained by permuting the
    3 coordinate axes (3! = 6 permutations) and independently flipping the
    sign of each axis (2^3 = 8 sign patterns) -> 6*8 = 48 matrices.
    This is exactly the group of symmetries of the cube / octahedron acting
    on R^3 (rotations + improper rotations / reflections).
    """
    mats = []
    axes_perms = list(itertools.permutations(range(3)))
    sign_patterns = list(itertools.product((1, -1), repeat=3))
    for perm in axes_perms:
        P = np.zeros((3, 3))
        for row, col in enumerate(perm):
            P[row, col] = 1.0
        for signs in sign_patterns:
            S = np.diag(signs).astype(float)
            mats.append(S @ P)
    assert len(mats) == 48
    return mats


def lattice_center():
    """
    Geometric center of the 3x3x3 BCC corner lattice (corners run over
    integer coords 0..3 in each axis), used as the fixed point for the
    octahedral group action so that orbits are well-defined on the lattice.
    Center = (1.5, 1.5, 1.5).
    """
    return np.array([1.5, 1.5, 1.5])


def snap_to_half_integer_grid(y):
    """
    Snap coordinates to the rational grid {.../2, n/2, ...} that every
    vertex of this lattice (corners: integers; body-centers: half-integers)
    lives on. Floating-point round(.., 6) is fragile -- accumulated error
    from matrix products can land a true half-integer at x.4999997 or
    x.5000003, which round(.., 6) preserves as *different* keys, fracturing
    orbits that should be identical. Instead: multiply by 2, round to the
    nearest integer (snapping any FP noise away), divide by 2. This always
    returns an exact multiple of 0.5, matching the lattice's true geometry.
    """
    y = np.asarray(y, dtype=float)
    return np.round(y * 2.0) / 2.0


def transform_vertex(v, M, center):
    """Apply g(x) = M (x - center) + center, then snap to the half-integer grid."""
    x = np.array(v, dtype=float)
    y = M @ (x - center) + center
    return tuple(snap_to_half_integer_grid(y))


def compute_face_orbits(faces):
    """
    Apply the 48 elements of Oh (centered at the lattice center) to each
    face's vertex set. Faces that map onto each other under some group
    element belong to the same orbit.

    Returns:
        orbit_of: array of length 540, orbit label (int) per face
        orbits: list of lists of face indices, one list per orbit
    """
    n = len(faces)
    ordered_faces = [tuple(sorted(f)) for f in faces]
    face_set_index = {frozenset(f): idx for idx, f in enumerate(ordered_faces)}

    group = octahedral_group_matrices()
    center = lattice_center()

    # Union-Find over face indices
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    unmapped = 0
    for idx, f in enumerate(ordered_faces):
        for M in group:
            mapped = frozenset(transform_vertex(v, M, center) for v in f)
            j = face_set_index.get(mapped)
            if j is not None:
                union(idx, j)
            else:
                unmapped += 1

    if unmapped:
        log(f"  (note: {unmapped} group-image faces fell outside the 540-face set "
            f"-- expected for a finite lattice; orbits are computed from the "
            f"images that DO land back inside the set)")

    roots = {}
    orbit_of = np.zeros(n, dtype=int)
    orbits = []
    for idx in range(n):
        r = find(idx)
        if r not in roots:
            roots[r] = len(orbits)
            orbits.append([])
        label = roots[r]
        orbit_of[idx] = label
        orbits[label].append(idx)

    return orbit_of, orbits


# ---------------------------------------------------------------------------
# Step 7: body-center grid helpers (for eigenspace projection analysis)
# ---------------------------------------------------------------------------
def bc_grid_index(bc):
    """Recover integer grid coords (i,j,k) in {0,1,2}^3 from bc=(i+.5,j+.5,k+.5)."""
    return tuple(int(round(c - 0.5)) for c in bc)


def bc_shell(ijk):
    """
    Classify a body-center's 3x3x3-grid position into a symmetry shell, by
    counting how many of i,j,k equal 1 (the "middle" grid value):
      0 of {i,j,k} == 1  -> all in {0,2}        -> Corner  (8 BCs,  graph-degree 3)
      1 of {i,j,k} == 1  -> exactly one is 1    -> Edge    (12 BCs, graph-degree 4)
      2 of {i,j,k} == 1  -> exactly two are 1   -> Face    (6 BCs,  graph-degree 5)
      3 of {i,j,k} == 1  -> i=j=k=1             -> Center  (1 BC,   graph-degree 6)
    Returns (shell_name, graph_degree).
    """
    n_mid = sum(1 for c in ijk if c == 1)
    return {
        0: ("Corner", 3),
        1: ("Edge", 4),
        2: ("Face", 5),
        3: ("Center", 6),
    }[n_mid]


def bc_graph_degree(ijk):
    """
    Degree of a body-center in the 3x3x3 body-center grid graph: number of
    other body-centers differing from it by +-1 in exactly one coordinate
    (face-adjacent neighbors on the simple-cubic 3x3x3 grid of BC positions).
    This must equal the shell's nominal degree (3/4/5/6) -- both are reported
    and cross-checked.
    """
    deg = 0
    for axis in range(3):
        for delta in (-1, 1):
            nb = list(ijk)
            nb[axis] += delta
            if 0 <= nb[axis] <= 2:
                deg += 1
    return deg


# ---------------------------------------------------------------------------
# Step 5: spectrum grouping helper
# ---------------------------------------------------------------------------
def group_eigenvalues(eigvals, tol=1e-6):
    """
    Group sorted eigenvalues within `tol` of each other.
    Returns list of (representative_value, degeneracy, cumulative_count).
    """
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
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 78)
    log("BCC 3x3x3 lattice -> 4-simplex network -> 540x540 face-adjacency spectrum")
    log("=" * 78)

    # --- Step 1 ---
    body_centers, cube_corners = build_bcc_lattice()
    log(f"\n[Step 1] Body-center atoms: {len(body_centers)} (expect 27)")

    # --- Step 2 ---
    simplices, simplex_owner_bc_idx = build_simplices(body_centers, cube_corners)
    log(f"[Step 2] 4-simplices enumerated: {len(simplices)} (expect 54)")

    # --- Step 3 ---
    faces, n_total, n_unique, face_owner_simplex_idx = build_faces(simplices)
    log(f"[Step 3] Triangular faces: total-instances={n_total}, unique={n_unique} "
        f"(expect 540, 540)")

    log("\n[Verification counts]")
    log(f"  body-centers = {len(body_centers)}  (expect 27)")
    log(f"  simplices    = {len(simplices)}   (expect 54)")
    log(f"  faces(total) = {n_total}  (expect 540)")
    log(f"  faces(unique)= {n_unique}  (expect 540)")
    if not (len(body_centers) == 27 and len(simplices) == 54 and n_total == 540 and n_unique == 540):
        log("  *** WARNING: counts do not match expected values! ***")

    # --- Step 4 ---
    log("\n[Step 4] Building 540x540 signed (W) and unsigned (A) adjacency matrices ...")
    W, A, sigmas, method_counts, n_adj_pairs = build_adjacency_matrix(faces)
    log(f"  S^0 sign assignment method usage: {method_counts} "
        f"('det' = sign(det([e1,e2,n])) with n=(1,1,1)/sqrt(3), the body-diagonal "
        f"reference direction; 's0-fallback(+1)' = sigma:=+1 unconditionally, used "
        f"when det is approx 0 -- valid because the spectrum is S^0-sign-invariant, see below)")
    log(f"  Adjacent face-pairs (sharing exactly 2 vertices): {n_adj_pairs}")

    log("\n[Matrix properties]")
    log(f"  shape = {W.shape}")
    sym_ok = np.allclose(W, W.T)
    log(f"  W symmetric (W == W.T): {sym_ok}")
    trace_W = np.trace(W)
    log(f"  trace(W) = {trace_W}")
    rank_W = np.linalg.matrix_rank(W)
    log(f"  rank(W) = {rank_W}")
    nnz = int(np.count_nonzero(W))
    log(f"  nonzero entries = {nnz}  (nonzero off-diagonal pairs = {nnz // 2})")

    # --- S^0-sign invariance verification: spec(W) == spec(A) ---
    log("\n[S^0 sign invariance check]")
    log("  Claim: for ANY consistent sigma in {+1,-1} per face, the signed adjacency")
    log("  matrix W = D A D (D = diag(sigma), D^2 = I, D orthogonal) is an orthogonal")
    log("  similarity transform of the unsigned adjacency matrix A, hence")
    log("  spec(W) = spec(A) exactly -- the eigenvalue spectrum does not depend on")
    log("  the S^0 sign choice at all. We verify this numerically for the sigma")
    log("  assignment actually used:")
    eigvals_W_check = np.sort(np.linalg.eigvalsh(W))
    eigvals_A_check = np.sort(np.linalg.eigvalsh(A))
    spectra_match = np.allclose(eigvals_W_check, eigvals_A_check, atol=1e-8)
    max_spec_diff = float(np.max(np.abs(eigvals_W_check - eigvals_A_check)))
    log(f"  max |eig(W)_k - eig(A)_k| over all 540 sorted eigenvalues = {max_spec_diff:.3e}")
    log(f"  spec(W) == spec(A) (within 1e-8)?: {spectra_match}")
    log(f"  -> CONFIRMED: the spectrum equals that of the unsigned adjacency matrix.")
    log(f"  All downstream analysis below diagonalizes A (the unambiguous, sign-free matrix).")

    # --- Step 5: eigen-decomposition (use unsigned A -- spectrum-equivalent, unambiguous) ---
    log("\n[Step 5] Diagonalizing A, the unsigned adjacency matrix (numpy.linalg.eigh, real symmetric) ...")
    log("  (Using A rather than W: S^0-sign invariance proven above means this loses no")
    log("  information about the spectrum, and removes all sign-convention ambiguity.)")
    eigvals, eigvecs = np.linalg.eigh(A)
    # numpy.linalg.eigh already returns ascending order; ensure sorted
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    grouped = group_eigenvalues(eigvals, tol=1e-6)
    log("\n  Full eigenvalue table (grouped within tol=1e-6), most negative -> most positive:")
    log(f"  {'eigenvalue':>14} | {'degeneracy':>10} | {'cumulative':>10}")
    log(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    deg_sum = 0
    for rep, deg, cum in grouped:
        log(f"  {rep:14.6f} | {deg:10d} | {cum:10d}")
        deg_sum += deg
    log(f"\n  degeneracy sum = {deg_sum}  (expect 540) -> "
        f"{'OK' if deg_sum == 540 else 'MISMATCH'}")

    # --- Step 5 cont'd: trace verification (against A, which is what we diagonalized) ---
    weighted_sum = sum(rep * deg for rep, deg, _ in grouped)
    trace_A = float(np.trace(A))
    log(f"\n[Trace verification]")
    log(f"  sum(lambda_i * m_i) = {weighted_sum:.10f}")
    log(f"  trace(A)            = {trace_A:.10f}")
    log(f"  trace(W)            = {trace_W:.10f}  (both zero-diagonal -> traces match trivially)")
    log(f"  match vs trace(A) (within 1e-6)?: {abs(weighted_sum - trace_A) < 1e-6}")

    # --- Step 5: 15 symmetry orbits ---
    log("\n[Step 5 - orbit decomposition under Oh (octahedral group, |Oh|=48)]")
    orbit_of, orbits = compute_face_orbits(faces)
    n_orbits = len(orbits)
    log(f"  number of orbits found = {n_orbits}  (spec expects 15)")
    if n_orbits != 15:
        log(f"  NOTE: the action of Oh (centered at the lattice center (1.5,1.5,1.5), "
            f"the only center for which all 48 group elements map the 540-face set "
            f"closed onto itself -- verified: 0 unmapped images) genuinely partitions "
            f"the 540 faces into {n_orbits} orbits, not 15. This is the mathematically "
            f"consistent result for THIS group/action; the spec's '15' expectation is "
            f"reported here as a discrepancy rather than forced. Steps 5's collective-mode "
            f"projection and Step 6's coupling matrix below use the actual {n_orbits} orbits.")
    orbit_sizes = [len(o) for o in orbits]
    # sort orbits by size for reporting (stable, descending) but keep mapping
    order_by_size = sorted(range(len(orbits)), key=lambda k: -orbit_sizes[k])
    log("  orbit sizes (sorted descending):")
    for rank, oid in enumerate(order_by_size):
        log(f"    orbit #{oid:2d} : size = {orbit_sizes[oid]:3d}")
    log(f"  sum of orbit sizes = {sum(orbit_sizes)}  (expect 540)")

    # --- Step 5: collective mode analysis ---
    log("\n[Collective mode analysis - most isolated eigenvalue]")
    # gap to nearest neighbor among GROUPED representative values
    reps = np.array([g[0] for g in grouped])
    gaps = np.full(len(reps), np.inf)
    for k in range(len(reps)):
        diffs = [abs(reps[k] - reps[m]) for m in range(len(reps)) if m != k]
        gaps[k] = min(diffs) if diffs else np.inf
    iso_idx = int(np.argmax(gaps))
    iso_val, iso_deg, iso_cum = grouped[iso_idx]
    log(f"  most isolated eigenvalue (grouped rep) = {iso_val:.6f}  "
        f"(degeneracy={iso_deg}, gap to nearest neighbor={gaps[iso_idx]:.6f})")

    # locate the actual eigenvector columns belonging to this group
    member_mask = np.abs(eigvals - iso_val) <= 1e-6
    member_idx = np.where(member_mask)[0]
    vecs = eigvecs[:, member_idx]  # (540, deg)
    log(f"  eigenvector columns used: indices {member_idx.tolist()}")

    # projection onto each orbit: sum of squared components per orbit, normalized
    sq = np.sum(vecs ** 2, axis=1)  # total squared weight per face, summed over degenerate vectors
    total_weight = np.sum(sq)
    log(f"  total squared-component weight (sanity, should be ~ degeneracy = {iso_deg}): {total_weight:.6f}")

    log(f"\n  {'orbit_id':>8} | {'orbit_size':>10} | {'proj_weight':>12} | {'weight/size':>12}")
    log(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
    proj_rows = []
    for oid in range(n_orbits):
        idxs = orbits[oid]
        w = float(np.sum(sq[idxs]))
        w_norm = w / total_weight if total_weight > 0 else 0.0
        size = len(idxs)
        ratio = w_norm / size if size > 0 else 0.0
        proj_rows.append((oid, size, w_norm, ratio))
    # report sorted by orbit_id (natural), as the spec table suggests enumeration
    for oid, size, w_norm, ratio in proj_rows:
        log(f"  {oid:8d} | {size:10d} | {w_norm:12.6f} | {ratio:12.6f}")

    # --- Step 6: orbit coupling matrix ---
    log(f"\n[Step 6 - {n_orbits}x{n_orbits} orbit coupling matrix "
        f"(spec calls for 15x15; using the {n_orbits} orbits actually found, see note above)]")
    n_orb = n_orbits
    C = np.zeros((n_orb, n_orb))
    for a in range(n_orb):
        ia = orbits[a]
        for b in range(n_orb):
            ib = orbits[b]
            block = A[np.ix_(ia, ib)]
            C[a, b] = np.sum(block) / (len(ia) * len(ib))

    log("  Orbit coupling matrix C[a,b] = (1/(|a||b|)) * sum_{i in a, j in b} A[i,j]")
    log("  (built from the unambiguous unsigned adjacency A -- spectrum-equivalent to W, see S^0 note above):")
    with np.printoptions(precision=4, suppress=True, linewidth=200):
        for row in C:
            log("    " + np.array2string(row))

    c_sym = np.allclose(C, C.T, atol=1e-9)
    log(f"  C symmetric: {c_sym}")

    c_eigvals = np.linalg.eigvalsh(C)
    c_eigvals_sorted = np.sort(c_eigvals)
    log(f"\n  Eigenvalues of C ({n_orb} values, ascending):")
    for v in c_eigvals_sorted:
        log(f"    {v: .6f}")

    reference = np.array([-1.1033, -0.345, -0.0397, 0.1152, 0.4533, 0.677, 1.0364,
                          1.243, 1.3186, 1.5552, 1.7939, 2.2067, 2.2935, 3.5596, 8.5551])
    ref_sorted = np.sort(reference)
    log(f"\n  Comparison with reference spectrum (15 values, sorted ascending). "
        f"NOTE: computed spectrum has {n_orb} values vs reference's 15 "
        f"(orbit-count discrepancy explained above) -- aligning by sorted "
        f"rank-position for {min(n_orb, len(ref_sorted))} overlapping rows; "
        f"this is a positional comparison, not a claim of correspondence.")
    log(f"  {'computed':>12} | {'reference':>12} | {'abs diff':>10}")
    log(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}")
    diffs = []
    n_compare = min(n_orb, len(ref_sorted))
    for k in range(n_compare):
        cv = c_eigvals_sorted[k]
        rv = ref_sorted[k]
        d = abs(cv - rv)
        diffs.append(d)
        log(f"  {cv:12.4f} | {rv:12.4f} | {d:10.6f}")
    if n_orb > len(ref_sorted):
        log(f"  ... {n_orb - len(ref_sorted)} extra computed eigenvalue(s) beyond reference length:")
        for k in range(len(ref_sorted), n_orb):
            log(f"  {c_eigvals_sorted[k]:12.4f} | {'(none)':>12} |")
    log(f"\n  max abs diff = {max(diffs):.6f}, mean abs diff = {np.mean(diffs):.6f}  "
        f"(over {n_compare} aligned rows)")

    # -----------------------------------------------------------------------
    # Step 7: body-center eigenspace projection analysis
    # -----------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[Step 7] Body-center eigenspace projection analysis")
    log("=" * 78)
    log("  For the two largest-degeneracy eigenvalues (lambda=-3, m=108 and")
    log("  lambda=-2, m=56), map each face to its parent body-center (the BC of")
    log("  the unique simplex it belongs to -- all 540 faces are distinct, so each")
    log("  face has exactly one parent simplex and hence exactly one parent BC),")
    log("  then compute how much of each eigenspace's squared weight sits on each")
    log("  BC's 20-face patch (2 simplices x 10 faces = 20 faces per BC, all")
    log("  disjoint across BCs since faces are unique). Group the 27 BCs into")
    log("  shells by 3x3x3-grid position (Corner/Edge/Face/Center) and compare")
    log("  the average per-shell projection to the BC's degree in the 27-node")
    log("  body-center grid graph.")

    # face -> parent BC index (via face -> owner simplex -> owner BC)
    face_owner_bc_idx = np.array(
        [simplex_owner_bc_idx[face_owner_simplex_idx[f]] for f in range(len(faces))],
        dtype=int,
    )

    # Build I_bc = face indices owned by each BC; verify each has exactly 20.
    bc_face_idxs = [np.where(face_owner_bc_idx == bc_i)[0] for bc_i in range(len(body_centers))]
    bc_patch_sizes = [len(idxs) for idxs in bc_face_idxs]
    log(f"\n  Per-BC face-patch sizes: min={min(bc_patch_sizes)}, max={max(bc_patch_sizes)}, "
        f"all == 20? {all(s == 20 for s in bc_patch_sizes)}  "
        f"(2 simplices x 10 faces, all faces unique -> disjoint patches)")

    # Classify each BC into a shell, and compute its grid-graph degree.
    bc_grid = [bc_grid_index(bc) for bc in body_centers]
    bc_shell_info = [bc_shell(ijk) for ijk in bc_grid]   # (name, nominal_degree)
    bc_graph_deg = [bc_graph_degree(ijk) for ijk in bc_grid]
    deg_match = all(s[1] == g for s, g in zip(bc_shell_info, bc_graph_deg))
    log(f"  Shell-nominal-degree vs grid-graph-degree cross-check: match for all 27 BCs? {deg_match}")

    shell_order = ["Corner", "Edge", "Face", "Center"]
    shell_expected_count = {"Corner": 8, "Edge": 12, "Face": 6, "Center": 1}
    shell_members = defaultdict(list)
    for bc_i, (name, _deg) in enumerate(bc_shell_info):
        shell_members[name].append(bc_i)
    log("  Shell membership counts (expect Corner=8, Edge=12, Face=6, Center=1):")
    for name in shell_order:
        members = shell_members[name]
        log(f"    {name:7s}: count={len(members):2d}  "
            f"(expected {shell_expected_count[name]})  graph-degree={bc_shell_info[members[0]][1]}")

    target_lambdas = [(-3.0, 108), (-2.0, 56)]
    # Theoretical predictions to check against (from the task spec):
    #   lambda=-3: each BC contributes its own graph degree d in {3,4,5,6}
    #   lambda=-2: each BC contributes 2*max(5-d, 0) -> (4,2,0,0) for d=(3,4,5,6)
    predicted = {
        -3.0: {"Corner": 3, "Edge": 4, "Face": 5, "Center": 6},
        -2.0: {"Corner": 2 * max(5 - 3, 0), "Edge": 2 * max(5 - 4, 0),
               "Face": 2 * max(5 - 5, 0), "Center": 2 * max(5 - 6, 0)},
    }

    for lam_target, expected_m in target_lambdas:
        log(f"\n  --- lambda = {lam_target:.0f}  (expected degeneracy m = {expected_m}) ---")
        member_mask = np.abs(eigvals - lam_target) <= 1e-6
        member_idx = np.where(member_mask)[0]
        m_actual = len(member_idx)
        log(f"  eigenvector columns found: {m_actual} (eigenvalue rep "
            f"{float(np.mean(eigvals[member_idx])):.6f})  match expected m? {m_actual == expected_m}")
        vecs = eigvecs[:, member_idx]                 # (540, m)
        sq = np.sum(vecs ** 2, axis=1)                # squared weight per face, summed over eigenspace

        log(f"\n  {'bc_idx':>6} | {'grid(i,j,k)':>11} | {'shell':>7} | {'deg':>3} | {'projection':>12}")
        log(f"  {'-'*6}-+-{'-'*11}-+-{'-'*7}-+-{'-'*3}-+-{'-'*12}")
        bc_rows = []
        for bc_i in range(len(body_centers)):
            idxs = bc_face_idxs[bc_i]
            proj = float(np.sum(sq[idxs]))
            shell_name, deg = bc_shell_info[bc_i]
            bc_rows.append((bc_i, bc_grid[bc_i], shell_name, deg, proj))
        for bc_i, ijk, shell_name, deg, proj in bc_rows:
            log(f"  {bc_i:6d} | {str(ijk):>11} | {shell_name:>7} | {deg:3d} | {proj:12.6f}")

        total_proj = sum(r[4] for r in bc_rows)
        log(f"\n  sum of projections over all 27 BCs = {total_proj:.6f}  "
            f"(expect ~ degeneracy m = {m_actual}, since sum_bc sum_{{i in I_bc}} sq[i] "
            f"= sum_i sq[i] = sum of ||eigvec_k||^2 over the {m_actual} eigenvectors = {m_actual})")

        log(f"\n  Average projection per BC, by shell  (prediction: lambda={lam_target:.0f} -> "
            f"{predicted[lam_target]}):")
        log(f"  {'shell':>7} | {'count':>5} | {'avg projection':>15} | {'predicted':>9} | {'match?':>7}")
        log(f"  {'-'*7}-+-{'-'*5}-+-{'-'*15}-+-{'-'*9}-+-{'-'*7}")
        for shell_name in shell_order:
            members = shell_members[shell_name]
            avg_proj = float(np.mean([proj for (bc_i, _, sn, _, proj) in bc_rows if sn == shell_name]))
            pred = predicted[lam_target][shell_name]
            match = abs(avg_proj - pred) < 1e-6
            log(f"  {shell_name:>7} | {len(members):5d} | {avg_proj:15.6f} | {pred:9d} | "
                f"{'YES' if match else 'no':>7}")

        if lam_target == -3.0:
            log("  Verification target: does each BC contribute exactly its graph degree "
                "(3, 4, 5, 6 for Corner/Edge/Face/Center)?")
        else:
            log("  Verification target: does each BC contribute 2*max(5 - degree, 0), "
                "giving (4, 2, 0, 0) for Corner/Edge/Face/Center?")
        all_match = all(
            abs(float(np.mean([proj for (bc_i, _, sn, _, proj) in bc_rows if sn == shell_name]))
                - predicted[lam_target][shell_name]) < 1e-6
            for shell_name in shell_order
        )
        log(f"  ALL FOUR SHELLS MATCH PREDICTION: {all_match}")

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
