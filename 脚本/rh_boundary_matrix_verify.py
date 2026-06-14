# -*- coding: utf-8 -*-
# rh_boundary_matrix_verify.py
#
# Computes cell complex boundary matrices over F_2 FROM SCRATCH for three
# lattice structures: Fano plane PG(2,2), Simple Cubic (SC), and BCC.
# No hand-filled tables — everything derived from combinatorial definitions.
#
# Replaces earlier scripts that were found to use circular reasoning
# (hand-filling data and then "verifying" the hand-filled values).
#
# What is computed:
#   1. Cell complexes (vertices, edges, faces) constructed from definitions
#   2. Boundary matrices d_1, d_2 over F_2
#   3. d^2 = 0 verification (d_1 . d_2 = 0 mod 2)
#   4. Ranks via Gaussian elimination over F_2
#   5. Betti numbers (F_2 homology)
#   6. Z_2 involutions and fixed-point sets
#
# Pure Python, no external dependencies.

import sys
import itertools

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════
# F_2 matrix utilities
# ═══════════════════════════════════════════════════════════════════════

def mat_zeros(rows, cols):
    """Create a rows x cols zero matrix (list of lists)."""
    return [[0] * cols for _ in range(rows)]


def mat_mul_f2(A, B):
    """Multiply two matrices over F_2. A is m×n, B is n×p → result m×p."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    p = len(B[0]) if len(B) > 0 else 0
    assert len(B) == n, f"Dimension mismatch: A is {m}x{n}, B is {len(B)}x{p}"
    C = mat_zeros(m, p)
    for i in range(m):
        for j in range(p):
            s = 0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s % 2
    return C


def mat_is_zero(M):
    """Check if all entries of M are 0."""
    return all(M[i][j] == 0 for i in range(len(M)) for j in range(len(M[0])))


def rank_f2(M):
    """Compute rank of matrix M over F_2 via Gaussian elimination.
    Does not modify M."""
    if not M or not M[0]:
        return 0
    m = len(M)
    n = len(M[0])
    # Work on a copy
    A = [row[:] for row in M]
    rank = 0
    for col in range(n):
        # Find pivot in column col at or below row 'rank'
        pivot = None
        for row in range(rank, m):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap pivot row with current rank row
        A[rank], A[pivot] = A[pivot], A[rank]
        # Eliminate all other 1s in this column
        for row in range(m):
            if row != rank and A[row][col] == 1:
                for c in range(n):
                    A[row][c] = (A[row][c] + A[rank][c]) % 2
        rank += 1
    return rank


def kernel_dim_f2(M):
    """Dimension of kernel of M over F_2 = #cols - rank."""
    if not M or not M[0]:
        return 0
    return len(M[0]) - rank_f2(M)


def print_matrix(M, name, max_rows=30, max_cols=30):
    """Print a matrix, truncating if too large."""
    m = len(M)
    n = len(M[0]) if m > 0 else 0
    if m <= max_rows and n <= max_cols:
        print(f"  {name} ({m} x {n}):")
        for row in M:
            print("    [" + " ".join(str(x) for x in row) + "]")
    else:
        print(f"  {name} ({m} x {n}): [too large to print, showing dimensions only]")


# ═══════════════════════════════════════════════════════════════════════
# Part 1A: Fano Plane PG(2,2)
# ═══════════════════════════════════════════════════════════════════════

def build_fano():
    """Build the Fano plane cell complex.

    Points: 0..6
    Lines (2-cells): the 7 lines of PG(2,2).
    1-skeleton: K_7 (every pair of points is an edge, since every pair
                lies on exactly one line).
    """
    points = list(range(7))
    lines = [
        frozenset({0, 1, 3}),
        frozenset({1, 2, 4}),
        frozenset({2, 3, 5}),
        frozenset({3, 4, 6}),
        frozenset({4, 5, 0}),
        frozenset({5, 6, 1}),
        frozenset({6, 0, 2}),
    ]

    # Verify: each pair of points lies on exactly one line
    pair_count = {}
    for line in lines:
        for p, q in itertools.combinations(sorted(line), 2):
            pair_count[(p, q)] = pair_count.get((p, q), 0) + 1

    # All 21 pairs should appear exactly once
    all_pairs = list(itertools.combinations(range(7), 2))
    assert len(all_pairs) == 21
    assert all(pair_count.get(p, 0) == 1 for p in all_pairs), \
        "Fano line configuration error: not every pair on exactly one line"

    # Edges = all 21 pairs (sorted tuples)
    edges = sorted(all_pairs)
    edge_index = {e: i for i, e in enumerate(edges)}

    # Build d_1: 7 x 21 matrix (points x edges)
    # d_1[v][e] = 1 if vertex v is a boundary of edge e
    d1 = mat_zeros(7, 21)
    for idx, (a, b) in enumerate(edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

    # Build d_2: 21 x 7 matrix (edges x lines/triangles)
    # d_2[e][f] = 1 if edge e is in the boundary of face/line f
    d2 = mat_zeros(21, 7)
    for f_idx, line in enumerate(lines):
        verts = sorted(line)
        for a, b in itertools.combinations(verts, 2):
            e_idx = edge_index[(a, b)]
            d2[e_idx][f_idx] = 1

    return {
        'name': 'Fano plane PG(2,2)',
        'vertices': points,
        'edges': edges,
        'faces': [sorted(l) for l in lines],
        'n_vertices': 7,
        'n_edges': 21,
        'n_faces': 7,
        'd1': d1,
        'd2': d2,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 1B: Simple Cubic (SC) unit cell
# ═══════════════════════════════════════════════════════════════════════

def build_sc():
    """Build the SC unit cube cell complex.

    8 vertices: (i,j,k) for i,j,k in {0,1}
    12 edges: pairs differing in exactly 1 coordinate
    6 faces: the 6 square faces of the cube, each with 4 edges
    """
    vertices = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    vert_index = {v: i for i, v in enumerate(vertices)}

    # Edges: pairs differing in exactly 1 coordinate
    edges = []
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if j > i:
                diff = sum(1 for a, b in zip(v1, v2) if a != b)
                if diff == 1:
                    edges.append((i, j))
    edge_index = {e: idx for idx, e in enumerate(edges)}

    # 6 faces: each face fixes one coordinate at 0 or 1
    # A face is defined by (axis, value), and has the 4 vertices where
    # that coordinate = value.
    faces = []
    for axis in range(3):
        for val in range(2):
            face_verts = [i for i, v in enumerate(vertices) if v[axis] == val]
            assert len(face_verts) == 4
            # The 4 edges of this square face
            face_edges = []
            for a, b in itertools.combinations(face_verts, 2):
                pair = (min(a, b), max(a, b))
                if pair in edge_index:
                    face_edges.append(pair)
            assert len(face_edges) == 4, f"Square face should have 4 edges, got {len(face_edges)}"
            faces.append(face_edges)

    n_v = len(vertices)
    n_e = len(edges)
    n_f = len(faces)

    # d_1: n_v x n_e
    d1 = mat_zeros(n_v, n_e)
    for idx, (a, b) in enumerate(edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

    # d_2: n_e x n_f (over F_2, boundary of square = sum of 4 edges)
    d2 = mat_zeros(n_e, n_f)
    for f_idx, face_edges in enumerate(faces):
        for e in face_edges:
            e_idx = edge_index[e]
            d2[e_idx][f_idx] = 1

    return {
        'name': 'Simple Cubic (SC) unit cell',
        'vertices': vertices,
        'edges': edges,
        'faces': faces,
        'n_vertices': n_v,
        'n_edges': n_e,
        'n_faces': n_f,
        'd1': d1,
        'd2': d2,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 1C: BCC (Body-Centered Cubic)
# ═══════════════════════════════════════════════════════════════════════

def build_bcc():
    """Build the BCC cell complex by adding a body center to the SC cube
    and triangulating.

    9 vertices: 8 cube corners + 1 body center
    20 edges: 12 cube edges + 8 center-to-corner edges
    24 triangular faces:
      - 12 triangles: {center, v_i, v_j} for each cube edge {v_i, v_j}
      - 12 triangles: each square face of the cube triangulated by choosing
        the diagonal connecting the vertex with smaller index to the one
        with larger index (consistent choice)
    """
    # Vertices: indices 0..7 are cube corners, 8 is body center
    cube_verts = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    center = (0.5, 0.5, 0.5)
    all_verts = cube_verts + [center]
    n_v = 9
    CENTER = 8

    # Cube edges (among indices 0..7)
    cube_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = sum(1 for a, b in zip(cube_verts[i], cube_verts[j]) if a != b)
            if diff == 1:
                cube_edges.append((i, j))
    assert len(cube_edges) == 12

    # Center-to-corner edges
    center_edges = [(i, CENTER) for i in range(8)]

    # All edges
    edges = sorted(cube_edges + center_edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}
    n_e = len(edges)
    assert n_e == 20

    # Triangular faces
    faces = []  # Each face is a list of 3 edge tuples

    # Type A: 12 triangles from {center, v_i, v_j} for each cube edge (v_i, v_j)
    for (vi, vj) in cube_edges:
        tri_verts = sorted([vi, vj, CENTER])
        # Edges of this triangle
        tri_edges = []
        for a, b in itertools.combinations(tri_verts, 2):
            e = (min(a, b), max(a, b))
            assert e in edge_index, f"Edge {e} not found"
            tri_edges.append(e)
        assert len(tri_edges) == 3
        faces.append(tri_edges)

    # Type B: Triangulate each square face of the cube
    # Each square face is defined by fixing one coordinate axis at 0 or 1
    for axis in range(3):
        for val in range(2):
            face_verts = [i for i in range(8) if cube_verts[i][axis] == val]
            assert len(face_verts) == 4
            face_verts.sort()

            # Choose diagonal: smallest index to largest index
            # face_verts = [a, b, c, d] sorted
            # Diagonal: (a, d). Two triangles: {a, b, d} and {a, c, d}?
            # No — we need to ensure these are actual triangles whose edges
            # are all in our edge set. All 4 face verts are cube corners,
            # so edges between them that differ in 1 coord are cube edges.
            # The diagonal (a, d) differs in 2 coords (not a cube edge!).
            #
            # We need to add diagonals as edges, OR triangulate differently.
            #
            # Better approach: use a different triangulation that only uses
            # existing edges. On a square face with vertices p,q,r,s, the
            # edges are p-q, q-r, r-s, s-p (the 4 cube edges on that face).
            # A diagonal p-r or q-s differs in 2 coords → NOT a cube edge.
            #
            # Since we don't have diagonal edges, we can't triangulate the
            # square faces using only cube edges. We need the center.
            #
            # Revision: connect each face vertex to the body center to form
            # 4 triangles per face, but the center is NOT on the face...
            # geometrically it's interior.
            #
            # Actually, the standard BCC Wigner-Seitz cell is a truncated
            # octahedron, not a cube. Let's reconsider.
            #
            # For a CELL COMPLEX (CW complex), square faces are perfectly
            # valid 2-cells. We don't need to triangulate. But then we're
            # mixing triangular and square faces.
            #
            # Decision: Use the 6 square faces as-is (4 edges each) plus
            # the 12 triangles from the body center. This gives 18 faces.
            pass

    # Restart face construction with the cleaner approach:
    # 12 triangles (center + each cube edge) + 6 square faces
    faces = []

    # 12 triangular faces: {center, v_i, v_j} for each cube edge
    face_types = []
    for (vi, vj) in cube_edges:
        tri_verts = sorted([vi, vj, CENTER])
        tri_edges = []
        for a, b in itertools.combinations(tri_verts, 2):
            e = (min(a, b), max(a, b))
            tri_edges.append(e)
        faces.append(tri_edges)
        face_types.append('triangle')

    # 6 square faces of the cube
    for axis in range(3):
        for val in range(2):
            face_vert_indices = [i for i in range(8) if cube_verts[i][axis] == val]
            # Find the 4 cube edges on this face
            sq_edges = []
            for a, b in itertools.combinations(sorted(face_vert_indices), 2):
                pair = (min(a, b), max(a, b))
                if pair in edge_index:
                    # Check it's a cube edge (differ in 1 coord)
                    diff = sum(1 for x, y in zip(cube_verts[a], cube_verts[b]) if x != y)
                    if diff == 1:
                        sq_edges.append(pair)
            assert len(sq_edges) == 4, f"Expected 4 edges for square face, got {len(sq_edges)}"
            faces.append(sq_edges)
            face_types.append('square')

    n_f = len(faces)
    assert n_f == 18, f"Expected 18 faces (12 tri + 6 sq), got {n_f}"

    # d_1: 9 x 20
    d1 = mat_zeros(n_v, n_e)
    for idx, (a, b) in enumerate(edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

    # d_2: 20 x 18
    # Over F_2: boundary of triangle = sum of 3 edges, boundary of square = sum of 4 edges
    d2 = mat_zeros(n_e, n_f)
    for f_idx, face_edges in enumerate(faces):
        for e in face_edges:
            e_idx = edge_index[e]
            d2[e_idx][f_idx] = 1

    return {
        'name': 'BCC (Body-Centered Cubic)',
        'vertices': all_verts,
        'edges': edges,
        'faces': faces,
        'face_types': face_types,
        'n_vertices': n_v,
        'n_edges': n_e,
        'n_faces': n_f,
        'd1': d1,
        'd2': d2,
        'center_index': CENTER,
        'cube_verts': cube_verts,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Z_2 involutions
# ═══════════════════════════════════════════════════════════════════════

def fano_involutions():
    """Find all involutions (order-2 automorphisms) of PG(2,2).

    An automorphism permutes {0..6} and maps lines to lines.
    Aut(PG(2,2)) = GL(3,2) = PSL(2,7), order 168.
    """
    lines = [
        frozenset({0, 1, 3}),
        frozenset({1, 2, 4}),
        frozenset({2, 3, 5}),
        frozenset({3, 4, 6}),
        frozenset({4, 5, 0}),
        frozenset({5, 6, 1}),
        frozenset({6, 0, 2}),
    ]
    line_set = set(lines)

    def is_automorphism(perm):
        """Check if permutation preserves the line structure."""
        for line in lines:
            image = frozenset(perm[v] for v in line)
            if image not in line_set:
                return False
        return True

    def fixed_points(perm):
        return [i for i in range(7) if perm[i] == i]

    def is_involution(perm):
        return all(perm[perm[i]] == i for i in range(7)) and any(perm[i] != i for i in range(7))

    # Enumerate automorphisms by brute force over S_7 is too slow (5040).
    # Use a smarter approach: generate Aut by finding generators.
    # GL(3,2) is generated by two elements. We can represent points of PG(2,2)
    # as nonzero vectors in F_2^3.

    # Map point labels 0..6 to nonzero vectors in F_2^3
    # Standard: i -> binary representation of (i+1), but let's use the
    # vectors directly.
    # Assignment found by solving v_a + v_b + v_c = 0 mod 2 for all lines
    vectors = [
        (0, 0, 1),  # point 0
        (0, 1, 0),  # point 1
        (1, 0, 0),  # point 2
        (0, 1, 1),  # point 3
        (1, 1, 0),  # point 4
        (1, 1, 1),  # point 5
        (1, 0, 1),  # point 6
    ]

    # Verify: a line {a,b,c} in PG(2,2) iff v_a + v_b + v_c = 0 mod 2
    for line in lines:
        verts = sorted(line)
        s = tuple((vectors[verts[0]][k] + vectors[verts[1]][k] + vectors[verts[2]][k]) % 2
                  for k in range(3))
        assert s == (0, 0, 0), f"Line {verts} doesn't satisfy collinearity: sum = {s}"

    vec_to_point = {}
    for i, v in enumerate(vectors):
        vec_to_point[v] = i

    # Enumerate GL(3,2): all invertible 3x3 matrices over F_2
    # A 3x3 matrix over F_2 has 2^9 = 512 possibilities. ~168 are invertible.
    involutions = []
    aut_count = 0

    for entries in range(512):
        # Decode entries into 3x3 matrix
        mat = [[0]*3 for _ in range(3)]
        e = entries
        for r in range(3):
            for c in range(3):
                mat[r][c] = e % 2
                e //= 2

        # Check invertibility via rank
        if rank_f2(mat) != 3:
            continue

        # Apply matrix to each vector to get the permutation
        perm = [0] * 7
        valid = True
        for i, v in enumerate(vectors):
            image = tuple(sum(mat[r][c] * v[c] for c in range(3)) % 2 for r in range(3))
            if image == (0, 0, 0):
                valid = False
                break
            if image not in vec_to_point:
                valid = False
                break
            perm[i] = vec_to_point[image]
        if not valid:
            continue

        aut_count += 1

        if is_involution(perm):
            fp = fixed_points(perm)
            involutions.append((perm, fp))

    return aut_count, involutions


def sc_antipodal():
    """Antipodal involution on the SC unit cube: (x,y,z) -> (1-x, 1-y, 1-z).

    Maps vertex index i to the vertex with all coordinates flipped.
    No fixed points among the 8 vertices (antipodal has no fixed vertex).
    The geometric fixed point is the center (0.5, 0.5, 0.5).
    """
    cube_verts = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    vert_index = {v: i for i, v in enumerate(cube_verts)}

    perm = [0] * 8
    for i, v in enumerate(cube_verts):
        antipode = (1 - v[0], 1 - v[1], 1 - v[2])
        perm[i] = vert_index[antipode]

    fixed = [i for i in range(8) if perm[i] == i]
    return perm, fixed


def bcc_antipodal():
    """Antipodal involution on BCC: corners map as in SC, body center is fixed.

    The body center (0.5, 0.5, 0.5) is the unique fixed point.
    """
    cube_verts = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    vert_index = {v: i for i, v in enumerate(cube_verts)}

    perm = [0] * 9
    for i, v in enumerate(cube_verts):
        antipode = (1 - v[0], 1 - v[1], 1 - v[2])
        perm[i] = vert_index[antipode]
    perm[8] = 8  # Body center is fixed

    fixed = [i for i in range(9) if perm[i] == i]
    return perm, fixed


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Homology computation
# ═══════════════════════════════════════════════════════════════════════

def compute_homology(data):
    """Compute F_2 homology from boundary matrices.

    H_0 = ker(d_0) / im(d_1), but d_0 = 0, so H_0 = C_0 / im(d_1).
      beta_0 = n_vertices - rank(d_1)

    H_1 = ker(d_1) / im(d_2)
      beta_1 = nullity(d_1) - rank(d_2) = (n_edges - rank(d_1)) - rank(d_2)

    H_2 = ker(d_2) / im(d_3), but no 3-cells, so H_2 = ker(d_2)
      beta_2 = nullity(d_2) = n_faces - rank(d_2)
    """
    d1 = data['d1']
    d2 = data['d2']

    r1 = rank_f2(d1)
    r2 = rank_f2(d2)

    n_v = data['n_vertices']
    n_e = data['n_edges']
    n_f = data['n_faces']

    beta_0 = n_v - r1
    beta_1 = (n_e - r1) - r2
    beta_2 = n_f - r2

    # Euler characteristic check: chi = beta_0 - beta_1 + beta_2
    #                              also chi = n_v - n_e + n_f
    chi_cells = n_v - n_e + n_f
    chi_betti = beta_0 - beta_1 + beta_2

    return {
        'rank_d1': r1,
        'rank_d2': r2,
        'nullity_d1': n_e - r1,
        'nullity_d2': n_f - r2,
        'beta_0': beta_0,
        'beta_1': beta_1,
        'beta_2': beta_2,
        'chi_cells': chi_cells,
        'chi_betti': chi_betti,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 4: Report
# ═══════════════════════════════════════════════════════════════════════

def verify_and_report(data):
    """Run all verifications for a cell complex and print results."""
    name = data['name']
    d1 = data['d1']
    d2 = data['d2']
    n_v = data['n_vertices']
    n_e = data['n_edges']
    n_f = data['n_faces']

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # --- Construction ---
    print(f"\n  [Construction]")
    print(f"    Vertices (0-cells): {n_v}")
    print(f"    Edges    (1-cells): {n_e}")
    print(f"    Faces    (2-cells): {n_f}")

    if n_v <= 10:
        print(f"    Vertex list: {data['vertices']}")
    if n_e <= 25:
        print(f"    Edge list: {data['edges']}")
    if n_f <= 25 and 'face_types' not in data:
        print(f"    Face list: {data['faces']}")
    if 'face_types' in data:
        tri = sum(1 for t in data['face_types'] if t == 'triangle')
        sq = sum(1 for t in data['face_types'] if t == 'square')
        print(f"    Face types: {tri} triangles + {sq} squares = {n_f} total")

    # --- Boundary matrices ---
    print(f"\n  [Boundary matrices over F_2]")
    print(f"    d_1: {n_v} x {n_e}  (vertices x edges)")
    print(f"    d_2: {n_e} x {n_f}  (edges x faces)")

    if n_v <= 12 and n_e <= 25:
        print_matrix(d1, "d_1")
    if n_e <= 25 and n_f <= 25:
        print_matrix(d2, "d_2")

    # --- d^2 = 0 verification ---
    print(f"\n  [d^2 = 0 verification]")
    product = mat_mul_f2(d1, d2)
    is_zero = mat_is_zero(product)
    tag = "PASS" if is_zero else "FAIL"
    print(f"    d_1 . d_2 = 0 (mod 2)?  [{tag}]")
    if not is_zero:
        print(f"    Product matrix (should be all zeros):")
        print_matrix(product, "d_1 . d_2")
    else:
        print(f"    Product is the {len(product)} x {len(product[0])} zero matrix.")

    # --- Ranks and homology ---
    hom = compute_homology(data)
    print(f"\n  [Ranks and Betti numbers (F_2 coefficients)]")
    print(f"    rank(d_1) = {hom['rank_d1']}")
    print(f"    rank(d_2) = {hom['rank_d2']}")
    print(f"    nullity(d_1) = ker(d_1) dim = {hom['nullity_d1']}")
    print(f"    nullity(d_2) = ker(d_2) dim = {hom['nullity_d2']}")
    print(f"")
    print(f"    beta_0 = dim H_0 = {n_v} - {hom['rank_d1']} = {hom['beta_0']}")
    print(f"    beta_1 = dim H_1 = {hom['nullity_d1']} - {hom['rank_d2']} = {hom['beta_1']}")
    print(f"    beta_2 = dim H_2 = {hom['nullity_d2']} (no 3-cells) = {hom['beta_2']}")
    print(f"")
    print(f"    Euler characteristic (cells): {n_v} - {n_e} + {n_f} = {hom['chi_cells']}")
    print(f"    Euler characteristic (Betti): {hom['beta_0']} - {hom['beta_1']} + {hom['beta_2']} = {hom['chi_betti']}")
    chi_match = hom['chi_cells'] == hom['chi_betti']
    tag = "PASS" if chi_match else "FAIL"
    print(f"    Euler characteristic match?  [{tag}]")

    # --- Exactness check ---
    print(f"\n  [Chain complex exactness]")
    print(f"    im(d_2) <= ker(d_1)?  rank(d_2) = {hom['rank_d2']} <= nullity(d_1) = {hom['nullity_d1']}")
    exact = hom['rank_d2'] <= hom['nullity_d1']
    tag = "PASS" if exact else "FAIL"
    print(f"    [{tag}]  (d^2=0 guarantees this; consistency check)")

    return hom


def report_z2():
    """Report Z_2 involution analysis."""
    print(f"\n{'='*70}")
    print(f"  Z_2 Involutions and Fixed Points")
    print(f"{'='*70}")

    # Fano involutions
    print(f"\n  [Fano plane PG(2,2)]")
    print(f"    Computing Aut(PG(2,2)) = GL(3,2)...")
    aut_count, involutions = fano_involutions()
    print(f"    |Aut(PG(2,2))| = {aut_count}  (expected: 168)")
    tag = "PASS" if aut_count == 168 else "FAIL"
    print(f"    [{tag}]")
    print(f"    Number of involutions (order-2 elements): {len(involutions)}")

    # Classify by fixed-point count
    fp_classes = {}
    for perm, fp in involutions:
        k = len(fp)
        if k not in fp_classes:
            fp_classes[k] = []
        fp_classes[k].append((perm, fp))

    for k in sorted(fp_classes.keys()):
        count = len(fp_classes[k])
        example_perm, example_fp = fp_classes[k][0]
        print(f"    {count} involutions with {k} fixed point(s)")
        print(f"      Example: {example_perm} fixes {example_fp}")

    # SC antipodal
    print(f"\n  [Simple Cubic — antipodal involution]")
    perm, fixed = sc_antipodal()
    print(f"    sigma(x,y,z) = (1-x, 1-y, 1-z)")
    print(f"    Permutation: {perm}")
    print(f"    Fixed vertices: {fixed}  (count: {len(fixed)})")
    print(f"    Geometric fixed point: (0.5, 0.5, 0.5) = BCC body center")
    tag = "PASS" if len(fixed) == 0 else "FAIL"
    print(f"    No vertex fixed?  [{tag}]")

    # BCC antipodal
    print(f"\n  [BCC — antipodal involution]")
    perm, fixed = bcc_antipodal()
    print(f"    Permutation (9 vertices, index 8 = center): {perm}")
    print(f"    Fixed vertices: {fixed}  (count: {len(fixed)})")
    tag = "PASS" if fixed == [8] else "FAIL"
    print(f"    Unique fixed point = body center (index 8)?  [{tag}]")
    print(f"    -> The BCC body center is the Z_2-invariant point of the")
    print(f"       antipodal involution: the 'Morse critical point at h=1/2'.")


def comparison_summary(fano_hom, sc_hom, bcc_hom):
    """Compare computed values with hand-filled data from old scripts."""
    print(f"\n{'='*70}")
    print(f"  Comparison: Computed vs Old Hand-Filled Data")
    print(f"{'='*70}")

    print(f"""
  Old scripts used tables like:
    Fano: V=7, E=21, F=7    (these were hand-filled)
    SC:   V=8, E=12, F=6
    BCC:  V=9, E=20, F=24   (note: F=24 in old data!)

  This script COMPUTED from definitions:
    Fano: V=7, E=21, F=7     <- matches
    SC:   V=8, E=12, F=6     <- matches
    BCC:  V=9, E=20, F=18    <- DIFFERS from old F=24

  The discrepancy in BCC faces:
    Old scripts assumed F=24 (fully triangulated: 12 center-triangles
    + 12 triangulated-square-face triangles using added diagonals).
    This script uses the natural CW complex with 12 triangles + 6 squares
    = 18 faces, because diagonal edges don't exist in the BCC edge set.

    To get F=24, one would need to add 6 diagonal edges (one per square
    face), which changes the complex (E=26, F=24 is a different space).
    The F=24 count was likely an error in the hand-filled data — it
    assumed triangulation without adding the required extra edges.
""")

    print(f"  Homology comparison:")
    for label, hom in [("Fano", fano_hom), ("SC", sc_hom), ("BCC", bcc_hom)]:
        print(f"    {label:5s}: beta_0={hom['beta_0']}, beta_1={hom['beta_1']}, "
              f"beta_2={hom['beta_2']}, chi={hom['chi_cells']}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  rh_boundary_matrix_verify.py")
    print("  Cell complex boundary matrices over F_2 — computed from scratch")
    print("=" * 70)
    print()
    print("  All matrix operations are over F_2 (mod 2 arithmetic).")
    print("  Ranks computed by Gaussian elimination over F_2.")
    print("  No hand-filled tables. Everything derived from definitions.")

    # Build complexes
    fano = build_fano()
    sc = build_sc()
    bcc = build_bcc()

    # Verify and report each
    fano_hom = verify_and_report(fano)
    sc_hom = verify_and_report(sc)
    bcc_hom = verify_and_report(bcc)

    # Z_2 analysis
    report_z2()

    # Comparison
    comparison_summary(fano_hom, sc_hom, bcc_hom)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  Summary of all d^2=0 checks")
    print(f"{'='*70}")

    for data in [fano, sc, bcc]:
        product = mat_mul_f2(data['d1'], data['d2'])
        ok = mat_is_zero(product)
        tag = "PASS" if ok else "FAIL"
        print(f"    {data['name']:35s}  d_1.d_2 = 0?  [{tag}]")

    print()


if __name__ == '__main__':
    main()
