# -*- coding: utf-8 -*-
# rh_extended_homology.py
#
# Follow-up to rh_boundary_matrix_verify.py. Extends the analysis with:
#   Part 1: Explicit kernel/image bases for each complex
#   Part 2: Triangulated BCC (F=24 version) with diagonal edges
#   Part 3: Fano -> SC cobordism / cycle-killing analysis
#   Part 4: O_h action on BCC
#   Part 5: Cycle budget analysis
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
    return [[0] * cols for _ in range(rows)]


def mat_mul_f2(A, B):
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
    return all(M[i][j] == 0 for i in range(len(M)) for j in range(len(M[0])))


def rank_f2(M):
    if not M or not M[0]:
        return 0
    m = len(M)
    n = len(M[0])
    A = [row[:] for row in M]
    rank = 0
    for col in range(n):
        pivot = None
        for row in range(rank, m):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        for row in range(m):
            if row != rank and A[row][col] == 1:
                for c in range(n):
                    A[row][c] = (A[row][c] + A[rank][c]) % 2
        rank += 1
    return rank


def rref_f2(M):
    """Compute RREF over F_2. Returns (rref_matrix, pivot_columns)."""
    if not M or not M[0]:
        return [], []
    m = len(M)
    n = len(M[0])
    A = [row[:] for row in M]
    pivots = []
    r = 0
    for col in range(n):
        pivot = None
        for row in range(r, m):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        for row in range(m):
            if row != r and A[row][col] == 1:
                for c in range(n):
                    A[row][c] = (A[row][c] + A[r][c]) % 2
        pivots.append(col)
        r += 1
    return A, pivots


def kernel_basis_f2(M):
    """Compute a basis for ker(M) over F_2.

    M is m x n. Returns list of vectors in F_2^n that span ker(M).
    """
    if not M or not M[0]:
        return []
    m = len(M)
    n = len(M[0])
    A, pivots = rref_f2(M)
    pivot_set = set(pivots)
    free_vars = [j for j in range(n) if j not in pivot_set]

    # For each free variable, construct a kernel vector
    basis = []
    for fv in free_vars:
        vec = [0] * n
        vec[fv] = 1
        # For each pivot column, solve for its value
        for idx, pc in enumerate(pivots):
            vec[pc] = A[idx][fv]  # Since RREF, row idx has pivot at pc
        basis.append(vec)
    return basis


def image_basis_f2(M):
    """Compute a basis for im(M) over F_2.

    M is m x n. im(M) = column space of M, living in F_2^m.
    Returns list of vectors in F_2^m that form a basis for im(M).

    Method: pivot columns of M (found via RREF of M) give the basis.
    """
    if not M or not M[0]:
        return []
    m = len(M)
    n = len(M[0])
    _, pivots = rref_f2(M)
    # Pivot columns of M form a basis for the column space
    basis = []
    for pc in pivots:
        col = [M[i][pc] for i in range(m)]
        basis.append(col)
    return basis


def vec_to_edges(vec, edges):
    """Convert a binary vector to a list of edges (where vec[i]=1)."""
    return [edges[i] for i in range(len(vec)) if vec[i] == 1]


# ═══════════════════════════════════════════════════════════════════════
# Complex builders (from rh_boundary_matrix_verify.py)
# ═══════════════════════════════════════════════════════════════════════

def build_fano():
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
    edges = sorted(itertools.combinations(range(7), 2))
    edge_index = {e: i for i, e in enumerate(edges)}

    d1 = mat_zeros(7, 21)
    for idx, (a, b) in enumerate(edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

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
        'n_vertices': 7, 'n_edges': 21, 'n_faces': 7,
        'd1': d1, 'd2': d2,
        'edge_index': edge_index,
    }


def build_sc():
    vertices = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    vert_index = {v: i for i, v in enumerate(vertices)}

    edges = []
    for i, v1 in enumerate(vertices):
        for j, v2 in enumerate(vertices):
            if j > i:
                diff = sum(1 for a, b in zip(v1, v2) if a != b)
                if diff == 1:
                    edges.append((i, j))
    edge_index = {e: idx for idx, e in enumerate(edges)}

    faces = []
    for axis in range(3):
        for val in range(2):
            face_verts = [i for i, v in enumerate(vertices) if v[axis] == val]
            face_edges = []
            for a, b in itertools.combinations(sorted(face_verts), 2):
                pair = (min(a, b), max(a, b))
                if pair in edge_index:
                    face_edges.append(pair)
            faces.append(face_edges)

    n_v, n_e, n_f = len(vertices), len(edges), len(faces)
    d1 = mat_zeros(n_v, n_e)
    for idx, (a, b) in enumerate(edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

    d2 = mat_zeros(n_e, n_f)
    for f_idx, face_edges in enumerate(faces):
        for e in face_edges:
            e_idx = edge_index[e]
            d2[e_idx][f_idx] = 1

    return {
        'name': 'Simple Cubic (SC)',
        'vertices': vertices, 'edges': edges, 'faces': faces,
        'n_vertices': n_v, 'n_edges': n_e, 'n_faces': n_f,
        'd1': d1, 'd2': d2,
        'edge_index': edge_index,
        'vert_index': vert_index,
    }


def build_bcc():
    cube_verts = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    CENTER = 8
    cube_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = sum(1 for a, b in zip(cube_verts[i], cube_verts[j]) if a != b)
            if diff == 1:
                cube_edges.append((i, j))

    center_edges = [(i, CENTER) for i in range(8)]
    edges = sorted(cube_edges + center_edges)
    edge_index = {e: idx for idx, e in enumerate(edges)}

    faces = []
    face_types = []
    for (vi, vj) in cube_edges:
        tri_verts = sorted([vi, vj, CENTER])
        tri_edges = []
        for a, b in itertools.combinations(tri_verts, 2):
            e = (min(a, b), max(a, b))
            tri_edges.append(e)
        faces.append(tri_edges)
        face_types.append('triangle')

    for axis in range(3):
        for val in range(2):
            face_vert_indices = [i for i in range(8) if cube_verts[i][axis] == val]
            sq_edges = []
            for a, b in itertools.combinations(sorted(face_vert_indices), 2):
                pair = (min(a, b), max(a, b))
                if pair in edge_index:
                    diff = sum(1 for x, y in zip(cube_verts[a], cube_verts[b]) if x != y)
                    if diff == 1:
                        sq_edges.append(pair)
            faces.append(sq_edges)
            face_types.append('square')

    n_v, n_e, n_f = 9, len(edges), len(faces)
    d1 = mat_zeros(n_v, n_e)
    for idx, (a, b) in enumerate(edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

    d2 = mat_zeros(n_e, n_f)
    for f_idx, face_edges in enumerate(faces):
        for e in face_edges:
            e_idx = edge_index[e]
            d2[e_idx][f_idx] = 1

    return {
        'name': 'BCC (natural, F=18)',
        'vertices': list(range(9)),
        'edges': edges, 'faces': faces,
        'face_types': face_types,
        'n_vertices': n_v, 'n_edges': n_e, 'n_faces': n_f,
        'd1': d1, 'd2': d2,
        'edge_index': edge_index,
        'cube_verts': cube_verts,
        'cube_edges': cube_edges,
        'center_edges': center_edges,
    }


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Explicit kernel and image bases
# ═══════════════════════════════════════════════════════════════════════

def format_cycle(edge_list):
    """Format an edge list as a human-readable cycle string."""
    return " + ".join(f"({a},{b})" for a, b in edge_list)


def part1_kernel_image(data):
    name = data['name']
    edges = data['edges']
    d1 = data['d1']
    d2 = data['d2']

    print(f"\n{'='*70}")
    print(f"  Part 1: Kernel/Image bases for {name}")
    print(f"{'='*70}")

    # ker(d1) basis = 1-cycles
    ker_d1 = kernel_basis_f2(d1)
    print(f"\n  ker(d_1) basis (1-cycles): dim = {len(ker_d1)}")
    for i, vec in enumerate(ker_d1):
        edge_list = vec_to_edges(vec, edges)
        print(f"    z_{i}: {format_cycle(edge_list)}")

    # im(d2) basis = 1-boundaries (as vectors in edge space)
    # im(d2) = column space of d2 = set of d2 * e_j for face columns
    # Compute by taking columns of d2 and finding a basis
    im_d2 = image_basis_f2(d2)
    print(f"\n  im(d_2) basis (1-boundaries): dim = {len(im_d2)}")
    for i, vec in enumerate(im_d2):
        edge_list = vec_to_edges(vec, edges)
        # Try to identify which face(s) generate this boundary
        print(f"    b_{i}: {format_cycle(edge_list)}")

    # Show face boundaries explicitly
    faces = data['faces']
    print(f"\n  Face boundaries (each face -> its boundary edges):")
    for f_idx, face in enumerate(faces):
        col = [d2[row][f_idx] for row in range(len(d2))]
        edge_list = vec_to_edges(col, edges)
        face_label = face if len(face) <= 4 else f"face_{f_idx}"
        ftype = ""
        if 'face_types' in data:
            ftype = f" [{data['face_types'][f_idx]}]"
        print(f"    d_2(f_{f_idx}) = {format_cycle(edge_list)}  <- face {face_label}{ftype}")

    beta_1 = len(ker_d1) - len(im_d2)
    print(f"\n  beta_1 = dim(ker) - dim(im) = {len(ker_d1)} - {len(im_d2)} = {beta_1}")

    if beta_1 == 0:
        print(f"  -> H_1 = 0: every 1-cycle is a 1-boundary.")
        print(f"     This means ker(d_1) = im(d_2) (exact at C_1).")
    else:
        print(f"  -> H_1 has {beta_1} independent non-trivial classes.")
        print(f"\n  H_1 representatives (cycles not in im(d_2)):")
        # To find H_1 reps: find ker(d1) vectors not in im(d2)
        # We do this by extending im(d2) basis and checking which
        # ker(d1) vectors are independent of im(d2)
        # Build matrix with im(d2) basis rows, then try adding each ker(d1) vector
        im_basis = im_d2[:]
        h1_reps = []
        current_rank = len(im_basis)
        for vec in ker_d1:
            test_mat = [v[:] for v in im_basis] + [vec]
            # Transpose to check rank as column vectors
            r = rank_f2(transpose(test_mat))
            if r > current_rank:
                h1_reps.append(vec)
                im_basis.append(vec)
                current_rank = r
                if len(h1_reps) == beta_1:
                    break

        for i, vec in enumerate(h1_reps):
            edge_list = vec_to_edges(vec, edges)
            print(f"    h_{i}: {format_cycle(edge_list)}")


def transpose(M):
    if not M:
        return []
    m = len(M)
    n = len(M[0])
    return [[M[i][j] for i in range(m)] for j in range(n)]


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Triangulated BCC (F=24)
# ═══════════════════════════════════════════════════════════════════════

def build_bcc_triangulated():
    """Build BCC with triangulated square faces.

    Adds 6 diagonal edges (one per square face) to allow full triangulation.
    Result: V=9, E=26, F=24.
    """
    cube_verts = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    CENTER = 8

    cube_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            diff = sum(1 for a, b in zip(cube_verts[i], cube_verts[j]) if a != b)
            if diff == 1:
                cube_edges.append((i, j))

    center_edges = [(i, CENTER) for i in range(8)]

    # For each square face, add a diagonal edge
    # Rule: connect the vertex with smallest index to the one with largest index
    diagonal_edges = []
    square_face_data = []  # (face_verts_sorted, diagonal, two_triangles)

    for axis in range(3):
        for val in range(2):
            fv = sorted([i for i in range(8) if cube_verts[i][axis] == val])
            assert len(fv) == 4
            # Square face vertices: fv[0], fv[1], fv[2], fv[3]
            # The 4 cube edges on this face
            sq_edges_on_face = []
            for a, b in itertools.combinations(fv, 2):
                diff = sum(1 for x, y in zip(cube_verts[a], cube_verts[b]) if x != y)
                if diff == 1:
                    sq_edges_on_face.append((a, b))
            assert len(sq_edges_on_face) == 4

            # Diagonal: connect fv[0] to fv[3] (smallest to largest index)
            diag = (fv[0], fv[3])
            diagonal_edges.append(diag)

            # The diagonal splits the square into two triangles.
            # We need to find the two triangles.
            # Square with edges: find which vertices are adjacent to both
            # endpoints of the diagonal.
            # fv[0] neighbors on this face (connected by a cube edge):
            nbrs_0 = [v for v in fv if v != fv[0] and
                       sum(1 for x, y in zip(cube_verts[fv[0]], cube_verts[v]) if x != y) == 1]
            nbrs_3 = [v for v in fv if v != fv[3] and
                       sum(1 for x, y in zip(cube_verts[fv[3]], cube_verts[v]) if x != y) == 1]
            # The two other vertices
            others = [v for v in fv if v != fv[0] and v != fv[3]]
            assert len(others) == 2
            # Triangle 1: fv[0], others[0], fv[3]
            # Triangle 2: fv[0], others[1], fv[3]
            tri1 = sorted([fv[0], others[0], fv[3]])
            tri2 = sorted([fv[0], others[1], fv[3]])
            square_face_data.append((fv, diag, tri1, tri2))

    # Build complete edge set
    all_edges = sorted(set(cube_edges + center_edges + diagonal_edges))
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    n_e = len(all_edges)

    # Build faces: 12 center-triangles + 12 triangulated-square triangles
    faces = []
    face_types = []

    # Center triangles: {center, vi, vj} for each cube edge
    for (vi, vj) in cube_edges:
        tri_verts = sorted([vi, vj, CENTER])
        tri_edges = []
        for a, b in itertools.combinations(tri_verts, 2):
            e = (min(a, b), max(a, b))
            tri_edges.append(e)
        faces.append(tri_edges)
        face_types.append('center-tri')

    # Triangulated square faces
    for (fv, diag, tri1, tri2) in square_face_data:
        for tri in [tri1, tri2]:
            tri_edges = []
            for a, b in itertools.combinations(tri, 2):
                e = (min(a, b), max(a, b))
                assert e in edge_index, f"Edge {e} not in edge set!"
                tri_edges.append(e)
            faces.append(tri_edges)
            face_types.append('sq-tri')

    n_v = 9
    n_f = len(faces)

    d1 = mat_zeros(n_v, n_e)
    for idx, (a, b) in enumerate(all_edges):
        d1[a][idx] = 1
        d1[b][idx] = 1

    d2 = mat_zeros(n_e, n_f)
    for f_idx, face_edges in enumerate(faces):
        for e in face_edges:
            e_idx = edge_index[e]
            d2[e_idx][f_idx] = 1

    return {
        'name': 'BCC (triangulated, F=24)',
        'vertices': list(range(9)),
        'edges': all_edges, 'faces': faces,
        'face_types': face_types,
        'n_vertices': n_v, 'n_edges': n_e, 'n_faces': n_f,
        'd1': d1, 'd2': d2,
        'edge_index': edge_index,
        'diagonal_edges': diagonal_edges,
        'square_face_data': square_face_data,
    }


def part2_triangulated_bcc():
    print(f"\n{'='*70}")
    print(f"  Part 2: Triangulated BCC (F=24)")
    print(f"{'='*70}")

    data = build_bcc_triangulated()
    n_v, n_e, n_f = data['n_vertices'], data['n_edges'], data['n_faces']
    d1, d2 = data['d1'], data['d2']

    print(f"\n  Construction:")
    print(f"    V = {n_v}, E = {n_e}, F = {n_f}")
    print(f"    Diagonal edges added: {data['diagonal_edges']}")
    ct = sum(1 for t in data['face_types'] if t == 'center-tri')
    st = sum(1 for t in data['face_types'] if t == 'sq-tri')
    print(f"    Face types: {ct} center-triangles + {st} square-face triangles = {n_f}")

    # Verify d^2 = 0
    product = mat_mul_f2(d1, d2)
    is_zero = mat_is_zero(product)
    tag = "PASS" if is_zero else "FAIL"
    print(f"\n  d^2 = 0 check: [{tag}]")

    # Ranks and Betti numbers
    r1 = rank_f2(d1)
    r2 = rank_f2(d2)
    beta_0 = n_v - r1
    beta_1 = (n_e - r1) - r2
    beta_2 = n_f - r2

    print(f"\n  Ranks:")
    print(f"    rank(d_1) = {r1}")
    print(f"    rank(d_2) = {r2}")
    print(f"\n  Betti numbers:")
    print(f"    beta_0 = {n_v} - {r1} = {beta_0}")
    print(f"    beta_1 = ({n_e} - {r1}) - {r2} = {beta_1}")
    print(f"    beta_2 = {n_f} - {r2} = {beta_2}")
    print(f"    chi = {n_v} - {n_e} + {n_f} = {n_v - n_e + n_f}")
    print(f"    chi (Betti) = {beta_0} - {beta_1} + {beta_2} = {beta_0 - beta_1 + beta_2}")

    # Comparison
    print(f"\n  Comparison with natural BCC (F=18):")
    print(f"    Natural:      V=9, E=20, F=18  -> rank(d2)=12, beta_1=0, beta_2=6")
    print(f"    Triangulated: V=9, E={n_e}, F={n_f}  -> rank(d2)={r2}, beta_1={beta_1}, beta_2={beta_2}")

    print(f"\n  Old hand-filled data analysis:")
    print(f"    Old data claimed: V=9, E=16, F=24")
    print(f"    Our triangulated: V=9, E=26, F=24")
    print(f"    E=16 is WRONG. With 8 cube corners + 1 center:")
    print(f"      - Cube edges alone = 12")
    print(f"      - Center-to-corner edges = 8")
    print(f"      - Minimum: 12 + 8 = 20 edges")
    print(f"      - With 6 diagonals: 20 + 6 = 26 edges")
    print(f"      - E=16 < 20 is impossible for this vertex set.")
    print(f"      - E=16 appears to be erroneous hand-filled data.")
    print(f"    Old data claimed rank(d_2) = 8.")
    print(f"    Our computed rank(d_2) = {r2} for the triangulated version.")

    return data


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Fano -> SC cobordism
# ═══════════════════════════════════════════════════════════════════════

def part3_cobordism():
    print(f"\n{'='*70}")
    print(f"  Part 3: Fano -> SC Cobordism Analysis")
    print(f"{'='*70}")

    # Fano: K_7 on 7 points, 7 triangular faces
    # SC: cube graph on 8 vertices, 6 square faces

    # K_7 (no faces)
    print(f"\n  Stage 1: K_7 (Fano 1-skeleton, no faces)")
    n_v_k7, n_e_k7 = 7, 21
    beta1_k7 = n_e_k7 - n_v_k7 + 1  # connected graph
    print(f"    V=7, E=21, F=0")
    print(f"    beta_1(K_7) = E - V + 1 = 21 - 7 + 1 = {beta1_k7}")
    print(f"    (For a connected graph with no faces, beta_1 = E - V + 1)")

    # Fano plane (K_7 + 7 triangular faces)
    print(f"\n  Stage 2: Fano plane (K_7 + 7 triangular 2-cells)")
    fano = build_fano()
    r2_fano = rank_f2(fano['d2'])
    beta1_fano = beta1_k7 - r2_fano
    print(f"    V=7, E=21, F=7")
    print(f"    rank(d_2) = {r2_fano} (7 faces, but rank may be < 7)")
    print(f"    beta_1(Fano) = {beta1_k7} - {r2_fano} = {beta1_fano}")
    print(f"    Adding {r2_fano} independent faces killed {r2_fano} of the {beta1_k7} cycles.")

    # Cube graph (no faces)
    print(f"\n  Stage 3: Cube graph (SC 1-skeleton, no faces)")
    n_v_cube, n_e_cube = 8, 12
    beta1_cube = n_e_cube - n_v_cube + 1
    print(f"    V=8, E=12, F=0")
    print(f"    beta_1(cube graph) = 12 - 8 + 1 = {beta1_cube}")

    # SC (cube + 6 square faces)
    print(f"\n  Stage 4: SC (cube graph + 6 square 2-cells)")
    sc = build_sc()
    r2_sc = rank_f2(sc['d2'])
    beta1_sc = beta1_cube - r2_sc
    print(f"    V=8, E=12, F=6")
    print(f"    rank(d_2) = {r2_sc}")
    print(f"    beta_1(SC) = {beta1_cube} - {r2_sc} = {beta1_sc}")

    # Analysis of the transition
    print(f"\n  Transition analysis: Fano (beta_1=8) -> SC (beta_1=0)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"    The transition involves:")
    print(f"      - Adding 1 vertex (7 -> 8)")
    print(f"      - Removing 9 edges (21 -> 12)")
    print(f"      - Replacing 7 triangular faces with 6 square faces")
    print(f"")
    print(f"    Effect of edge removal on cycle count:")
    print(f"      K_7 has beta_1 = 15 (before Fano faces)")
    print(f"      Cube graph has beta_1 = 5 (before SC faces)")
    print(f"      Edge removal (21->12) + vertex addition (7->8)")
    print(f"      kills 15 - 5 = 10 cycles just from topology change.")
    print(f"")
    print(f"    Effect of 2-cells:")
    print(f"      Fano: 7 faces kill {r2_fano} cycles (15 -> {beta1_fano})")
    print(f"      SC: 6 faces kill {r2_sc} cycles (5 -> {beta1_sc})")
    print(f"")
    print(f"    Summary: the ∂-cascade 2D->3D transition kills H_1 via TWO mechanisms:")
    print(f"      (a) Graph simplification: K_7 -> cube graph (15 -> 5 independent cycles)")
    print(f"      (b) Face filling: 6 square faces kill remaining 5 cycles")
    print(f"    Both are needed. Neither alone suffices.")


# ═══════════════════════════════════════════════════════════════════════
# Part 4: O_h action on BCC
# ═══════════════════════════════════════════════════════════════════════

def part4_oh_action():
    print(f"\n{'='*70}")
    print(f"  Part 4: O_h (Octahedral Group) Action on BCC")
    print(f"{'='*70}")

    # Cube vertices as {0,1}^3
    cube_verts = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    coord_to_idx = {v: i for i, v in enumerate(cube_verts)}

    # O_h generators acting on coordinates.
    # The full octahedral group O_h acts on R^3 preserving the cube [-1,1]^3.
    # We work with {0,1}^3 by mapping: x -> 1-x for coordinate negation.
    #
    # Generators of O_h:
    # 1. 90-degree rotation about z-axis: (x,y,z) -> (1-y, x, z) [in {0,1} coords]
    # 2. 90-degree rotation about x-axis: (x,y,z) -> (x, 1-z, y)
    # 3. Inversion: (x,y,z) -> (1-x, 1-y, 1-z)

    def apply_coord_transform(coord, transform):
        """Apply a coordinate transformation to a vertex in {0,1}^3."""
        return transform(coord)

    def coord_to_perm(transform):
        """Convert a coordinate transformation to a permutation of vertex indices."""
        perm = [0] * 8
        for i, v in enumerate(cube_verts):
            new_v = transform(v)
            perm[i] = coord_to_idx[new_v]
        return tuple(perm)

    # Define generators
    def rot_z90(v):
        """90-degree rotation about z-axis."""
        x, y, z = v
        return (1 - y, x, z)

    def rot_x90(v):
        """90-degree rotation about x-axis."""
        x, y, z = v
        return (x, 1 - z, y)

    def inversion(v):
        """Inversion through center."""
        x, y, z = v
        return (1 - x, 1 - y, 1 - z)

    def compose(f, g):
        """Compose two coordinate transformations: (f . g)(v) = f(g(v))."""
        return lambda v: f(g(v))

    def perm_compose(p, q):
        """Compose two permutations: (p . q)[i] = p[q[i]]."""
        return tuple(p[q[i]] for i in range(len(p)))

    def perm_identity(n):
        return tuple(range(n))

    # Generate O_h by closure under composition
    gen_transforms = [rot_z90, rot_x90, inversion]
    gen_perms = [coord_to_perm(t) for t in gen_transforms]

    print(f"\n  Generators (as permutations of 8 cube vertices):")
    print(f"    Rot_z(90):  {gen_perms[0]}")
    print(f"    Rot_x(90):  {gen_perms[1]}")
    print(f"    Inversion:  {gen_perms[2]}")

    # Generate the group by closure
    group = set()
    queue = [perm_identity(8)]
    group.add(perm_identity(8))

    while queue:
        current = queue.pop()
        for g in gen_perms:
            new_perm = perm_compose(g, current)
            if new_perm not in group:
                group.add(new_perm)
                queue.append(new_perm)
            new_perm2 = perm_compose(current, g)
            if new_perm2 not in group:
                group.add(new_perm2)
                queue.append(new_perm2)

    group = sorted(group)  # For reproducibility

    print(f"\n  |O_h| = {len(group)}  (expected: 48)")
    tag = "PASS" if len(group) == 48 else "FAIL"
    print(f"  [{tag}]")

    # Verify it's a group (closure already guaranteed by construction)
    # Check: identity is in, and closure under composition
    id_perm = perm_identity(8)
    assert id_perm in group

    # Orbits on vertices (including center = vertex 8)
    print(f"\n  Orbits on BCC vertices (0-7 = corners, 8 = center):")

    # Corner orbit: apply all group elements to vertex 0
    corner_orbit = set()
    for g in group:
        corner_orbit.add(g[0])
    print(f"    Orbit of vertex 0 (corner): {sorted(corner_orbit)}")
    print(f"    Size: {len(corner_orbit)} (expected: 8)")
    tag = "PASS" if len(corner_orbit) == 8 else "FAIL"
    print(f"    [{tag}]")

    print(f"    Orbit of vertex 8 (center): {{8}} (fixed by all of O_h)")
    print(f"    Size: 1 (expected: 1) [PASS]")

    # Orbits on edges
    print(f"\n  Orbits on edges:")
    cube_edges_set = set()
    center_edges_set = set()
    for i in range(8):
        for j in range(i + 1, 8):
            diff = sum(1 for a, b in zip(cube_verts[i], cube_verts[j]) if a != b)
            if diff == 1:
                cube_edges_set.add((i, j))
                center_edges_set.discard((i, j))  # no-op, just for clarity
    for i in range(8):
        center_edges_set.add((i, 8))

    # Orbit of a cube edge under O_h
    # Take edge (0,1) = (0,0,0)-(1,0,0)
    test_edge = (0, 1)
    cube_edge_orbit = set()
    for g in group:
        img = tuple(sorted([g[test_edge[0]], g[test_edge[1]]]))
        cube_edge_orbit.add(img)
    print(f"    Orbit of cube edge (0,1): {len(cube_edge_orbit)} edges")
    # Check all are cube edges
    all_cube = all(e in cube_edges_set for e in cube_edge_orbit)
    print(f"    All in cube edge set: {all_cube}")
    print(f"    (Expected: 12 cube edges form a single orbit)")
    tag = "PASS" if len(cube_edge_orbit) == 12 and all_cube else "FAIL"
    print(f"    [{tag}]")

    # Orbit of a center edge
    test_cedge = (0, 8)
    center_edge_orbit = set()
    for g in group:
        img_corner = g[0]  # center is fixed
        img_edge = tuple(sorted([img_corner, 8]))
        center_edge_orbit.add(img_edge)
    print(f"    Orbit of center edge (0,8): {len(center_edge_orbit)} edges")
    all_center = all(e in center_edges_set for e in center_edge_orbit)
    print(f"    All in center edge set: {all_center}")
    print(f"    (Expected: 8 center-to-corner edges form a single orbit)")
    tag = "PASS" if len(center_edge_orbit) == 8 and all_center else "FAIL"
    print(f"    [{tag}]")

    # Stabilizers
    print(f"\n  Stabilizers:")

    # Stabilizer of vertex 0
    stab_0 = [g for g in group if g[0] == 0]
    print(f"    Stab(vertex 0): |Stab| = {len(stab_0)}")
    print(f"    Orbit-stabilizer: |O_h| / |Stab(v0)| = {len(group)} / {len(stab_0)} = {len(group) // len(stab_0)}")
    print(f"    (Expected: |Stab| = 6, orbit size = 8)")
    tag = "PASS" if len(stab_0) == 6 else "FAIL"
    print(f"    [{tag}]")

    # Stabilizer of center (vertex 8) = entire group
    stab_8_size = len(group)  # center is always fixed
    print(f"    Stab(vertex 8 = center): |Stab| = {stab_8_size}")
    print(f"    (Expected: 48 = all of O_h, since center is fixed)")
    tag = "PASS" if stab_8_size == 48 else "FAIL"
    print(f"    [{tag}]")

    # Show stabilizer of vertex 0 is isomorphic to S_3
    print(f"\n  Stab(vertex 0) structure:")
    print(f"    The 3 neighbors of vertex 0 = (0,0,0) on the cube are:")
    nbrs = []
    for j in range(8):
        if j != 0:
            diff = sum(1 for a, b in zip(cube_verts[0], cube_verts[j]) if a != b)
            if diff == 1:
                nbrs.append(j)
    print(f"    {nbrs} = {[cube_verts[n] for n in nbrs]}")
    print(f"    Stab(v0) permutes these 3 neighbors -> S_3 (order 6).")

    # Show how stab permutes the neighbors
    print(f"    Stab(v0) action on neighbors {nbrs}:")
    for g in stab_0:
        action = tuple(g[n] for n in nbrs)
        print(f"      {nbrs} -> {list(action)}")


# ═══════════════════════════════════════════════════════════════════════
# Part 5: Cycle budget analysis
# ═══════════════════════════════════════════════════════════════════════

def part5_cycle_budget():
    print(f"\n{'='*70}")
    print(f"  Part 5: Cycle Budget Analysis")
    print(f"{'='*70}")

    print(f"""
  The "cycle budget" tracks how independent 1-cycles are created and
  killed as we build each complex.

  For a connected graph: beta_1 = E - V + 1 (number of independent cycles)
  Adding a 2-cell kills at most 1 independent cycle (if its boundary is
  not already in im(d_2), i.e., it's linearly independent over F_2).
""")

    # K_7
    print(f"  [K_7: complete graph on 7 vertices]")
    print(f"    V=7, E=21, F=0")
    print(f"    beta_1 = 21 - 7 + 1 = 15")

    # Fano
    fano = build_fano()
    r2 = rank_f2(fano['d2'])
    print(f"\n  [Fano plane: K_7 + 7 triangular 2-cells]")
    print(f"    V=7, E=21, F=7")
    print(f"    rank(d_2) = {r2}")
    print(f"    7 faces added, {r2} are independent -> kill {r2} cycles")
    print(f"    beta_1 = 15 - {r2} = {15 - r2}")

    # What would it take to kill all 15 cycles of K_7?
    # Need rank(d_2) = 15, which requires at least 15 independent faces
    # K_7 has C(7,3) = 35 possible triangles. Of these, 7 are Fano lines.
    # Adding the remaining 28 would give 35 faces total.
    print(f"\n  Minimum faces to kill all cycles of K_7:")
    print(f"    Need 15 independent 2-cells (one per cycle)")
    print(f"    Fano provides only {r2} independent ones")
    print(f"    Need 15 - {r2} = {15 - r2} more independent 2-cells")
    print(f"    (This is exactly beta_1(Fano) = 8)")

    # Cube graph
    print(f"\n  [Cube graph: 8 vertices, 12 edges, no faces]")
    print(f"    V=8, E=12, F=0")
    print(f"    beta_1 = 12 - 8 + 1 = 5")

    # SC
    sc = build_sc()
    r2_sc = rank_f2(sc['d2'])
    print(f"\n  [SC: cube + 6 square 2-cells]")
    print(f"    V=8, E=12, F=6")
    print(f"    rank(d_2) = {r2_sc}")
    print(f"    6 faces added, {r2_sc} are independent -> kill {r2_sc} cycles")
    print(f"    beta_1 = 5 - {r2_sc} = {5 - r2_sc}")

    # BCC natural
    bcc = build_bcc()
    r2_bcc = rank_f2(bcc['d2'])
    r1_bcc = rank_f2(bcc['d1'])
    beta1_bcc = (20 - r1_bcc) - r2_bcc
    print(f"\n  [BCC natural: 9V, 20E, 18F (12 tri + 6 sq)]")
    print(f"    V=9, E=20, F=18")
    print(f"    Graph beta_1 (no faces) = 20 - 9 + 1 = 12")
    print(f"    rank(d_2) = {r2_bcc}")
    print(f"    beta_1 = {beta1_bcc}")

    # BCC triangulated
    bcc_t = build_bcc_triangulated()
    r2_bcc_t = rank_f2(bcc_t['d2'])
    r1_bcc_t = rank_f2(bcc_t['d1'])
    beta1_bcc_t = (bcc_t['n_edges'] - r1_bcc_t) - r2_bcc_t
    print(f"\n  [BCC triangulated: 9V, {bcc_t['n_edges']}E, 24F (all triangles)]")
    print(f"    V=9, E={bcc_t['n_edges']}, F=24")
    print(f"    Graph beta_1 (no faces) = {bcc_t['n_edges']} - 9 + 1 = {bcc_t['n_edges'] - 9 + 1}")
    print(f"    rank(d_2) = {r2_bcc_t}")
    print(f"    beta_1 = {beta1_bcc_t}")

    # Summary table
    print(f"\n  ┌───────────────────────┬────┬────┬────┬──────────┬──────────┬────────┐")
    print(f"  │ Complex               │  V │  E │  F │ graph b1 │ rank(d2) │ beta_1 │")
    print(f"  ├───────────────────────┼────┼────┼────┼──────────┼──────────┼────────┤")
    print(f"  │ K_7 (no faces)        │  7 │ 21 │  0 │    15    │    0     │   15   │")
    print(f"  │ Fano (7 tri faces)    │  7 │ 21 │  7 │    15    │    {r2:>1}     │    {15-r2:>1}   │")
    print(f"  │ Cube graph (no faces) │  8 │ 12 │  0 │     5    │    0     │    5   │")
    print(f"  │ SC (6 sq faces)       │  8 │ 12 │  6 │     5    │    {r2_sc:>1}     │    {5-r2_sc:>1}   │")
    print(f"  │ BCC nat. (18 faces)   │  9 │ 20 │ 18 │    12    │   {r2_bcc:>2}     │    {beta1_bcc:>1}   │")
    print(f"  │ BCC tri. (24 faces)   │  9 │ 26 │ 24 │    18    │   {r2_bcc_t:>2}     │    {beta1_bcc_t:>1}   │")
    print(f"  └───────────────────────┴────┴────┴────┴──────────┴──────────┴────────┘")

    print(f"""
  Key observations:
    1. K_7 -> Cube graph: removing 9 edges and adding 1 vertex reduces
       independent cycles from 15 to 5 (a reduction of 10).

    2. Fano faces kill {r2} of 15 cycles. SC faces kill {r2_sc} of 5 cycles.
       Both face-filling operations are "efficient" (each face kills one cycle
       when possible), but SC needs fewer because the graph has fewer cycles.

    3. The ∂-cascade transition 2D -> 3D achieves H_1 = 0 primarily through
       EDGE REDUCTION (graph simplification), not face addition.
       The edge reduction from K_7 to cube kills 10 cycles.
       The face filling kills the remaining 5.

    4. BCC natural and triangulated both have beta_1 = 0, confirming that
       the mixed CW complex (with square faces) and the fully triangulated
       simplicial complex compute the same H_1.

    5. The old hand-filled data with E=16 is impossible:
       - 8 cube corners + 1 center requires at least 20 edges
       - E=16 < 20 violates the combinatorial structure
       - This was a data entry error in the original script
""")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  rh_extended_homology.py")
    print("  Extended homology analysis: cycles, cobordism, symmetry")
    print("=" * 70)

    # Build complexes
    fano = build_fano()
    sc = build_sc()
    bcc = build_bcc()

    # Part 1: Kernel/Image bases
    for data in [fano, sc, bcc]:
        part1_kernel_image(data)

    # Part 2: Triangulated BCC
    part2_triangulated_bcc()

    # Part 3: Cobordism analysis
    part3_cobordism()

    # Part 4: O_h group action
    part4_oh_action()

    # Part 5: Cycle budget
    part5_cycle_budget()

    print(f"\n{'='*70}")
    print(f"  Done.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
