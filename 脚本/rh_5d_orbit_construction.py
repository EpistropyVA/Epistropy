# -*- coding: utf-8 -*-
"""
rh_5d_orbit_construction.py

Construct the actual 5D graph (V=22, E=22) from O_h orbits.

The 5D layer in the d-cascade is built from the 22 O_h orbits of the 540
triangular faces of the 3x3x3 BCC complex.  These 22 orbits become the
22 VERTICES of the 5D orbit graph.  The question is: what are the 22
EDGES, and does the resulting graph have faces (hence beta_1=0 or 1)?

Approach
--------
1. Build the 3x3x3 BCC lattice (27 body-centers + 64 corner sites).
2. Generate all 540 triangular faces.
3. Generate the O_h group (48 signed permutation matrices about center).
4. Partition faces into O_h orbits -> 22 orbits.
5. Build the orbit adjacency graph: two orbit-vertices are connected if
   faces from distinct orbits share an edge (are face-adjacent).
6. Threshold the coupling to find an edge set of size 22.
7. Analyze: find cycles, determine if natural 2-cells exist, compute beta_1.

Pure Python + stdlib only (no numpy).
"""

import sys
import io
import itertools
from collections import defaultdict
from fractions import Fraction

# Force UTF-8 stdout for Windows GBK terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ═══════════════════════════════════════════════════════════════════════
# Section 1: BCC lattice construction
# ═══════════════════════════════════════════════════════════════════════

def build_bcc_3x3x3():
    """Build the 3x3x3 BCC complex.

    Vertices:
      - 64 corner sites: ('c', x, y, z) with x,y,z in {0,1,2,3}
      - 27 body-centers: ('bc', i, j, k) with i,j,k in {0,1,2}

    Returns (vertex_list, face_list, face_set)
      face_list: list of frozensets, each containing 3 vertex tuples
      face_set: frozenset -> index
    """
    corners = [('c', x, y, z) for x in range(4) for y in range(4) for z in range(4)]
    bcs = [('bc', i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    vertex_list = corners + bcs

    face_list = []
    face_set = {}

    for i, j, k in itertools.product(range(3), repeat=3):
        bc_v = ('bc', i, j, k)
        # The 8 corners of this unit cell
        cell_corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            cell_corners.append(('c', i + dx, j + dy, k + dz))

        # Partition corners by parity into two tetrahedra
        even_corners = [c for c in cell_corners if (c[1] + c[2] + c[3]) % 2 == 0]
        odd_corners = [c for c in cell_corners if (c[1] + c[2] + c[3]) % 2 == 1]

        # Each tet + body-center = 5 vertices -> C(5,3)=10 triangles per tet
        for tet_corners in (even_corners, odd_corners):
            simplex_verts = [bc_v] + tet_corners  # 5 vertices
            for combo in itertools.combinations(simplex_verts, 3):
                f = frozenset(combo)
                if f not in face_set:
                    face_set[f] = len(face_list)
                    face_list.append(f)

    return vertex_list, face_list, face_set


# ═══════════════════════════════════════════════════════════════════════
# Section 2: O_h symmetry group (48 signed permutation matrices)
# ═══════════════════════════════════════════════════════════════════════

# Center of the 3x3x3 lattice
CENTER = (Fraction(3, 2), Fraction(3, 2), Fraction(3, 2))


def generate_oh_group():
    """Generate all 48 signed permutation matrices as tuples-of-tuples."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            # M[row][col] = signs[row] if col == perm[row] else 0
            M = tuple(
                tuple(signs[row] if col == perm[row] else 0 for col in range(3))
                for row in range(3)
            )
            mats.append(M)
    assert len(mats) == 48, f"Expected 48 O_h elements, got {len(mats)}"
    return mats


def get_pos(v):
    """Get geometric position of a vertex as (Fraction, Fraction, Fraction)."""
    if v[0] == 'bc':
        _, i, j, k = v
        return (Fraction(2 * i + 1, 2), Fraction(2 * j + 1, 2), Fraction(2 * k + 1, 2))
    else:
        _, x, y, z = v
        return (Fraction(x), Fraction(y), Fraction(z))


def apply_oh(M, pos):
    """Apply O_h matrix M about CENTER to position pos. Returns new position."""
    # Translate to center, apply M, translate back
    dx = pos[0] - CENTER[0]
    dy = pos[1] - CENTER[1]
    dz = pos[2] - CENTER[2]

    nx = M[0][0] * dx + M[0][1] * dy + M[0][2] * dz + CENTER[0]
    ny = M[1][0] * dx + M[1][1] * dy + M[1][2] * dz + CENTER[1]
    nz = M[2][0] * dx + M[2][1] * dy + M[2][2] * dz + CENTER[2]

    return (nx, ny, nz)


def pos_to_vertex(pos):
    """Convert a Fraction position back to a vertex label, or None."""
    x, y, z = pos
    # Check body-center: half-integer coords
    hx, hy, hz = x - Fraction(1, 2), y - Fraction(1, 2), z - Fraction(1, 2)
    if hx.denominator == 1 and hy.denominator == 1 and hz.denominator == 1:
        i, j, k = int(hx), int(hy), int(hz)
        if 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2:
            return ('bc', i, j, k)
    # Check corner: integer coords
    if x.denominator == 1 and y.denominator == 1 and z.denominator == 1:
        ix, iy, iz = int(x), int(y), int(z)
        if 0 <= ix <= 3 and 0 <= iy <= 3 and 0 <= iz <= 3:
            return ('c', ix, iy, iz)
    return None


def apply_oh_to_face(face_fs, M):
    """Apply O_h element M to a face (frozenset of 3 vertices).
    Returns new frozenset or None if any vertex maps outside the lattice."""
    new_verts = []
    for v in face_fs:
        new_pos = apply_oh(M, get_pos(v))
        new_v = pos_to_vertex(new_pos)
        if new_v is None:
            return None
        new_verts.append(new_v)
    return frozenset(new_verts)


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Orbit classification
# ═══════════════════════════════════════════════════════════════════════

def build_oh_orbits(face_list, face_set, oh_group):
    """Partition the faces into O_h orbits.
    Returns (orbits, orbit_of).
      orbits: list of lists of face indices
      orbit_of: face_index -> orbit_index
    """
    N = len(face_list)
    orbit_of = [-1] * N
    orbits = []

    for fi in range(N):
        if orbit_of[fi] >= 0:
            continue
        orb_idx = len(orbits)
        orbit = []
        for M in oh_group:
            new_face = apply_oh_to_face(face_list[fi], M)
            if new_face is not None and new_face in face_set:
                new_fi = face_set[new_face]
                if orbit_of[new_fi] < 0:
                    orbit_of[new_fi] = orb_idx
                    orbit.append(new_fi)
        orbits.append(orbit)

    return orbits, orbit_of


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Face adjacency and orbit coupling
# ═══════════════════════════════════════════════════════════════════════

def build_face_adjacency(face_list):
    """Two faces are adjacent iff they share exactly 2 vertices (i.e. share an edge).
    Returns adjacency dict: face_index -> set of adjacent face_indices.
    """
    # Build vertex-to-faces index
    v2f = defaultdict(set)
    for fi, f in enumerate(face_list):
        for v in f:
            v2f[v].add(fi)

    adj = defaultdict(set)
    for fi, f in enumerate(face_list):
        # Candidate neighbors: faces sharing at least one vertex
        candidates = set()
        for v in f:
            candidates.update(v2f[v])
        candidates.discard(fi)
        for fj in candidates:
            if len(f & face_list[fj]) == 2:
                adj[fi].add(fj)
                adj[fj].add(fi)

    return adj


def build_orbit_coupling(orbits, orbit_of, face_adj):
    """Build the 22x22 orbit coupling matrix.
    C[i][j] = number of face-pairs (f_i in orbit i, f_j in orbit j) that are adjacent.
    Diagonal C[i][i] = intra-orbit adjacencies.
    """
    n_orb = len(orbits)
    C = [[0] * n_orb for _ in range(n_orb)]

    for fi, neighbors in face_adj.items():
        oi = orbit_of[fi]
        for fj in neighbors:
            oj = orbit_of[fj]
            C[oi][oj] += 1

    # Each adjacency counted twice (fi->fj and fj->fi), so halve
    # Actually, face_adj already has both directions, so C[i][j] counts
    # each pair once from fi side and once from fj side if i != j.
    # For i == j, same issue. So divide all by 2.
    for i in range(n_orb):
        for j in range(n_orb):
            C[i][j] //= 2

    return C


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Graph analysis (cycles, beta_1)
# ═══════════════════════════════════════════════════════════════════════

def f2_rank(matrix, nrows, ncols):
    """Rank of an nrows x ncols matrix over F2 (bitwise)."""
    if nrows == 0 or ncols == 0:
        return 0
    rows = []
    for r in range(nrows):
        val = 0
        for c in range(ncols):
            if matrix[r][c]:
                val |= (1 << c)
        rows.append(val)

    rank = 0
    for col in range(ncols):
        mask = 1 << col
        pivot = None
        for r in range(rank, len(rows)):
            if rows[r] & mask:
                pivot = r
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and (rows[r] & mask):
                rows[r] ^= rows[rank]
        rank += 1
    return rank


def connected_components(V, edges):
    """Find connected components via union-find. Returns number of components."""
    parent = list(range(V))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in edges:
        union(a, b)

    return len(set(find(i) for i in range(V)))


def find_all_cycles_dfs(V, adj_list):
    """Find a basis of independent cycles using DFS.
    Returns list of cycles, each cycle is a list of vertex indices."""
    visited = [False] * V
    parent = [-1] * V
    depth = [0] * V
    cycles = []

    def dfs(u, d):
        visited[u] = True
        depth[u] = d
        for w in adj_list[u]:
            if not visited[w]:
                parent[w] = u
                dfs(w, d + 1)
            elif w != parent[u] and depth[w] < depth[u]:
                # Back edge -> cycle
                cycle = [u]
                x = u
                while x != w:
                    x = parent[x]
                    cycle.append(x)
                cycles.append(cycle)

    for start in range(V):
        if not visited[start]:
            dfs(start, 0)

    return cycles


def find_cycle_edges(cycle, edge_map):
    """Given a cycle [v0, v1, ..., vk] (where vk connects back to v0),
    return list of edge indices."""
    result = []
    for i in range(len(cycle)):
        a, b = cycle[i], cycle[(i + 1) % len(cycle)]
        key = (min(a, b), max(a, b))
        if key in edge_map:
            result.append(edge_map[key])
    return result


def compute_beta1(V, edges, face_edge_lists):
    """Compute beta_1 over F2.
    V: number of vertices
    edges: list of (a, b) tuples
    face_edge_lists: list of lists of edge indices (each face's boundary)
    """
    E = len(edges)
    F = len(face_edge_lists)

    # d1: E x V
    d1 = [[0] * V for _ in range(E)]
    for ei, (a, b) in enumerate(edges):
        d1[ei][a] = 1
        d1[ei][b] = 1
    rank_d1 = f2_rank(d1, E, V)

    # d2: F x E
    if F > 0:
        d2 = [[0] * E for _ in range(F)]
        for fi, fe_list in enumerate(face_edge_lists):
            for ei in fe_list:
                d2[fi][ei] = 1
        rank_d2 = f2_rank(d2, F, E)
    else:
        rank_d2 = 0

    ker_d1 = E - rank_d1
    beta1 = ker_d1 - rank_d2

    return beta1, rank_d1, rank_d2, ker_d1


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Edge selection strategies
# ═══════════════════════════════════════════════════════════════════════

def select_edges_by_threshold(coupling_matrix, n_orbits, target_edges):
    """Try various thresholds on the coupling matrix to find an edge set
    of size target_edges. Reports all thresholds and their edge counts.
    Returns list of (threshold, edges) pairs."""
    # Collect all distinct off-diagonal coupling values
    values = set()
    for i in range(n_orbits):
        for j in range(i + 1, n_orbits):
            if coupling_matrix[i][j] > 0:
                values.add(coupling_matrix[i][j])

    results = []
    for thresh in sorted(values):
        edges = []
        for i in range(n_orbits):
            for j in range(i + 1, n_orbits):
                if coupling_matrix[i][j] >= thresh:
                    edges.append((i, j))
        results.append((thresh, edges))

    return results


def select_edges_spanning_tree_plus_k(coupling_matrix, n_orbits, target_edges):
    """Build a maximum-weight spanning tree, then add the next strongest
    edges to reach target_edges total.
    Uses Kruskal's algorithm with max weight."""
    # Collect all edges with weights
    weighted_edges = []
    for i in range(n_orbits):
        for j in range(i + 1, n_orbits):
            w = coupling_matrix[i][j]
            if w > 0:
                weighted_edges.append((w, i, j))

    # Sort descending by weight
    weighted_edges.sort(reverse=True)

    # Kruskal's for max spanning tree
    parent = list(range(n_orbits))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    tree_edges = []
    extra_edges = []

    for w, i, j in weighted_edges:
        if union(i, j):
            tree_edges.append((i, j, w))
        else:
            extra_edges.append((i, j, w))

    # How many tree edges?
    n_tree = len(tree_edges)
    # Need target_edges - n_tree extra edges
    needed = target_edges - n_tree

    selected = [(i, j) for i, j, w in tree_edges]
    for i, j, w in extra_edges[:max(0, needed)]:
        selected.append((i, j))

    return selected, tree_edges, extra_edges


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Natural geometric objects and O_h orbits on points
# ═══════════════════════════════════════════════════════════════════════

def oh_orbit_on_points(points, oh_group):
    """Compute O_h orbits on a set of points (as Fraction triples).
    Returns list of orbits (each orbit is a frozenset of points)."""
    remaining = set(points)
    orbits = []
    while remaining:
        p = next(iter(remaining))
        orbit = set()
        for M in oh_group:
            q = apply_oh(M, p)
            if q in remaining:
                orbit.add(q)
        orbits.append(frozenset(orbit))
        remaining -= orbit
    return orbits


def analyze_bcc_point_orbits(oh_group):
    """Analyze O_h orbits on various geometric objects in/around BCC unit cell."""
    print("=" * 72)
    print("PART 1: O_h orbits on geometric points of the 3x3x3 BCC lattice")
    print("=" * 72)
    print()

    # All corner positions
    corners = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                corners.append((Fraction(x), Fraction(y), Fraction(z)))
    print(f"Corner sites: {len(corners)}")
    c_orbits = oh_orbit_on_points(corners, oh_group)
    print(f"  O_h orbits: {len(c_orbits)}")
    for i, orb in enumerate(sorted(c_orbits, key=lambda o: -len(o))):
        rep = sorted(orb)[0]
        print(f"    orbit {i}: size {len(orb)}, rep = ({float(rep[0]):.1f}, {float(rep[1]):.1f}, {float(rep[2]):.1f})")
    total_corners = sum(len(o) for o in c_orbits)
    print(f"  Total: {total_corners}")
    print()

    # All body-center positions
    bcs = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bcs.append((Fraction(2*i+1, 2), Fraction(2*j+1, 2), Fraction(2*k+1, 2)))
    print(f"Body-center sites: {len(bcs)}")
    bc_orbits = oh_orbit_on_points(bcs, oh_group)
    print(f"  O_h orbits: {len(bc_orbits)}")
    for i, orb in enumerate(sorted(bc_orbits, key=lambda o: -len(o))):
        rep = sorted(orb)[0]
        print(f"    orbit {i}: size {len(orb)}, rep = ({float(rep[0]):.1f}, {float(rep[1]):.1f}, {float(rep[2]):.1f})")
    total_bcs = sum(len(o) for o in bc_orbits)
    print(f"  Total: {total_bcs}")
    print()

    # Edge midpoints of cube edges in 3x3x3
    edge_mids = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                p = (Fraction(x), Fraction(y), Fraction(z))
                # Edges along x
                if x + 1 <= 3:
                    edge_mids.add((Fraction(2*x+1, 2), Fraction(y), Fraction(z)))
                # Edges along y
                if y + 1 <= 3:
                    edge_mids.add((Fraction(x), Fraction(2*y+1, 2), Fraction(z)))
                # Edges along z
                if z + 1 <= 3:
                    edge_mids.add((Fraction(x), Fraction(y), Fraction(2*z+1, 2)))
    print(f"Cube edge midpoints: {len(edge_mids)}")
    em_orbits = oh_orbit_on_points(edge_mids, oh_group)
    print(f"  O_h orbits: {len(em_orbits)}")
    for i, orb in enumerate(sorted(em_orbits, key=lambda o: -len(o))):
        rep = sorted(orb)[0]
        print(f"    orbit {i}: size {len(orb)}, rep = ({float(rep[0]):.1f}, {float(rep[1]):.1f}, {float(rep[2]):.1f})")
    print()

    # Face centers of cube cells
    face_centers = set()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # 6 faces of cell (i,j,k)
                # x-faces at x=i and x=i+1
                face_centers.add((Fraction(i), Fraction(2*j+1, 2), Fraction(2*k+1, 2)))
                face_centers.add((Fraction(i+1), Fraction(2*j+1, 2), Fraction(2*k+1, 2)))
                # y-faces
                face_centers.add((Fraction(2*i+1, 2), Fraction(j), Fraction(2*k+1, 2)))
                face_centers.add((Fraction(2*i+1, 2), Fraction(j+1), Fraction(2*k+1, 2)))
                # z-faces
                face_centers.add((Fraction(2*i+1, 2), Fraction(2*j+1, 2), Fraction(k)))
                face_centers.add((Fraction(2*i+1, 2), Fraction(2*j+1, 2), Fraction(k+1)))
    print(f"Cube face centers: {len(face_centers)}")
    fc_orbits = oh_orbit_on_points(face_centers, oh_group)
    print(f"  O_h orbits: {len(fc_orbits)}")
    for i, orb in enumerate(sorted(fc_orbits, key=lambda o: -len(o))):
        rep = sorted(orb)[0]
        print(f"    orbit {i}: size {len(orb)}, rep = ({float(rep[0]):.1f}, {float(rep[1]):.1f}, {float(rep[2]):.1f})")
    print()

    print("  Possible 22-decompositions from these orbit sizes:")
    # Collect all orbit sizes
    all_orbit_sizes = []
    all_orbit_sizes.extend([(len(o), 'corner') for o in c_orbits])
    all_orbit_sizes.extend([(len(o), 'bc') for o in bc_orbits])
    all_orbit_sizes.extend([(len(o), 'edge-mid') for o in em_orbits])
    all_orbit_sizes.extend([(len(o), 'face-ctr') for o in fc_orbits])
    print(f"  All orbit sizes: {[s for s, _ in sorted(all_orbit_sizes)]}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Main analysis
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("5D Orbit Graph Construction")
    print("Constructing the V=22, E=22 graph from O_h orbits of BCC faces")
    print("=" * 72)
    print()

    # ── Step 1: Generate O_h group ──
    oh_group = generate_oh_group()
    print(f"[STEP 1] O_h group generated: {len(oh_group)} elements")
    tag = "PASS" if len(oh_group) == 48 else "FAIL"
    print(f"  [{tag}] |O_h| = {len(oh_group)} (expected 48)")
    print()

    # ── Step 1b: Point orbit analysis ──
    analyze_bcc_point_orbits(oh_group)

    # ── Step 2: Build BCC lattice ──
    print("=" * 72)
    print("PART 2: Build 3x3x3 BCC complex and classify face orbits")
    print("=" * 72)
    print()
    vertex_list, face_list, face_set = build_bcc_3x3x3()
    print(f"[STEP 2] BCC complex: {len(vertex_list)} vertices, {len(face_list)} faces")
    tag = "PASS" if len(face_list) == 540 else "FAIL"
    print(f"  [{tag}] |faces| = {len(face_list)} (expected 540)")
    print()

    # ── Step 3: Orbit classification ──
    print("[STEP 3] Classifying faces into O_h orbits...")
    orbits, orbit_of = build_oh_orbits(face_list, face_set, oh_group)
    n_orb = len(orbits)
    print(f"  Number of orbits: {n_orb}")
    tag = "PASS" if n_orb == 22 else "FAIL"
    print(f"  [{tag}] |orbits| = {n_orb} (expected 22)")

    orbit_sizes = [len(o) for o in orbits]
    orbit_sizes_sorted = sorted(orbit_sizes, reverse=True)
    print(f"  Orbit sizes (sorted): {orbit_sizes_sorted}")
    print(f"  Sum of orbit sizes: {sum(orbit_sizes)} (should be 540)")
    tag = "PASS" if sum(orbit_sizes) == 540 else "FAIL"
    print(f"  [{tag}] sum = {sum(orbit_sizes)}")
    print()

    # Size distribution
    from collections import Counter
    size_counts = Counter(orbit_sizes)
    print("  Size distribution:")
    for sz in sorted(size_counts.keys(), reverse=True):
        count = size_counts[sz]
        print(f"    size {sz:3d}: {count} orbits (contributes {sz * count})")
    print()

    # ── Step 4: Face adjacency ──
    print("[STEP 4] Building face adjacency (shared-edge relation)...")
    face_adj = build_face_adjacency(face_list)
    total_adj = sum(len(v) for v in face_adj.values()) // 2
    print(f"  Total face-adjacency pairs: {total_adj}")
    print()

    # ── Step 5: Orbit coupling matrix ──
    print("[STEP 5] Building orbit coupling matrix...")
    C = build_orbit_coupling(orbits, orbit_of, face_adj)

    # Print the coupling matrix
    print("  Orbit coupling matrix C[i][j] (number of cross-orbit face-adjacencies):")
    print()
    # Header
    header = "     " + " ".join(f"{j:5d}" for j in range(n_orb))
    print(header)
    for i in range(n_orb):
        row = f"{i:3d}: " + " ".join(f"{C[i][j]:5d}" for j in range(n_orb))
        print(row)
    print()

    # Off-diagonal nonzero counts
    off_diag_nonzero = 0
    off_diag_pairs = []
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            if C[i][j] > 0:
                off_diag_nonzero += 1
                off_diag_pairs.append((C[i][j], i, j))
    print(f"  Off-diagonal nonzero pairs: {off_diag_nonzero}")
    print()

    # ── Step 6: Edge selection ──
    print("=" * 72)
    print("PART 3: Edge selection -- find 22 edges from coupling structure")
    print("=" * 72)
    print()

    # Strategy A: Threshold scan
    print("[Strategy A] Threshold scan on coupling values:")
    thresh_results = select_edges_by_threshold(C, n_orb, 22)
    for thresh, edges in thresh_results:
        n_comp = connected_components(n_orb, edges)
        print(f"  threshold >= {thresh:4d}: {len(edges):3d} edges, {n_comp} components")
    print()

    # Find which threshold gives 22
    exact_22 = [(t, e) for t, e in thresh_results if len(e) == 22]
    if exact_22:
        print(f"  [PASS] Threshold {exact_22[0][0]} gives exactly 22 edges!")
        chosen_edges = exact_22[0][1]
        chosen_threshold = exact_22[0][0]
    else:
        print("  [INFO] No threshold gives exactly 22 edges.")
        # Find closest
        closest = min(thresh_results, key=lambda x: abs(len(x[1]) - 22))
        print(f"  Closest: threshold {closest[0]} -> {len(closest[1])} edges")
        chosen_edges = None
        chosen_threshold = None

    # Strategy B: Spanning tree + extras
    print()
    print("[Strategy B] Maximum spanning tree + strongest extra edges:")
    mst_edges, tree_edges, extra_edges = select_edges_spanning_tree_plus_k(C, n_orb, 22)
    print(f"  Spanning tree: {len(tree_edges)} edges")
    print(f"  Extra edges needed for 22: {22 - len(tree_edges)}")
    print(f"  Available extra edges: {len(extra_edges)}")
    n_comp_mst = connected_components(n_orb, mst_edges)
    print(f"  MST+extra total: {len(mst_edges)} edges, {n_comp_mst} components")

    # Tree edge weights
    tree_weights = sorted([w for _, _, w in tree_edges], reverse=True)
    print(f"  Tree edge weights: {tree_weights}")
    if extra_edges:
        extra_weights = [w for _, _, w in extra_edges[:5]]
        print(f"  Top extra edge weights: {extra_weights}")
    print()

    # ── Step 7: Graph analysis of chosen edge set ──
    print("=" * 72)
    print("PART 4: Graph analysis")
    print("=" * 72)
    print()

    # Use threshold-based if available, otherwise MST-based
    if chosen_edges is not None:
        analysis_edges = chosen_edges
        print(f"Using threshold-based edges (threshold >= {chosen_threshold})")
    else:
        analysis_edges = mst_edges
        print(f"Using MST-based edges ({len(mst_edges)} edges)")
    print(f"  V = {n_orb}, E = {len(analysis_edges)}")
    print()

    n_comp = connected_components(n_orb, analysis_edges)
    print(f"  Connected components: {n_comp}")
    print()

    # Degree sequence
    deg = [0] * n_orb
    for a, b in analysis_edges:
        deg[a] += 1
        deg[b] += 1
    deg_sorted = sorted(deg, reverse=True)
    print(f"  Degree sequence: {deg_sorted}")
    print(f"  Sum of degrees: {sum(deg)} (should be 2 * E = {2 * len(analysis_edges)})")
    print()

    # beta_1 without faces
    E_count = len(analysis_edges)
    if n_comp > 0:
        beta1_no_face = E_count - n_orb + n_comp
    else:
        beta1_no_face = E_count - n_orb + 1
    print(f"  beta_1 (F=0) = E - V + c = {E_count} - {n_orb} + {n_comp} = {beta1_no_face}")
    print()

    # Verify with boundary matrix
    b1, r_d1, r_d2, ker_d1 = compute_beta1(n_orb, analysis_edges, [])
    print(f"  Verified via F2 boundary matrix:")
    print(f"    rank(d1) = {r_d1}, ker(d1) = {ker_d1}")
    print(f"    rank(d2) = {r_d2} (no faces)")
    print(f"    beta_1 = {b1}")
    tag = "PASS" if b1 == beta1_no_face else "FAIL"
    print(f"    [{tag}] matches formula: {b1} == {beta1_no_face}")
    print()

    # Find cycles
    adj_list = defaultdict(list)
    for a, b in analysis_edges:
        adj_list[a].append(b)
        adj_list[b].append(a)

    sys.setrecursionlimit(10000)
    cycles = find_all_cycles_dfs(n_orb, adj_list)
    print(f"  Independent cycles found: {len(cycles)}")
    for ci, cyc in enumerate(cycles):
        print(f"    Cycle {ci}: length {len(cyc)}, vertices = {cyc}")

        # Show orbit sizes along cycle
        if ci < 5:  # Print details for first 5 cycles
            sizes = [orbit_sizes[v] for v in cyc]
            print(f"      Orbit sizes along cycle: {sizes}")
    print()

    # ── Step 8: Can faces seal the cycles? ──
    print("=" * 72)
    print("PART 5: Face existence and beta_1 verdict")
    print("=" * 72)
    print()

    if len(cycles) == 0:
        print("  No cycles -> beta_1 = 0 (already sealed, F=0 suffices)")
        print("  [PASS] 5D is sealed with F=0")
    else:
        # Build edge map
        edge_map = {}
        for ei, (a, b) in enumerate(analysis_edges):
            edge_map[(min(a, b), max(a, b))] = ei

        all_face_lists = []
        for ci, cyc in enumerate(cycles):
            cyc_edges = find_cycle_edges(cyc, edge_map)
            print(f"  Cycle {ci}: {len(cyc)} vertices, boundary has {len(cyc_edges)} edges")

            if len(cyc_edges) == len(cyc):
                print(f"    This cycle is a valid 2-cell boundary (all edges present)")
                all_face_lists.append(cyc_edges)
            else:
                print(f"    WARNING: cycle edges don't close ({len(cyc_edges)} != {len(cyc)})")

        # Check if filling all cycles seals the complex
        if all_face_lists:
            b1_filled, r1, r2, k1 = compute_beta1(n_orb, analysis_edges, all_face_lists)
            print()
            print(f"  After filling {len(all_face_lists)} cycle(s) as 2-cells:")
            print(f"    rank(d1) = {r1}, ker(d1) = {k1}")
            print(f"    rank(d2) = {r2}")
            print(f"    beta_1 = {b1_filled}")
            print()

            if b1_filled == 0:
                print(f"  [PASS] Filling {len(all_face_lists)} face(s) seals 5D: beta_1 = 0")
                print(f"  => F = {len(all_face_lists)} gives a sealed 5D complex")
            else:
                print(f"  [WARN] Even after filling, beta_1 = {b1_filled} > 0")
                print(f"  => Additional faces needed or structure is genuinely open")

        print()
        print("--- Geometric interpretation ---")
        print()
        print(f"  5D orbit graph has V={n_orb}, E={len(analysis_edges)}")
        print(f"  Without faces: beta_1 = {b1}")
        if b1 == 1:
            print("  Exactly ONE independent cycle exists.")
            print("  This means the 5D layer has a single 1-dimensional hole.")
            print()
            print("  Physical meaning: a closed loop of orbit-couplings that")
            print("  cannot be contracted to a point within the orbit graph.")
            print("  This is the j-axis (second-order relation) structure:")
            print("  a cycle of relations-between-relations.")
        elif b1 > 1:
            print(f"  {b1} independent cycles exist.")
        print()

        # Euler characteristic
        chi_no_face = n_orb - len(analysis_edges)
        print(f"  Euler characteristic (F=0): chi = V - E = {n_orb} - {len(analysis_edges)} = {chi_no_face}")
        if all_face_lists:
            chi_with_face = n_orb - len(analysis_edges) + len(all_face_lists)
            print(f"  Euler characteristic (F={len(all_face_lists)}): chi = {chi_with_face}")
        print()

    # ── Summary ──
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print(f"  3x3x3 BCC complex: {len(face_list)} triangular faces")
    print(f"  O_h orbits: {n_orb} (sizes: {orbit_sizes_sorted})")
    print(f"  Orbit coupling: {off_diag_nonzero} nonzero off-diagonal pairs")
    print()
    if chosen_edges is not None:
        print(f"  Threshold {chosen_threshold} -> exactly {len(chosen_edges)} edges")
        print(f"  Graph: V={n_orb}, E={len(chosen_edges)}")
    else:
        print(f"  No threshold gives exactly 22 edges.")
        print(f"  Closest match: {len(analysis_edges)} edges")

    print(f"  Connected components: {n_comp}")
    print(f"  beta_1 (F=0) = {b1}")
    if b1 > 0 and all_face_lists:
        print(f"  beta_1 (F={len(all_face_lists)}) = {b1_filled}")
    print()

    # Final verdict
    if b1 == 0:
        print("  VERDICT: 5D is SEALED (beta_1 = 0). No open holes.")
    elif b1 == 1:
        print("  VERDICT: 5D has exactly ONE open 1-cycle (beta_1 = 1).")
        print("  This can be sealed by adding 1 face (any polygon filling the cycle).")
        print("  Whether it SHOULD be sealed depends on the cascade's")
        print("  cobordism rules for the 4D->5D transition.")
    else:
        print(f"  VERDICT: 5D has {b1} open cycles (beta_1 = {b1}).")
    print()


if __name__ == "__main__":
    main()
