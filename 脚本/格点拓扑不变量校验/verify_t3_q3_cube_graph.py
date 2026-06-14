# verify_t3_q3_cube_graph.py
# Verify claim from 互洽(二) §10: the 8 simplices surrounding a deep-interior SC vertex
# have a pairwise shared-vertex-count matrix whose positive-weight adjacency (count>=2)
# is isomorphic to the cube graph Q3.
#
# Claim: 8-hub pairwise intersection matrix — each row has 3 entries "2" + 4 entries "1"
# which defines adjacency = Q3 (8 nodes, 12 edges, 3-regular, bipartite, diameter 3).

import itertools
import sys

def build_bcc_simplices():
    """
    Build BCC 3x3x3 simplices (same construction as bcc_540_spectrum.py).
    Returns list of 54 frozensets of 5 vertices (float body-center + int corner tuples).
    """
    simplices = []
    simplex_corners = []  # 4 corners per simplex (without BC)
    for i, j, k in itertools.product(range(3), repeat=3):
        bc = (i + 0.5, j + 0.5, k + 0.5)
        corners = [(i + dx, j + dy, k + dz) for dx, dy, dz in itertools.product((0,1), repeat=3)]
        even_tet = [c for c in corners if (c[0]+c[1]+c[2]) % 2 == 0]
        odd_tet  = [c for c in corners if (c[0]+c[1]+c[2]) % 2 == 1]
        assert len(even_tet) == 4 and len(odd_tet) == 4
        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))
        for tet in (even_tet, odd_tet):
            tet_f = [(float(c[0]), float(c[1]), float(c[2])) for c in tet]
            simplex = frozenset([bc_f] + tet_f)
            simplices.append(simplex)
            simplex_corners.append(frozenset(tet_f))
    return simplices, simplex_corners


def find_deep_interior_sc_vertices(simplices):
    """
    A 'deep interior' SC vertex is a corner vertex that appears in exactly 8 simplices
    (surrounded by all 8 body-centers that share that corner).
    In the 3x3x3 periodic-like open BCC lattice the truly interior SC corners
    have coords in {1,2}^3 (away from boundary of the 3x3x3 box).
    We find SC vertices by membership count.
    """
    from collections import defaultdict
    vertex_to_simplices = defaultdict(list)
    for s_idx, simplex in enumerate(simplices):
        for v in simplex:
            # SC vertices have integer coords, BC vertices have .5 coords
            if all(x == int(x) for x in v):
                vertex_to_simplices[v].append(s_idx)
    # deep interior: appears in exactly 8 simplices
    hub_vertices = {v: idxs for v, idxs in vertex_to_simplices.items() if len(idxs) == 8}
    return hub_vertices


def build_intersection_matrix(simplex_list):
    """
    Given 8 simplices (as frozensets of 5 vertices), build 8x8 symmetric matrix
    where M[i,j] = |simplex_i ∩ simplex_j| for i!=j (diagonal = 5).
    """
    n = len(simplex_list)
    M = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            M[i][j] = len(simplex_list[i] & simplex_list[j])
    return M


def intersection_to_adjacency(M, threshold=2):
    """
    Adjacency: A[i,j] = 1 if M[i,j] >= threshold and i != j.
    """
    n = len(M)
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and M[i][j] >= threshold:
                A[i][j] = 1
    return A


def graph_properties(A):
    n = len(A)
    edges = [(i,j) for i in range(n) for j in range(i+1,n) if A[i][j]]
    degrees = [sum(A[i]) for i in range(n)]
    # BFS diameter
    def bfs_dist(src):
        dist = [-1]*n
        dist[src] = 0
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if A[u][v] and dist[v] == -1:
                    dist[v] = dist[u]+1
                    queue.append(v)
        return dist
    max_dist = 0
    connected = True
    for i in range(n):
        d = bfs_dist(i)
        if -1 in d:
            connected = False
        else:
            max_dist = max(max_dist, max(d))
    # bipartite check via 2-coloring
    color = [-1]*n
    is_bipartite = True
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            for v in range(n):
                if A[u][v]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
    return {
        'n_nodes': n,
        'n_edges': len(edges),
        'degrees': degrees,
        'regular': len(set(degrees)) == 1,
        'degree': degrees[0] if len(set(degrees)) == 1 else None,
        'diameter': max_dist if connected else None,
        'connected': connected,
        'bipartite': is_bipartite,
    }


def q3_adjacency():
    """
    Q3 cube graph: vertices = bit strings of length 3, edges = Hamming distance 1.
    Returns 8x8 adjacency as list-of-lists.
    """
    n = 8
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = i ^ j
                if diff and (diff & (diff-1)) == 0:  # exactly one bit differs
                    A[i][j] = 1
    return A


def canonical_degree_sequence_check(A_cand):
    """Quick necessary condition: same degree sequence as Q3 (all degrees = 3)."""
    degs = sorted([sum(row) for row in A_cand], reverse=True)
    return degs


def brute_force_isomorphism(A1, A2):
    """
    Check graph isomorphism by trying all 8! = 40320 permutations of node labels.
    A1, A2: 8x8 adjacency lists.
    Returns (is_iso, permutation_or_None)
    """
    n = len(A1)
    from itertools import permutations
    for perm in permutations(range(n)):
        ok = True
        for i in range(n):
            for j in range(n):
                if A1[i][j] != A2[perm[i]][perm[j]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True, perm
    return False, None


def main():
    print("=" * 65)
    print("VERIFY T3: 8-hub intersection matrix isomorphic to Q3")
    print("=" * 65)
    print()

    # Build simplices
    simplices, simplex_corners = build_bcc_simplices()
    print(f"Total simplices: {len(simplices)} (expect 54)")
    assert len(simplices) == 54

    # Find deep-interior SC vertices (those surrounded by exactly 8 simplices)
    hub_vertices = find_deep_interior_sc_vertices(simplices)
    print(f"Deep-interior SC vertices (|membership|=8): {len(hub_vertices)}")
    if not hub_vertices:
        print()
        print("NOT-FOUND: No SC vertex with exactly 8 simplices found.")
        print("Searched: bcc_540_spectrum.py construction (27 BCs x 2 tets = 54 simplices),")
        print("          vertex membership counts across all 54 frozensets.")
        print("The open BCC 3x3x3 lattice has no truly interior vertex surrounded by 8 simplices.")
        # Report membership distribution
        from collections import Counter, defaultdict
        v2s = defaultdict(list)
        for s_idx, simplex in enumerate(simplices):
            for v in simplex:
                if all(x == int(x) for x in v):
                    v2s[v].append(s_idx)
        counts = Counter(len(idxs) for idxs in v2s.values())
        print(f"  SC vertex membership distribution: {dict(sorted(counts.items()))}")
        return

    # Use the first hub vertex
    hub_v = sorted(hub_vertices.keys())[0]
    hub_simplex_idxs = sorted(hub_vertices[hub_v])
    print(f"Using hub vertex: {hub_v}")
    print(f"Simplex indices: {hub_simplex_idxs}")

    hub_simplices = [simplices[i] for i in hub_simplex_idxs]

    # Build intersection matrix
    M = build_intersection_matrix(hub_simplices)
    print()
    print("Pairwise intersection matrix (shared vertex count):")
    for row in M:
        print("  " + " ".join(f"{x:2d}" for x in row))

    # Check row pattern: diagonal=5, off-diagonal should be {1,2}
    row_counts = {}
    for i in range(8):
        counts_2 = sum(1 for j in range(8) if i != j and M[i][j] == 2)
        counts_1 = sum(1 for j in range(8) if i != j and M[i][j] == 1)
        counts_0 = sum(1 for j in range(8) if i != j and M[i][j] == 0)
        row_counts[i] = (counts_2, counts_1, counts_0)

    print()
    print("Row pattern (off-diagonal): count_2, count_1, count_0:")
    consistent_3_4 = True
    for i in range(8):
        c2, c1, c0 = row_counts[i]
        print(f"  Row {i}: {c2} twos, {c1} ones, {c0} zeros")
        if c2 != 3 or c1 != 4:
            consistent_3_4 = False

    print()
    if consistent_3_4:
        print("CHECK: Each row has exactly 3 twos + 4 ones — matches claim.")
    else:
        print("CHECK: Row pattern does NOT match claimed '3 twos + 4 ones'.")

    # Build adjacency from intersection >= 2
    A_hub = intersection_to_adjacency(M, threshold=2)
    props = graph_properties(A_hub)
    print()
    print("Graph properties (adjacency = intersection >= 2):")
    print(f"  Nodes:     {props['n_nodes']} (Q3 expects 8)")
    print(f"  Edges:     {props['n_edges']} (Q3 expects 12)")
    print(f"  Regular:   {props['regular']} (Q3 expects True)")
    print(f"  Degree:    {props['degree']} (Q3 expects 3)")
    print(f"  Connected: {props['connected']} (Q3 expects True)")
    print(f"  Bipartite: {props['bipartite']} (Q3 expects True)")
    print(f"  Diameter:  {props['diameter']} (Q3 expects 3)")

    # Build reference Q3
    A_q3 = q3_adjacency()
    q3_props = graph_properties(A_q3)
    print()
    print("Reference Q3 properties:")
    print(f"  Nodes: {q3_props['n_nodes']}, Edges: {q3_props['n_edges']}, "
          f"Degree: {q3_props['degree']}, Diameter: {q3_props['diameter']}, "
          f"Bipartite: {q3_props['bipartite']}")

    # Necessary conditions check
    necessary = (
        props['n_nodes'] == 8 and
        props['n_edges'] == 12 and
        props['regular'] and props['degree'] == 3 and
        props['connected'] and
        props['bipartite'] and
        props['diameter'] == 3
    )
    print()
    if not necessary:
        print("FAIL: Necessary conditions for Q3 not met. Skipping isomorphism check.")
        return

    print("Necessary conditions (8 nodes, 12 edges, 3-regular, bipartite, diameter 3): MET")
    print()
    print("Running brute-force isomorphism check (8! = 40320 permutations)...")
    is_iso, perm = brute_force_isomorphism(A_hub, A_q3)

    if is_iso:
        print(f"Isomorphism found: node mapping = {perm}")
        print()
        print("PASS: 8-hub intersection adjacency is isomorphic to Q3.")
    else:
        print()
        print("FAIL: No isomorphism found — graph is NOT isomorphic to Q3.")


if __name__ == "__main__":
    main()
