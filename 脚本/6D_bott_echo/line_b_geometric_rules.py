"""
Line B: Period 2 geometric connection rules
=============================================
Derive period 2's insertion rules from the same geometric framework
that generated period 1.

Period 1 rules (known):
  - BCC base: cube (Hamming-1 edges) + body center
  - v9→{3,5,6}: non-adjacent (Hamming≥2 among 3,5,6), ijk direction 1
  - v10→{1,2,4,9}: non-adjacent, ijk direction 2
  - v11→{0,3,5,6}: non-adjacent, ijk direction 3
  - v12→{0-6,9-11}: K-closure (all except v7,v8)

Period 2 question: what's the analogous geometry at the 8D+ level?

Approach:
  1. Characterize the 8D complex's connectivity graph
  2. Define "adjacency" at this level (what's the period-2 analog of Hamming distance?)
  3. Test: do the period-1 insertion sets follow a pattern that generalizes?
  4. For period 2: which subsets maintain β₁=0 AND increase β₂
     while following geometric rules (not just brute-force optimization)?
  5. What determines the "natural" partial K-closure?
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
import time


def rank_f2(mat):
    if mat.size == 0:
        return 0
    m = mat.copy() % 2
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        found = -1
        for row in range(pivot_row, rows):
            if m[row, col] == 1:
                found = row
                break
        if found == -1:
            continue
        if pivot_row != found:
            temp = m[pivot_row].copy()
            m[pivot_row] = m[found]
            m[found] = temp
        for row in range(rows):
            if row != pivot_row and m[row, col] == 1:
                m[row] = (m[row] + m[pivot_row]) % 2
        pivot_row += 1
    return pivot_row


class SimplicialComplex:
    def __init__(self):
        self.vertices = []
        self.edge_idx = {}
        self.tri_idx = {}
        self.tet_idx = {}

    def copy(self):
        c = SimplicialComplex()
        c.vertices = list(self.vertices)
        c.edge_idx = dict(self.edge_idx)
        c.tri_idx = dict(self.tri_idx)
        c.tet_idx = dict(self.tet_idx)
        return c

    def add_vertex(self, v): self.vertices.append(v)
    def add_edge(self, a, b):
        key = frozenset({a, b})
        if key not in self.edge_idx: self.edge_idx[key] = len(self.edge_idx)
    def add_triangle(self, a, b, c):
        key = frozenset({a, b, c})
        if key not in self.tri_idx: self.tri_idx[key] = len(self.tri_idx)
    def add_tetrahedron(self, a, b, c, d):
        key = frozenset({a, b, c, d})
        if key not in self.tet_idx: self.tet_idx[key] = len(self.tet_idx)

    def insert_vertex_with_cone(self, v, S):
        self.add_vertex(v)
        S = list(S)
        for s in S:
            self.add_edge(v, s)
        for a, b in combinations(S, 2):
            if frozenset({a, b}) in self.edge_idx:
                self.add_triangle(v, a, b)
        for a, b, c in combinations(S, 3):
            if frozenset({a, b, c}) in self.tri_idx:
                self.add_tetrahedron(v, a, b, c)

    def boundary_1(self):
        nv = len(self.vertices)
        ne = len(self.edge_idx)
        if ne == 0: return np.zeros((nv, 0), dtype=np.int8)
        v_idx = {v: i for i, v in enumerate(self.vertices)}
        mat = np.zeros((nv, ne), dtype=np.int8)
        for edge, ei in self.edge_idx.items():
            for v in edge:
                mat[v_idx[v], ei] = 1
        return mat

    def boundary_2(self):
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        if nt == 0: return np.zeros((ne, 0), dtype=np.int8)
        mat = np.zeros((ne, nt), dtype=np.int8)
        for tri, ti in self.tri_idx.items():
            for a, b in combinations(sorted(tri), 2):
                ei = self.edge_idx.get(frozenset({a, b}))
                if ei is not None: mat[ei, ti] = 1
        return mat

    def boundary_3(self):
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        if ntet == 0: return np.zeros((nt, 0), dtype=np.int8)
        mat = np.zeros((nt, ntet), dtype=np.int8)
        for tet, teti in self.tet_idx.items():
            for face in combinations(sorted(tet), 3):
                fi = self.tri_idx.get(frozenset(face))
                if fi is not None: mat[fi, teti] = 1
        return mat

    def betti(self):
        nv = len(self.vertices)
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        d1 = self.boundary_1()
        d2 = self.boundary_2()
        d3 = self.boundary_3()
        r1 = rank_f2(d1); r2 = rank_f2(d2); r3 = rank_f2(d3)
        return (nv - r1, (ne - r1) - r2, (nt - r2) - r3, ntet - r3)

    def stats(self):
        return (len(self.vertices), len(self.edge_idx),
                len(self.tri_idx), len(self.tet_idx))

    def neighbors(self, v):
        nbrs = set()
        for edge in self.edge_idx:
            if v in edge:
                nbrs.update(edge - {v})
        return nbrs

    def adjacency_matrix(self):
        n = len(self.vertices)
        v_idx = {v: i for i, v in enumerate(self.vertices)}
        mat = np.zeros((n, n), dtype=int)
        for edge in self.edge_idx:
            vl = list(edge)
            i, j = v_idx[vl[0]], v_idx[vl[1]]
            mat[i, j] = mat[j, i] = 1
        return mat, {v: i for i, v in enumerate(self.vertices)}


def build_8d_base():
    sc = SimplicialComplex()
    for i in range(9): sc.add_vertex(i)
    cube_edges = []
    for a in range(8):
        for b in range(a+1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))
                sc.add_edge(a, b)
    for i in range(8): sc.add_edge(8, i)
    for a, b in cube_edges: sc.add_triangle(a, b, 8)
    sc.insert_vertex_with_cone(9, {3, 5, 6})
    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})
    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})
    sc.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})
    return sc


def main():
    t0 = time.time()
    sc = build_8d_base()
    print(f"8D base: {sc.stats()}, beta={sc.betti()}")

    # ═══════════════════════════════════════════════════════════════
    # 1. Adjacency structure of the 8D complex
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("1. ADJACENCY STRUCTURE")
    print("="*60)

    adj, v_idx = sc.adjacency_matrix()
    idx_v = {i: v for v, i in v_idx.items()}
    n = len(sc.vertices)

    # Graph distance matrix
    dist = np.full((n, n), 999, dtype=int)
    for i in range(n):
        dist[i, i] = 0
    for edge in sc.edge_idx:
        vl = list(edge)
        i, j = v_idx[vl[0]], v_idx[vl[1]]
        dist[i, j] = dist[j, i] = 1

    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    print("\n  Graph distance matrix:")
    header = "    " + "  ".join(f"v{idx_v[i]:>2}" for i in range(n))
    print(header)
    for i in range(n):
        row = f"v{idx_v[i]:>2}" + "  ".join(f"{dist[i,j]:>4}" for j in range(n))
        print(f"  {row}")

    # ═══════════════════════════════════════════════════════════════
    # 2. "Adjacency" at the period-2 level
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("2. PERIOD-2 ADJACENCY CONCEPT")
    print("="*60)

    # In period 1, "adjacent" = Hamming distance 1 (share all but one bit)
    # "non-adjacent" = Hamming distance ≥ 2

    # At the 8D level, we have 13 vertices. What's the analog?
    # Hypothesis 1: graph distance in the 8D complex
    # Hypothesis 2: shared-neighbor count (Jaccard-like)
    # Hypothesis 3: "triangle distance" (are they in a common triangle?)

    # Shared neighbor count for all pairs
    print("\n  Shared neighbor count (|N(i) ∩ N(j)|):")
    shared = np.zeros((n, n), dtype=int)
    for i in range(n):
        ni = sc.neighbors(idx_v[i])
        for j in range(n):
            nj = sc.neighbors(idx_v[j])
            shared[i, j] = len(ni & nj)

    header = "    " + "  ".join(f"v{idx_v[i]:>2}" for i in range(n))
    print(header)
    for i in range(n):
        row = f"v{idx_v[i]:>2}" + "  ".join(f"{shared[i,j]:>4}" for j in range(n))
        print(f"  {row}")

    # Triangle co-membership
    print("\n  Triangle co-membership (are v_i and v_j in a common triangle?):")
    tri_co = np.zeros((n, n), dtype=int)
    for tri in sc.tri_idx:
        vl = list(tri)
        for a, b in combinations(vl, 2):
            i, j = v_idx[a], v_idx[b]
            tri_co[i, j] += 1
            tri_co[j, i] += 1

    header = "    " + "  ".join(f"v{idx_v[i]:>2}" for i in range(n))
    print(header)
    for i in range(n):
        row = f"v{idx_v[i]:>2}" + "  ".join(f"{tri_co[i,j]:>4}" for j in range(n))
        print(f"  {row}")

    # ═══════════════════════════════════════════════════════════════
    # 3. Period 1 insertion pattern analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("3. PERIOD 1 INSERTION PATTERN ANALYSIS")
    print("="*60)

    # For each period-1 insertion, analyze the target set's internal structure
    insertions = [
        ("v9→{3,5,6}", {3, 5, 6}),
        ("v10→{1,2,4,9}", {1, 2, 4, 9}),
        ("v11→{0,3,5,6}", {0, 3, 5, 6}),
        ("v12→{0-6,9-11}", {0,1,2,3,4,5,6,9,10,11}),
    ]

    # Build complex incrementally to check at each stage
    sc_stage = SimplicialComplex()
    for i in range(9): sc_stage.add_vertex(i)
    cube_edges = []
    for a in range(8):
        for b in range(a+1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))
                sc_stage.add_edge(a, b)
    for i in range(8): sc_stage.add_edge(8, i)
    for a, b in cube_edges: sc_stage.add_triangle(a, b, 8)

    for label, S in insertions:
        # Check internal edge density of S in CURRENT complex (before insertion)
        internal_edges = []
        for a, b in combinations(S, 2):
            if frozenset({a, b}) in sc_stage.edge_idx:
                internal_edges.append((a, b))

        internal_tris = []
        for a, b, c in combinations(S, 3):
            if frozenset({a, b, c}) in sc_stage.tri_idx:
                internal_tris.append((a, b, c))

        # Hamming distances within S (for cube vertices only)
        hamming = []
        for a, b in combinations(S, 2):
            if a < 8 and b < 8:
                hamming.append(bin(a ^ b).count('1'))

        print(f"\n  {label}:")
        print(f"    |S| = {len(S)}")
        print(f"    Internal edges: {len(internal_edges)} of {len(S)*(len(S)-1)//2} possible")
        print(f"    Internal triangles: {len(internal_tris)}")
        if hamming:
            print(f"    Hamming distances (cube verts): {hamming}")
        print(f"    Edge details: {internal_edges}")

        # Do the insertion
        vid = {"{3,5,6}": 9, "{1,2,4,9}": 10, "{0,3,5,6}": 11,
               "{0-6,9-11}": 12}
        v_id = [9, 10, 11, 12][insertions.index((label, S))]
        sc_stage.insert_vertex_with_cone(v_id, S)

    # ═══════════════════════════════════════════════════════════════
    # 4. Period 2 candidate rules
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("4. PERIOD 2: TESTING GEOMETRIC INSERTION RULES")
    print("="*60)

    # Rule candidates for period-2 "non-adjacent" insertions:
    # A) Connect to vertices with graph distance ≥ 2 from each other
    # B) Connect to vertices NOT in any common triangle
    # C) Connect to vertices with low shared-neighbor count

    sc = build_8d_base()

    # Find all subsets of size 3-4 where NO pair shares a triangle
    print("\n  Subsets (size 3) where no pair shares a triangle:")
    no_tri_subsets_3 = []
    for combo in combinations(sc.vertices, 3):
        ok = True
        for a, b in combinations(combo, 2):
            if tri_co[v_idx[a], v_idx[b]] > 0:
                ok = False
                break
        if ok:
            no_tri_subsets_3.append(combo)

    # Test each as a period-2 insertion
    results_notri = []
    for S in no_tri_subsets_3:
        sc_t = sc.copy()
        sc_t.insert_vertex_with_cone(13, frozenset(S))
        b = sc_t.betti()
        if b[1] == 0:
            results_notri.append((S, b))

    results_notri.sort(key=lambda x: x[1][2], reverse=True)
    print(f"  Total no-tri-shared subsets (size 3): {len(no_tri_subsets_3)}")
    print(f"  With β₁=0: {len(results_notri)}")
    if results_notri:
        print(f"  Top 5 by β₂:")
        for S, b in results_notri[:5]:
            print(f"    S={sorted(S)}, beta={b}")
        print(f"  Bottom 5 by β₂:")
        for S, b in results_notri[-5:]:
            print(f"    S={sorted(S)}, beta={b}")

    # Find subsets (size 3) where all pairs are NON-adjacent (graph dist ≥ 2)
    print("\n  Subsets (size 3) where all pairs have graph distance >= 2:")
    nonadj_subsets_3 = []
    for combo in combinations(sc.vertices, 3):
        ok = True
        for a, b in combinations(combo, 2):
            if dist[v_idx[a], v_idx[b]] < 2:
                ok = False
                break
        if ok:
            nonadj_subsets_3.append(combo)

    results_nonadj = []
    for S in nonadj_subsets_3:
        sc_t = sc.copy()
        sc_t.insert_vertex_with_cone(13, frozenset(S))
        b = sc_t.betti()
        if b[1] == 0:
            results_nonadj.append((S, b))

    results_nonadj.sort(key=lambda x: x[1][2], reverse=True)
    print(f"  Total non-adjacent subsets (size 3): {len(nonadj_subsets_3)}")
    print(f"  With β₁=0: {len(results_nonadj)}")
    if results_nonadj:
        print(f"  Top 5 by β₂:")
        for S, b in results_nonadj[:5]:
            print(f"    S={sorted(S)}, beta={b}")

    # ═══════════════════════════════════════════════════════════════
    # 5. What makes v12's K-closure "partial"?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("5. v12's PARTIAL K-CLOSURE STRUCTURE")
    print("="*60)

    # v12 → {0,1,2,3,4,5,6,9,10,11}, skipping v7 and v8
    # The BCC before v12 has vertices {0,...,11}
    # v12 connects to 10 of 12 vertices

    # What's special about v7 and v8?
    # v7 = 111 = the "opposite corner" from v0 = 000
    # v8 = body center

    # In terms of the BCC + ijk structure:
    # v8 connects to ALL cube corners (0-7)
    # v7 connects to {3,5,6,8} and indirectly to v9 (which connects to {3,5,6})

    # Hypothesis: v12 skips vertices that would create a CONE
    # If v12 connected to everything, β would be (1,0,0,0) trivially
    # The skip preserves the non-trivial topology

    # Test: for each vertex v12 DOES connect to, what would happen
    # if we removed that connection?
    print("  Effect of removing each connection from v12:")
    base_before_v12 = SimplicialComplex()
    for i in range(9): base_before_v12.add_vertex(i)
    for a in range(8):
        for b in range(a+1, 8):
            if bin(a ^ b).count('1') == 1:
                base_before_v12.add_edge(a, b)
    for i in range(8): base_before_v12.add_edge(8, i)
    cube_edges_list = [(a,b) for a in range(8) for b in range(a+1,8) if bin(a^b).count('1')==1]
    for a, b in cube_edges_list: base_before_v12.add_triangle(a, b, 8)
    base_before_v12.insert_vertex_with_cone(9, {3, 5, 6})
    base_before_v12.insert_vertex_with_cone(10, {1, 2, 4, 9})
    base_before_v12.insert_vertex_with_cone(11, {0, 3, 5, 6})

    full_S = {0,1,2,3,4,5,6,9,10,11}
    for skip_v in sorted(full_S):
        S_minus = full_S - {skip_v}
        sc_t = base_before_v12.copy()
        sc_t.insert_vertex_with_cone(12, S_minus)
        b = sc_t.betti()
        print(f"    Skip v{skip_v}: beta={b}")

    # What about adding v7 or v8 to v12's connections?
    print("\n  Effect of ADDING connections to v12's set:")
    for add_v in [7, 8]:
        S_plus = full_S | {add_v}
        sc_t = base_before_v12.copy()
        sc_t.insert_vertex_with_cone(12, S_plus)
        b = sc_t.betti()
        st = sc_t.stats()
        print(f"    Add v{add_v}: S={sorted(S_plus)}, beta={b}, Tet={st[3]}")

    S_all = full_S | {7, 8}
    sc_t = base_before_v12.copy()
    sc_t.insert_vertex_with_cone(12, S_all)
    b = sc_t.betti()
    st = sc_t.stats()
    print(f"    Add v7+v8: S={sorted(S_all)}, beta={b}, Tet={st[3]}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
