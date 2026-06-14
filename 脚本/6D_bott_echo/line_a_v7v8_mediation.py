"""
Line A: v7 and v8 mediation analysis
=====================================
Question: v12 doesn't connect to v7(111) and v8(body center).
Hypothesis: the relationship EXISTS but is mediated through the network,
not absent. The β₂=3 residue comes from the topology of mediation,
not from missing connections.

Analysis:
1. What are v7 and v8's roles in the existing complex?
2. Path structure: how does v12 reach v7/v8 through existing edges?
3. Adding v7/v8 connections one at a time — what changes homologically?
4. What ARE the 3 surviving β₂ cycles? What do they look like?
5. Internal cancellation: do v7/v8 already participate in ∂₂ chains
   that reach v12's cone?
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


def kernel_basis_f2(mat):
    """Return basis vectors of ker(mat) over F₂."""
    if mat.size == 0:
        return []
    m = mat.copy() % 2
    rows, cols = m.shape
    pivot_col = [-1] * rows
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
        pivot_col[pivot_row] = col
        for row in range(rows):
            if row != pivot_row and m[row, col] == 1:
                m[row] = (m[row] + m[pivot_row]) % 2
        pivot_row += 1

    rank = pivot_row
    pivot_cols = set(pivot_col[r] for r in range(rank))
    free_cols = [c for c in range(cols) if c not in pivot_cols]

    basis = []
    for fc in free_cols:
        vec = np.zeros(cols, dtype=np.int8)
        vec[fc] = 1
        for r in range(rank):
            if m[r, fc] == 1:
                vec[pivot_col[r]] = 1
        basis.append(vec)
    return basis


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

    def edge_list(self):
        return [sorted(e) for e in self.edge_idx]

    def tri_list(self):
        return [sorted(t) for t in self.tri_idx]

    def get_edge_index(self, a, b):
        return self.edge_idx.get(frozenset({a, b}))

    def get_tri_index(self, a, b, c):
        return self.tri_idx.get(frozenset({a, b, c}))


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
    print(f"8D base: V={sc.stats()[0]}, E={sc.stats()[1]}, T={sc.stats()[2]}, Tet={sc.stats()[3]}")
    print(f"  beta = {sc.betti()}")

    # ═══════════════════════════════════════════════════════════════
    # 1. v7 and v8's structural roles
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("1. VERTEX ROLES IN 8D BASE")
    print("="*60)

    vertex_names = {
        0: "000", 1: "001", 2: "010", 3: "011",
        4: "100", 5: "101", 6: "110", 7: "111",
        8: "center", 9: "ijk-1(3,5,6)", 10: "ijk-2(1,2,4,9)",
        11: "ijk-3(0,3,5,6)", 12: "K-close"
    }

    for v in sc.vertices:
        nbrs = sorted(sc.neighbors(v))
        # Count triangles containing v
        tri_count = sum(1 for t in sc.tri_idx if v in t)
        print(f"  v{v} ({vertex_names.get(v,'')}): degree={len(nbrs)}, triangles={tri_count}, neighbors={nbrs}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Path analysis: v12 to v7, v12 to v8
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("2. PATH ANALYSIS: v12 to v7 and v8")
    print("="*60)

    # BFS shortest paths
    def bfs_paths(sc, start, end, max_depth=4):
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        all_paths = []
        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth + 1:
                break
            if node == end and len(path) > 1:
                all_paths.append(path)
                continue
            for nbr in sc.neighbors(node):
                if nbr not in visited or nbr == end:
                    queue.append((nbr, path + [nbr]))
            visited.add(node)
        return all_paths

    # All shortest paths from v12 to v7
    paths_12_7 = bfs_paths(sc, 12, 7)
    print(f"\n  Paths v12 → v7 (up to depth 4):")
    for p in paths_12_7:
        print(f"    {' → '.join(f'v{v}' for v in p)} (length {len(p)-1})")

    # All shortest paths from v12 to v8
    paths_12_8 = bfs_paths(sc, 12, 8)
    print(f"\n  Paths v12 → v8 (up to depth 4):")
    for p in paths_12_8:
        print(f"    {' → '.join(f'v{v}' for v in p)} (length {len(p)-1})")

    # v7 to v8
    paths_7_8 = bfs_paths(sc, 7, 8)
    print(f"\n  Paths v7 → v8:")
    for p in paths_7_8:
        print(f"    {' → '.join(f'v{v}' for v in p)} (length {len(p)-1})")

    # ═══════════════════════════════════════════════════════════════
    # 3. Homological effect of adding v7/v8 connections to v12
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("3. ADDING v7/v8 CONNECTIONS TO v12")
    print("="*60)

    # Base: v12 connected to {0,1,2,3,4,5,6,9,10,11}
    # Test: add v12-v7, v12-v8, both

    # +v7 only
    sc_v7 = sc.copy()
    sc_v7.add_edge(12, 7)
    # Check for new triangles: {12, 7, x} where {7,x} is an existing edge
    new_tri_v7 = []
    for nbr in sc.neighbors(7):
        if sc.get_edge_index(12, nbr) is not None:
            sc_v7.add_triangle(12, 7, nbr)
            new_tri_v7.append((12, 7, nbr))
    # Check for new tetrahedra
    new_tet_v7 = []
    for t in list(sc_v7.tri_idx):
        if 7 in t and 12 in t:
            third = (t - {7, 12}).pop()
            # Need {12, 7, third, x} where all 4 faces exist
            for x in sc_v7.vertices:
                if x not in {7, 12, third}:
                    if (sc_v7.get_tri_index(12, 7, x) is not None and
                        sc_v7.get_tri_index(12, third, x) is not None and
                        sc_v7.get_tri_index(7, third, x) is not None):
                        sc_v7.add_tetrahedron(12, 7, third, x)
                        new_tet_v7.append((12, 7, third, x))

    b_v7 = sc_v7.betti()
    print(f"  +v12-v7 edge:")
    print(f"    new triangles: {new_tri_v7}")
    print(f"    new tetrahedra: {new_tet_v7}")
    print(f"    beta = {b_v7}")

    # +v8 only
    sc_v8 = sc.copy()
    sc_v8.add_edge(12, 8)
    new_tri_v8 = []
    for nbr in sc.neighbors(8):
        if sc.get_edge_index(12, nbr) is not None:
            sc_v8.add_triangle(12, 8, nbr)
            new_tri_v8.append((12, 8, nbr))
    new_tet_v8 = []
    for a, b in combinations([v for v in sc_v8.vertices if v not in {8, 12}], 2):
        if (sc_v8.get_tri_index(12, 8, a) is not None and
            sc_v8.get_tri_index(12, 8, b) is not None and
            sc_v8.get_tri_index(12, a, b) is not None and
            sc_v8.get_tri_index(8, a, b) is not None):
            sc_v8.add_tetrahedron(12, 8, a, b)
            new_tet_v8.append((12, 8, a, b))

    b_v8 = sc_v8.betti()
    print(f"\n  +v12-v8 edge:")
    print(f"    new triangles ({len(new_tri_v8)}): {new_tri_v8}")
    print(f"    new tetrahedra ({len(new_tet_v8)}): {new_tet_v8}")
    print(f"    beta = {b_v8}")

    # +both
    sc_both = sc.copy()
    sc_both.add_edge(12, 7)
    sc_both.add_edge(12, 8)
    for nbr in sc.neighbors(7):
        if sc.get_edge_index(12, nbr) is not None:
            sc_both.add_triangle(12, 7, nbr)
    for nbr in sc.neighbors(8):
        if sc.get_edge_index(12, nbr) is not None:
            sc_both.add_triangle(12, 8, nbr)
    # 7-8 edge exists (cube corner to center)
    if sc.get_edge_index(7, 8) is not None and sc.get_edge_index(12, 7) is not None and sc.get_edge_index(12, 8) is not None:
        sc_both.add_triangle(12, 7, 8)
    # tetrahedra
    for a, b, c in combinations(sc_both.vertices, 3):
        if 12 in {a,b,c}:
            others = [x for x in {a,b,c} if x != 12]
            if len(others) == 2:
                if (sc_both.get_tri_index(12, others[0], others[1]) is not None):
                    for x in sc_both.vertices:
                        if x not in {a,b,c}:
                            faces = [frozenset({a,b,x}), frozenset({a,c,x}), frozenset({b,c,x}), frozenset({a,b,c})]
                            if all(f in sc_both.tri_idx for f in faces):
                                sc_both.add_tetrahedron(a,b,c,x)

    b_both = sc_both.betti()
    st_both = sc_both.stats()
    print(f"\n  +v12-v7 AND v12-v8:")
    print(f"    stats: V={st_both[0]}, E={st_both[1]}, T={st_both[2]}, Tet={st_both[3]}")
    print(f"    beta = {b_both}")

    # ═══════════════════════════════════════════════════════════════
    # 4. What ARE the 3 surviving β₂ cycles?
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("4. THE 3 SURVIVING beta2 CYCLES")
    print("="*60)

    d2 = sc.boundary_2()
    d3 = sc.boundary_3()  # should be empty (no tet)

    # ker ∂₂ basis
    ker_d2_basis = kernel_basis_f2(d2)
    print(f"  dim(ker ∂₂) = {len(ker_d2_basis)}")

    # im ∂₃ = 0 (no tetrahedra)
    # So H₂ = ker ∂₂ / im ∂₃ = ker ∂₂
    # But we also need to check: some of ker ∂₂ might be im ∂₃... no, ∂₃ is zero matrix

    # Actually β₂ = dim(ker ∂₂) - rank(∂₃) = dim(ker ∂₂) since no tet
    # But β₂ = 3, and dim(ker ∂₂) might be larger... let me check

    rank_d2 = rank_f2(d2)
    nt = len(sc.tri_idx)
    print(f"  |T| = {nt}, rank(∂₂) = {rank_d2}, dim(ker ∂₂) = {nt - rank_d2}")
    print(f"  β₂ = {nt - rank_d2} (= dim ker ∂₂ since no ∂₃)")

    # Identify the 3 cycle representatives
    tri_list = sorted(sc.tri_idx.items(), key=lambda x: x[1])
    edge_list_sorted = sorted(sc.edge_idx.items(), key=lambda x: x[1])

    print(f"\n  Triangle index:")
    for tri, idx in tri_list:
        print(f"    T{idx}: {sorted(tri)}")

    print(f"\n  β₂ cycle representatives (as triangle combinations over F₂):")
    for i, vec in enumerate(ker_d2_basis):
        tris_in_cycle = [sorted(tri_list[j][0]) for j in range(len(vec)) if vec[j] == 1]
        # Which vertices appear?
        verts = set()
        for t in tris_in_cycle:
            verts.update(t)
        print(f"    Cycle {i}: {len(tris_in_cycle)} triangles, vertices {sorted(verts)}")
        for t in tris_in_cycle:
            print(f"      {t}")

    # ═══════════════════════════════════════════════════════════════
    # 5. Mediation structure
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("5. MEDIATION STRUCTURE")
    print("="*60)

    # v9 connects to {3,5,6} = v7's Hamming-1 neighbors
    v7_h1 = [v for v in range(8) if bin(v ^ 7).count('1') == 1]
    print(f"  v7 (111) Hamming-1 neighbors: {v7_h1}")
    print(f"  v9 connects to: {{3, 5, 6}} = v7's H1 neighbors exactly")

    # v11 connects to {0,3,5,6} — includes v7's H1 neighbors + v0
    print(f"  v11 connects to: {{0, 3, 5, 6}} — v7's H1 + v0")

    # Common neighbors of v12 and v7 (through existing edges)
    n12 = sc.neighbors(12)
    n7 = sc.neighbors(7)
    common = n12 & n7
    print(f"\n  v12 neighbors: {sorted(n12)}")
    print(f"  v7 neighbors: {sorted(n7)}")
    print(f"  Common neighbors: {sorted(common)}")

    # Common neighbors of v12 and v8
    n8 = sc.neighbors(8)
    common_8 = n12 & n8
    print(f"\n  v8 neighbors: {sorted(n8)}")
    print(f"  Common neighbors v12∩v8: {sorted(common_8)}")

    # Triangles containing v7
    t_v7 = [sorted(t) for t in sc.tri_idx if 7 in t]
    print(f"\n  Triangles containing v7: {t_v7}")

    # Triangles containing v8
    t_v8 = [sorted(t) for t in sc.tri_idx if 8 in t]
    print(f"  Triangles containing v8 ({len(t_v8)} total)")

    # Do any β₂ cycles pass through v7 or v8?
    print(f"\n  β₂ cycles passing through v7 or v8:")
    for i, vec in enumerate(ker_d2_basis):
        tris_in_cycle = [tri_list[j][0] for j in range(len(vec)) if vec[j] == 1]
        has_v7 = any(7 in t for t in tris_in_cycle)
        has_v8 = any(8 in t for t in tris_in_cycle)
        print(f"    Cycle {i}: v7={has_v7}, v8={has_v8}")

    # ═══════════════════════════════════════════════════════════════
    # 6. The cancellation hypothesis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("6. CANCELLATION HYPOTHESIS")
    print("="*60)

    # If v7's role is already mediated by v9 (which connects to v7's H1 neighbors),
    # then v9 is the "internal representative" of v7.
    # Check: is v9 homologically equivalent to v7 in some sense?

    # v7's star: edges to {3,5,6,8}, triangles {3,5,8},{3,6,8},{5,6,8} (BCC cones), {3,5,6} if exists
    # v9's star: edges to {3,5,6,10,12}, triangles involving v9

    t_v9 = [sorted(t) for t in sc.tri_idx if 9 in t]
    print(f"  v9 triangles: {t_v9}")

    t_v7_detail = [sorted(t) for t in sc.tri_idx if 7 in t]
    print(f"  v7 triangles: {t_v7_detail}")

    # Edge overlap between v7-star and v9-star
    print(f"\n  v7 edges: {sorted([sorted(e) for e in sc.edge_idx if 7 in e])}")
    print(f"  v9 edges: {sorted([sorted(e) for e in sc.edge_idx if 9 in e])}")

    # Key question: if we contract v9 onto v7 (identify them),
    # would the complex change homologically?
    # This is related to whether v9 and v7 are in the same "position" in the nerve

    print(f"\n  v7 degree: {len(sc.neighbors(7))}")
    print(f"  v9 degree: {len(sc.neighbors(9))}")
    print(f"  v7 neighbors: {sorted(sc.neighbors(7))}")
    print(f"  v9 neighbors: {sorted(sc.neighbors(9))}")
    print(f"  Symmetric difference: {sorted(sc.neighbors(7).symmetric_difference(sc.neighbors(9)))}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
