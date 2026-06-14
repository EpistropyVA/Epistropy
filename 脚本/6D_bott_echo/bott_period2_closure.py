"""
Bott Period 2 — Closure search
================================
Instead of maximizing β₂, test K-closure (connect to all) at each step.
Also test targeted closures connecting to v8 (body center) and v12 (K-point).

Key insight: β₂ closure requires tetrahedra (im ∂₃ > 0).
Tetrahedra form when new vertex connects to 3+ vertices that form a triangle.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
import time
import random


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

    def add_vertex(self, v):
        self.vertices.append(v)
    def add_edge(self, a, b):
        key = frozenset({a, b})
        if key not in self.edge_idx:
            self.edge_idx[key] = len(self.edge_idx)
    def add_triangle(self, a, b, c):
        key = frozenset({a, b, c})
        if key not in self.tri_idx:
            self.tri_idx[key] = len(self.tri_idx)
    def add_tetrahedron(self, a, b, c, d):
        key = frozenset({a, b, c, d})
        if key not in self.tet_idx:
            self.tet_idx[key] = len(self.tet_idx)

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
        if ne == 0:
            return np.zeros((nv, 0), dtype=np.int8)
        v_idx = {v: i for i, v in enumerate(self.vertices)}
        mat = np.zeros((nv, ne), dtype=np.int8)
        for edge, ei in self.edge_idx.items():
            for v in edge:
                mat[v_idx[v], ei] = 1
        return mat

    def boundary_2(self):
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        if nt == 0:
            return np.zeros((ne, 0), dtype=np.int8)
        mat = np.zeros((ne, nt), dtype=np.int8)
        for tri, ti in self.tri_idx.items():
            verts = sorted(tri)
            for a, b in combinations(verts, 2):
                ei = self.edge_idx.get(frozenset({a, b}))
                if ei is not None:
                    mat[ei, ti] = 1
        return mat

    def boundary_3(self):
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        if ntet == 0:
            return np.zeros((nt, 0), dtype=np.int8)
        mat = np.zeros((nt, ntet), dtype=np.int8)
        for tet, teti in self.tet_idx.items():
            verts = sorted(tet)
            for face in combinations(verts, 3):
                fi = self.tri_idx.get(frozenset(face))
                if fi is not None:
                    mat[fi, teti] = 1
        return mat

    def betti(self):
        nv = len(self.vertices)
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        d1 = self.boundary_1()
        d2 = self.boundary_2()
        d3 = self.boundary_3()
        rank_d1 = rank_f2(d1)
        rank_d2 = rank_f2(d2)
        rank_d3 = rank_f2(d3)
        b0 = nv - rank_d1
        b1 = (ne - rank_d1) - rank_d2
        b2 = (nt - rank_d2) - rank_d3
        b3 = ntet - rank_d3
        return (b0, b1, b2, b3)

    def stats(self):
        return (len(self.vertices), len(self.edge_idx),
                len(self.tri_idx), len(self.tet_idx))

    def triangle_count_for_vertices(self, vset):
        """Count how many existing triangles have all 3 vertices in vset."""
        count = 0
        for tri in self.tri_idx:
            if tri.issubset(vset):
                count += 1
        return count


def build_8d_base():
    sc = SimplicialComplex()
    for i in range(9):
        sc.add_vertex(i)
    cube_edges = []
    for a in range(8):
        for b in range(a+1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))
                sc.add_edge(a, b)
    for i in range(8):
        sc.add_edge(8, i)
    for a, b in cube_edges:
        sc.add_triangle(a, b, 8)
    sc.insert_vertex_with_cone(9, {3, 5, 6})
    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})
    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})
    sc.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})
    return sc


def find_best_b2_insertion(base_sc, new_vertex, max_subsets=4096, seed=42):
    """Find insertion maximizing β₂ with β₁=0."""
    existing = base_sc.vertices
    n = len(existing)
    total = 2**n - 1
    rng = random.Random(seed)

    if total <= 8192:
        subsets = [frozenset(combo) for size in range(1, n+1)
                   for combo in combinations(existing, size)]
    else:
        subsets = set()
        subsets.add(frozenset(existing))
        for skip in existing:
            subsets.add(frozenset(v for v in existing if v != skip))
        for size in [2, 3, 4, 5, 6]:
            all_c = list(combinations(existing, size))
            if len(all_c) <= 400:
                subsets.update(frozenset(c) for c in all_c)
            else:
                subsets.update(frozenset(c) for c in rng.sample(all_c, 400))
        remaining = max_subsets - len(subsets)
        if remaining > 0:
            pool = []
            for size in range(1, n+1):
                for combo in combinations(existing, size):
                    fs = frozenset(combo)
                    if fs not in subsets:
                        pool.append(fs)
            if len(pool) > remaining:
                subsets.update(rng.sample(pool, remaining))
            else:
                subsets.update(pool)
        subsets = list(subsets)

    best_b2 = -1
    best_S = None
    best_betti = None
    for S in subsets:
        sc = base_sc.copy()
        sc.insert_vertex_with_cone(new_vertex, S)
        b = sc.betti()
        if b[1] == 0 and b[2] > best_b2:
            best_b2 = b[2]
            best_S = S
            best_betti = b
    if best_S is None:
        for S in subsets:
            sc = base_sc.copy()
            sc.insert_vertex_with_cone(new_vertex, S)
            b = sc.betti()
            if b[2] > best_b2:
                best_b2 = b[2]
                best_S = S
                best_betti = b
    return best_S, best_betti


def main():
    t_start = time.time()
    base = build_8d_base()
    print(f"8D base: beta={base.betti()}, stats={base.stats()}")

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: K-closure at each dimension
    # Build 3 steps of β₂-maximizing, then K-close at 4th step (period 1 mirror)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 1: 3 build + 1 K-close (period 1 mirror)")
    print("="*60)

    sc = base.copy()
    vid = 13

    # 3 building steps (maximize β₂)
    for step in range(3):
        dim = 9 + step
        best_S, best_b = find_best_b2_insertion(sc, vid)
        sc.insert_vertex_with_cone(vid, best_S)
        b = sc.betti()
        st = sc.stats()
        print(f"  {dim}D build: V={st[0]}, beta={b}, Tet={st[3]}, S={sorted(best_S)}")
        vid += 1

    # K-closure at 12D: connect to ALL vertices
    dim = 12
    all_verts = frozenset(sc.vertices)
    sc_k = sc.copy()
    sc_k.insert_vertex_with_cone(vid, all_verts)
    b_k = sc_k.betti()
    st_k = sc_k.stats()
    print(f"  {dim}D K-CLOSE (all): V={st_k[0]}, beta={b_k}, Tet={st_k[3]}")

    # Also try connecting to all EXCEPT v8, v12
    sc_k2 = sc.copy()
    s2 = frozenset(v for v in sc.vertices if v not in {8, 12})
    sc_k2.insert_vertex_with_cone(vid, s2)
    b_k2 = sc_k2.betti()
    st_k2 = sc_k2.stats()
    print(f"  {dim}D K-CLOSE (no v8,v12): V={st_k2[0]}, beta={b_k2}, Tet={st_k2[3]}")

    # Try v8+v12 only
    sc_k3 = sc.copy()
    sc_k3.insert_vertex_with_cone(vid, frozenset({8, 12}))
    b_k3 = sc_k3.betti()
    print(f"  {dim}D (v8+v12 only): beta={b_k3}")

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: K-closure from 8D directly at each step
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 2: K-closure at EACH step from 8D base")
    print("="*60)

    sc = base.copy()
    vid = 13
    for dim in range(9, 17):
        sc_test = sc.copy()
        all_v = frozenset(sc.vertices)
        sc_test.insert_vertex_with_cone(vid, all_v)
        b = sc_test.betti()
        st = sc_test.stats()
        tri_in_S = sc.triangle_count_for_vertices(all_v)
        print(f"  {dim}D K-close: V={st[0]}, beta={b}, Tet={st[3]}, triangles_in_S={tri_in_S}")

        # Accept the K-closure and continue
        sc = sc_test
        vid += 1

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Minimum β₂ search
    # At each step from 8D, find insertion that MINIMIZES β₂
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 3: Minimize β₂ at each step")
    print("="*60)

    sc = base.copy()
    vid = 13
    for dim in range(9, 17):
        existing = sc.vertices
        n = len(existing)
        total = 2**n - 1
        rng = random.Random(42 + dim)

        if total <= 8192:
            subsets = [frozenset(combo) for size in range(1, n+1)
                       for combo in combinations(existing, size)]
        else:
            subsets = set()
            subsets.add(frozenset(existing))
            for skip in existing:
                subsets.add(frozenset(v for v in existing if v != skip))
            for size in range(2, n):
                all_c = list(combinations(existing, size))
                if len(all_c) <= 300:
                    subsets.update(frozenset(c) for c in all_c)
                else:
                    subsets.update(frozenset(c) for c in rng.sample(all_c, 300))
            subsets = list(subsets)

        min_b2 = 999999
        min_S = None
        min_betti = None
        min_tet = 0
        for S in subsets:
            sc_t = sc.copy()
            sc_t.insert_vertex_with_cone(vid, S)
            b = sc_t.betti()
            st = sc_t.stats()
            if b[2] < min_b2 or (b[2] == min_b2 and st[3] > min_tet):
                min_b2 = b[2]
                min_S = S
                min_betti = b
                min_tet = st[3]

        sc.insert_vertex_with_cone(vid, min_S)
        st = sc.stats()
        print(f"  {dim}D min-β₂: V={st[0]}, beta={min_betti}, Tet={st[3]}, |S|={len(min_S)}, S={sorted(min_S)}")
        vid += 1

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: Diagnostic — WHY no tetrahedra?
    # Check triangle density in the 8D base
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("EXPERIMENT 4: Triangle structure diagnostic")
    print("="*60)

    sc = base.copy()
    print(f"  8D base: {sc.stats()}")
    print(f"  Triangles: {len(sc.tri_idx)}")

    # For each triple of vertices that form a triangle,
    # check if they share edges with a 4th vertex (potential tetrahedron)
    print("\n  Triangle list:")
    for tri in sorted(sc.tri_idx, key=lambda t: sorted(t)):
        verts = sorted(tri)
        # Find vertices adjacent to all 3
        common_neighbors = set(sc.vertices)
        for v in verts:
            neighbors = set()
            for edge in sc.edge_idx:
                if v in edge:
                    neighbors.update(edge - {v})
            common_neighbors &= neighbors
        common_neighbors -= set(verts)
        if common_neighbors:
            print(f"    {verts} — common neighbors: {sorted(common_neighbors)}")

    # Check: if we connect a new vertex to vertices that form triangles,
    # do we get tetrahedra?
    print("\n  Testing: connect new vertex to all of {0,1,2,3,4,5,6,7,8}")
    sc_test = base.copy()
    sc_test.insert_vertex_with_cone(13, frozenset(range(9)))
    st = sc_test.stats()
    b = sc_test.betti()
    print(f"    V={st[0]}, E={st[1]}, T={st[2]}, Tet={st[3]}, beta={b}")

    print(f"\n  Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
