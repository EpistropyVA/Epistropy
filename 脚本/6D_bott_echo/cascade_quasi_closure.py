# -*- coding: utf-8 -*-
"""
Quasi-K-Close Minus One Scan (Non-Symmetrized and Symmetrized)
==============================================================
Performs the "quasi-K-close minus one" scan at the closure step of Period 2.
Excludes each vertex (or Z2 pair) to locate the critical closure center
analogous to v8 in Period 1 (the seed for Period 3).
"""

import sys
import io
import numpy as np
from itertools import combinations

# Ensure UTF-8 stdout encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


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


def g_map(v):
    if v < 8:
        return v ^ 7
    if v == 8:
        return 8
    if 9 <= v <= 16:
        return v + 10
    if 19 <= v <= 26:
        return v - 10
    raise ValueError(f"Unknown vertex: {v}")


def perm_sign(vals):
    n = len(vals)
    inv = 0
    for i in range(n):
        for j in range(i+1, n):
            if vals[i] > vals[j]:
                inv += 1
    return 1 if inv % 2 == 0 else -1


def rank_f3(mat):
    if mat.size == 0:
        return 0
    m = mat.copy() % 3
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        found = -1
        for row in range(pivot_row, rows):
            if m[row, col] != 0:
                found = row
                break
        if found == -1:
            continue
        if pivot_row != found:
            temp = m[pivot_row].copy()
            m[pivot_row] = m[found]
            m[found] = temp
        pivot_val = m[pivot_row, col]
        if pivot_val == 2:
            m[pivot_row] = (m[pivot_row] * 2) % 3
        for row in range(rows):
            if row != pivot_row and m[row, col] != 0:
                factor = m[row, col]
                m[row] = (m[row] - factor * m[pivot_row]) % 3
        pivot_row += 1
    return pivot_row


def get_betti_f3(sc):
    v_list = sorted(list(sc.vertices))
    edge_list = sorted([sorted(list(e)) for e in sc.edge_idx.keys()])
    tri_list = sorted([sorted(list(t)) for t in sc.tri_idx.keys()])
    tet_list = sorted([sorted(list(tet)) for tet in sc.tet_idx.keys()])

    m0, m1, m2, m3 = len(v_list), len(edge_list), len(tri_list), len(tet_list)

    v_map = {v: i for i, v in enumerate(v_list)}
    edge_map = {frozenset(e): i for i, e in enumerate(edge_list)}
    tri_map = {frozenset(t): i for i, t in enumerate(tri_list)}

    d1 = np.zeros((m0, m1), dtype=int)
    for j, e in enumerate(edge_list):
        a, b = e
        d1[v_map[a], j] = -1
        d1[v_map[b], j] = 1

    d2 = np.zeros((m1, m2), dtype=int)
    for j, t in enumerate(tri_list):
        a, b, c = t
        d2[edge_map[frozenset({b, c})], j] = 1
        d2[edge_map[frozenset({a, c})], j] = -1
        d2[edge_map[frozenset({a, b})], j] = 1

    d3 = np.zeros((m2, m3), dtype=int)
    for j, tet in enumerate(tet_list):
        a, b, c, d = tet
        d3[tri_map[frozenset({b, c, d})], j] = 1
        d3[tri_map[frozenset({a, c, d})], j] = -1
        d3[tri_map[frozenset({a, b, d})], j] = 1
        d3[tri_map[frozenset({a, b, c})], j] = -1

    r1 = rank_f3(d1)
    r2 = rank_f3(d2)
    r3 = rank_f3(d3)

    return (m0 - r1, m1 - r1 - r2, m2 - r2 - r3, m3 - r3), (m0, m1, m2, m3)


def build_base_original():
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

    sc.insert_vertex_with_cone(13, {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11})
    sc.insert_vertex_with_cone(14, {0, 1, 2, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(15, {0, 1, 2, 3, 4, 5, 6, 7, 9, 11})
    return sc


def build_base_symmetric():
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
    sc.insert_vertex_with_cone(19, {4, 2, 1})

    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})
    sc.insert_vertex_with_cone(20, {6, 5, 3, 19})

    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})
    sc.insert_vertex_with_cone(21, {7, 4, 2, 1})

    sc.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(22, {7, 6, 5, 4, 3, 2, 1, 19, 20, 21})

    cube_faces = [
        (0,1,3), (0,2,3),
        (4,5,7), (4,6,7),
        (0,2,6), (0,4,6),
        (1,3,7), (1,5,7),
        (2,3,7), (2,6,7),
        (0,1,5), (0,4,5)
    ]
    for a, b, c in cube_faces:
        sc.add_edge(a, b)
        sc.add_edge(b, c)
        sc.add_edge(a, c)
        sc.add_triangle(a, b, c)

    sc.insert_vertex_with_cone(13, {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11})
    sc.insert_vertex_with_cone(23, {0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21})

    sc.insert_vertex_with_cone(14, {0, 1, 2, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(24, {1, 2, 3, 5, 6, 7, 19, 20, 21})

    sc.insert_vertex_with_cone(15, {0, 1, 2, 3, 4, 5, 6, 7, 9, 11})
    sc.insert_vertex_with_cone(25, {0, 1, 2, 3, 4, 5, 6, 7, 19, 21})

    return sc


def main():
    # ── PART 1: Original Non-Symmetrized Scan ──
    print("=" * 80)
    print(" PART 1: ORIGINAL NON-SYMMETRIZED QUASI-K-CLOSE MINUS ONE SCAN")
    print("=" * 80)
    
    sc_orig = build_base_original()
    existing_orig = sorted(list(sc_orig.vertices))
    
    for v in existing_orig:
        sc_test = sc_orig.copy()
        S = frozenset(x for x in existing_orig if x != v)
        sc_test.insert_vertex_with_cone(16, S)
        b, st = get_betti_f3(sc_test)
        print(f"  Exclude v{v:<2} | Simplices: V={st[0]:<2}, E={st[1]:<3}, T={st[2]:<3}, Tet={st[3]:<3} | beta = ({b[0]},{b[1]},{b[2]},{b[3]})")
    print()

    # ── PART 2: Symmetrized Scan (Antipodal Pairs Excluded) ──
    print("=" * 80)
    print(" PART 2: SYMMETRIZED QUASI-K-CLOSE MINUS ANTIPODAL PAIR SCAN")
    print("=" * 80)
    
    sc_sym = build_base_symmetric()
    existing_sym = sorted(list(sc_sym.vertices))
    pairs = [
        frozenset({0, 7}), frozenset({1, 6}), frozenset({2, 5}), frozenset({3, 4}),
        frozenset({8}),
        frozenset({9, 19}), frozenset({10, 20}), frozenset({11, 21}), frozenset({12, 22}),
        frozenset({13, 23}), frozenset({14, 24}), frozenset({15, 25})
    ]
    
    for p in pairs:
        sc_test = sc_sym.copy()
        S = frozenset(v for v in existing_sym if v not in p)
        sc_test.insert_vertex_with_cone(16, S)
        sc_test.insert_vertex_with_cone(26, S)
        b, st = get_betti_f3(sc_test)
        pair_str = ", ".join(map(str, sorted(list(p))))
        print(f"  Exclude {{{pair_str:<5}}} | Simplices: V={st[0]:<2}, E={st[1]:<3}, T={st[2]:<3}, Tet={st[3]:<3} | beta = ({b[0]},{b[1]},{b[2]},{b[3]})")
    print("=" * 80)


if __name__ == "__main__":
    main()
