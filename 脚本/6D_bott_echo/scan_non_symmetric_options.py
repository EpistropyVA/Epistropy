# -*- coding: utf-8 -*-
"""
Scan Non-Symmetric Options for Period 2 "Quasi-K-Close Minus One"
================================================================
Checks the non-symmetrized complex with and without cube faces.
Excludes each vertex from the coning set of v16 to find the minimal non-zero beta_3.
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


def build_base_step7(with_cube_faces):
    sc = SimplicialComplex()
    # 8D Base (13V, indices 0..12)
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

    if with_cube_faces:
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

    # Steps 5-7 (Period 2 Build)
    sc.insert_vertex_with_cone(13, {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11})
    sc.insert_vertex_with_cone(14, {0, 1, 2, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(15, {0, 1, 2, 3, 4, 5, 6, 7, 9, 11})
    
    return sc


def run_scan(with_cube_faces):
    sc_base = build_base_step7(with_cube_faces)
    b_base, st_base = get_betti_f3(sc_base)
    print(f"Base complex (with_cube_faces={with_cube_faces}):")
    print(f"  Simplices: V={st_base[0]}, E={st_base[1]}, T={st_base[2]}, Tet={st_base[3]}")
    print(f"  F3-Betti:  beta = {b_base}")
    
    existing_vertices = sorted(list(sc_base.vertices))
    results = []
    
    # Also calculate K-closure (no exclusions)
    sc_k = sc_base.copy()
    sc_k.insert_vertex_with_cone(16, frozenset(existing_vertices))
    b_k, st_k = get_betti_f3(sc_k)
    print(f"  K-Closure   | Simplices: V={st_k[0]:<2}, E={st_k[1]:<3}, T={st_k[2]:<3}, Tet={st_k[3]:<3} | beta = {b_k}")
    
    for v_exclude in existing_vertices:
        sc_test = sc_base.copy()
        S_exclude = frozenset(v for v in existing_vertices if v != v_exclude)
        sc_test.insert_vertex_with_cone(16, S_exclude)
        b, st = get_betti_f3(sc_test)
        results.append((v_exclude, b, st))
        print(f"  Exclude v{v_exclude:<2} | Simplices: V={st[0]:<2}, E={st[1]:<3}, T={st[2]:<3}, Tet={st[3]:<3} | beta = {b}")
    
    print("-" * 60)
    # Find results with beta_3 > 0 or minimal beta_2
    min_b2 = min(r[1][2] for r in results)
    best_candidates = [r for r in results if r[1][2] == min_b2]
    print(f"Candidates with minimal beta_2 ({min_b2}):")
    for v_exclude, b, st in best_candidates:
        print(f"  Exclude v{v_exclude:<2} | beta = {b} | Tet = {st[3]}")
    print("\n")


def main():
    print("=" * 80)
    print(" SCAN 1: WITHOUT CUBE FACES (ORIGINAL)")
    print("=" * 80)
    run_scan(False)

    print("=" * 80)
    print(" SCAN 2: WITH CUBE FACES")
    print("=" * 80)
    run_scan(True)


if __name__ == "__main__":
    main()
