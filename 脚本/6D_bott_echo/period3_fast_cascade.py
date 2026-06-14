# -*- coding: utf-8 -*-
import sys
import io
import numpy as np
import time
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class SimplicialComplex:
    def __init__(self):
        self.vertices = []
        self.edge_idx = {}
        self.tri_idx = {}
        self.tet_idx = {}

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

    def copy(self):
        c = SimplicialComplex()
        c.vertices = list(self.vertices)
        c.edge_idx = dict(self.edge_idx)
        c.tri_idx = dict(self.tri_idx)
        c.tet_idx = dict(self.tet_idx)
        return c


def g_map(v):
    if v < 8:
        return v ^ 7
    if v == 8:
        return 8
    # Period 1 & 2
    if 9 <= v <= 16:
        return v + 10
    if 19 <= v <= 26:
        return v - 10
    # Period 3
    if 29 <= v <= 32:
        return v + 10
    if 39 <= v <= 42:
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
        # Find pivot in column col from pivot_row to rows
        col_slice = m[pivot_row:rows, col]
        nonzero = np.where(col_slice != 0)[0]
        if len(nonzero) > 0:
            found = nonzero[0] + pivot_row
        if found == -1:
            continue
        if pivot_row != found:
            temp = m[pivot_row].copy()
            m[pivot_row] = m[found]
            m[found] = temp
        pivot_val = m[pivot_row, col]
        if pivot_val == 2:
            m[pivot_row] = (m[pivot_row] * 2) % 3
        # Vectorized elimination
        col_vals = m[:, col].copy()
        col_vals[pivot_row] = 0
        nonzero_rows = np.where(col_vals != 0)[0]
        if len(nonzero_rows) > 0:
            factors = col_vals[nonzero_rows, np.newaxis]
            m[nonzero_rows] = (m[nonzero_rows] - factors * m[pivot_row]) % 3
        pivot_row += 1
    return pivot_row


def build_base_step7():
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


def analyze_homology_representation(sc, name):
    t0 = time.time()
    v_list = sorted(list(sc.vertices))
    edge_list = sorted([sorted(list(e)) for e in sc.edge_idx.keys()])
    tri_list = sorted([sorted(list(t)) for t in sc.tri_idx.keys()])
    tet_list = sorted([sorted(list(tet)) for tet in sc.tet_idx.keys()])

    m0, m1, m2, m3 = len(v_list), len(edge_list), len(tri_list), len(tet_list)

    v_map = {v: i for i, v in enumerate(v_list)}
    edge_map = {frozenset(e): i for i, e in enumerate(edge_list)}
    tri_map = {frozenset(t): i for i, t in enumerate(tri_list)}
    tet_map = {frozenset(tet): i for i, tet in enumerate(tet_list)}

    # Boundaries
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

    # Representation matrices G_k (mod 3)
    G0 = np.zeros((m0, m0), dtype=int)
    for i, v in enumerate(v_list):
        G0[v_map[g_map(v)], i] = 1

    G1 = np.zeros((m1, m1), dtype=int)
    for i, e in enumerate(edge_list):
        a, b = e
        ga, gb = g_map(a), g_map(b)
        G1[edge_map[frozenset({ga, gb})], i] = perm_sign([ga, gb]) % 3

    G2 = np.zeros((m2, m2), dtype=int)
    for i, t in enumerate(tri_list):
        a, b, c = t
        ga, gb, gc = g_map(a), g_map(b), g_map(c)
        G2[tri_map[frozenset({ga, gb, gc})], i] = perm_sign([ga, gb, gc]) % 3

    G3 = np.zeros((m3, m3), dtype=int)
    for i, tet in enumerate(tet_list):
        a, b, c, d = tet
        ga, gb, gc, gd = g_map(a), g_map(b), g_map(c), g_map(d)
        G3[tet_map[frozenset({ga, gb, gc, gd})], i] = perm_sign([ga, gb, gc, gd]) % 3

    # Projections
    I0, I1, I2, I3 = np.eye(m0, dtype=int), np.eye(m1, dtype=int), np.eye(m2, dtype=int), np.eye(m3, dtype=int)
    P0_plus, P0_minus = (2 * (I0 + G0)) % 3, (2 * (I0 - G0)) % 3
    P1_plus, P1_minus = (2 * (I1 + G1)) % 3, (2 * (I1 - G1)) % 3
    P2_plus, P2_minus = (2 * (I2 + G2)) % 3, (2 * (I2 - G2)) % 3
    P3_plus, P3_minus = (2 * (I3 + G3)) % 3, (2 * (I3 - G3)) % 3

    d1_mod3 = d1 % 3
    d2_mod3 = d2 % 3
    d3_mod3 = d3 % 3

    # Subspace dimensions
    dim0_p, dim0_m = rank_f3(P0_plus), rank_f3(P0_minus)
    dim1_p, dim1_m = rank_f3(P1_plus), rank_f3(P1_minus)
    dim2_p, dim2_m = rank_f3(P2_plus), rank_f3(P2_minus)
    dim3_p, dim3_m = rank_f3(P3_plus), rank_f3(P3_minus)

    # Restricted boundaries
    d1_plus = (P0_plus @ d1_mod3 @ P1_plus) % 3
    d1_minus = (P0_minus @ d1_mod3 @ P1_minus) % 3
    d2_plus = (P1_plus @ d2_mod3 @ P2_plus) % 3
    d2_minus = (P1_minus @ d2_mod3 @ P2_minus) % 3
    d3_plus = (P2_plus @ d3_mod3 @ P3_plus) % 3
    d3_minus = (P2_minus @ d3_mod3 @ P3_minus) % 3

    r1_p, r1_m = rank_f3(d1_plus), rank_f3(d1_minus)
    r2_p, r2_m = rank_f3(d2_plus), rank_f3(d2_minus)
    r3_p, r3_m = rank_f3(d3_plus), rank_f3(d3_minus)

    # Betti numbers
    beta0_p = dim0_p - r1_p
    beta1_p = dim1_p - r1_p - r2_p
    beta2_p = dim2_p - r2_p - r3_p
    beta3_p = dim3_p - r3_p

    beta0_m = dim0_m - r1_m
    beta1_m = dim1_m - r1_m - r2_m
    beta2_m = dim2_m - r2_m - r3_m
    beta3_m = dim3_m - r3_m

    # Total Betti
    r1_tot = rank_f3(d1_mod3)
    r2_tot = rank_f3(d2_mod3)
    r3_tot = rank_f3(d3_mod3)
    beta0_tot = m0 - r1_tot
    beta1_tot = m1 - r1_tot - r2_tot
    beta2_tot = m2 - r2_tot - r3_tot
    beta3_tot = m3 - r3_tot

    dt = time.time() - t0
    print(f"\n--- {name} ({dt:.2f}s) ---")
    print(f"  Simplices: V={m0:<2}, E={m1:<3}, T={m2:<3}, Tet={m3:<3}")
    print(f"  beta:      ({beta0_tot}, {beta1_tot}, {beta2_tot}, {beta3_tot})")
    print(f"  beta(C+):  ({beta0_p}, {beta1_p}, {beta2_p}, {beta3_p})")
    print(f"  beta(C-):  ({beta0_m}, {beta1_m}, {beta2_m}, {beta3_m})")


def main():
    print("=" * 80)
    print(" PERIOD 3 FAST CASCADE (K-CLOSURE)")
    print("=" * 80)

    # Step 0 to 7
    sc = build_base_step7()
    
    # Step 8 (K-close)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(16, existing)
    sc.insert_vertex_with_cone(26, existing)
    analyze_homology_representation(sc, "Step 8: K-close (V=25)")

    # Step 9 (K-close)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(29, existing)
    sc.insert_vertex_with_cone(39, existing)
    analyze_homology_representation(sc, "Step 9: K-close (V=27)")

    # Step 10 (K-close)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(30, existing)
    sc.insert_vertex_with_cone(40, existing)
    analyze_homology_representation(sc, "Step 10: K-close (V=29)")

    # Step 11 (K-close)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(31, existing)
    sc.insert_vertex_with_cone(41, existing)
    analyze_homology_representation(sc, "Step 11: K-close (V=31)")

    # Step 12 (K-close / Period 3 Closure)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(32, existing)
    sc.insert_vertex_with_cone(42, existing)
    analyze_homology_representation(sc, "Step 12: K-close / Period 3 Closure (V=33)")

    print("\n" + "=" * 80)
    print(" PERIOD 3 FAST CASCADE (EXCLUDE PAIR {8})")
    print("=" * 80)

    sc_ex8 = build_base_step7()
    existing_ex8 = sorted(list(sc_ex8.vertices))
    
    # Step 8
    sc_ex8.insert_vertex_with_cone(16, [v for v in existing_ex8 if v != 8])
    sc_ex8.insert_vertex_with_cone(26, [v for v in existing_ex8 if v != 8])
    analyze_homology_representation(sc_ex8, "Step 8: Exclude pair {8} (V=25)")

    # Step 9
    existing_ex8 = sorted(list(sc_ex8.vertices))
    sc_ex8.insert_vertex_with_cone(29, [v for v in existing_ex8 if v != 8])
    sc_ex8.insert_vertex_with_cone(39, [v for v in existing_ex8 if v != 8])
    analyze_homology_representation(sc_ex8, "Step 9: Exclude pair {8} (V=27)")

    # Step 10
    existing_ex8 = sorted(list(sc_ex8.vertices))
    sc_ex8.insert_vertex_with_cone(30, [v for v in existing_ex8 if v != 8])
    sc_ex8.insert_vertex_with_cone(40, [v for v in existing_ex8 if v != 8])
    analyze_homology_representation(sc_ex8, "Step 10: Exclude pair {8} (V=29)")

    # Step 11
    existing_ex8 = sorted(list(sc_ex8.vertices))
    sc_ex8.insert_vertex_with_cone(31, [v for v in existing_ex8 if v != 8])
    sc_ex8.insert_vertex_with_cone(41, [v for v in existing_ex8 if v != 8])
    analyze_homology_representation(sc_ex8, "Step 11: Exclude pair {8} (V=31)")

    # Step 12 (Closure)
    existing_ex8 = sorted(list(sc_ex8.vertices))
    sc_ex8.insert_vertex_with_cone(32, [v for v in existing_ex8 if v != 8])
    sc_ex8.insert_vertex_with_cone(42, [v for v in existing_ex8 if v != 8])
    analyze_homology_representation(sc_ex8, "Step 12: Exclude pair {8} / Period 3 Closure (V=33)")


if __name__ == "__main__":
    main()
