# -*- coding: utf-8 -*-
"""
Cascade Z2 Homology Representation Tracking (Targeted Closure Path)
==================================================================
Tracks the Z2 representation splitting (C+/C-) and homology evolution
at each step of the cascade, where Step 8 is a targeted closure path
(adding only v8-v12 and v8-v22 edges + induced triangles/tetrahedra).
Uses integer boundary matrices reduced to F3, with safe row-swapping.
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

    def neighbors(self, v):
        nbrs = set()
        for edge in self.edge_idx:
            if v in edge:
                nbrs.update(edge - {v})
        return nbrs


def g_map(v):
    """
    Z2 action g on vertices:
    g(a) = a ^ 7 for a in {0..7}
    g(8) = 8
    g(v) = v + 10 for v in {9..16}
    g(v) = v - 10 for v in {19..26}
    """
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
        # Swap rows safely
        if pivot_row != found:
            temp = m[pivot_row].copy()
            m[pivot_row] = m[found]
            m[found] = temp
        pivot_val = m[pivot_row, col]
        if pivot_val == 2:
            m[pivot_row] = (m[pivot_row] * 2) % 3
        # Eliminate entries
        for row in range(rows):
            if row != pivot_row and m[row, col] != 0:
                factor = m[row, col]
                m[row] = (m[row] - factor * m[pivot_row]) % 3
        pivot_row += 1
    return pivot_row


def analyze_step(sc, step_name):
    # Ordered bases
    v_list = sorted(list(sc.vertices))
    edge_list = sorted([sorted(list(e)) for e in sc.edge_idx.keys()])
    tri_list = sorted([sorted(list(t)) for t in sc.tri_idx.keys()])
    tet_list = sorted([sorted(list(tet)) for tet in sc.tet_idx.keys()])

    m0, m1, m2, m3 = len(v_list), len(edge_list), len(tri_list), len(tet_list)

    v_map = {v: i for i, v in enumerate(v_list)}
    edge_map = {frozenset(e): i for i, e in enumerate(edge_list)}
    tri_map = {frozenset(t): i for i, t in enumerate(tri_list)}
    tet_map = {frozenset(tet): i for i, tet in enumerate(tet_list)}

    # Boundary matrices over Z
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

    # Reduce boundaries to F3
    d1_mod3 = d1 % 3
    d2_mod3 = d2 % 3
    d3_mod3 = d3 % 3

    # Projections under F3
    I0, I1, I2, I3 = np.eye(m0, dtype=int), np.eye(m1, dtype=int), np.eye(m2, dtype=int), np.eye(m3, dtype=int)
    P0_plus, P0_minus = (2 * (I0 + G0)) % 3, (2 * (I0 - G0)) % 3
    P1_plus, P1_minus = (2 * (I1 + G1)) % 3, (2 * (I1 - G1)) % 3
    P2_plus, P2_minus = (2 * (I2 + G2)) % 3, (2 * (I2 - G2)) % 3
    P3_plus, P3_minus = (2 * (I3 + G3)) % 3, (2 * (I3 - G3)) % 3

    # Check commutativity
    c1 = np.all((d1_mod3 @ G1 - G0 @ d1_mod3) % 3 == 0)
    c2 = np.all((d2_mod3 @ G2 - G1 @ d2_mod3) % 3 == 0)
    c3 = np.all((d3_mod3 @ G3 - G2 @ d3_mod3) % 3 == 0)
    assert c1 and c2 and c3, "Representation matrices fail commutativity!"

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

    # Restricted boundary ranks
    r1_p, r1_m = rank_f3(d1_plus), rank_f3(d1_minus)
    r2_p, r2_m = rank_f3(d2_plus), rank_f3(d2_minus)
    r3_p, r3_m = rank_f3(d3_plus), rank_f3(d3_minus)

    # Betti numbers for C+ and C-
    beta0_p = dim0_p - r1_p
    beta1_p = dim1_p - r1_p - r2_p
    beta2_p = dim2_p - r2_p - r3_p
    beta3_p = dim3_p - r3_p

    beta0_m = dim0_m - r1_m
    beta1_m = dim1_m - r1_m - r2_m
    beta2_m = dim2_m - r2_m - r3_m
    beta3_m = dim3_m - r3_m

    # Total Betti
    r1_f3 = rank_f3(d1_mod3)
    r2_f3 = rank_f3(d2_mod3)
    r3_f3 = rank_f3(d3_mod3)
    beta0_f3 = m0 - r1_f3
    beta1_f3 = m1 - r1_f3 - r2_f3
    beta2_f3 = m2 - r2_f3 - r3_f3
    beta3_f3 = m3 - r3_f3

    # Check direct sum decomposition
    assert dim0_p + dim0_m == m0
    assert dim1_p + dim1_m == m1
    assert dim2_p + dim2_m == m2
    assert dim3_p + dim3_m == m3
    assert r1_f3 == r1_p + r1_m
    assert r2_f3 == r2_p + r2_m
    assert r3_f3 == r3_p + r3_m
    assert beta0_f3 == beta0_p + beta0_m
    assert beta1_f3 == beta1_p + beta1_m
    assert beta2_f3 == beta2_p + beta2_m
    assert beta3_f3 == beta3_p + beta3_m

    return {
        "stats": (m0, m1, m2, m3),
        "dim_p": (dim0_p, dim1_p, dim2_p, dim3_p),
        "dim_m": (dim0_m, dim1_m, dim2_m, dim3_m),
        "beta_p": (beta0_p, beta1_p, beta2_p, beta3_p),
        "beta_m": (beta0_m, beta1_m, beta2_m, beta3_m),
        "beta_tot": (beta0_f3, beta1_f3, beta2_f3, beta3_f3)
    }


def main():
    # ── Step 0: 8D Base (9V) ──
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

    steps = []
    steps.append(("Step 0: 8D base (9V)", sc.copy()))

    # ── Step 1: Add v9 and companion v19 ──
    sc.insert_vertex_with_cone(9, {3, 5, 6})
    sc.insert_vertex_with_cone(19, {4, 2, 1})
    steps.append(("Step 1: Add v9/v19", sc.copy()))

    # ── Step 2: Add v10 and companion v20 ──
    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})
    sc.insert_vertex_with_cone(20, {6, 5, 3, 19})
    steps.append(("Step 2: Add v10/v20", sc.copy()))

    # ── Step 3: Add v11 and companion v21 ──
    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})
    sc.insert_vertex_with_cone(21, {7, 4, 2, 1})
    steps.append(("Step 3: Add v11/v21", sc.copy()))

    # ── Step 4: Add v12/v22 and cube faces ──
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
    steps.append(("Step 4: Add v12/v22 + cube faces", sc.copy()))

    # ── Step 5: Period 2 build (v13/v23) ──
    sc.insert_vertex_with_cone(13, {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11})
    sc.insert_vertex_with_cone(23, {0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21})
    steps.append(("Step 5: Add v13/v23", sc.copy()))

    # ── Step 6: Period 2 build (v14/v24) ──
    sc.insert_vertex_with_cone(14, {0, 1, 2, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(24, {1, 2, 3, 5, 6, 7, 19, 20, 21})
    steps.append(("Step 6: Add v14/v24", sc.copy()))

    # ── Step 7: Period 2 build (v15/v25) ──
    sc.insert_vertex_with_cone(15, {0, 1, 2, 3, 4, 5, 6, 7, 9, 11})
    sc.insert_vertex_with_cone(25, {0, 1, 2, 3, 4, 5, 6, 7, 19, 21})
    steps.append(("Step 7: Add v15/v25", sc.copy()))

    # ── Step 8: Targeted close (Add v8-v12 and v8-v22 edges + induced triangles/tetrahedra) ──
    sc_c = sc.copy()
    
    # Add edge (12, 8)
    sc_c.add_edge(12, 8)
    common_12 = sc.neighbors(12) & sc.neighbors(8)
    for x in common_12:
        sc_c.add_triangle(12, 8, x)
    for a, b in combinations(sorted(common_12), 2):
        if frozenset({8, a, b}) in sc_c.tri_idx:
            if frozenset({12, a, b}) in sc_c.tri_idx:
                if frozenset({12, 8, a}) in sc_c.tri_idx and frozenset({12, 8, b}) in sc_c.tri_idx:
                    sc_c.add_tetrahedron(12, 8, a, b)
                    
    # Add edge (22, 8)
    sc_c.add_edge(22, 8)
    common_22 = sc.neighbors(22) & sc.neighbors(8)
    for x in common_22:
        sc_c.add_triangle(22, 8, x)
    for a, b in combinations(sorted(common_22), 2):
        if frozenset({8, a, b}) in sc_c.tri_idx:
            if frozenset({22, a, b}) in sc_c.tri_idx:
                if frozenset({22, 8, a}) in sc_c.tri_idx and frozenset({22, 8, b}) in sc_c.tri_idx:
                    sc_c.add_tetrahedron(22, 8, a, b)
                    
    steps.append(("Step 8: Add v8-v12 / v8-v22 (Targeted Close)", sc_c))

    # Perform analysis at each step
    results = []
    for name, scomplex in steps:
        res = analyze_step(scomplex, name)
        results.append((name, res))

    # Print Report
    print("\n" + "=" * 80)
    print("                CASCADE TARGETED CLOSURE DECOMPOSITION TABLE")
    print("=" * 80)
    
    for i, (name, res) in enumerate(results):
        print(f"\n--- {name} ---")
        st = res["stats"]
        print(f"  Simplices Count: V={st[0]:<3} | E={st[1]:<3} | T={st[2]:<3} | Tet={st[3]:<3}")
        
        # Dimensions table
        print(f"  Subspace dims:   C0+ = {res['dim_p'][0]:<2}, C0- = {res['dim_m'][0]:<2} | "
              f"C1+ = {res['dim_p'][1]:<2}, C1- = {res['dim_m'][1]:<2} | "
              f"C2+ = {res['dim_p'][2]:<2}, C2- = {res['dim_m'][2]:<2} | "
              f"C3+ = {res['dim_p'][3]:<2}, C3- = {res['dim_m'][3]:<2}")
              
        # Betti numbers
        print(f"  Betti numbers:   beta0(C+) = {res['beta_p'][0]:<2}, beta0(C-) = {res['beta_m'][0]:<2} | Total = {res['beta_tot'][0]}")
        print(f"                   beta1(C+) = {res['beta_p'][1]:<2}, beta1(C-) = {res['beta_m'][1]:<2} | Total = {res['beta_tot'][1]}")
        print(f"                   beta2(C+) = {res['beta_p'][2]:<2}, beta2(C-) = {res['beta_m'][2]:<2} | Total = {res['beta_tot'][2]}")
        print(f"                   beta3(C+) = {res['beta_p'][3]:<2}, beta3(C-) = {res['beta_m'][3]:<2} | Total = {res['beta_tot'][3]}")

        # Label directions of growth
        if i > 0:
            prev_res = results[i-1][1]
            diff_beta2_p = res['beta_p'][2] - prev_res['beta_p'][2]
            diff_beta2_m = res['beta_m'][2] - prev_res['beta_m'][2]
            diff_beta1_p = res['beta_p'][1] - prev_res['beta_p'][1]
            diff_beta1_m = res['beta_m'][1] - prev_res['beta_m'][1]
            
            changes = []
            if diff_beta2_p > 0: changes.append(f"beta2(C+) increases (+{diff_beta2_p})")
            elif diff_beta2_p < 0: changes.append(f"beta2(C+) decreases ({diff_beta2_p})")
            
            if diff_beta2_m > 0: changes.append(f"beta2(C-) increases (+{diff_beta2_m})")
            elif diff_beta2_m < 0: changes.append(f"beta2(C-) decreases ({diff_beta2_m})")
            
            if diff_beta1_p > 0: changes.append(f"beta1(C+) increases (+{diff_beta1_p})")
            elif diff_beta1_p < 0: changes.append(f"beta1(C+) decreases ({diff_beta1_p})")
            
            if diff_beta1_m > 0: changes.append(f"beta1(C-) increases (+{diff_beta1_m})")
            elif diff_beta1_m < 0: changes.append(f"beta1(C-) decreases ({diff_beta1_m})")
            
            if changes:
                print(f"  Evolution:       " + ", ".join(changes))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
