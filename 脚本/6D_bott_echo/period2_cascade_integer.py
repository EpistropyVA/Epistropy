"""
Bott Period 2 — Integer Homology Cascade (Experiment 2)
======================================================
Tests K-closure (connect to all) at each step from 8D to 16D.
Uses ℤ-homology with oriented boundaries and Smith normal form,
correcting the torsion extraction index bug.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_form
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
            return np.zeros((nv, 0), dtype=int)
        mat = np.zeros((nv, ne), dtype=int)
        v_to_idx = {v: i for i, v in enumerate(self.vertices)}
        for edge, ei in self.edge_idx.items():
            a, b = sorted(edge)
            # d([a,b]) = b - a
            mat[v_to_idx[a], ei] = -1
            mat[v_to_idx[b], ei] = 1
        return mat

    def boundary_2(self):
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        if nt == 0:
            return np.zeros((ne, 0), dtype=int)
        mat = np.zeros((ne, nt), dtype=int)
        for tri, ti in self.tri_idx.items():
            a, b, c = sorted(tri)
            # d([a,b,c]) = +[b,c] - [a,c] + [a,b]
            ei_bc = self.edge_idx.get(frozenset({b, c}))
            ei_ac = self.edge_idx.get(frozenset({a, c}))
            ei_ab = self.edge_idx.get(frozenset({a, b}))
            if ei_bc is not None:
                mat[ei_bc, ti] = 1
            if ei_ac is not None:
                mat[ei_ac, ti] = -1
            if ei_ab is not None:
                mat[ei_ab, ti] = 1
        return mat

    def boundary_3(self):
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        if ntet == 0:
            return np.zeros((nt, 0), dtype=int)
        mat = np.zeros((nt, ntet), dtype=int)
        for tet, teti in self.tet_idx.items():
            a, b, c, d = sorted(tet)
            # d([a,b,c,d]) = +[b,c,d] - [a,c,d] + [a,b,d] - [a,b,c]
            ti_bcd = self.tri_idx.get(frozenset({b, c, d}))
            ti_acd = self.tri_idx.get(frozenset({a, c, d}))
            ti_abd = self.tri_idx.get(frozenset({a, b, d}))
            ti_abc = self.tri_idx.get(frozenset({a, b, c}))
            if ti_bcd is not None:
                mat[ti_bcd, teti] = 1
            if ti_acd is not None:
                mat[ti_acd, teti] = -1
            if ti_abd is not None:
                mat[ti_abd, teti] = 1
            if ti_abc is not None:
                mat[ti_abc, teti] = -1
        return mat

    def betti_and_torsion_z(self):
        nv = len(self.vertices)
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        
        d1 = self.boundary_1()
        d2 = self.boundary_2()
        d3 = self.boundary_3()
        
        # SNF for D1
        M1 = Matrix(d1)
        S1 = smith_normal_form(M1)
        diag1 = [int(S1[i, i]) for i in range(min(S1.shape)) if S1[i, i] != 0]
        r1 = len(diag1)
        
        # SNF for D2
        M2 = Matrix(d2)
        S2 = smith_normal_form(M2)
        diag2 = [int(S2[i, i]) for i in range(min(S2.shape)) if S2[i, i] != 0]
        r2 = len(diag2)
        
        # SNF for D3
        M3 = Matrix(d3)
        if d3.shape[1] > 0:
            S3 = smith_normal_form(M3)
            diag3 = [int(S3[i, i]) for i in range(min(S3.shape)) if S3[i, i] != 0]
            r3 = len(diag3)
        else:
            diag3 = []
            r3 = 0
            
        beta0 = nv - r1
        beta1 = ne - r1 - r2
        beta2 = nt - r2 - r3
        beta3 = ntet - r3
        
        # Corrected torsion extraction:
        # Torsion of H_k comes from diagonal entries > 1 of D_{k+1}
        t0 = [d for d in diag1 if d > 1]
        t1 = [d for d in diag2 if d > 1]
        t2 = [d for d in diag3 if d > 1]
        t3 = [] # since C4 = 0
        
        return (beta0, beta1, beta2, beta3), (t0, t1, t2, t3)

    def betti_f2(self):
        nv = len(self.vertices)
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        
        d1 = self.boundary_1()
        d2 = self.boundary_2()
        d3 = self.boundary_3()
        
        r1 = rank_f2(d1)
        r2 = rank_f2(d2)
        r3 = rank_f2(d3)
        
        b0 = nv - r1
        b1 = ne - r1 - r2
        b2 = nt - r2 - r3
        b3 = ntet - r3
        
        return (b0, b1, b2, b3)

    def stats(self):
        return len(self.vertices), len(self.edge_idx), len(self.tri_idx), len(self.tet_idx)

    def neighbors(self, v):
        nbrs = set()
        for edge in self.edge_idx:
            if v in edge:
                nbrs.update(edge - {v})
        return nbrs


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


def format_torsion(torsion_list):
    if not torsion_list:
        return "0 (trivial)"
    return " + ".join([f"Z_{d}" for d in sorted(torsion_list)])


def main():
    t_start = time.time()
    print("=" * 70)
    print("EXPERIMENT 2: K-CLOSURE CASCADE FROM 8D TO 16D (ℤ vs 𝔽₂ HOMOLOGY)")
    print("=" * 70)

    # Initialize from the 8D base complex
    sc = build_8d_base()
    
    # Compute initial state
    print("\nInitial 8D Base Complex:")
    st = sc.stats()
    print(f"  Simplices: V={st[0]}, E={st[1]}, T={st[2]}, Tet={st[3]}")
    b_z, tor = sc.betti_and_torsion_z()
    b_f2 = sc.betti_f2()
    mismatch_str = ""
    for k in range(4):
        if b_z[k] != b_f2[k]:
            mismatch_str += f" beta_{k}(ℤ={b_z[k]}!=𝔽₂={b_f2[k]})"
    
    print(f"  ℤ-Betti numbers: beta₀={b_z[0]}, beta₁={b_z[1]}, beta₂={b_z[2]}, beta₃={b_z[3]}")
    print(f"  𝔽₂-Betti numbers: beta₀={b_f2[0]}, beta₁={b_f2[1]}, beta₂={b_f2[2]}, beta₃={b_f2[3]}")
    if mismatch_str:
        print(f"  * MISMATCH *:{mismatch_str}")
    else:
        print("  Betti check: ℤ == 𝔽₂ in all dimensions")
    print(f"  Torsion subgroups:")
    print(f"    Torsion H₀: {format_torsion(tor[0])}")
    print(f"    Torsion H₁: {format_torsion(tor[1])}")
    print(f"    Torsion H₂: {format_torsion(tor[2])}")
    print(f"    Torsion H₃: {format_torsion(tor[3])}")

    vid = 13
    for dim in range(9, 17):
        print(f"\n" + "-" * 60)
        print(f"STEP: {dim}D K-close (Adding vertex v{vid} coned to all existing)")
        print(f"-" * 60)
        
        sc_test = sc.copy()
        all_v = frozenset(sc.vertices)
        sc_test.insert_vertex_with_cone(vid, all_v)
        
        # Analyze
        st_test = sc_test.stats()
        print(f"  Simplices: V={st_test[0]}, E={st_test[1]}, T={st_test[2]}, Tet={st_test[3]}")
        
        t0 = time.time()
        b_z, tor = sc_test.betti_and_torsion_z()
        b_f2 = sc_test.betti_f2()
        dt = time.time() - t0
        
        print(f"  ℤ-Betti numbers: beta₀={b_z[0]}, beta₁={b_z[1]}, beta₂={b_z[2]}, beta₃={b_z[3]}")
        print(f"  𝔽₂-Betti numbers: beta₀={b_f2[0]}, beta₁={b_f2[1]}, beta₂={b_f2[2]}, beta₃={b_f2[3]}")
        
        mismatch_str = ""
        for k in range(4):
            if b_z[k] != b_f2[k]:
                mismatch_str += f" beta_{k}(ℤ={b_z[k]}!=𝔽₂={b_f2[k]})"
        if mismatch_str:
            print(f"  * WARNING: MISMATCH *:{mismatch_str}")
        else:
            print("  Betti check: ℤ == 𝔽₂ in all dimensions")
            
        print(f"  Torsion subgroups:")
        print(f"    Torsion H₀: {format_torsion(tor[0])}")
        print(f"    Torsion H₁: {format_torsion(tor[1])}")
        print(f"    Torsion H₂: {format_torsion(tor[2])}")
        print(f"    Torsion H₃: {format_torsion(tor[3])}")
        print(f"  (Calculation took {dt:.3f} seconds)")
        
        # Accept the step
        sc = sc_test
        vid += 1

    print("\n" + "=" * 70)
    print(f"Total cascade runtime: {time.time() - t_start:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
