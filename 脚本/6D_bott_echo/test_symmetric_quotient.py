"""
Z2 Quotient Homology Analysis (Simplicial Quotient)
===================================================
Constructs Z2 quotient simplicial complexes, filtering out degenerate
simplices (loops, double edges, etc.) to ensure valid Betti numbers (beta_k >= 0).
Calculates Betti numbers and torsion subgroups for:
(A) Original 8D base (13V)
(B) Symmetric base + cube faces (21V)
(C) Complex B + v8-v12 edge (21V)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_form


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


def q_map(v):
    q = {
        0:0, 7:0, 1:1, 6:1, 2:2, 5:2, 3:3, 4:3, 
        8:8, 
        9:9, 19:9, 
        10:10, 20:10, 
        11:11, 21:11, 
        12:12, 22:12
    }
    return q.get(v, v)


def make_quotient_complex(sc):
    qsc = SimplicialComplex()
    
    # Vertices mapping
    unique_verts = sorted(list(set(q_map(v) for v in sc.vertices)))
    for v in unique_verts:
        qsc.add_vertex(v)
        
    degenerate_edges = []
    degenerate_tris = []
    degenerate_tets = []
    
    # Edges mapping & filtration
    unique_edges = set()
    for edge in sc.edge_idx:
        a, b = list(edge)
        qa, qb = q_map(a), q_map(b)
        if qa != qb:
            unique_edges.add(frozenset({qa, qb}))
        else:
            degenerate_edges.append(sorted([a, b]))
    for edge in sorted(list(unique_edges), key=lambda x: sorted(list(x))):
        a, b = list(edge)
        qsc.add_edge(a, b)
        
    # Triangles mapping & filtration
    unique_tris = set()
    for tri in sc.tri_idx:
        a, b, c = list(tri)
        qa, qb, qc = q_map(a), q_map(b), q_map(c)
        if len({qa, qb, qc}) == 3:
            unique_tris.add(frozenset({qa, qb, qc}))
        else:
            degenerate_tris.append(sorted([a, b, c]))
    for tri in sorted(list(unique_tris), key=lambda x: sorted(list(x))):
        a, b, c = list(tri)
        qsc.add_triangle(a, b, c)
        
    # Tetrahedra mapping & filtration
    unique_tets = set()
    for tet in sc.tet_idx:
        a, b, c, d = list(tet)
        qa, qb, qc, qd = q_map(a), q_map(b), q_map(c), q_map(d)
        if len({qa, qb, qc, qd}) == 4:
            unique_tets.add(frozenset({qa, qb, qc, qd}))
        else:
            degenerate_tets.append(sorted([a, b, c, d]))
    for tet in sorted(list(unique_tets), key=lambda x: sorted(list(x))):
        a, b, c, d = list(tet)
        qsc.add_tetrahedron(a, b, c, d)
        
    return qsc, degenerate_edges, degenerate_tris, degenerate_tets


def format_torsion(t_list):
    if not t_list:
        return "0 (trivial)"
    return " + ".join([f"Z_{d}" for d in sorted(t_list)])


def compute_and_report(sc, name):
    print("=" * 60)
    print(f"REPORT FOR: {name}")
    print("=" * 60)
    
    qsc, deg_edges, deg_tris, deg_tets = make_quotient_complex(sc)
    
    m0, m1, m2, m3 = len(qsc.vertices), len(qsc.edge_idx), len(qsc.tri_idx), len(qsc.tet_idx)
    print(f"Quotient simplices: m0={m0}, m1={m1}, m2={m2}, m3={m3}")
    print(f"Degenerated counts: edges={len(deg_edges)}, triangles={len(deg_tris)}, tetrahedra={len(deg_tets)}")
    if deg_edges:
        print(f"  Degenerated edges: {deg_edges}")
    if deg_tris:
        print(f"  Degenerated triangles: {deg_tris}")
    if deg_tets:
        print(f"  Degenerated tetrahedra: {deg_tets}")
        
    d1 = qsc.boundary_1()
    d2 = qsc.boundary_2()
    d3 = qsc.boundary_3()
    
    M1 = Matrix(d1)
    M2 = Matrix(d2)
    M3 = Matrix(d3)
    
    S1 = smith_normal_form(M1)
    diag1 = [int(S1[i, i]) for i in range(min(S1.shape)) if S1[i, i] != 0]
    r1 = len(diag1)
    
    S2 = smith_normal_form(M2)
    diag2 = [int(S2[i, i]) for i in range(min(S2.shape)) if S2[i, i] != 0]
    r2 = len(diag2)
    
    if d3.shape[1] > 0:
        S3 = smith_normal_form(M3)
        diag3 = [int(S3[i, i]) for i in range(min(S3.shape)) if S3[i, i] != 0]
        r3 = len(diag3)
    else:
        diag3 = []
        r3 = 0
        
    beta0 = m0 - r1
    beta1 = m1 - r1 - r2
    beta2 = m2 - r2 - r3
    beta3 = m3 - r3
    
    print(f"Ranks of boundaries: r1={r1}, r2={r2}, r3={r3}")
    print(f"ℤ-Betti numbers: beta₀={beta0}, beta₁={beta1}, beta₂={beta2}, beta₃={beta3}")
    
    # Corrected torsion indexing (torsion of H_k comes from D_{k+1})
    t0 = [d for d in diag1 if d > 1]
    t1 = [d for d in diag2 if d > 1]
    t2 = [d for d in diag3 if d > 1]
    t3 = []
    
    print(f"Torsion subgroups:")
    print(f"  Torsion H₀: {format_torsion(t0)}")
    print(f"  Torsion H₁: {format_torsion(t1)}")
    print(f"  Torsion H₂: {format_torsion(t2)}")
    print(f"  Torsion H₃: {format_torsion(t3)}")
    print("\n")
    
    # Verification
    for val, label in [(beta0, "beta_0"), (beta1, "beta_1"), (beta2, "beta_2"), (beta3, "beta_3")]:
        if val < 0:
            print("ERROR: Negative Betti number detected!")
            print(f"Degenerated edges ({len(deg_edges)}): {deg_edges}")
            print(f"Degenerated triangles ({len(deg_tris)}): {deg_tris}")
            print(f"Degenerated tetrahedra ({len(deg_tets)}): {deg_tets}")
            raise ValueError(f"Negative Betti number: {label} = {val}")


def main():
    # ── (A) Original 8D base (13V) ──
    sc_a = build_8d_base()
    compute_and_report(sc_a, "Complex A: Original 8D Base (13V)")

    # ── (B) Symmetric base + cube faces (21V) ──
    sc_b = SimplicialComplex()
    for i in range(9):
        sc_b.add_vertex(i)
    cube_edges = []
    for a in range(8):
        for b in range(a+1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))
                sc_b.add_edge(a, b)
    for i in range(8):
        sc_b.add_edge(8, i)
    for a, b in cube_edges:
        sc_b.add_triangle(a, b, 8)
        
    sc_b.insert_vertex_with_cone(9, {3, 5, 6})
    sc_b.insert_vertex_with_cone(19, {4, 2, 1})

    sc_b.insert_vertex_with_cone(10, {1, 2, 4, 9})
    sc_b.insert_vertex_with_cone(20, {6, 5, 3, 19})

    sc_b.insert_vertex_with_cone(11, {0, 3, 5, 6})
    sc_b.insert_vertex_with_cone(21, {7, 4, 2, 1})

    sc_b.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})
    sc_b.insert_vertex_with_cone(22, {7, 6, 5, 4, 3, 2, 1, 19, 20, 21})

    cube_faces = [
        (0,1,3), (0,2,3),
        (4,5,7), (4,6,7),
        (0,2,6), (0,4,6),
        (1,3,7), (1,5,7),
        (2,3,7), (2,6,7),
        (0,1,5), (0,4,5)
    ]
    for a, b, c in cube_faces:
        sc_b.add_edge(a, b)
        sc_b.add_edge(b, c)
        sc_b.add_edge(a, c)
        sc_b.add_triangle(a, b, c)

    compute_and_report(sc_b, "Complex B: Symmetric base + cube faces (21V)")

    # ── (C) Complex B + v8-v12 edge ──
    sc_c = sc_b.copy()
    sc_c.add_edge(12, 8)
    sc_c.add_edge(22, 8)
    
    zsc_temp = sc_c.copy()
    common_12 = zsc_temp.neighbors(12) & zsc_temp.neighbors(8)
    for x in common_12:
        sc_c.add_triangle(12, 8, x)
    for a, b in combinations(sorted(common_12), 2):
        if frozenset({8, a, b}) in sc_c.tri_idx:
            if frozenset({12, a, b}) in sc_c.tri_idx:
                if frozenset({12, 8, a}) in sc_c.tri_idx and frozenset({12, 8, b}) in sc_c.tri_idx:
                    sc_c.add_tetrahedron(12, 8, a, b)
                    
    common_22 = zsc_temp.neighbors(22) & zsc_temp.neighbors(8)
    for x in common_22:
        sc_c.add_triangle(22, 8, x)
    for a, b in combinations(sorted(common_22), 2):
        if frozenset({8, a, b}) in sc_c.tri_idx:
            if frozenset({22, a, b}) in sc_c.tri_idx:
                if frozenset({22, 8, a}) in sc_c.tri_idx and frozenset({22, 8, b}) in sc_c.tri_idx:
                    sc_c.add_tetrahedron(22, 8, a, b)

    compute_and_report(sc_c, "Complex C: Complex B + v8-v12 edge")


if __name__ == "__main__":
    main()
