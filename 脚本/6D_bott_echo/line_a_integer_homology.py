"""
Line A integer homology analysis: tracking orientation and torsion over Z
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


def format_torsion(torsion_list):
    if not torsion_list:
        return "0 (trivial)"
    return " + ".join([f"Z_{d}" for d in sorted(torsion_list)])


def compute_homology(sc, name):
    print("=" * 60)
    print(f"ZHOMOLOGY ANALYSIS FOR: {name}")
    print("=" * 60)
    
    d1 = sc.boundary_1()
    d2 = sc.boundary_2()
    d3 = sc.boundary_3()
    
    m0 = len(sc.vertices)
    m1 = len(sc.edge_idx)
    m2 = len(sc.tri_idx)
    m3 = len(sc.tet_idx)
    
    print(f"Simplices count: m0(vertices)={m0}, m1(edges)={m1}, m2(triangles)={m2}, m3(tetrahedra)={m3}")
    
    M1 = Matrix(d1)
    M2 = Matrix(d2)
    M3 = Matrix(d3)
    
    print(f"Computing Smith Normal Form for D1 ({M1.shape[0]}x{M1.shape[1]})...")
    S1 = smith_normal_form(M1)
    diag1 = [int(S1[i, i]) for i in range(min(S1.shape)) if S1[i, i] != 0]
    r1 = len(diag1)
    
    print(f"Computing Smith Normal Form for D2 ({M2.shape[0]}x{M2.shape[1]})...")
    S2 = smith_normal_form(M2)
    diag2 = [int(S2[i, i]) for i in range(min(S2.shape)) if S2[i, i] != 0]
    r2 = len(diag2)
    
    print(f"Computing Smith Normal Form for D3 ({M3.shape[0]}x{M3.shape[1]})...")
    S3 = smith_normal_form(M3)
    diag3 = [int(S3[i, i]) for i in range(min(S3.shape)) if S3[i, i] != 0]
    r3 = len(diag3)
    
    # Betti numbers
    beta0 = m0 - r1
    beta1 = m1 - r1 - r2
    beta2 = m2 - r2 - r3
    beta3 = m3 - r3
    
    print("\nZ-Betti numbers:")
    print(f"  beta_0 = {beta0}")
    print(f"  beta_1 = {beta1}")
    print(f"  beta_2 = {beta2}")
    print(f"  beta_3 = {beta3}")
    
    # Torsion subgroups
    t0 = [d for d in diag1 if d > 1]
    t1 = [d for d in diag2 if d > 1]
    t2 = [d for d in diag3 if d > 1]
    t3 = []
    
    print("\nTorsion subgroups:")
    print(f"  Torsion H_0: {format_torsion(t0)}")
    print(f"  Torsion H_1: {format_torsion(t1)}")
    print(f"  Torsion H_2: {format_torsion(t2)}")
    print(f"  Torsion H_3: {format_torsion(t3)}")
    
    # Find beta_2 cycles if beta_2 > 0 and no higher boundaries block it
    if beta2 > 0:
        ns2 = M2.nullspace()
        basis_cycles = []
        for vec in ns2:
            vals = [vec[i] for i in range(len(vec))]
            denoms = [val.q for val in vals if hasattr(val, 'q')]
            import math
            lcm = 1
            for d in denoms:
                lcm = (lcm * d) // math.gcd(lcm, d)
            int_vals = [int(val * lcm) for val in vals]
            g = int_vals[0]
            for val in int_vals[1:]:
                g = math.gcd(g, val)
            if g != 0:
                int_vals = [val // g for val in int_vals]
            for val in int_vals:
                if val != 0:
                    if val < 0:
                        int_vals = [-v for v in int_vals]
                    break
            basis_cycles.append(int_vals)
            
        print("\nbeta_2 cycle representatives (oriented generators):")
        tri_list = sorted(sc.tri_idx.items(), key=lambda x: x[1])
        for idx, cycle in enumerate(basis_cycles):
            nonzero_tris = []
            for j, coeff in enumerate(cycle):
                if coeff != 0:
                    tri, _ = tri_list[j]
                    nonzero_tris.append((sorted(list(tri)), coeff))
            nonzero_tris.sort(key=lambda x: x[0])
            terms = []
            for tri, coeff in nonzero_tris:
                sign = "+" if coeff > 0 else "-"
                abs_coeff = abs(coeff)
                coeff_str = "" if abs_coeff == 1 else str(abs_coeff)
                terms.append(f"{sign} {coeff_str}{tri}")
            print(f"  Cycle {idx}: " + " ".join(terms))
    print("\n")


def main():
    # 1. Base Complex
    sc_base = build_8d_base()
    compute_homology(sc_base, "8D Base Complex (13V)")
    
    # 2. Modified Complex (+ edge v12-v8)
    sc_mod = sc_base.copy()
    sc_mod.add_edge(12, 8)
    common = sc_mod.neighbors(12) & sc_mod.neighbors(8)
    for x in common:
        sc_mod.add_triangle(12, 8, x)
    for a, b in combinations(sorted(common), 2):
        if frozenset({8, a, b}) in sc_mod.tri_idx:
            if frozenset({12, a, b}) in sc_mod.tri_idx:
                if frozenset({12, 8, a}) in sc_mod.tri_idx and frozenset({12, 8, b}) in sc_mod.tri_idx:
                    sc_mod.add_tetrahedron(12, 8, a, b)
                    
    compute_homology(sc_mod, "Modified Complex (Base + Edge v8-v12)")


if __name__ == "__main__":
    main()
