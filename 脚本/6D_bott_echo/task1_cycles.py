# -*- coding: utf-8 -*-
import sys
import io
import numpy as np
from itertools import combinations
from sympy import Matrix

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


def build_complex():
    sc = SimplicialComplex()
    # 8D Base
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

    # Steps 5-7 (Period 2 Build)
    sc.insert_vertex_with_cone(13, {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11})
    sc.insert_vertex_with_cone(14, {0, 1, 2, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(15, {0, 1, 2, 3, 4, 5, 6, 7, 9, 11})

    # Step 8 (v16 coned to all except 8)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(16, [v for v in existing if v != 8])
    return sc


def rref_f3(mat):
    """Computes RREF over F3."""
    m = mat.copy() % 3
    rows, cols = m.shape
    pivot_row = 0
    pivots = []
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
        pivots.append((pivot_row, col))
        pivot_row += 1
    return m, pivots


def nullspace_f3(mat):
    """Computes a basis for the nullspace of mat over F3."""
    rref_mat, pivots = rref_f3(mat)
    rows, cols = mat.shape
    pivot_cols = {c: r for r, c in pivots}
    basis = []
    for col in range(cols):
        if col not in pivot_cols:
            # This is a free variable
            vec = np.zeros(cols, dtype=int)
            vec[col] = 1
            for r, c_pivot in pivots:
                val = rref_mat[r, col]
                vec[c_pivot] = (-val) % 3
            basis.append(vec)
    return np.array(basis).T if basis else np.zeros((cols, 0), dtype=int)


def main():
    sc = build_complex()
    v_list = sorted(list(sc.vertices))
    edge_list = sorted([sorted(list(e)) for e in sc.edge_idx.keys()])
    tri_list = sorted([sorted(list(t)) for t in sc.tri_idx.keys()])
    tet_list = sorted([sorted(list(tet)) for tet in sc.tet_idx.keys()])

    m0, m1, m2, m3 = len(v_list), len(edge_list), len(tri_list), len(tet_list)
    print(f"Simplices count: V={m0}, E={m1}, T={m2}, Tet={m3}")

    v_map = {v: i for i, v in enumerate(v_list)}
    edge_map = {frozenset(e): i for i, e in enumerate(edge_list)}
    tri_map = {frozenset(t): i for i, t in enumerate(tri_list)}

    # Boundaries
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

    # Find kernel of d2 mod 3
    print("Computing nullspace of d2 over F3...")
    ns2 = nullspace_f3(d2)

    print(f"Dimension of ker(d2) over F3 = {ns2.shape[1]}")

    # Combine image of d3 and kernel of d2
    # Matrix: [d3 | ns2]
    # Reduce it to find which columns of ns2 are pivots (linearly independent mod im(d3))
    combined = np.hstack((d3 % 3, ns2))
    rref_mat, pivots = rref_f3(combined)

    # The pivots in the second part (cols >= m3) are the generator columns of H_2
    h2_generators = []
    for r, c in pivots:
        if c >= m3:
            # This corresponds to column c - m3 in ns2
            h2_generators.append(ns2[:, c - m3])

    print(f"Dimension of H_2 over F3 = {len(h2_generators)} (expected 5)")

    # For each generator, print the support triangles
    for idx, gen in enumerate(h2_generators):
        print(f"\n--- Generator {idx+1} ---")
        support = []
        for j in range(m2):
            val = gen[j]
            if val != 0:
                support.append((tri_list[j], val))
        
        # Sort support triangles by vertices
        support.sort(key=lambda x: x[0])
        
        print(f"Support size: {len(support)} triangles")
        terms = []
        for tri, val in support:
            sign = "+" if val == 1 else "-"
            terms.append(f"{sign}{tri}")
        print(" ".join(terms))

        # Check if it uses vertex 8, 12, 13, 14, 15, 16
        vertices_used = set()
        for tri, _ in support:
            vertices_used.update(tri)
        print(f"Vertices used: {sorted(list(vertices_used))}")


if __name__ == "__main__":
    main()
