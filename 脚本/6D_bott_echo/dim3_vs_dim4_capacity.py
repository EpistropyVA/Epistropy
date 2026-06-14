# -*- coding: utf-8 -*-
"""
dim3_vs_dim4_capacity.py
========================
Tests the "3D capacity lock" hypothesis in a simplicial homology cascade.

Core question: when the cascade is 3D-capped (no pentatopes), beta3 = nullity(d3)
because rank(d4) = 0 by construction. If we allow 4-simplices (pentatopes),
does d4 kill some of those beta3 classes?

Two modes per step:
  3D-capped : no pentatopes, beta3 = nullity(d3)
  4D-open   : pentatopes allowed, beta3 = nullity(d3) - rank(d4)

BCC cascade: non-symmetrized complex, vertices 0-16, F3 coefficients.
"""

import sys
import io
import numpy as np
import time
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Vectorized F3 rank (from period3_fast_cascade.py)
# ---------------------------------------------------------------------------

def rank_f3(mat):
    if mat.size == 0:
        return 0
    m = mat.copy() % 3
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        col_slice = m[pivot_row:rows, col]
        nonzero = np.where(col_slice != 0)[0]
        if len(nonzero) == 0:
            continue
        found = nonzero[0] + pivot_row
        if pivot_row != found:
            temp = m[pivot_row].copy()
            m[pivot_row] = m[found]
            m[found] = temp
        pivot_val = m[pivot_row, col]
        if pivot_val == 2:
            m[pivot_row] = (m[pivot_row] * 2) % 3
        col_vals = m[:, col].copy()
        col_vals[pivot_row] = 0
        nonzero_rows = np.where(col_vals != 0)[0]
        if len(nonzero_rows) > 0:
            factors = col_vals[nonzero_rows, np.newaxis]
            m[nonzero_rows] = (m[nonzero_rows] - factors * m[pivot_row]) % 3
        pivot_row += 1
    return pivot_row


# ---------------------------------------------------------------------------
# SimplicialComplex tracking up to dimension 4
# ---------------------------------------------------------------------------

class SimplicialComplex:
    def __init__(self):
        self.vertices = []          # list of vertex labels
        self.edge_idx = {}          # frozenset(2) -> index
        self.tri_idx = {}           # frozenset(3) -> index
        self.tet_idx = {}           # frozenset(4) -> index
        self.pent_idx = {}          # frozenset(5) -> index

    # --- adders ---

    def add_vertex(self, v):
        if v not in self.vertices:
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

    def add_pentatope(self, a, b, c, d, e):
        key = frozenset({a, b, c, d, e})
        if key not in self.pent_idx:
            self.pent_idx[key] = len(self.pent_idx)

    # --- cone insertion ---

    def insert_vertex_with_cone(self, v, S, allow_pentatopes=True):
        """
        Cone v over subset S of existing simplices.
          edges     : {v,s} for each s in S
          triangles : {v,a,b} when edge {a,b} exists in S
          tetrahedra: {v,a,b,c} when triangle {a,b,c} exists in S
          pentatopes: {v,a,b,c,d} when tetrahedron {a,b,c,d} exists in S
                      (only if allow_pentatopes=True)
        """
        self.add_vertex(v)
        S = list(S)

        # edges
        for s in S:
            self.add_edge(v, s)

        # triangles
        for a, b in combinations(S, 2):
            if frozenset({a, b}) in self.edge_idx:
                self.add_triangle(v, a, b)

        # tetrahedra
        for a, b, c in combinations(S, 3):
            if frozenset({a, b, c}) in self.tri_idx:
                self.add_tetrahedron(v, a, b, c)

        # pentatopes — the new dimension
        if allow_pentatopes:
            for a, b, c, d in combinations(S, 4):
                if frozenset({a, b, c, d}) in self.tet_idx:
                    self.add_pentatope(v, a, b, c, d)

    def copy(self):
        c = SimplicialComplex()
        c.vertices = list(self.vertices)
        c.edge_idx = dict(self.edge_idx)
        c.tri_idx = dict(self.tri_idx)
        c.tet_idx = dict(self.tet_idx)
        c.pent_idx = dict(self.pent_idx)
        return c


# ---------------------------------------------------------------------------
# Homology computation (F3, full complex)
# ---------------------------------------------------------------------------

def compute_betti(sc):
    """
    Returns (beta0, beta1, beta2, beta3, nullity_d3, rank_d4, n_pent)
    using F3 coefficients.
    """
    v_list  = sorted(sc.vertices)
    e_list  = sorted([sorted(list(e)) for e in sc.edge_idx.keys()])
    t_list  = sorted([sorted(list(t)) for t in sc.tri_idx.keys()])
    tet_list= sorted([sorted(list(t)) for t in sc.tet_idx.keys()])
    p_list  = sorted([sorted(list(p)) for p in sc.pent_idx.keys()])

    m0, m1, m2, m3, m4 = len(v_list), len(e_list), len(t_list), len(tet_list), len(p_list)

    v_map   = {v: i for i, v in enumerate(v_list)}
    e_map   = {frozenset(e): i for i, e in enumerate(e_list)}
    t_map   = {frozenset(t): i for i, t in enumerate(t_list)}
    tet_map = {frozenset(t): i for i, t in enumerate(tet_list)}
    p_map   = {frozenset(p): i for i, p in enumerate(p_list)}

    # d1: vertices <- edges
    d1 = np.zeros((m0, m1), dtype=np.int8)
    for j, e in enumerate(e_list):
        a, b = e
        d1[v_map[a], j] = 2   # -1 mod 3
        d1[v_map[b], j] = 1

    # d2: edges <- triangles
    d2 = np.zeros((m1, m2), dtype=np.int8)
    for j, t in enumerate(t_list):
        a, b, c = t
        d2[e_map[frozenset({b, c})], j] = 1
        d2[e_map[frozenset({a, c})], j] = 2  # -1
        d2[e_map[frozenset({a, b})], j] = 1

    # d3: triangles <- tetrahedra
    d3 = np.zeros((m2, m3), dtype=np.int8)
    for j, tet in enumerate(tet_list):
        a, b, c, d = tet
        d3[t_map[frozenset({b, c, d})], j] = 1
        d3[t_map[frozenset({a, c, d})], j] = 2  # -1
        d3[t_map[frozenset({a, b, d})], j] = 1
        d3[t_map[frozenset({a, b, c})], j] = 2  # -1

    # d4: tetrahedra <- pentatopes
    d4 = np.zeros((m3, m4), dtype=np.int8)
    for j, p in enumerate(p_list):
        a, b, c, d, e = p
        d4[tet_map[frozenset({b, c, d, e})], j] = 1
        d4[tet_map[frozenset({a, c, d, e})], j] = 2  # -1
        d4[tet_map[frozenset({a, b, d, e})], j] = 1
        d4[tet_map[frozenset({a, b, c, e})], j] = 2  # -1
        d4[tet_map[frozenset({a, b, c, d})], j] = 1

    r1 = rank_f3(d1.astype(int))
    r2 = rank_f3(d2.astype(int))
    r3 = rank_f3(d3.astype(int))
    r4 = rank_f3(d4.astype(int))

    nullity_d3 = m3 - r3   # kernel of d3 = all 3-cycles
    # beta3 = nullity(d3) - rank(d4)  [rank(d4) fills in boundaries from 4-simplices]
    beta3 = nullity_d3 - r4

    beta0 = m0 - r1
    beta1 = m1 - r1 - r2
    beta2 = m2 - r2 - r3
    # beta3 as above

    return beta0, beta1, beta2, beta3, nullity_d3, r4, m4


# ---------------------------------------------------------------------------
# BCC cascade builder — non-symmetrized, vertices 0-16
# ---------------------------------------------------------------------------

def build_cascade():
    """
    Yields (step_label, sc_3d, sc_4d) at each step.
    sc_3d: 3D-capped (no pentatopes)
    sc_4d: 4D-open (pentatopes allowed)
    """

    def make_step0():
        """
        Cube (v0-v7) + body center v8.
        Cube edges (Hamming-1 neighbors) + spoke edges v8<->all cube vertices.
        """
        sc = SimplicialComplex()
        for i in range(9):
            sc.add_vertex(i)

        # cube edges
        cube_edges = []
        for a in range(8):
            for b in range(a+1, 8):
                if bin(a ^ b).count('1') == 1:
                    cube_edges.append((a, b))
                    sc.add_edge(a, b)

        # spoke edges
        for i in range(8):
            sc.add_edge(8, i)

        # triangles: cube face diagonals with center
        for a, b in cube_edges:
            sc.add_triangle(a, b, 8)

        return sc

    sc3 = make_step0()
    sc4 = make_step0()
    # Step 0 has no tetrahedra yet, so pentatope question is moot
    yield "Step 0 [Cube+center]", sc3.copy(), sc4.copy()

    # Helper: cone a new vertex to all existing vertices EXCEPT v8
    def cone_no8(sc3, sc4, v):
        existing = [x for x in sc3.vertices if x != 8]
        sc3.insert_vertex_with_cone(v, existing, allow_pentatopes=False)
        sc4.insert_vertex_with_cone(v, existing, allow_pentatopes=True)

    # Helper: K-close (cone to ALL existing vertices)
    def k_close(sc3, sc4, v):
        existing = list(sc3.vertices)
        sc3.insert_vertex_with_cone(v, existing, allow_pentatopes=False)
        sc4.insert_vertex_with_cone(v, existing, allow_pentatopes=True)

    # Steps 1-3: build vertices 9, 10, 11 (coned to all except v8)
    for v in [9, 10, 11]:
        cone_no8(sc3, sc4, v)
        yield f"Step {v-8} [add v{v}, cone\\v8]", sc3.copy(), sc4.copy()

    # Step 4: P1 close — v12 coned to ALL
    k_close(sc3, sc4, 12)
    yield "Step 4 [P1 K-close, v12]", sc3.copy(), sc4.copy()

    # Steps 5-7: build vertices 13, 14, 15 (coned to all except v8)
    for step, v in enumerate([13, 14, 15], start=5):
        cone_no8(sc3, sc4, v)
        yield f"Step {step} [add v{v}, cone\\v8]", sc3.copy(), sc4.copy()

    # Step 8: P2 close — v16 coned to ALL
    k_close(sc3, sc4, 16)
    yield "Step 8 [P2 K-close, v16]", sc3.copy(), sc4.copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  3D CAPACITY LOCK HYPOTHESIS — dim3_vs_dim4_capacity")
    print("  F3 coefficients | non-symmetrized BCC cascade | v0-v16")
    print("=" * 70)
    print()
    print("  3D-capped: beta3 = nullity(d3)          [rank(d4)=0 by design]")
    print("  4D-open  : beta3 = nullity(d3) - rank(d4)  [pentatopes active]")
    print()

    header = (
        f"{'Step':<32} | {'Mode':<10} | "
        f"{'beta':<20} | {'Tet':>5} | {'Pent':>5} | {'rank(d4)':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    t_global = time.time()

    for label, sc3, sc4 in build_cascade():
        t0 = time.time()

        b0_3, b1_3, b2_3, b3_3, null3_3, r4_3, np3 = compute_betti(sc3)
        b0_4, b1_4, b2_4, b3_4, null3_4, r4_4, np4 = compute_betti(sc4)

        dt = time.time() - t0

        n_tet3 = len(sc3.tet_idx)
        n_tet4 = len(sc4.tet_idx)

        print(
            f"{label:<32} | {'3D-capped':<10} | "
            f"({b0_3},{b1_3},{b2_3},{b3_3}){'':<10} | "
            f"{n_tet3:>5} | {'—':>5} | {'0':>8}"
        )
        print(
            f"{'':<32} | {'4D-open':<10} | "
            f"({b0_4},{b1_4},{b2_4},{b3_4}){'':<10} | "
            f"{n_tet4:>5} | {np4:>5} | {r4_4:>8}   ({dt:.2f}s)"
        )

        # Flag divergence
        if b3_3 != b3_4:
            diff = b3_3 - b3_4
            print(
                f"  *** BETA3 DIVERGENCE: 3D-cap={b3_3}, 4D-open={b3_4}, "
                f"killed={diff}, rank(d4)={r4_4} ***"
            )
        print()

    print(sep)
    print(f"\nTotal time: {time.time()-t_global:.2f}s")
    print()
    print("KEY QUESTION — Step 8 [P2 K-close]:")
    print("  If 4D-open beta3 < 3D-capped beta3, d4 kills 3-cycles.")
    print("  If equal, the BCC geometry generates no pentatopes that fill 3-cycles.")


if __name__ == "__main__":
    main()
