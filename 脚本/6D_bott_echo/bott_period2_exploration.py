"""
Bott Period 2 Exploration — Simplicial Cascade over F₂
========================================================
Explores the second Bott period starting from the verified 8D base complex
(13 vertices, β=(1,0,3,0)).

Structure:
  Part 0: F₂ homology calculator (up to dimension 3)
  Part 1: Verify 8D base complex
  Part 2: Enumerate all 9D insertions (v13)
  Part 3: Enumerate 10D insertions (v14) from best 9D state
  Part 4: Pattern analysis
"""

import sys
import io
# Force UTF-8 output to avoid GBK encoding errors on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
import time
import random


# ─────────────────────────────────────────────────────────────────────────────
# F₂ linear algebra
# ─────────────────────────────────────────────────────────────────────────────

def rank_f2(mat):
    """Rank of a matrix over GF(2) via Gaussian elimination."""
    if mat.size == 0:
        return 0
    m = mat.copy() % 2
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        # find pivot
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


def kernel_dim(mat, ncols):
    """dim(ker(mat)) = ncols - rank(mat)."""
    return ncols - rank_f2(mat)


# ─────────────────────────────────────────────────────────────────────────────
# Simplicial complex data structure
# ─────────────────────────────────────────────────────────────────────────────

class SimplicialComplex:
    """
    Tracks vertices, edges, triangles, tetrahedra with indexed dicts.
    All operations are over F₂.
    """

    def __init__(self):
        self.vertices = []          # list of vertex ids
        self.edge_idx = {}          # frozenset({a,b}) -> int index
        self.tri_idx = {}           # frozenset({a,b,c}) -> int index
        self.tet_idx = {}           # frozenset({a,b,c,d}) -> int index

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
        """
        Insert vertex v connected to subset S.
        Adds all induced simplices:
          edges: {v,s} for s in S
          triangles: {v,a,b} for {a,b} in S that's an existing edge
          tetrahedra: {v,a,b,c} for {a,b,c} in S that's an existing triangle
        """
        self.add_vertex(v)
        S = list(S)

        # new edges
        for s in S:
            self.add_edge(v, s)

        # new triangles from existing edges within S
        for a, b in combinations(S, 2):
            if frozenset({a, b}) in self.edge_idx:
                self.add_triangle(v, a, b)

        # new tetrahedra from existing triangles within S
        for a, b, c in combinations(S, 3):
            if frozenset({a, b, c}) in self.tri_idx:
                self.add_tetrahedron(v, a, b, c)

    # ── boundary matrices ──────────────────────────────────────────────────

    def boundary_1(self):
        """∂₁: edges → vertices, shape = (|V|, |E|)."""
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
        """∂₂: triangles → edges, shape = (|E|, |T|)."""
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        if nt == 0:
            return np.zeros((ne, 0), dtype=np.int8)
        mat = np.zeros((ne, nt), dtype=np.int8)
        for tri, ti in self.tri_idx.items():
            verts = sorted(tri)
            # 3 edges of triangle
            for a, b in combinations(verts, 2):
                ei = self.edge_idx.get(frozenset({a, b}))
                if ei is not None:
                    mat[ei, ti] = 1
        return mat

    def boundary_3(self):
        """∂₃: tetrahedra → triangles, shape = (|T|, |Tet|)."""
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        if ntet == 0:
            return np.zeros((nt, 0), dtype=np.int8)
        mat = np.zeros((nt, ntet), dtype=np.int8)
        for tet, teti in self.tet_idx.items():
            verts = sorted(tet)
            # 4 faces of tetrahedron
            for face in combinations(verts, 3):
                fi = self.tri_idx.get(frozenset(face))
                if fi is not None:
                    mat[fi, teti] = 1
        return mat

    def betti(self):
        """Compute β₀, β₁, β₂, β₃ over F₂."""
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

        # ker ∂_k
        ker0 = nv            # ∂₀ = 0
        ker1 = ne - rank_d1  # ker ∂₁
        ker2 = nt - rank_d2  # ker ∂₂
        ker3 = ntet - rank_d3  # ker ∂₃

        # im ∂_{k+1}
        im1 = rank_d1   # im ∂₁ ⊂ C₀
        im2 = rank_d2   # im ∂₂ ⊂ C₁
        im3 = rank_d3   # im ∂₃ ⊂ C₂

        b0 = ker0 - im1
        b1 = ker1 - im2
        b2 = ker2 - im3
        b3 = ker3  # no ∂₄

        return (b0, b1, b2, b3)

    def stats(self):
        return (len(self.vertices), len(self.edge_idx),
                len(self.tri_idx), len(self.tet_idx))


# ─────────────────────────────────────────────────────────────────────────────
# Part 0 test: tiny sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check():
    print("=== Sanity checks ===")

    # Single triangle: β=(1,0,0,0) (filled) or (1,1,0,0) (empty loop)
    sc = SimplicialComplex()
    for i in range(3):
        sc.add_vertex(i)
    sc.add_edge(0,1); sc.add_edge(1,2); sc.add_edge(0,2)
    sc.add_triangle(0,1,2)
    print(f"  Filled triangle β={sc.betti()} (expect (1,0,0,0))")

    sc2 = SimplicialComplex()
    for i in range(3):
        sc2.add_vertex(i)
    sc2.add_edge(0,1); sc2.add_edge(1,2); sc2.add_edge(0,2)
    print(f"  Empty triangle β={sc2.betti()} (expect (1,1,0,0))")

    # Hollow tetrahedron: S² → β=(1,0,1,0)
    sc3 = SimplicialComplex()
    for i in range(4):
        sc3.add_vertex(i)
    for a,b in combinations(range(4),2):
        sc3.add_edge(a,b)
    for a,b,c in combinations(range(4),3):
        sc3.add_triangle(a,b,c)
    print(f"  Hollow tetrahedron β={sc3.betti()} (expect (1,0,1,0))")

    # Filled tetrahedron: β=(1,0,0,0)
    sc4 = sc3.copy()
    sc4.add_tetrahedron(0,1,2,3)
    print(f"  Filled tetrahedron β={sc4.betti()} (expect (1,0,0,0))")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Build 8D base complex
# ─────────────────────────────────────────────────────────────────────────────

def build_8d_base():
    """
    Build the verified 8D complex (13 vertices, β=(1,0,3,0)).
    Vertices 0-7: cube corners, 8: body center
    BCC: 12 cube edges + 8 center-to-corner edges + 12 cone triangles
    Then insertions: v9→{3,5,6}, v10→{1,2,4,9}, v11→{0,3,5,6}
    Then K-closure: v12→{0,1,2,3,4,5,6,9,10,11}
    """
    sc = SimplicialComplex()

    # Vertices 0-8
    for i in range(9):
        sc.add_vertex(i)

    # Cube edges (pairs differing in exactly one bit)
    cube_edges = []
    for a in range(8):
        for b in range(a+1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))
                sc.add_edge(a, b)

    # Center-to-corner edges
    for i in range(8):
        sc.add_edge(8, i)

    # BCC cone triangles: {a, b, 8} for each cube edge {a,b}
    for a, b in cube_edges:
        sc.add_triangle(a, b, 8)

    # v9 → edges to {3,5,6}, no triangles (no pre-existing edges among {3,5,6})
    sc.insert_vertex_with_cone(9, {3, 5, 6})

    # v10 → edges to {1,2,4,9}
    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})

    # v11 → edges to {0,3,5,6}
    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})

    # v12 → K-closure: {0,1,2,3,4,5,6,9,10,11}
    sc.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})

    return sc


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 & 3: Enumerate insertions
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_insertions(base_sc, new_vertex, max_subsets=None, seed=42):
    """
    For each non-empty subset S of base_sc.vertices, insert new_vertex with cone S.
    Returns list of (S, betti, stats).
    If max_subsets given, randomly sample that many subsets.
    """
    existing = base_sc.vertices
    n = len(existing)
    results = []

    if max_subsets is None:
        # Full enumeration: all 2^n - 1 non-empty subsets
        subset_iter = []
        for size in range(1, n+1):
            for combo in combinations(existing, size):
                subset_iter.append(frozenset(combo))
    else:
        # Random sample
        rng = random.Random(seed)
        all_subsets = []
        for size in range(1, n+1):
            for combo in combinations(existing, size):
                all_subsets.append(frozenset(combo))
        subset_iter = rng.sample(all_subsets, min(max_subsets, len(all_subsets)))

    for S in subset_iter:
        sc = base_sc.copy()
        sc.insert_vertex_with_cone(new_vertex, S)
        b = sc.betti()
        st = sc.stats()
        results.append((S, b, st))

    return results


def report_enumeration(results, label):
    print(f"=== {label} ===")
    print(f"  Total configurations: {len(results)}")

    from collections import Counter
    b1_dist = Counter(b[1] for _, b, _ in results)
    b2_dist = Counter(b[2] for _, b, _ in results)
    b3_dist = Counter(b[3] for _, b, _ in results)

    print(f"\n  β₁ distribution:")
    for k in sorted(b1_dist):
        print(f"    β₁={k}: {b1_dist[k]} configs")

    print(f"\n  β₂ distribution:")
    for k in sorted(b2_dist):
        print(f"    β₂={k}: {b2_dist[k]} configs")

    print(f"\n  β₃ distribution:")
    for k in sorted(b3_dist):
        print(f"    β₃={k}: {b3_dist[k]} configs")

    # Top 10 by β₂
    sorted_by_b2 = sorted(results, key=lambda x: (x[1][2], -x[1][1], x[1][3]), reverse=True)
    print(f"\n  Top 10 by β₂:")
    print(f"  {'β':<20} {'|S|':<5} {'V,E,T,Tet':<20} S")
    for S, b, st in sorted_by_b2[:10]:
        print(f"  β={b}  |S|={len(S):<3} {str(st):<20} {sorted(S)}")

    # β₁=0 AND β₂ > 3 (increased from base)
    candidates = [(S, b, st) for S, b, st in results if b[1] == 0 and b[2] > 3]
    candidates.sort(key=lambda x: (x[1][2], x[1][3]), reverse=True)
    print(f"\n  Configs with β₁=0 AND β₂>3: {len(candidates)}")
    if candidates:
        print(f"  Top 5:")
        for S, b, st in candidates[:5]:
            print(f"    β={b}  |S|={len(S)}  {sorted(S)}")

    # Best candidate for next stage
    # Priority: β₁=0 AND max β₂, then β₃
    if candidates:
        best = candidates[0]
    else:
        best = sorted_by_b2[0]

    print(f"\n  Best candidate: β={best[1]}  S={sorted(best[0])}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # ── Sanity ────────────────────────────────────────────────────────────────
    sanity_check()

    # ── Part 1: 8D base ───────────────────────────────────────────────────────
    print("=== Part 1: 8D Base Complex ===")
    base = build_8d_base()
    b = base.betti()
    st = base.stats()
    print(f"  Stats: V={st[0]}, E={st[1]}, T={st[2]}, Tet={st[3]}")
    print(f"  β = {b}")
    if b == (1, 0, 3, 0):
        print("  [OK] Verified: betti=(1,0,3,0)")
    else:
        print(f"  [FAIL] MISMATCH -- expected (1,0,3,0), got {b}")
    print()

    # ── Part 2: v13 enumeration ────────────────────────────────────────────────
    print("=== Part 2: Enumerating v13 insertions (2^13 = 8192 subsets) ===")
    t2 = time.time()
    results_13 = enumerate_insertions(base, 13)
    print(f"  Enumeration time: {time.time()-t2:.1f}s")
    print()
    best_13 = report_enumeration(results_13, "v13 Results")
    print()

    # ── Part 3: v14 enumeration from best 13-vertex state ─────────────────────
    print("=== Part 3: Enumerating v14 insertions ===")
    best_S13, best_b13, _ = best_13

    # Build the best 13-vertex complex
    sc13 = base.copy()
    sc13.insert_vertex_with_cone(13, best_S13)
    b13 = sc13.betti()
    st13 = sc13.stats()
    print(f"  Starting from 13-vertex complex:")
    print(f"    v13 connected to: {sorted(best_S13)}")
    print(f"    Stats: V={st13[0]}, E={st13[1]}, T={st13[2]}, Tet={st13[3]}")
    print(f"    β = {b13}")
    print()

    # 2^14 = 16384 subsets — enumerate fully, but time-check first
    n14 = len(sc13.vertices)  # should be 14
    total_subsets = 2**n14 - 1
    print(f"  Total subsets to check: {total_subsets}")

    t3 = time.time()
    # Use sampling if too large
    MAX_SUBSETS_14 = 4096
    use_sample = total_subsets > 8192
    if use_sample:
        print(f"  Sampling {MAX_SUBSETS_14} random subsets...")
        results_14 = enumerate_insertions(sc13, 14, max_subsets=MAX_SUBSETS_14)
    else:
        results_14 = enumerate_insertions(sc13, 14)
    print(f"  Enumeration time: {time.time()-t3:.1f}s")
    print()
    best_14 = report_enumeration(results_14, "v14 Results")
    print()

    # ── Part 4: Pattern analysis ───────────────────────────────────────────────
    print("=== Part 4: Pattern Analysis ===")
    print()
    print("  Period 1 β₁ evolution (4D→8D):")
    print("    4D:  β₁=0  (BCC base)")
    print("    5D:  β₁=2  (v9 → {3,5,6}, ijk-like)")
    print("    6D:  β₁=5  (v10 → {1,2,4,9})")
    print("    7D:  β₁=8  (v11 → {0,3,5,6})")
    print("    8D:  β₁=0  (v12 K-closure, β₂ jumps to 3)")
    print()
    print("  Period 2 β₂ evolution (8D→10D, observed so far):")

    # Collect β₂ values across the stages
    b2_at_8d = b[2]   # from base
    b2_at_9d = best_b13[2]
    b2_at_10d = best_14[1][2]

    print(f"    8D:  β₂={b2_at_8d}  (period-2 start)")
    print(f"    9D:  β₂={b2_at_9d}  (best v13 insertion, β₁={best_b13[1]})")
    print(f"    10D: β₂={b2_at_10d}  (best v14 insertion, β₁={best_14[1][1]})")
    print()

    # β₃ presence
    any_b3_13 = any(b[3] > 0 for _, b, _ in results_13)
    any_b3_14 = any(b[3] > 0 for _, b, _ in results_14)
    max_b3_13 = max(b[3] for _, b, _ in results_13)
    max_b3_14 = max(b[3] for _, b, _ in results_14)
    print(f"  β₃ presence:")
    print(f"    v13 stage: max β₃={max_b3_13}, any β₃>0: {any_b3_13}")
    print(f"    v14 stage: max β₃={max_b3_14}, any β₃>0: {any_b3_14}")
    print()

    print("  Structural comparison:")
    print("  Period 1 pattern: ijk insertions build β₁ (cycles), K-closure collapses β₁ to 0, β₂ appears")
    print("  Period 2 expectation: insertions build β₂ (2-cycles/surfaces), K-closure collapses β₂ to 0, β₃ appears")
    print()

    # Check if β₂ is increasing in period 2
    increasing = b2_at_9d >= b2_at_8d
    print(f"  β₂ increasing from 8D to 9D: {increasing} ({b2_at_8d} → {b2_at_9d})")

    if any_b3_13 or any_b3_14:
        print(f"  β₃ has appeared! This signals 3-dimensional structure forming.")
        if any_b3_13:
            b3_configs = [(S, b, st) for S, b, st in results_13 if b[3] > 0]
            b3_configs.sort(key=lambda x: x[1][3], reverse=True)
            print(f"  Top β₃ configs at 9D:")
            for S, b, st in b3_configs[:3]:
                print(f"    β={b}  |S|={len(S)}  {sorted(S)}")
    else:
        print(f"  β₃=0 throughout — 3D structure not yet closing")

    print()
    print(f"  Total runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
