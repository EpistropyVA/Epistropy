"""
Period 2 Mixed Strategy — Cascade from 8D to 16D
=================================================
Test mixed strategies: alternating build (maximize β₂) and K-closure steps.

Key insight: β₃ residue requires tetrahedra from at least TWO different cone
vertices. Single-cone tetrahedra are contractible. We need multi-cone structure.

8D base: BCC cube(0-7) + v8(center) + v9→{3,5,6} + v10→{1,2,4,9}
       + v11→{0,3,5,6} + v12→{0,1,2,3,4,5,6,9,10,11}
Expected base: β=(1,0,3,0)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
import time
import random


# ─────────────────────────────────────────────────────────────────────────────
# F₂ rank
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# SimplicialComplex over F₂
# ─────────────────────────────────────────────────────────────────────────────
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
            return np.zeros((nv, 0), dtype=np.int8)
        v_idx = {v: i for i, v in enumerate(self.vertices)}
        mat = np.zeros((nv, ne), dtype=np.int8)
        for edge, ei in self.edge_idx.items():
            for v in edge:
                mat[v_idx[v], ei] = 1
        return mat

    def boundary_2(self):
        ne = len(self.edge_idx)
        nt = len(self.tri_idx)
        if nt == 0:
            return np.zeros((ne, 0), dtype=np.int8)
        mat = np.zeros((ne, nt), dtype=np.int8)
        for tri, ti in self.tri_idx.items():
            verts = sorted(tri)
            for a, b in combinations(verts, 2):
                ei = self.edge_idx.get(frozenset({a, b}))
                if ei is not None:
                    mat[ei, ti] = 1
        return mat

    def boundary_3(self):
        nt = len(self.tri_idx)
        ntet = len(self.tet_idx)
        if ntet == 0:
            return np.zeros((nt, 0), dtype=np.int8)
        mat = np.zeros((nt, ntet), dtype=np.int8)
        for tet, teti in self.tet_idx.items():
            verts = sorted(tet)
            for face in combinations(verts, 3):
                fi = self.tri_idx.get(frozenset(face))
                if fi is not None:
                    mat[fi, teti] = 1
        return mat

    def betti(self):
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
        b0 = nv - rank_d1
        b1 = (ne - rank_d1) - rank_d2
        b2 = (nt - rank_d2) - rank_d3
        b3 = ntet - rank_d3
        return (b0, b1, b2, b3)

    def stats(self):
        return (len(self.vertices), len(self.edge_idx),
                len(self.tri_idx), len(self.tet_idx))


# ─────────────────────────────────────────────────────────────────────────────
# Build 8D base
# ─────────────────────────────────────────────────────────────────────────────
def build_8d_base():
    sc = SimplicialComplex()
    for i in range(9):
        sc.add_vertex(i)
    # BCC cube: Hamming-1 edges
    cube_edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))
                sc.add_edge(a, b)
    # v8 = body center: connect to all cube corners
    for i in range(8):
        sc.add_edge(8, i)
    # Triangles: each cube edge + center forms a triangle
    for a, b in cube_edges:
        sc.add_triangle(a, b, 8)
    # v9 → {3,5,6}
    sc.insert_vertex_with_cone(9, {3, 5, 6})
    # v10 → {1,2,4,9}
    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})
    # v11 → {0,3,5,6}
    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})
    # v12 → K-closure of {0..6,9,10,11}
    sc.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})
    return sc


# ─────────────────────────────────────────────────────────────────────────────
# Best β₂ insertion search
# ─────────────────────────────────────────────────────────────────────────────
def find_best_b2_insertion(base_sc, new_vertex, max_subsets=4096, seed=42):
    """Find insertion set S maximizing β₂ with β₁=0 preferred."""
    existing = base_sc.vertices
    n = len(existing)
    total = 2**n - 1
    rng = random.Random(seed)

    if total <= 8192:
        subsets = [frozenset(combo) for size in range(1, n + 1)
                   for combo in combinations(existing, size)]
    else:
        subsets = set()
        # Always include all-connect variant
        subsets.add(frozenset(existing))
        # Skip one vertex variants
        for skip in existing:
            subsets.add(frozenset(v for v in existing if v != skip))
        # Small subsets exhaustively
        for size in [2, 3, 4, 5, 6]:
            all_c = list(combinations(existing, size))
            if len(all_c) <= 400:
                subsets.update(frozenset(c) for c in all_c)
            else:
                subsets.update(frozenset(c) for c in rng.sample(all_c, 400))
        # Fill remaining quota with random samples
        remaining = max_subsets - len(subsets)
        if remaining > 0:
            pool = []
            for size in range(1, n + 1):
                for combo in combinations(existing, size):
                    fs = frozenset(combo)
                    if fs not in subsets:
                        pool.append(fs)
            if len(pool) > remaining:
                subsets.update(rng.sample(pool, remaining))
            else:
                subsets.update(pool)
        subsets = list(subsets)

    best_b2 = -1
    best_S = None
    best_betti = None

    # Prefer β₁=0
    for S in subsets:
        sc = base_sc.copy()
        sc.insert_vertex_with_cone(new_vertex, S)
        b = sc.betti()
        if b[1] == 0 and b[2] > best_b2:
            best_b2 = b[2]
            best_S = S
            best_betti = b

    # Fall back to any β₂ maximizer
    if best_S is None:
        for S in subsets:
            sc = base_sc.copy()
            sc.insert_vertex_with_cone(new_vertex, S)
            b = sc.betti()
            if b[2] > best_b2:
                best_b2 = b[2]
                best_S = S
                best_betti = b

    return best_S, best_betti


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def k_close(sc, new_vertex):
    """Insert new_vertex connected to ALL existing vertices (K-closure)."""
    all_v = frozenset(sc.vertices)
    sc2 = sc.copy()
    sc2.insert_vertex_with_cone(new_vertex, all_v)
    return sc2


def fmt(sc):
    b = sc.betti()
    st = sc.stats()
    return f"V={st[0]}, beta={b}, Tet={st[3]}"


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Controlled multi-phase cascade
# ─────────────────────────────────────────────────────────────────────────────
def experiment1(base):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Controlled multi-phase cascade (8D → 20D)")
    print("  Phase A (2 build): v13, v14 — maximize β₂, β₁=0")
    print("  Phase B (1 K-close): v15 — connect ALL (first cone)")
    print("  Phase C (2 build): v16, v17 — maximize β₂ on closed base")
    print("  Phase D (1 K-close): v18 — connect ALL (second cone)")
    print("  Phase E (explore): v19, v20 — report both max-β₂ and K-close")
    print("=" * 70)

    sc = base.copy()
    vid = 13

    # Phase A: 2 build steps
    print("\n--- Phase A: Build ---")
    for step in range(2):
        dim = 9 + step
        best_S, best_b = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=42 + step)
        sc.insert_vertex_with_cone(vid, best_S)
        b = sc.betti()
        st = sc.stats()
        print(f"  {dim}D build v{vid}: V={st[0]}, beta={b}, Tet={st[3]}, |S|={len(best_S)}")
        vid += 1

    # Phase B: K-close
    print("\n--- Phase B: First K-closure ---")
    sc = k_close(sc, vid)
    b = sc.betti()
    st = sc.stats()
    print(f"  11D K-close v{vid}: {fmt(sc)}")
    vid += 1

    # Phase C: 2 build steps on closed base
    print("\n--- Phase C: Build on closed base ---")
    for step in range(2):
        dim = 12 + step
        best_S, best_b = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=100 + step)
        sc.insert_vertex_with_cone(vid, best_S)
        b = sc.betti()
        st = sc.stats()
        print(f"  {dim}D build v{vid}: V={st[0]}, beta={b}, Tet={st[3]}, |S|={len(best_S)}")
        vid += 1

    # Phase D: Second K-closure
    print("\n--- Phase D: Second K-closure ---")
    sc = k_close(sc, vid)
    b = sc.betti()
    st = sc.stats()
    print(f"  14D K-close v{vid}: {fmt(sc)}")
    vid += 1

    # Phase E: Explore both options for v19, v20
    print("\n--- Phase E: Dual exploration ---")
    for step in range(2):
        dim = 15 + step
        # Option 1: maximize β₂
        best_S, best_b = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=200 + step)
        sc_build = sc.copy()
        sc_build.insert_vertex_with_cone(vid, best_S)
        b_build = sc_build.betti()
        st_build = sc_build.stats()

        # Option 2: K-close
        sc_kc = k_close(sc, vid)
        b_kc = sc_kc.betti()
        st_kc = sc_kc.stats()

        print(f"  {dim}D v{vid}:")
        print(f"    max-β₂: V={st_build[0]}, beta={b_build}, Tet={st_build[3]}, |S|={len(best_S)}")
        print(f"    K-close: V={st_kc[0]}, beta={b_kc}, Tet={st_kc[3]}")

        # Advance with max-β₂ for next step (arbitrary — both paths explored above)
        sc = sc_build
        vid += 1

    print()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Systematic partial closure timing
# ─────────────────────────────────────────────────────────────────────────────
def experiment2(base):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Systematic partial closure timing")
    print("  For each first-closure point D in {9D..12D}:")
    print("    build up to D (maximize β₂)")
    print("    K-close at D")
    print("    build one more step (maximize β₂)")
    print("    K-close again")
    print("  Focus: β₃ after double K-close")
    print("=" * 70)

    # first_close_dim = 9,10,11,12 means we build 0,1,2,3 steps before first close
    for first_close_dim in range(9, 13):
        build_steps = first_close_dim - 9  # 0..3

        sc = base.copy()
        vid = 13

        # Build up to first_close_dim - 1
        for step in range(build_steps):
            best_S, _ = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=42 + step)
            sc.insert_vertex_with_cone(vid, best_S)
            vid += 1

        # First K-close at first_close_dim
        sc = k_close(sc, vid)
        b_after_close1 = sc.betti()
        vid += 1

        # Build one more step
        best_S, _ = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=99)
        sc.insert_vertex_with_cone(vid, best_S)
        vid += 1

        # Second K-close
        sc = k_close(sc, vid)
        b_final = sc.betti()
        st_final = sc.stats()
        vid += 1

        dim_after = len(base.vertices) + build_steps + 1 + 1 + 1 + 1  # vertices added
        print(f"  first_close@{first_close_dim}D: after_close1_beta={b_after_close1}, "
              f"double_close_beta={b_final}, V={st_final[0]}, Tet={st_final[3]}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Natural partial double-closure with skip variants
# ─────────────────────────────────────────────────────────────────────────────
def experiment3(base):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Partial double-closure — skip variants")
    print("  State: 2 build + 1 K-close + 1 build (4 insertions from 8D)")
    print("  At second closure: test all single-skip and double-skip variants")
    print("  Report which produce β₃ > 0; sort by: β₂ asc, then β₃ desc")
    print("=" * 70)

    sc = base.copy()
    vid = 13

    # Step 1: build (maximize β₂)
    best_S, _ = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=42)
    sc.insert_vertex_with_cone(vid, best_S)
    b1 = sc.betti()
    print(f"  v{vid} build: beta={b1}, Tet={sc.stats()[3]}, |S|={len(best_S)}")
    vid += 1

    # Step 2: build (maximize β₂)
    best_S, _ = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=43)
    sc.insert_vertex_with_cone(vid, best_S)
    b2 = sc.betti()
    print(f"  v{vid} build: beta={b2}, Tet={sc.stats()[3]}, |S|={len(best_S)}")
    vid += 1

    # Step 3: K-close (first closure)
    sc = k_close(sc, vid)
    b3 = sc.betti()
    print(f"  v{vid} K-close: beta={b3}, Tet={sc.stats()[3]}")
    vid += 1

    # Step 4: build one more
    best_S, _ = find_best_b2_insertion(sc, vid, max_subsets=4096, seed=100)
    sc.insert_vertex_with_cone(vid, best_S)
    b4 = sc.betti()
    print(f"  v{vid} build: beta={b4}, Tet={sc.stats()[3]}, |S|={len(best_S)}")
    vid += 1

    print(f"\n  State before second closure: V={sc.stats()[0]}, beta={b4}")
    existing = sc.vertices
    all_v = frozenset(existing)
    n = len(existing)
    print(f"  Testing second closure variants (all-v, single-skip, double-skip): {n} vertices")

    results = []

    # All-connect
    sc_test = sc.copy()
    sc_test.insert_vertex_with_cone(vid, all_v)
    b = sc_test.betti()
    st = sc_test.stats()
    results.append((b[2], -b[3], 'all-connect', b, st[3], len(all_v)))

    # Single-skip variants
    for skip in existing:
        S = all_v - {skip}
        sc_test = sc.copy()
        sc_test.insert_vertex_with_cone(vid, S)
        b = sc_test.betti()
        st = sc_test.stats()
        results.append((b[2], -b[3], f'skip v{skip}', b, st[3], len(S)))

    # Double-skip variants
    for s1, s2 in combinations(existing, 2):
        S = all_v - {s1, s2}
        sc_test = sc.copy()
        sc_test.insert_vertex_with_cone(vid, S)
        b = sc_test.betti()
        st = sc_test.stats()
        results.append((b[2], -b[3], f'skip v{s1},v{s2}', b, st[3], len(S)))

    # Sort: β₂ asc, then β₃ desc (=-β₃ asc)
    results.sort(key=lambda x: (x[0], x[1]))

    print(f"\n  Results sorted by (β₂ asc, β₃ desc) — showing β₃>0 first:")
    b3_positive = [(desc, b, tet, s_len) for b2, neg_b3, desc, b, tet, s_len in results if b[3] > 0]
    b3_zero = [(desc, b, tet, s_len) for b2, neg_b3, desc, b, tet, s_len in results if b[3] == 0]

    if b3_positive:
        print(f"  [β₃ > 0 variants: {len(b3_positive)}]")
        for desc, b, tet, s_len in b3_positive:
            print(f"    {desc}: beta={b}, Tet={tet}, |S|={s_len}")
    else:
        print("  [No β₃ > 0 variants found]")

    print(f"\n  [β₃ = 0 sample (first 10 by β₂ asc)]")
    for desc, b, tet, s_len in b3_zero[:10]:
        print(f"    {desc}: beta={b}, Tet={tet}, |S|={s_len}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()

    print("Building 8D base complex...")
    base = build_8d_base()
    b_base = base.betti()
    st_base = base.stats()
    print(f"8D base: V={st_base[0]}, beta={b_base}, Tet={st_base[3]}")
    print(f"  (expected: beta=(1,0,3,0))")

    if b_base != (1, 0, 3, 0):
        print(f"  WARNING: beta mismatch! Got {b_base}, expected (1,0,3,0)")

    # Run experiments
    experiment1(base)
    print(f"  [after Exp1: {time.time()-t_start:.1f}s]")

    experiment2(base)
    print(f"  [after Exp2: {time.time()-t_start:.1f}s]")

    experiment3(base)
    print(f"  [after Exp3: {time.time()-t_start:.1f}s]")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
