"""
Period 2 Partial Closure — Systematic skip-pattern test at v16 (12D)
=====================================================================
1. Build 8D base (13 vertices, beta=(1,0,3,0))
2. 3 building steps: v13, v14, v15 — each maximizing beta2 with beta1=0
3. Step 4: v16 — exhaustive partial K-closure test
   - Full K-closure (all 16 vertices)
   - Single skip: skip each vertex once
   - Double skip: skip each pair
   - Triple skip: 500 random triples
4. Report: configs with beta3>0 (sorted), beta2>0 flagged
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
import time
import random


# ── F2 linear algebra ──────────────────────────────────────────────────────

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


# ── Simplicial complex ─────────────────────────────────────────────────────

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


# ── Build 8D base ──────────────────────────────────────────────────────────

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


# ── Find beta2-maximizing insertion ───────────────────────────────────────

def find_best_b2_insertion(base_sc, new_vertex, max_subsets=4096, seed=42):
    """Find insertion maximizing beta2 with beta1=0."""
    existing = base_sc.vertices
    n = len(existing)
    total = 2**n - 1
    rng = random.Random(seed)

    if total <= 8192:
        subsets = [frozenset(combo) for size in range(1, n+1)
                   for combo in combinations(existing, size)]
    else:
        subsets = set()
        # Always include full K-closure
        subsets.add(frozenset(existing))
        # Single-skip variants
        for skip in existing:
            subsets.add(frozenset(v for v in existing if v != skip))
        # Double-skip variants
        for v1, v2 in combinations(existing, 2):
            subsets.add(frozenset(v for v in existing if v not in {v1, v2}))
        # Random fill
        remaining = max_subsets - len(subsets)
        if remaining > 0:
            pool = []
            for size in range(1, n+1):
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

    for S in subsets:
        sc = base_sc.copy()
        sc.insert_vertex_with_cone(new_vertex, S)
        b = sc.betti()
        if b[1] == 0 and b[2] > best_b2:
            best_b2 = b[2]
            best_S = S
            best_betti = b

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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # === Phase 1: Build 8D base ===
    print("=" * 60)
    print("PHASE 1: 8D BASE")
    print("=" * 60)
    base = build_8d_base()
    b = base.betti()
    st = base.stats()
    print(f"  V={st[0]}, E={st[1]}, T={st[2]}, Tet={st[3]}")
    print(f"  beta = {b}")
    assert b == (1, 0, 3, 0), f"8D base mismatch: {b}"
    sys.stdout.flush()

    # === Phase 2: 3 building steps ===
    print("\n" + "=" * 60)
    print("PHASE 2: BUILDING STEPS (v13, v14, v15) — maximize beta2")
    print("=" * 60)

    sc = base.copy()
    build_states = [(8, 13, b, None)]  # (dim, next_vid, betti, S)

    for step, (new_vid, label) in enumerate([(13, "9D"), (14, "10D"), (15, "11D")]):
        t = time.time()
        n = len(sc.vertices)
        total_subsets = 2**n - 1
        use_sample = total_subsets > 8192
        print(f"\n  [{label}] v{new_vid}: n={n}, subsets={min(total_subsets, 4096)} ({'sampled' if use_sample else 'full'}) ...", end=' ')
        sys.stdout.flush()

        best_S, best_b = find_best_b2_insertion(sc, new_vid, max_subsets=4096, seed=42 + step)
        sc.insert_vertex_with_cone(new_vid, best_S)
        actual_b = sc.betti()
        actual_st = sc.stats()
        print(f"done ({time.time()-t:.1f}s)")
        print(f"    beta={actual_b}, V={actual_st[0]}, Tet={actual_st[3]}")
        print(f"    S={sorted(best_S)}")
        build_states.append((int(label[:-1]), new_vid + 1, actual_b, sorted(best_S)))
        sys.stdout.flush()

    # sc now has 16 vertices (v0..v15)
    all_existing = list(sc.vertices)
    n_existing = len(all_existing)
    print(f"\n  Pre-closure state: V={n_existing}, beta={sc.betti()}")
    sys.stdout.flush()

    # === Phase 3: Partial K-closure test at v16 ===
    print("\n" + "=" * 60)
    print("PHASE 3: PARTIAL K-CLOSURE TEST — v16 (12D)")
    print(f"  Existing vertices: {all_existing}")
    print("=" * 60)
    sys.stdout.flush()

    results = []  # (skip_set, betti)
    new_vid = 16

    # -- Full K-closure --
    t = time.time()
    sc_full = sc.copy()
    sc_full.insert_vertex_with_cone(new_vid, frozenset(all_existing))
    b_full = sc_full.betti()
    results.append((frozenset(), b_full))
    print(f"\n  [FULL] skip=none  beta={b_full}  ({time.time()-t:.2f}s)")
    sys.stdout.flush()

    # -- Single skips --
    print(f"\n  [SINGLE SKIP] {n_existing} variants:")
    for v in all_existing:
        S = frozenset(u for u in all_existing if u != v)
        sc_t = sc.copy()
        sc_t.insert_vertex_with_cone(new_vid, S)
        b = sc_t.betti()
        results.append((frozenset({v}), b))
        marker = " ***beta3>0***" if b[3] > 0 else ("  [beta2>0]" if b[2] > 0 else "")
        print(f"    skip={{v{v}}}  beta={b}{marker}")
        sys.stdout.flush()

    # -- Double skips --
    pairs = list(combinations(all_existing, 2))
    print(f"\n  [DOUBLE SKIP] {len(pairs)} variants:")
    for v1, v2 in pairs:
        S = frozenset(u for u in all_existing if u not in {v1, v2})
        sc_t = sc.copy()
        sc_t.insert_vertex_with_cone(new_vid, S)
        b = sc_t.betti()
        results.append((frozenset({v1, v2}), b))
        if b[3] > 0 or b[2] > 0:
            marker = " ***beta3>0***" if b[3] > 0 else "  [beta2>0]"
            print(f"    skip={{v{v1},v{v2}}}  beta={b}{marker}")
        sys.stdout.flush()

    print(f"  (double skip done)")

    # -- Triple skips (sample 500) --
    rng = random.Random(137)
    triples_all = list(combinations(all_existing, 3))
    if len(triples_all) > 500:
        triples_sample = rng.sample(triples_all, 500)
        triple_label = f"500/{len(triples_all)} sampled"
    else:
        triples_sample = triples_all
        triple_label = f"{len(triples_all)} (full)"
    print(f"\n  [TRIPLE SKIP] {triple_label}:")
    triple_hits = []
    for v1, v2, v3 in triples_sample:
        S = frozenset(u for u in all_existing if u not in {v1, v2, v3})
        sc_t = sc.copy()
        sc_t.insert_vertex_with_cone(new_vid, S)
        b = sc_t.betti()
        results.append((frozenset({v1, v2, v3}), b))
        if b[3] > 0 or b[2] > 0:
            triple_hits.append((frozenset({v1, v2, v3}), b))
    if triple_hits:
        for skip_set, b in triple_hits:
            marker = " ***beta3>0***" if b[3] > 0 else "  [beta2>0]"
            print(f"    skip={set(skip_set)}  beta={b}{marker}")
    else:
        print(f"  (no beta3>0 or beta2>0 in sample)")
    sys.stdout.flush()

    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY: beta3 > 0 configurations")
    print("=" * 60)

    b3_positive = [(skip, b) for skip, b in results if b[3] > 0]
    b3_positive.sort(key=lambda x: (-x[1][3], -x[1][2]))

    if b3_positive:
        print(f"  Found {len(b3_positive)} configs with beta3 > 0:\n")
        print(f"  {'skip_size':<12} {'skipped':<25} {'beta':<20} note")
        print(f"  {'-'*70}")
        for skip_set, b in b3_positive:
            skip_list = sorted(skip_set)
            note = ""
            # Check analogy to period-1 skips (v7, v8)
            if 7 in skip_set or 8 in skip_set:
                note += " [cube-body analog]"
            if 12 in skip_set:
                note += " [K-point analog]"
            print(f"  {len(skip_set):<12} {str(skip_list):<25} {str(b):<20} {note}")
    else:
        print("  None found.")

    print("\n" + "=" * 60)
    print("SUMMARY: beta2 > 0 configurations (partial closure)")
    print("=" * 60)

    b2_positive = [(skip, b) for skip, b in results if b[2] > 0 and b[3] == 0]
    b2_positive.sort(key=lambda x: (-x[1][2], len(x[0])))

    if b2_positive:
        # Show top 20
        show = b2_positive[:20]
        print(f"  Found {len(b2_positive)} configs with beta2 > 0 (and beta3 = 0), showing top 20:\n")
        print(f"  {'skip_size':<12} {'skipped':<25} {'beta'}")
        print(f"  {'-'*60}")
        for skip_set, b in show:
            skip_list = sorted(skip_set)
            print(f"  {len(skip_set):<12} {str(skip_list):<25} {b}")
    else:
        print("  None found (all configs either trivial or have beta3>0).")

    print("\n" + "=" * 60)
    print("BETTI DISTRIBUTION ACROSS ALL TESTED CONFIGS")
    print("=" * 60)
    from collections import Counter
    b_counter = Counter(b for _, b in results)
    for betti, count in sorted(b_counter.items(), key=lambda x: x[0]):
        print(f"  {betti}: {count} configs")

    print(f"\n  Total configs tested: {len(results)}")
    print(f"  Total runtime: {time.time()-t_start:.1f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
