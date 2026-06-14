"""
Bott Period 2 Extension — Push cascade from 10D to 16D
=======================================================
Starts from the verified 10D state (15 vertices) and extends
step by step to find β₂ closure.

Strategy: at each step, enumerate subsets (sample if >8192),
pick best candidate (β₁=0, max β₂), continue.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
import time
import random


# ── F₂ linear algebra ──────────────────────────────────────────────────────

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
        ker1 = ne - rank_d1
        ker2 = nt - rank_d2
        ker3 = ntet - rank_d3
        b0 = nv - rank_d1
        b1 = ker1 - rank_d2
        b2 = ker2 - rank_d3
        b3 = ker3
        return (b0, b1, b2, b3)

    def stats(self):
        return (len(self.vertices), len(self.edge_idx),
                len(self.tri_idx), len(self.tet_idx))


# ── Build verified states ──────────────────────────────────────────────────

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


def find_best_insertion(base_sc, new_vertex, max_subsets=4096, seed=42):
    """
    Enumerate subsets, find best candidate with β₁=0 and max β₂.
    Returns (best_S, best_betti, all_results_summary).
    """
    existing = base_sc.vertices
    n = len(existing)
    total = 2**n - 1

    rng = random.Random(seed)

    if total <= 8192:
        # Full enumeration
        subsets = []
        for size in range(1, n+1):
            for combo in combinations(existing, size):
                subsets.append(frozenset(combo))
        sampled = False
    else:
        # Smart sampling: mix targeted subsets with random
        subsets = []

        # Always include: connect to all (K-closure analog)
        subsets.append(frozenset(existing))

        # Include: connect to all except one vertex (near-closure)
        for skip in existing:
            subsets.append(frozenset(v for v in existing if v != skip))

        # Include: connect to specific structural vertices
        # v8=body center, v12=K-closure point - these are predicted closure triggers
        structural = {8, 12}
        for s in structural:
            if s in existing:
                subsets.append(frozenset({s}))
                # s + each other vertex
                for v in existing:
                    if v != s:
                        subsets.append(frozenset({s, v}))

        # Include: small subsets (size 2-4) - pure cycle builders
        for size in [2, 3, 4]:
            all_combos = list(combinations(existing, size))
            if len(all_combos) <= 500:
                for combo in all_combos:
                    subsets.append(frozenset(combo))
            else:
                subsets.extend(frozenset(c) for c in rng.sample(all_combos, 500))

        # Include: medium subsets (size 5-8)
        for size in [5, 6, 7, 8]:
            all_combos = list(combinations(existing, size))
            if len(all_combos) <= 300:
                for combo in all_combos:
                    subsets.append(frozenset(combo))
            else:
                subsets.extend(frozenset(c) for c in rng.sample(all_combos, 300))

        # Fill remaining budget with random
        remaining = max_subsets - len(subsets)
        if remaining > 0:
            all_subsets_set = set(subsets)
            pool = []
            for size in range(1, n+1):
                for combo in combinations(existing, size):
                    fs = frozenset(combo)
                    if fs not in all_subsets_set:
                        pool.append(fs)
            if len(pool) > remaining:
                subsets.extend(rng.sample(pool, remaining))
            else:
                subsets.extend(pool)

        # Deduplicate
        subsets = list(set(subsets))
        sampled = True

    # Evaluate
    from collections import Counter
    best_b2 = -1
    best_S = None
    best_betti = None
    b1_dist = Counter()
    b2_dist = Counter()
    b3_dist = Counter()
    b1_zero_max_b2 = -1
    b1_zero_count = 0

    for S in subsets:
        sc = base_sc.copy()
        sc.insert_vertex_with_cone(new_vertex, S)
        b = sc.betti()
        b1_dist[b[1]] += 1
        b2_dist[b[2]] += 1
        b3_dist[b[3]] += 1

        if b[1] == 0:
            b1_zero_count += 1
            if b[2] > b1_zero_max_b2:
                b1_zero_max_b2 = b[2]
                best_S = S
                best_betti = b

        if b[2] > best_b2:
            best_b2 = b[2]

    # If no β₁=0 candidate, use overall max β₂
    if best_S is None:
        for S in subsets:
            sc = base_sc.copy()
            sc.insert_vertex_with_cone(new_vertex, S)
            b = sc.betti()
            if b[2] == best_b2:
                best_S = S
                best_betti = b
                break

    summary = {
        'total_checked': len(subsets),
        'sampled': sampled,
        'b1_dist': dict(b1_dist),
        'b2_dist': dict(b2_dist),
        'b3_dist': dict(b3_dist),
        'b1_zero_count': b1_zero_count,
        'max_b2_overall': best_b2,
        'max_b2_with_b1_zero': b1_zero_max_b2,
    }

    return best_S, best_betti, summary


def main():
    t_start = time.time()

    # Build and verify 8D base
    print("=== Building 8D base ===")
    base = build_8d_base()
    b = base.betti()
    print(f"  8D: V={base.stats()[0]}, beta={b}")
    assert b == (1, 0, 3, 0), f"8D base mismatch: {b}"

    # Replay v13 (9D) and v14 (10D) with known best subsets from prior run
    # v13 best: need to re-derive or hardcode from prior output
    # For now, re-enumerate to find best
    print("\n=== Replaying 9D (v13) ===")
    t = time.time()
    best_S13, best_b13, summary13 = find_best_insertion(base, 13, max_subsets=8192)
    sc_9d = base.copy()
    sc_9d.insert_vertex_with_cone(13, best_S13)
    print(f"  9D: V={sc_9d.stats()[0]}, beta={sc_9d.betti()}, S={sorted(best_S13)}")
    print(f"  ({summary13['total_checked']} subsets, {time.time()-t:.1f}s)")

    print("\n=== Replaying 10D (v14) ===")
    t = time.time()
    best_S14, best_b14, summary14 = find_best_insertion(sc_9d, 14, max_subsets=4096)
    sc_10d = sc_9d.copy()
    sc_10d.insert_vertex_with_cone(14, best_S14)
    print(f"  10D: V={sc_10d.stats()[0]}, beta={sc_10d.betti()}, S={sorted(best_S14)}")
    print(f"  ({summary14['total_checked']} subsets, {time.time()-t:.1f}s)")

    # Now extend: 11D through 16D
    current = sc_10d
    dim = 10
    next_vid = 15

    evolution = [
        (8, 13, (1,0,3,0), None),
        (9, 14, sc_9d.betti(), sorted(best_S13)),
        (10, 15, sc_10d.betti(), sorted(best_S14)),
    ]

    for target_dim in range(11, 17):  # 11D through 16D
        print(f"\n=== {target_dim}D (v{next_vid}) ===")
        t = time.time()
        best_S, best_b, summary = find_best_insertion(current, next_vid, max_subsets=4096)

        new_sc = current.copy()
        new_sc.insert_vertex_with_cone(next_vid, best_S)
        actual_b = new_sc.betti()
        st = new_sc.stats()

        print(f"  {target_dim}D: V={st[0]}, E={st[1]}, T={st[2]}, Tet={st[3]}")
        print(f"  beta = {actual_b}")
        print(f"  S = {sorted(best_S)} (|S|={len(best_S)})")
        print(f"  Checked: {summary['total_checked']} subsets ({'sampled' if summary['sampled'] else 'full'})")
        print(f"  beta1=0 configs: {summary['b1_zero_count']}")
        print(f"  max beta2 (with b1=0): {summary['max_b2_with_b1_zero']}")
        print(f"  max beta2 (overall): {summary['max_b2_overall']}")

        # Show β₃ distribution
        b3d = summary['b3_dist']
        if any(k > 0 for k in b3d):
            print(f"  *** beta3 distribution: {b3d}")
        else:
            print(f"  beta3 = 0 throughout")

        print(f"  Time: {time.time()-t:.1f}s")

        evolution.append((target_dim, st[0], actual_b, sorted(best_S)))
        current = new_sc
        next_vid += 1

    # Final summary
    print("\n\n" + "="*60)
    print("PERIOD 2 CASCADE SUMMARY")
    print("="*60)
    print(f"{'dim':<5} {'V':<4} {'beta0':<6} {'beta1':<6} {'beta2':<6} {'beta3':<6} S")
    print("-"*60)
    for dim, nv, betti, S in evolution:
        print(f"{dim}D    {nv:<4} {betti[0]:<6} {betti[1]:<6} {betti[2]:<6} {betti[3]:<6} {S}")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
