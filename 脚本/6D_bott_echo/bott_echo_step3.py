"""
bott_echo_step3.py
Simplicial homology over F2 for BCC complex cascade.
Vertices 0-7: cube corners, 8: body center, 9,10,11: successive insertions.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations

# ── F₂ rank via Gaussian elimination ──────────────────────────────────────────

def rank_f2(matrix):
    """Rank of a binary matrix over F₂ via Gaussian elimination."""
    if matrix.size == 0:
        return 0
    M = matrix.copy() % 2
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        # Find pivot
        found = -1
        for r in range(pivot_row, rows):
            if M[r, col] == 1:
                found = r
                break
        if found == -1:
            continue
        if pivot_row != found:
            temp = M[pivot_row].copy()
            M[pivot_row] = M[found]
            M[found] = temp
        # Eliminate
        for r in range(rows):
            if r != pivot_row and M[r, col] == 1:
                M[r] = (M[r] + M[pivot_row]) % 2
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


# ── Complex builder ────────────────────────────────────────────────────────────

class SimplicialComplex:
    """Tracks vertices, edges, triangles and computes F₂ Betti numbers."""

    def __init__(self):
        self.vertices = set()
        self.edges = set()      # frozensets of size 2
        self.triangles = set()  # frozensets of size 3

    def add_vertex(self, v):
        self.vertices.add(v)

    def add_edge(self, a, b):
        self.vertices.add(a)
        self.vertices.add(b)
        self.edges.add(frozenset([a, b]))

    def add_triangle(self, a, b, c):
        f = frozenset([a, b, c])
        self.triangles.add(f)
        # ensure all sub-edges present
        for e in combinations(sorted(f), 2):
            self.edges.add(frozenset(e))
        for v in f:
            self.vertices.add(v)

    def betti(self):
        """Return (β₀, β₁, β₂)."""
        V = sorted(self.vertices)
        E = sorted([tuple(sorted(e)) for e in self.edges])
        F = sorted([tuple(sorted(f)) for f in self.triangles])

        nV, nE, nF = len(V), len(E), len(F)
        v_idx = {v: i for i, v in enumerate(V)}
        e_idx = {e: i for i, e in enumerate(E)}

        # ∂₁: nV × nE
        if nE > 0:
            d1 = np.zeros((nV, nE), dtype=np.int8)
            for j, (a, b) in enumerate(E):
                d1[v_idx[a], j] = 1
                d1[v_idx[b], j] = 1
            r1 = rank_f2(d1)
        else:
            r1 = 0

        # ∂₂: nE × nF
        if nF > 0 and nE > 0:
            d2 = np.zeros((nE, nF), dtype=np.int8)
            for k, (a, b, c) in enumerate(F):
                d2[e_idx[(a, b)], k] = 1
                d2[e_idx[(a, c)], k] = 1
                d2[e_idx[(b, c)], k] = 1
            r2 = rank_f2(d2)
        else:
            r2 = 0

        beta0 = nV - r1
        beta1 = nE - r1 - r2
        beta2 = nF - r2
        return beta0, beta1, beta2


# ── BCC base complex ───────────────────────────────────────────────────────────

def build_bcc():
    """
    Build BCC complex: vertices 0-8, 20 edges, 12 cone triangles.
    - 12 cube edges (pairs differing in exactly 1 bit)
    - 8 center edges (vertex 8 to each corner)
    - 12 triangles: {a, b, 8} for each cube edge {a, b}
    This gives beta=(1,0,0).
    """
    sc = SimplicialComplex()

    # Cube edges: pairs differing in exactly 1 bit
    cube_edges = []
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.append((a, b))

    # Add all vertices 0-8
    for v in range(9):
        sc.add_vertex(v)

    # Cube edges
    for (a, b) in cube_edges:
        sc.add_edge(a, b)

    # Center edges: vertex 8 connects to all 0-7
    for v in range(8):
        sc.add_edge(8, v)

    # Cone triangles: {a, b, 8} for each cube edge {a,b}
    for (a, b) in cube_edges:
        sc.add_triangle(a, b, 8)

    return sc, cube_edges


def copy_complex(sc):
    """Return a deep copy."""
    new = SimplicialComplex()
    new.vertices = set(sc.vertices)
    new.edges = set(sc.edges)
    new.triangles = set(sc.triangles)
    return new


# ── Add a new vertex with neighbor set S ──────────────────────────────────────

def add_vertex_to_complex(sc, new_v, neighbors):
    """
    Add new_v connected to each vertex in neighbors.
    For each pair {a,b} ⊆ neighbors that is already an edge, add triangle {new_v,a,b}.
    Returns a new complex (does not mutate sc).
    """
    nc = copy_complex(sc)
    nc.add_vertex(new_v)
    neighbor_list = list(neighbors)
    for s in neighbor_list:
        nc.add_edge(new_v, s)
    # Add triangles
    for a, b in combinations(sorted(neighbor_list), 2):
        if frozenset([a, b]) in sc.edges:
            nc.add_triangle(new_v, a, b)
    return nc


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("BCC Cascade — Simplicial Homology over F₂")
print("=" * 60)

# ── Build BCC ──────────────────────────────────────────────────────────────────
bcc, cube_edges = build_bcc()
b = bcc.betti()
print(f"\nBCC alone (vertices 0-8):  β₀={b[0]}, β₁={b[1]}, β₂={b[2]}")
print(f"  |V|={len(bcc.vertices)}, |E|={len(bcc.edges)}, |F|={len(bcc.triangles)}")


# ── Standard ordering: 9→{3,5,6}, 10→{1,2,4,9} ────────────────────────────────
print("\n── Standard ordering ──────────────────────────────────────────")

sc_v9 = add_vertex_to_complex(bcc, 9, [3, 5, 6])
b9 = sc_v9.betti()
print(f"After adding v9→{{3,5,6}}:  β₀={b9[0]}, β₁={b9[1]}, β₂={b9[2]}")

sc_v10 = add_vertex_to_complex(sc_v9, 10, [1, 2, 4, 9])
b10 = sc_v10.betti()
print(f"After adding v10→{{1,2,4,9}}: β₀={b10[0]}, β₁={b10[1]}, β₂={b10[2]}")

assert b10[1] == 5, f"Expected β₁=5, got {b10[1]}"
print("  ✓ β₁=5 verified")


# ── Exhaustive search for vertex 11 (standard ordering) ───────────────────────
print("\n── Exhaustive search: vertex 11 (standard ordering) ────────────")

existing_verts = sorted(sc_v10.vertices)  # 0..10
n_exist = len(existing_verts)
assert n_exist == 11

from collections import defaultdict

dist_standard = defaultdict(int)
best_standard = []   # (beta1, S)
max_b1_standard = -1

beta8_standard = []

total = 2**n_exist - 1
print(f"  Searching {total} subsets...")

for mask in range(1, 2**n_exist):
    S = [existing_verts[i] for i in range(n_exist) if (mask >> i) & 1]
    nc = add_vertex_to_complex(sc_v10, 11, S)
    b0, b1, b2 = nc.betti()
    dist_standard[b1] += 1
    if b1 == 8:
        beta8_standard.append((S, b0, b1, b2))
    if b1 > max_b1_standard:
        max_b1_standard = b1
        best_standard = [(S, b0, b1, b2)]
    elif b1 == max_b1_standard:
        best_standard.append((S, b0, b1, b2))

print(f"\n  Max β₁ achieved: {max_b1_standard}")
print(f"  β₁ distribution:")
for k in sorted(dist_standard):
    print(f"    β₁={k}: {dist_standard[k]} configs")

print(f"\n  Configs with β₁=8: {len(beta8_standard)}")
if beta8_standard:
    print("  All β₁=8 subsets S (vertex 11 neighbors):")
    for S, b0, b1, b2 in beta8_standard:
        print(f"    S={S}  →  β₀={b0}, β₁={b1}, β₂={b2}")
else:
    print(f"  Top configs (β₁={max_b1_standard}):")
    for S, b0, b1, b2 in best_standard[:5]:
        print(f"    S={S}  →  β₀={b0}, β₁={b1}, β₂={b2}")


# ── Config A: vertex 9→{1,2,4}, vertex 10→{3,5,6,9} ──────────────────────────
print("\n── Config A ordering: v9→{1,2,4}, v10→{3,5,6,9} ──────────────")

sc_a9 = add_vertex_to_complex(bcc, 9, [1, 2, 4])
ba9 = sc_a9.betti()
print(f"After adding v9→{{1,2,4}}:     β₀={ba9[0]}, β₁={ba9[1]}, β₂={ba9[2]}")

sc_a10 = add_vertex_to_complex(sc_a9, 10, [3, 5, 6, 9])
ba10 = sc_a10.betti()
print(f"After adding v10→{{3,5,6,9}}: β₀={ba10[0]}, β₁={ba10[1]}, β₂={ba10[2]}")

existing_verts_a = sorted(sc_a10.vertices)
assert len(existing_verts_a) == 11

dist_A = defaultdict(int)
max_b1_A = -1
beta8_A = []
best_A = []

print(f"  Searching {total} subsets for Config A...")

for mask in range(1, 2**11):
    S = [existing_verts_a[i] for i in range(11) if (mask >> i) & 1]
    nc = add_vertex_to_complex(sc_a10, 11, S)
    b0, b1, b2 = nc.betti()
    dist_A[b1] += 1
    if b1 == 8:
        beta8_A.append((S, b0, b1, b2))
    if b1 > max_b1_A:
        max_b1_A = b1
        best_A = [(S, b0, b1, b2)]
    elif b1 == max_b1_A:
        best_A.append((S, b0, b1, b2))

print(f"\n  Max β₁ achieved (Config A): {max_b1_A}")
print(f"  β₁ distribution (Config A):")
for k in sorted(dist_A):
    print(f"    β₁={k}: {dist_A[k]} configs")

print(f"\n  Configs with β₁=8 (Config A): {len(beta8_A)}")
if beta8_A:
    print("  All β₁=8 subsets (Config A):")
    for S, b0, b1, b2 in beta8_A:
        print(f"    S={S}  →  β₀={b0}, β₁={b1}, β₂={b2}")
else:
    print(f"  Top configs (β₁={max_b1_A}, Config A):")
    for S, b0, b1, b2 in best_A[:5]:
        print(f"    S={S}  →  β₀={b0}, β₁={b1}, β₂={b2}")

# ── Ordering comparison ────────────────────────────────────────────────────────
print("\n── Ordering comparison ─────────────────────────────────────────")
print(f"  Standard:  max β₁={max_b1_standard}, β₁=8 count={len(beta8_standard)}")
print(f"  Config A:  max β₁={max_b1_A},  β₁=8 count={len(beta8_A)}")
if dist_standard == dist_A:
    print("  Distributions IDENTICAL — ordering does not affect β₁ statistics.")
else:
    print("  Distributions DIFFER — ordering matters.")

print("\nDone.")
