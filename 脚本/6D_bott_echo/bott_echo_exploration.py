# -*- coding: utf-8 -*-
"""
Bott Echo Exploration: Simplicial Complex Homology over F2
Exploring cascade of complexes through dimension insertion
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations

# ============================================================
# CORE: F2 Homology computation
# ============================================================

def gf2_rank(matrix):
    """Compute rank of a matrix over GF(2) using Gaussian elimination."""
    if matrix.size == 0:
        return 0
    M = matrix.copy().astype(np.uint8)
    rows, cols = M.shape
    rank = 0
    pivot_row = 0
    for col in range(cols):
        found = -1
        for row in range(pivot_row, rows):
            if M[row, col] == 1:
                found = row
                break
        if found == -1:
            continue
        if pivot_row != found:
            temp = M[pivot_row].copy()
            M[pivot_row] = M[found]
            M[found] = temp
        for row in range(rows):
            if row != pivot_row and M[row, col] == 1:
                M[row] = (M[row] + M[pivot_row]) % 2
        rank += 1
        pivot_row += 1
    return rank


def compute_homology(vertices, edges, faces):
    """
    Compute H0, H1, H2 over F2.
    All inputs: lists. Edges = sorted 2-tuples. Faces = sorted 3-tuples.
    Returns (beta0, beta1, beta2).
    """
    n_v = len(vertices)
    n_e = len(edges)
    n_f = len(faces)

    v_idx = {v: i for i, v in enumerate(vertices)}
    e_idx = {e: i for i, e in enumerate(edges)}

    # d1: n_v x n_e
    d1 = np.zeros((n_v, n_e), dtype=np.uint8)
    for j, (a, b) in enumerate(edges):
        d1[v_idx[a], j] = 1
        d1[v_idx[b], j] = 1

    # d2: n_e x n_f
    d2 = np.zeros((n_e, n_f), dtype=np.uint8)
    for k, (a, b, c) in enumerate(faces):
        for e in [tuple(sorted([a,b])), tuple(sorted([a,c])), tuple(sorted([b,c]))]:
            if e in e_idx:
                d2[e_idx[e], k] = 1
            # Note: if edge missing from complex, it won't appear -- this would be
            # an "abstract" face not embedded in our complex. We want only embedded.

    rank_d1 = gf2_rank(d1)
    rank_d2 = gf2_rank(d2)

    beta0 = n_v - rank_d1
    beta1 = (n_e - rank_d1) - rank_d2
    beta2 = n_f - rank_d2

    return beta0, beta1, beta2


# ============================================================
# PART 1: Build and validate known complexes
# ============================================================

def build_fano():
    """Fano plane = P^2(F2): K7 + 7 faces"""
    vertices = list(range(7))
    edges = list(combinations(range(7), 2))  # K7, all 21 pairs
    fano_lines = [
        (0,1,3),(1,2,4),(2,3,5),(3,4,6),(4,5,0),(5,6,1),(6,0,2)
    ]
    faces = [tuple(sorted(t)) for t in fano_lines]
    return vertices, edges, faces


def coords3(i):
    """Cube vertex i as (x,y,z) with x=bit0, y=bit1, z=bit2."""
    return ((i>>0)&1, (i>>1)&1, (i>>2)&1)


def build_sc_cube():
    """
    Simple Cubic cube: 8 vertices, 12 edges.
    6 square faces, each triangulated by adding the diagonal that spans
    the two axis-directions. Each diagonal is added as an extra edge.
    This gives 12 cube edges + 6 diagonal edges = 18 edges total,
    and 12 triangular faces.
    """
    vertices = list(range(8))

    # Cube edges: pairs differing in exactly 1 coord
    cube_edges = []
    for i, j in combinations(range(8), 2):
        ci, cj = coords3(i), coords3(j)
        diff = sum(a != b for a, b in zip(ci, cj))
        if diff == 1:
            cube_edges.append(tuple(sorted([i,j])))
    cube_edges.sort()

    # 6 square faces with their 4 boundary edges and the chosen diagonal
    # For each face: 4 vertices forming a square
    # The two diagonals are the pairs with hamming=2. We pick one consistently.
    # For face with varying axes (a,b): diagonal goes from (0,0) corner to (1,1) corner
    cube_face_quads = [
        [0,2,4,6],  # x=0, varying y,z
        [1,3,5,7],  # x=1, varying y,z
        [0,1,4,5],  # y=0, varying x,z
        [2,3,6,7],  # y=1, varying x,z
        [0,1,2,3],  # z=0, varying x,y
        [4,5,6,7],  # z=1, varying x,y
    ]

    # For a quad [v0,v1,v2,v3] where the varying axes are a,b:
    # Sort vertices by (bit_a, bit_b): 00=low-low, 01=low-high, 10=high-low, 11=high-high
    # Then the square is:  00---10
    #                       |   |
    #                      01---11
    # The "00-11" diagonal (hamming=2 between 00 and 11) is one triangulation.
    # Triangles: {00,01,11} and {00,10,11}
    # This diagonal (from corner_00 to corner_11) must be added as an extra edge.

    def get_varying_axes(vlist):
        cs = [coords3(v) for v in vlist]
        return [a for a in range(3) if len(set(c[a] for c in cs)) > 1]

    def sort_quad(vlist):
        axes = get_varying_axes(vlist)
        def key(v):
            c = coords3(v)
            return (c[axes[0]], c[axes[1]])
        return sorted(vlist, key=key)  # order: 00, 01, 10, 11

    edges = list(cube_edges)
    faces = []

    for quad in cube_face_quads:
        sorted_q = sort_quad(quad)
        v00, v01, v10, v11 = sorted_q  # (a=0,b=0), (a=0,b=1), (a=1,b=0), (a=1,b=1)

        # Diagonal: v00 -- v11  (hamming=2, not a cube edge)
        diag = tuple(sorted([v00, v11]))
        if diag not in edges:
            edges.append(diag)

        # Two triangles using this diagonal
        faces.append(tuple(sorted([v00, v01, v11])))
        faces.append(tuple(sorted([v00, v10, v11])))

    edges = sorted(set(edges))
    faces = sorted(set(faces))
    return vertices, edges, faces


def build_bcc():
    """BCC = SC cube + center vertex 8 connected to all 8 corners."""
    sc_v, sc_e, sc_f = build_sc_cube()

    # Separate original cube edges from diagonals (for face construction)
    cube_edges_only = []
    for i, j in combinations(range(8), 2):
        ci, cj = coords3(i), coords3(j)
        diff = sum(a != b for a, b in zip(ci, cj))
        if diff == 1:
            cube_edges_only.append(tuple(sorted([i,j])))

    vertices = list(range(9))

    # Edges: SC edges (including diagonals) + 8 center-to-corner edges
    edges = list(sc_e)
    for i in range(8):
        edges.append(tuple(sorted([8, i])))
    edges = sorted(set(edges))

    # Faces: SC faces + for each original cube edge (a,b), triangle {8,a,b}
    faces = list(sc_f)
    for (a, b) in cube_edges_only:
        faces.append(tuple(sorted([8, a, b])))
    faces = sorted(set(faces))

    return vertices, edges, faces


print("=" * 60)
print("PART 1: Validation of Known Complexes")
print("=" * 60)

# Fano
fv, fe, ff = build_fano()
b0, b1, b2 = compute_homology(fv, fe, ff)
print(f"\nFano plane P^2(F2):")
print(f"  Vertices: {len(fv)}, Edges: {len(fe)}, Faces: {len(ff)}")
print(f"  b0={b0}, b1={b1}, b2={b2}")
print(f"  Expected: b0=1, b1=8, b2=0")
print(f"  {'PASS' if (b0,b1,b2)==(1,8,0) else 'FAIL'}")

# SC Cube
sv, se, sf = build_sc_cube()
b0, b1, b2 = compute_homology(sv, se, sf)
print(f"\nSC Cube (triangulated with diagonal edges):")
print(f"  Vertices: {len(sv)}, Edges: {len(se)}, Faces: {len(sf)}")
print(f"  b0={b0}, b1={b1}, b2={b2}")
print(f"  Expected: b0=1, b1=0, b2=1")
print(f"  {'PASS' if (b0,b1,b2)==(1,0,1) else 'FAIL'}")

# BCC
bv, be, bf = build_bcc()
b0, b1, b2 = compute_homology(bv, be, bf)
print(f"\nBCC (SC + body center):")
print(f"  Vertices: {len(bv)}, Edges: {len(be)}, Faces: {len(bf)}")
print(f"  b0={b0}, b1={b1}, b2={b2}")
print(f"  Expected: b0=1, b1=0, b2=6")
print(f"  {'PASS' if (b0,b1,b2)==(1,0,6) else 'FAIL'}")

# Debug if BCC fails
if not (b0,b1,b2)==(1,0,6):
    print(f"\n  [BCC debug] Euler: V-E+F = {len(bv)-len(be)+len(bf)} (should be 1 for b0=1,b1=0,b2=6 -> chi=1-0+6=7... wait)")
    # chi = b0 - b1 + b2 = 1 - 0 + 6 = 7
    print(f"  Expected chi = 1 - 0 + 6 = 7")
    print(f"  Actual V-E+F = {len(bv)}-{len(be)}+{len(bf)} = {len(bv)-len(be)+len(bf)}")


# ============================================================
# PART 2: Point insertion from BCC -- all 2^9 subsets
# ============================================================

print("\n" + "=" * 60)
print("PART 2: 10th Vertex Insertion from BCC")
print("=" * 60)

bv_base, be_base, bf_base = build_bcc()
be_set = set(be_base)

new_vertex = 9
nonzero_b1 = []
total_tested = 0

for mask in range(1, 512):
    S = [i for i in range(9) if (mask >> i) & 1]
    if len(S) < 2:
        continue
    total_tested += 1

    new_edges = [tuple(sorted([new_vertex, s])) for s in S]

    new_faces = []
    for a, b in combinations(S, 2):
        if tuple(sorted([a,b])) in be_set:
            new_faces.append(tuple(sorted([new_vertex, a, b])))

    vertices = bv_base + [new_vertex]
    edges = sorted(be_base + new_edges)
    faces = sorted(bf_base + new_faces)

    b0_r, b1_r, b2_r = compute_homology(vertices, edges, faces)

    if b1_r > 0:
        nonzero_b1.append((b1_r, b0_r, b2_r, S))

nonzero_b1.sort(reverse=True)

print(f"\nTotal subsets tested (|S|>=2): {total_tested}")
print(f"Configurations with b1>0: {len(nonzero_b1)}")

b1_counts = {}
for (b1, b0, b2, S) in nonzero_b1:
    b1_counts[b1] = b1_counts.get(b1, 0) + 1

if b1_counts:
    print(f"\nb1 distribution among b1>0 configs:")
    for val in sorted(b1_counts.keys(), reverse=True):
        print(f"  b1={val}: {b1_counts[val]} configurations")

    print(f"\nAll b1>0 configurations:")
    for idx, (b1, b0, b2, S) in enumerate(nonzero_b1):
        print(f"  [{idx+1:3d}] b0={b0}, b1={b1}, b2={b2} | S={S}")

    fano_echo = [(b1, b0, b2, S) for (b1, b0, b2, S) in nonzero_b1 if b1 == 8]
    if fano_echo:
        print(f"\n*** Fano Echo (b1=8) found! ***")
        for b1, b0, b2, S in fano_echo:
            print(f"  b0={b0}, b1={b1}, b2={b2} | S={S}")
    else:
        print(f"\n  (No Fano echo b1=8 found)")
else:
    print("  (All configurations yield b1=0)")


# ============================================================
# PART 3: 11th vertex from top b1 configs
# ============================================================

print("\n" + "=" * 60)
print("PART 3: 11th Vertex Insertion (from top Part 2 configs)")
print("=" * 60)

if nonzero_b1:
    top5 = nonzero_b1[:5]
    new_vertex_11 = 10

    for rank_idx, (b1_base, b0_base, b2_base, S_base) in enumerate(top5):
        print(f"\nConfig #{rank_idx+1}: b1={b1_base}, S={S_base}")

        new_edges_10 = [tuple(sorted([9, s])) for s in S_base]
        new_faces_10 = []
        for a, b in combinations(S_base, 2):
            if tuple(sorted([a,b])) in be_set:
                new_faces_10.append(tuple(sorted([9, a, b])))

        v10 = bv_base + [9]
        e10 = sorted(be_base + new_edges_10)
        f10 = sorted(bf_base + new_faces_10)
        e10_set = set(e10)

        results_11 = []
        tested_11 = 0
        for mask2 in range(1, 1024):
            S2 = [i for i in range(10) if (mask2 >> i) & 1]
            if len(S2) < 2:
                continue
            tested_11 += 1
            new_e11 = [tuple(sorted([new_vertex_11, s])) for s in S2]
            new_f11 = []
            for a, b in combinations(S2, 2):
                if tuple(sorted([a,b])) in e10_set:
                    new_f11.append(tuple(sorted([new_vertex_11, a, b])))

            v11 = v10 + [new_vertex_11]
            e11 = sorted(e10 + new_e11)
            f11 = sorted(f10 + new_f11)

            b0n, b1n, b2n = compute_homology(v11, e11, f11)
            if b1n > 0:
                results_11.append((b1n, b0n, b2n, S2))

        results_11.sort(reverse=True)
        r_counts = {}
        for (b1n,_,_,_) in results_11:
            r_counts[b1n] = r_counts.get(b1n, 0) + 1

        print(f"  11th vertex: {len(results_11)} configs with b1>0 out of {tested_11} tested")
        for val in sorted(r_counts.keys(), reverse=True)[:5]:
            print(f"    b1={val}: {r_counts[val]} configs")

        if results_11:
            print(f"  Top configs:")
            for b1n, b0n, b2n, S2 in results_11[:3]:
                print(f"    b0={b0n}, b1={b1n}, b2={b2n} | S={S2}")

        fano_11 = [(b1n,b0n,b2n,S2) for (b1n,b0n,b2n,S2) in results_11 if b1n==8]
        if fano_11:
            print(f"  *** Fano echo (b1=8) at 11-vertex level! ***")
            for b1n,b0n,b2n,S2 in fano_11[:3]:
                print(f"    b0={b0n}, b1={b1n}, b2={b2n} | S={S2}")
else:
    print("  No b1>0 configs found in Part 2 -- skipping Part 3.")


# ============================================================
# PART 4: 6D Hypercube Cross-section
# ============================================================

print("\n" + "=" * 60)
print("PART 4: 6D Hypercube Equatorial Cross-section")
print("=" * 60)

eq_verts = [v for v in range(64) if bin(v).count('1') == 3]
print(f"\nEquatorial vertices (exactly 3 ones in 6-bit string): {len(eq_verts)}")
assert len(eq_verts) == 20

def hamming(a, b):
    return bin(a ^ b).count('1')

# H=1 edges (will be empty for weight-3 vectors)
eq_edges_h1 = []
for i, j in combinations(eq_verts, 2):
    if hamming(i, j) == 1:
        eq_edges_h1.append(tuple(sorted([i, j])))

print(f"Edges with Hamming distance 1: {len(eq_edges_h1)}")
if len(eq_edges_h1) == 0:
    print("  (Weight-3 vectors cannot differ in exactly 1 bit -- confirmed zero)")

# H=2 edges: Johnson graph J(6,3)
eq_edges_h2 = []
for i, j in combinations(eq_verts, 2):
    if hamming(i, j) == 2:
        eq_edges_h2.append(tuple(sorted([i, j])))
print(f"\nJohnson graph J(6,3) -- edges with Hamming distance 2: {len(eq_edges_h2)}")
b0, b1, b2 = compute_homology(eq_verts, eq_edges_h2, [])
print(f"  J(6,3) graph only (no faces): b0={b0}, b1={b1}, b2={b2}")
print(f"  Expected b1: edges - vertices + 1 = {len(eq_edges_h2)} - {len(eq_verts)} + 1 = {len(eq_edges_h2)-len(eq_verts)+1}")

# Natural triangles within J(6,3)
eq_edge_h2_set = set(eq_edges_h2)
eq_triangles = []
for i, j, k in combinations(eq_verts, 3):
    if (tuple(sorted([i,j])) in eq_edge_h2_set and
        tuple(sorted([i,k])) in eq_edge_h2_set and
        tuple(sorted([j,k])) in eq_edge_h2_set):
        eq_triangles.append(tuple(sorted([i,j,k])))
print(f"\nJ(6,3) + natural triangles: {len(eq_triangles)} triangles")
b0, b1, b2 = compute_homology(eq_verts, eq_edges_h2, eq_triangles)
print(f"  b0={b0}, b1={b1}, b2={b2}")

# H=4 edges
eq_edges_h4 = []
for i, j in combinations(eq_verts, 2):
    if hamming(i, j) == 4:
        eq_edges_h4.append(tuple(sorted([i, j])))
print(f"\nEdges with Hamming distance 4: {len(eq_edges_h4)}")
b0, b1, b2 = compute_homology(eq_verts, eq_edges_h4, [])
print(f"  b0={b0}, b1={b1}, b2={b2}")

# H=6 edges
eq_edges_h6 = []
for i, j in combinations(eq_verts, 2):
    if hamming(i, j) == 6:
        eq_edges_h6.append(tuple(sorted([i, j])))
print(f"\nEdges with Hamming distance 6 (antipodal): {len(eq_edges_h6)}")

# K_20: all C(20,2)=190 edges, no faces
all_eq_edges = list(combinations(eq_verts, 2))
print(f"\nK_20 (all {len(all_eq_edges)} edges, no faces):")
b0, b1, b2 = compute_homology(eq_verts, all_eq_edges, [])
print(f"  b0={b0}, b1={b1}, b2={b2}")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
