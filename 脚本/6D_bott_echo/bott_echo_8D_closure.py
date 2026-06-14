"""
bott_echo_8D_closure.py
Explore vertex 12 insertion into BCC + 3 prior insertions (v9,v10,v11).
Exhaustive search over all 2^12 - 1 subsets S within {0..11}.
All print statements use ASCII only to avoid GBK encoding errors.
"""

import numpy as np
from itertools import combinations
from collections import defaultdict
import time


# --- F2 rank via Gaussian elimination ---

def rank_f2(mat):
    """Rank of a binary matrix over F2."""
    M = mat.copy() % 2
    rows, cols = M.shape
    pivot_row = 0
    for col in range(cols):
        found = -1
        for r in range(pivot_row, rows):
            if M[r, col]:
                found = r
                break
        if found == -1:
            continue
        if found != pivot_row:
            temp = M[pivot_row].copy()
            M[pivot_row] = M[found]
            M[found] = temp
        for r in range(rows):
            if r != pivot_row and M[r, col]:
                M[r] = (M[r] + M[pivot_row]) % 2
        pivot_row += 1
        if pivot_row == rows:
            break
    return pivot_row


def betti(vertices, edges, triangles):
    """
    Compute b0, b1, b2 over F2.
    vertices: list of ints
    edges: list of frozensets of 2 ints
    triangles: list of frozensets of 3 ints
    """
    V = len(vertices)
    E = len(edges)
    F = len(triangles)

    v_idx = {v: i for i, v in enumerate(vertices)}
    e_idx = {e: i for i, e in enumerate(edges)}

    # d1: E columns, V rows
    if E == 0:
        r1 = 0
    else:
        d1 = np.zeros((V, E), dtype=np.int8)
        for j, e in enumerate(edges):
            for v in e:
                d1[v_idx[v], j] = 1
        r1 = rank_f2(d1)

    # d2: F columns, E rows
    if F == 0:
        r2 = 0
    else:
        d2 = np.zeros((E, F), dtype=np.int8)
        for k, tri in enumerate(triangles):
            verts = sorted(tri)
            tri_edges = [frozenset({verts[0], verts[1]}),
                         frozenset({verts[0], verts[2]}),
                         frozenset({verts[1], verts[2]})]
            for te in tri_edges:
                if te in e_idx:
                    d2[e_idx[te], k] = 1
        r2 = rank_f2(d2)

    b0 = V - r1
    b1 = E - r1 - r2
    b2 = F - r2
    return b0, b1, b2


# --- Build 12-vertex complex ---

def build_base_complex(v11_neighbors):
    """
    Build 12-vertex complex (vertices 0-11).
    v11_neighbors: iterable of ints for vertex 11's connections.
    Returns (vertices, edges_frozenset_set, triangles_frozenset_set)
    """
    vertices = list(range(12))

    # BCC cube edges: pairs differing in exactly 1 bit
    cube_edges = set()
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count('1') == 1:
                cube_edges.add(frozenset({a, b}))
    assert len(cube_edges) == 12

    # BCC center edges
    center_edges = {frozenset({8, i}) for i in range(8)}

    # BCC cone triangles: {a, b, 8} for each cube edge {a,b}
    bcc_triangles = set()
    for e in cube_edges:
        a, b = tuple(e)
        bcc_triangles.add(frozenset({a, b, 8}))
    assert len(bcc_triangles) == 12

    all_edges = cube_edges | center_edges

    # v9 edges to {3,5,6}
    for n in [3, 5, 6]:
        all_edges.add(frozenset({9, n}))

    # v10 edges to {1,2,4,9}
    for n in [1, 2, 4, 9]:
        all_edges.add(frozenset({10, n}))

    # v11 edges
    for n in v11_neighbors:
        all_edges.add(frozenset({11, n}))

    return vertices, all_edges, bcc_triangles


def verify_base(v11_neighbors, label=""):
    vertices, edges, triangles = build_base_complex(v11_neighbors)
    b0, b1, b2 = betti(vertices, list(edges), list(triangles))
    print("  Base complex %s: V=%d, E=%d, F=%d" % (label, len(vertices), len(edges), len(triangles)))
    print("  b0=%d, b1=%d, b2=%d" % (b0, b1, b2))
    return vertices, edges, triangles


# --- Exhaustive search ---

def exhaustive_search(base_vertices, base_edges, base_triangles):
    """
    Try all non-empty S within {0..11} as neighbors for vertex 12.
    Returns list of (S_tuple, b0, b1, b2, n_new_edges, n_new_tris).
    """
    n = len(base_vertices)  # 12
    results = []
    dist = defaultdict(int)

    for mask in range(1, 1 << n):
        S = tuple(i for i in range(n) if mask & (1 << i))

        new_edges = {frozenset({12, s}) for s in S}
        all_edges = base_edges | new_edges

        new_tris = set()
        for a, b in combinations(S, 2):
            if frozenset({a, b}) in base_edges:
                new_tris.add(frozenset({12, a, b}))
        all_tris = base_triangles | new_tris

        all_verts = list(range(13))
        b0, b1, b2 = betti(all_verts, list(all_edges), list(all_tris))
        dist[b1] += 1
        results.append((S, b0, b1, b2, len(new_edges), len(new_tris)))

    return results, dist


# --- Reporting ---

def analyze_results(results, dist, label=""):
    print("")
    print("=" * 60)
    print("SEARCH RESULTS: %s" % label)
    print("Total configurations: %d" % len(results))
    print("")

    print("b1 distribution:")
    for k in sorted(dist):
        print("  b1=%d: %d configs" % (k, dist[k]))
    print("")

    min_b1 = min(dist.keys())
    print("Minimum b1 achieved: %d" % min_b1)
    print("")

    zero_configs = [(S, b0, b1, b2, ne, nt) for S, b0, b1, b2, ne, nt in results if b1 == 0]
    print("Configs with b1=0: %d" % len(zero_configs))
    for S, b0, b1, b2, ne, nt in zero_configs:
        print("  S=%s, b0=%d, b1=%d, b2=%d, new_edges=%d, new_tris=%d" % (list(S), b0, b1, b2, ne, nt))
    print("")

    low_configs = [(S, b0, b1, b2, ne, nt) for S, b0, b1, b2, ne, nt in results if b1 < 4]
    low_configs.sort(key=lambda x: (x[2], -x[4]))
    print("Configs with b1<4: %d" % len(low_configs))
    for S, b0, b1, b2, ne, nt in low_configs:
        print("  S=%s, b0=%d, b1=%d, b2=%d, new_edges=%d, new_tris=%d" % (list(S), b0, b1, b2, ne, nt))
    print("")

    min_configs = [(S, b0, b1, b2, ne, nt) for S, b0, b1, b2, ne, nt in results if b1 == min_b1]
    min_configs.sort(key=lambda x: -x[4])
    print("All configs with b1=%d (total %d, showing up to 20):" % (min_b1, len(min_configs)))
    for S, b0, b1, b2, ne, nt in min_configs[:20]:
        print("  S=%s, b0=%d, b1=%d, b2=%d, new_edges=%d, new_tris=%d" % (list(S), b0, b1, b2, ne, nt))

    return min_b1, zero_configs, low_configs


# --- Main ---

def main():
    print("=" * 60)
    print("PART 1: Verify base complexes")
    print("=" * 60)

    print("\n[A] v11 -> {0,3,5,6}")
    verts_A, edges_A, tris_A = verify_base([0, 3, 5, 6], "v11->{0,3,5,6}")

    print("\n[B] v11 -> {1,2,4,7}")
    verts_B, edges_B, tris_B = verify_base([1, 2, 4, 7], "v11->{1,2,4,7}")

    print("\n" + "=" * 60)
    print("PART 2 & 3: Exhaustive search for vertex 12")
    print("=" * 60)
    print("Searching 4095 subsets for each base complex...")
    print("(This may take ~30-60s)")

    t0 = time.time()
    results_A, dist_A = exhaustive_search(verts_A, edges_A, tris_A)
    t1 = time.time()
    print("Search A done in %.1fs" % (t1 - t0))

    results_B, dist_B = exhaustive_search(verts_B, edges_B, tris_B)
    t2 = time.time()
    print("Search B done in %.1fs" % (t2 - t1))

    min_A, zero_A, low_A = analyze_results(results_A, dist_A, "Base A: v11->{0,3,5,6}")
    min_B, zero_B, low_B = analyze_results(results_B, dist_B, "Base B: v11->{1,2,4,7}")

    print("\n" + "=" * 60)
    print("PART 4: Cross-base comparison")
    print("=" * 60)

    if zero_A:
        print("\nBase A has %d b1=0 configs." % len(zero_A))
        zero_A_sets = {S for S, *_ in zero_A}
        cross = [(S, b0, b1, b2, ne, nt) for S, b0, b1, b2, ne, nt in results_B
                 if S in zero_A_sets and b1 == 0]
        print("Of those S values, %d also achieve b1=0 in Base B:" % len(cross))
        for S, b0, b1, b2, ne, nt in cross:
            print("  S=%s, b0=%d, b1=%d, b2=%d" % (list(S), b0, b1, b2))
    else:
        print("\nBase A min b1=%d. No b1=0 achieved." % min_A)
        print("Base B min b1=%d." % min_B)
        best_A = {S for S, b0, b1, b2, ne, nt in results_A if b1 == min_A}
        best_B = {S for S, b0, b1, b2, ne, nt in results_B if b1 == min_B}
        overlap = best_A & best_B
        print("\nConfigs achieving min b1 in BOTH bases: %d" % len(overlap))
        if overlap:
            print("Examples (up to 5):")
            for S in list(overlap)[:5]:
                rA = next(r for r in results_A if r[0] == S)
                rB = next(r for r in results_B if r[0] == S)
                print("  S=%s" % list(S))
                print("    Base A: b0=%d, b1=%d, b2=%d, edges=%d, tris=%d" % (rA[1], rA[2], rA[3], rA[4], rA[5]))
                print("    Base B: b0=%d, b1=%d, b2=%d, edges=%d, tris=%d" % (rB[1], rB[2], rB[3], rB[4], rB[5]))

    print("\n" + "=" * 60)
    print("TOPOLOGICAL PROTECTION ANALYSIS")
    print("=" * 60)

    def euler(v, e, f):
        return v - e + f

    best_A_list = sorted([(S, b0, b1, b2, ne, nt) for S, b0, b1, b2, ne, nt in results_A
                          if b1 == min_A], key=lambda x: (-x[5], x[4]))
    if best_A_list:
        S, b0, b1, b2, ne, nt = best_A_list[0]
        V_tot = 13
        E_tot = len(edges_A) + ne
        F_tot = len(tris_A) + nt
        chi = euler(V_tot, E_tot, F_tot)
        print("\nBest config (Base A): S=%s" % list(S))
        print("  V=%d, E=%d, F=%d, chi=%d" % (V_tot, E_tot, F_tot, chi))
        print("  b0=%d, b1=%d, b2=%d" % (b0, b1, b2))
        print("  Check: chi = b0-b1+b2 = %d-%d+%d = %d" % (b0, b1, b2, b0 - b1 + b2))

    max_tris_A = max(nt for S, b0, b1, b2, ne, nt in results_A)
    print("\nMax triangles a single v12 insertion can add (Base A): %d" % max_tris_A)
    print("  (from %d existing edges among neighbors)" % len(edges_A))

    heavy_A = sorted(results_A, key=lambda x: -x[5])[:10]
    print("\nTop 10 by triangles added (Base A):")
    for S, b0, b1, b2, ne, nt in heavy_A:
        print("  S=%s, ne=%d, nt=%d, b1=%d" % (list(S), ne, nt, b1))

    print("\nDone.")


if __name__ == "__main__":
    main()
