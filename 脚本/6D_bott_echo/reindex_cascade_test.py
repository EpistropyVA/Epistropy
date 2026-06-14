# -*- coding: utf-8 -*-
"""
Reindex Cascade Test
====================
Hypothesis: After Period 1 closes beta_1 to 0, the residual structure
(with beta_2 as dominant) looks like the START of Period 1 under a
dimension shift: beta_2 -> "new beta_1", beta_3 -> "new beta_2".
The cascade operator only has 2 active modes at any time.

We test this by:
1. Tracking all Betti numbers at each cascade step
2. At Period 1 close (Step 4): record active pair (beta_2, beta_3)
3. At Period 2 close (Step 8): record active pair (beta_3, beta_4)
4. Comparing shifted pairs vs Period 1 absolute pairs
5. Computing relative homology H*(X_step8, X_step4) to isolate what P2 added
"""

import sys
import io
import numpy as np
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ─────────────────────────────────────────────
#  Core data structure
# ─────────────────────────────────────────────

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

    def copy(self):
        c = SimplicialComplex()
        c.vertices = list(self.vertices)
        c.edge_idx = dict(self.edge_idx)
        c.tri_idx = dict(self.tri_idx)
        c.tet_idx = dict(self.tet_idx)
        return c

    def simplex_counts(self):
        return (len(self.vertices), len(self.edge_idx),
                len(self.tri_idx), len(self.tet_idx))


# ─────────────────────────────────────────────
#  F_3 rank (Gaussian elimination mod 3)
#  Safe row-swap: use temp copy, no advanced indexing
# ─────────────────────────────────────────────

def rank_f3(mat):
    if mat.size == 0:
        return 0
    m = mat.copy() % 3
    rows, cols = m.shape
    pivot_row = 0
    for col in range(cols):
        # Find pivot
        col_slice = m[pivot_row:rows, col]
        nonzero = np.where(col_slice != 0)[0]
        if len(nonzero) == 0:
            continue
        found = nonzero[0] + pivot_row
        # Safe swap
        if pivot_row != found:
            temp = m[pivot_row].copy()
            m[pivot_row] = m[found].copy()
            m[found] = temp
        # Normalize pivot row so pivot_val = 1
        pivot_val = int(m[pivot_row, col])
        if pivot_val == 2:
            m[pivot_row] = (m[pivot_row] * 2) % 3
        # Eliminate column
        col_vals = m[:, col].copy()
        col_vals[pivot_row] = 0
        nonzero_rows = np.where(col_vals != 0)[0]
        if len(nonzero_rows) > 0:
            factors = col_vals[nonzero_rows, np.newaxis]
            m[nonzero_rows] = (m[nonzero_rows] - factors * m[pivot_row]) % 3
        pivot_row += 1
    return pivot_row


# ─────────────────────────────────────────────
#  Build boundary matrices mod 3
# ─────────────────────────────────────────────

def build_boundary_matrices(sc):
    """Return (d1, d2, d3) boundary matrices mod 3, plus sorted simplex lists."""
    v_list  = sorted(sc.vertices)
    edge_list = sorted([sorted(list(e)) for e in sc.edge_idx.keys()])
    tri_list  = sorted([sorted(list(t)) for t in sc.tri_idx.keys()])
    tet_list  = sorted([sorted(list(t)) for t in sc.tet_idx.keys()])

    m0, m1, m2, m3 = len(v_list), len(edge_list), len(tri_list), len(tet_list)
    v_map    = {v: i for i, v in enumerate(v_list)}
    edge_map = {frozenset(e): i for i, e in enumerate(edge_list)}
    tri_map  = {frozenset(t): i for i, t in enumerate(tri_list)}
    tet_map  = {frozenset(t): i for i, t in enumerate(tet_list)}

    # d1: (m0 x m1)
    d1 = np.zeros((m0, m1), dtype=int)
    for j, e in enumerate(edge_list):
        a, b = e
        d1[v_map[a], j] = 2  # -1 mod 3
        d1[v_map[b], j] = 1

    # d2: (m1 x m2)
    d2 = np.zeros((m1, m2), dtype=int)
    for j, t in enumerate(tri_list):
        a, b, c = t
        d2[edge_map[frozenset({b, c})], j] = 1
        d2[edge_map[frozenset({a, c})], j] = 2  # -1 mod 3
        d2[edge_map[frozenset({a, b})], j] = 1

    # d3: (m2 x m3)
    d3 = np.zeros((m2, m3), dtype=int)
    for j, t in enumerate(tet_list):
        a, b, c, d = t
        d3[tri_map[frozenset({b, c, d})], j] = 1
        d3[tri_map[frozenset({a, c, d})], j] = 2  # -1
        d3[tri_map[frozenset({a, b, d})], j] = 1
        d3[tri_map[frozenset({a, b, c})], j] = 2  # -1

    return (d1 % 3, d2 % 3, d3 % 3,
            v_list, edge_list, tri_list, tet_list,
            v_map, edge_map, tri_map, tet_map)


def betti_numbers(sc):
    """Compute (beta0, beta1, beta2, beta3) with F3 coefficients."""
    d1, d2, d3, *_ = build_boundary_matrices(sc)
    m0 = len(sc.vertices)
    m1 = len(sc.edge_idx)
    m2 = len(sc.tri_idx)
    m3 = len(sc.tet_idx)

    r1 = rank_f3(d1)
    r2 = rank_f3(d2)
    r3 = rank_f3(d3)

    b0 = m0 - r1
    b1 = m1 - r1 - r2
    b2 = m2 - r2 - r3
    b3 = m3 - r3

    return (b0, b1, b2, b3), (m0, m1, m2, m3)


# ─────────────────────────────────────────────
#  Relative homology H*(X, A) mod 3
#  X = full complex at step 8, A = subcomplex at step 4
#  Relative chain complex: generators = simplices in X \ A
#  Boundary map: boundary of sigma, zeroing faces that land in A
# ─────────────────────────────────────────────

def relative_betti(sc_X, sc_A):
    """
    Compute relative homology H_k(X, A; F3) for k=0,1,2,3.
    Uses the quotient chain complex C_k(X)/C_k(A).
    """
    # Get sorted simplex lists for X
    (d1_X, d2_X, d3_X,
     vX, eX, tX, tetX,
     vmap_X, emap_X, tmap_X, tetmap_X) = build_boundary_matrices(sc_X)

    # Sets for A simplices
    A_verts  = set(sc_A.vertices)
    A_edges  = sc_A.edge_idx   # frozenset -> idx
    A_tris   = sc_A.tri_idx
    A_tets   = sc_A.tet_idx

    # Relative generators: in X but not in A
    # k=0
    rel_v   = [v for v in vX  if v not in A_verts]
    # k=1
    rel_e   = [e for e in eX  if frozenset(e) not in A_edges]
    # k=2
    rel_t   = [t for t in tX  if frozenset(t) not in A_tris]
    # k=3
    rel_tet = [t for t in tetX if frozenset(t) not in A_tets]

    n0, n1, n2, n3 = len(rel_v), len(rel_e), len(rel_t), len(rel_tet)

    # Index maps for relative generators
    rv_map  = {v: i for i, v in enumerate(rel_v)}
    re_map  = {frozenset(e): i for i, e in enumerate(rel_e)}
    rt_map  = {frozenset(t): i for i, t in enumerate(rel_t)}
    rtet_map = {frozenset(t): i for i, t in enumerate(rel_tet)}

    # Relative d1: n0 x n1
    # For edge e = [a,b] in rel_e: boundary is +b - a
    # Face in A -> set to 0 (quotient)
    rd1 = np.zeros((n0, n1), dtype=int)
    for j, e in enumerate(rel_e):
        a, b = e
        # face a: coefficient -1 = 2 mod 3 if a is relative (not in A)
        if a in rv_map:
            rd1[rv_map[a], j] = 2  # -1 mod 3
        # face b: coefficient +1 if b is relative
        if b in rv_map:
            rd1[rv_map[b], j] = 1

    # Relative d2: n1 x n2
    rd2 = np.zeros((n1, n2), dtype=int)
    for j, t in enumerate(rel_t):
        a, b, c = t
        # Faces: {b,c} coeff +1, {a,c} coeff -1, {a,b} coeff +1
        fs = frozenset({b, c})
        if fs in re_map:
            rd2[re_map[fs], j] = 1
        fs = frozenset({a, c})
        if fs in re_map:
            rd2[re_map[fs], j] = 2  # -1
        fs = frozenset({a, b})
        if fs in re_map:
            rd2[re_map[fs], j] = 1

    # Relative d3: n2 x n3
    rd3 = np.zeros((n2, n3), dtype=int)
    for j, t in enumerate(rel_tet):
        a, b, c, d = t
        # Faces of tet with signs
        faces_signs = [
            (frozenset({b, c, d}),  1),
            (frozenset({a, c, d}),  2),  # -1
            (frozenset({a, b, d}),  1),
            (frozenset({a, b, c}),  2),  # -1
        ]
        for fs, sgn in faces_signs:
            if fs in rt_map:
                rd3[rt_map[fs], j] = sgn

    rd1 = rd1 % 3
    rd2 = rd2 % 3
    rd3 = rd3 % 3

    rr1 = rank_f3(rd1)
    rr2 = rank_f3(rd2)
    rr3 = rank_f3(rd3)

    rb0 = n0 - rr1
    rb1 = n1 - rr1 - rr2
    rb2 = n2 - rr2 - rr3
    rb3 = n3 - rr3

    return (rb0, rb1, rb2, rb3), (n0, n1, n2, n3)


# ─────────────────────────────────────────────
#  Build cascade step by step
#  Following period3_fast_cascade.py's build_base_step7 logic
#  but traced step by step for Period 1 (steps 1-4) and Period 2 (steps 5-8)
# ─────────────────────────────────────────────

def build_cascade_steps():
    """
    Returns list of (step_name, sc_snapshot) for each cascade milestone.
    Period 1: Steps 1-4 (closing beta_1 to 0)
    Period 2: Steps 5-8 (closing beta_2 to 0)
    """
    snapshots = []

    sc = SimplicialComplex()

    # ── BASE: cube (8 vertices, 12 edges) + apex v8 + cone triangles ──
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

    snapshots.append(("Step 0: Base (cube + apex, V=9)", sc.copy()))

    # ── PERIOD 1 STEPS ──
    # Step 1: cone v9 and v19 (g-pair)
    sc.insert_vertex_with_cone(9,  {3, 5, 6})
    sc.insert_vertex_with_cone(19, {4, 2, 1})
    snapshots.append(("Step 1: +v9,v19 (V=11)", sc.copy()))

    # Step 2: cone v10 and v20
    sc.insert_vertex_with_cone(10, {1, 2, 4, 9})
    sc.insert_vertex_with_cone(20, {6, 5, 3, 19})
    snapshots.append(("Step 2: +v10,v20 (V=13)", sc.copy()))

    # Step 3: cone v11 and v21
    sc.insert_vertex_with_cone(11, {0, 3, 5, 6})
    sc.insert_vertex_with_cone(21, {7, 4, 2, 1})
    snapshots.append(("Step 3: +v11,v21 (V=15)", sc.copy()))

    # Step 4: cone v12 and v22 — Period 1 close (beta_1 -> 0)
    sc.insert_vertex_with_cone(12, {0, 1, 2, 3, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(22, {7, 6, 5, 4, 3, 2, 1, 19, 20, 21})
    snapshots.append(("Step 4: +v12,v22 [P1 CLOSE] (V=17)", sc.copy()))

    # Save step4 complex for relative homology
    sc_step4 = sc.copy()

    # ── PERIOD 2 STEPS ──
    # Add cube face diagonals (needed for P2)
    cube_faces = [
        (0,1,3), (0,2,3),
        (4,5,7), (4,6,7),
        (0,2,6), (0,4,6),
        (1,3,7), (1,5,7),
        (2,3,7), (2,6,7),
        (0,1,5), (0,4,5)
    ]
    for a, b, c in cube_faces:
        sc.add_edge(a, b)
        sc.add_edge(b, c)
        sc.add_edge(a, c)
        sc.add_triangle(a, b, c)
    snapshots.append(("Step 4b: +cube faces (V=17, more simplices)", sc.copy()))

    # Step 5: cone v13 and v23
    sc.insert_vertex_with_cone(13, {0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11})
    sc.insert_vertex_with_cone(23, {0, 1, 2, 3, 4, 5, 6, 7, 19, 20, 21})
    snapshots.append(("Step 5: +v13,v23 (V=19)", sc.copy()))

    # Step 6: cone v14 and v24
    sc.insert_vertex_with_cone(14, {0, 1, 2, 4, 5, 6, 9, 10, 11})
    sc.insert_vertex_with_cone(24, {1, 2, 3, 5, 6, 7, 19, 20, 21})
    snapshots.append(("Step 6: +v14,v24 (V=21)", sc.copy()))

    # Step 7: cone v15 and v25
    sc.insert_vertex_with_cone(15, {0, 1, 2, 3, 4, 5, 6, 7, 9, 11})
    sc.insert_vertex_with_cone(25, {0, 1, 2, 3, 4, 5, 6, 7, 19, 21})
    snapshots.append(("Step 7: +v15,v25 (V=23)", sc.copy()))

    # Step 8: K-close v16 and v26 — Period 2 close (beta_2 -> 0)
    existing = sorted(list(sc.vertices))
    sc.insert_vertex_with_cone(16, existing)
    sc.insert_vertex_with_cone(26, existing)
    snapshots.append(("Step 8: +v16,v26 [P2/K CLOSE] (V=25)", sc.copy()))

    sc_step8 = sc.copy()

    return snapshots, sc_step4, sc_step8


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    print("=" * 72)
    print(" REINDEX CASCADE TEST — Shifted-Frame Hypothesis (F3 coefficients)")
    print("=" * 72)

    print("\nBuilding cascade steps...")
    snapshots, sc_step4, sc_step8 = build_cascade_steps()

    # ── STEP-BY-STEP BETTI TABLE ──
    print("\n" + "─" * 72)
    print(f"{'Step':<45} {'V':>4} {'E':>5} {'T':>5} {'Tet':>5}  "
          f"{'b0':>4} {'b1':>5} {'b2':>5} {'b3':>5}")
    print("─" * 72)

    step_betti = {}
    for name, sc in snapshots:
        betti, counts = betti_numbers(sc)
        b0, b1, b2, b3 = betti
        m0, m1, m2, m3 = counts
        print(f"  {name:<43} {m0:>4} {m1:>5} {m2:>5} {m3:>5}  "
              f"{b0:>4} {b1:>5} {b2:>5} {b3:>5}")
        step_betti[name] = betti

    # ── EXTRACT KEY SNAPSHOTS ──
    # Period 1 peak = Step 3 (beta_1 should be maximal)
    # Period 1 close = Step 4
    # Period 2 peak = Step 7 (beta_2 should be maximal)
    # Period 2 close = Step 8

    # Collect by tag
    betti_by_tag = {}
    for name, betti in step_betti.items():
        betti_by_tag[name] = betti

    p1_start  = step_betti.get("Step 0: Base (cube + apex, V=9)", (None,)*4)
    p1_step1  = step_betti.get("Step 1: +v9,v19 (V=11)", (None,)*4)
    p1_step2  = step_betti.get("Step 2: +v10,v20 (V=13)", (None,)*4)
    p1_peak   = step_betti.get("Step 3: +v11,v21 (V=15)", (None,)*4)
    p1_close  = step_betti.get("Step 4: +v12,v22 [P1 CLOSE] (V=17)", (None,)*4)
    p2_step5  = step_betti.get("Step 5: +v13,v23 (V=19)", (None,)*4)
    p2_step6  = step_betti.get("Step 6: +v14,v24 (V=21)", (None,)*4)
    p2_peak   = step_betti.get("Step 7: +v15,v25 (V=23)", (None,)*4)
    p2_close  = step_betti.get("Step 8: +v16,v26 [P2/K CLOSE] (V=25)", (None,)*4)

    # ── RELATIVE HOMOLOGY ──
    print("\n" + "─" * 72)
    print(" Computing relative homology H*(X_step8, X_step4; F3)...")
    print(" (Isolates what Period 2 added on top of Period 1 closed structure)")
    print("─" * 72)
    rel_betti, rel_counts = relative_betti(sc_step8, sc_step4)
    rb0, rb1, rb2, rb3 = rel_betti
    rn0, rn1, rn2, rn3 = rel_counts
    print(f"  Relative generators:  n0={rn0}, n1={rn1}, n2={rn2}, n3={rn3}")
    print(f"  Relative Betti H*(X,A): ({rb0}, {rb1}, {rb2}, {rb3})")

    # ── COMPARISON TABLE ──
    print("\n" + "=" * 72)
    print(" HYPOTHESIS COMPARISON: Shifted Frame")
    print("=" * 72)

    print("""
  HYPOTHESIS: After P1 closes beta_1 -> 0, the active pair shifts:
    Period 1: active modes = (beta_1, beta_2)
    Period 2: active modes = (beta_2, beta_3)  [same cascade, shifted +1 dim]
  If the cascade is self-similar under dimension shift, then the RATIO
  and structure of Period 2 should echo Period 1.
""")

    print(f"  Period 1 Betti at each step:")
    print(f"    Step 0 (start):  {p1_start}   <- base state")
    print(f"    Step 1:          {p1_step1}")
    print(f"    Step 2:          {p1_step2}")
    print(f"    Step 3 (peak):   {p1_peak}   <- beta_1 peak")
    print(f"    Step 4 (close):  {p1_close}   <- beta_1 -> 0, beta_2 residual")

    print(f"\n  Period 2 Betti at each step:")
    print(f"    Step 4b (start): (after cube faces added)")
    print(f"    Step 5:          {p2_step5}")
    print(f"    Step 6:          {p2_step6}")
    print(f"    Step 7 (peak):   {p2_peak}   <- beta_2 peak")
    print(f"    Step 8 (close):  {p2_close}   <- beta_2 -> 0, beta_3 residual")

    # ── SHIFTED FRAME ANALYSIS ──
    print("\n" + "─" * 72)
    print(" SHIFTED FRAME: Does P2 look like P1 with dim +1?")
    print("─" * 72)

    # P1 active pair: (beta_1, beta_2)
    # At each step in P1
    p1_pairs = {
        "start":  (p1_start[1],  p1_start[2]),
        "step1":  (p1_step1[1],  p1_step1[2]),
        "step2":  (p1_step2[1],  p1_step2[2]),
        "peak":   (p1_peak[1],   p1_peak[2]),
        "close":  (p1_close[1],  p1_close[2]),
    }
    # P2 active pair: (beta_2, beta_3) — shifted
    p2_pairs_shifted = {
        "step5":  (p2_step5[2],  p2_step5[3]),
        "step6":  (p2_step6[2],  p2_step6[3]),
        "peak":   (p2_peak[2],   p2_peak[3]),
        "close":  (p2_close[2],  p2_close[3]),
    }

    print(f"\n  P1 active pairs (beta_1, beta_2):")
    for k, v in p1_pairs.items():
        print(f"    {k:<8}: {v}")

    print(f"\n  P2 active pairs (beta_2, beta_3) [shifted frame]:")
    for k, v in p2_pairs_shifted.items():
        print(f"    {k:<8}: {v}")

    # Peak ratio comparison
    p1_b1_peak = p1_peak[1]
    p2_b2_peak = p2_peak[2]
    p1_b2_at_close = p1_close[2]   # residual at P1 close
    p2_b3_at_close = p2_close[3]   # residual at P2 close

    print(f"\n  Peak values:")
    print(f"    P1 beta_1 peak (Step 3):  {p1_b1_peak}")
    print(f"    P2 beta_2 peak (Step 7):  {p2_b2_peak}")
    if p1_b1_peak and p1_b1_peak > 0:
        ratio_peaks = p2_b2_peak / p1_b1_peak
        print(f"    Ratio (P2 peak / P1 peak): {ratio_peaks:.4f}  [{p2_b2_peak}/{p1_b1_peak}]")
    else:
        print(f"    Ratio: undefined (P1 peak = 0)")

    print(f"\n  Residuals at close:")
    print(f"    P1 close: beta_2 (residual) = {p1_b2_at_close}")
    print(f"    P2 close: beta_3 (residual) = {p2_b3_at_close}")

    print(f"\n  Initial conditions of each period (active pair at start):")
    print(f"    P1 start: (beta_1, beta_2) = {p1_pairs['start']}")
    print(f"    P2 start: (beta_2, beta_3) at P1 close = {(p1_close[2], p1_close[3])}")
    print(f"      <- P2 inherits beta_2={p1_close[2]} (NOT 0), so it starts SHIFTED")
    print(f"         from a non-zero base. Hypothesis predicts non-identical but")
    print(f"         structurally analogous cascade.")

    # ── VERDICT ──
    print("\n" + "=" * 72)
    print(" VERDICT")
    print("=" * 72)

    print(f"""
  The shifted-frame hypothesis makes two claims:

  CLAIM 1: The cascade has exactly 2 active modes at any time.
    P1 active: beta_1 (rising) + beta_2 (residual/preview)
    P2 active: beta_2 (rising) + beta_3 (residual/preview)
    -> Check: Is beta_3 = 0 throughout P1? Is beta_1 = 0 after P1 close?

    P1 close: beta = {p1_close}
    At P1 close: beta_1={p1_close[1]}, beta_3={p1_close[3]}
    {"CONFIRMED: beta_1=0 after P1 close" if p1_close[1] == 0 else f"PARTIAL: beta_1={p1_close[1]} after close"}
    {"beta_3=0 at P1 close (2 active modes)" if p1_close[3] == 0 else f"NOTE: beta_3={p1_close[3]} at P1 close (3+ modes)"}

  CLAIM 2: beta_3 is not independent — it's beta_1 in a shifted frame.
    P1 start: beta_1=0, period builds to {p1_b1_peak}, closes to 0, leaves beta_2={p1_b2_at_close}
    P2 start: beta_2={p1_close[2]} (inherited), builds to {p2_b2_peak}, closes to 0, leaves beta_3={p2_b3_at_close}

    Ratio of peak amplitudes: {p2_b2_peak}/{p1_b1_peak if p1_b1_peak else '?'} = {f"{ratio_peaks:.2f}" if p1_b1_peak else '?'}
    Ratio of initial conditions: {p1_close[2]}/{p1_pairs['start'][0] if p1_pairs['start'][0] else 0}

    If cascade is self-similar, expect: peak_ratio ~ initial_condition_ratio * some_amplification

  RELATIVE HOMOLOGY H*(X_step8, X_step4; F3):
    This is the "pure P2 contribution" stripped of the P1 skeleton.
    rel_beta = ({rb0}, {rb1}, {rb2}, {rb3})
    Compare to P1 absolute at close: {p1_close}
    {"STRUCTURAL ECHO: same shape up to shift!"
     if (rb1, rb2) == (p1_close[1], p1_close[2]) else
     f"SHAPE DIFFERENCE: rel ({rb0},{rb1},{rb2},{rb3}) vs P1 {p1_close}"}
""")

    print("=" * 72)


if __name__ == "__main__":
    main()
