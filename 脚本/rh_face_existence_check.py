"""
rh_face_existence_check.py — Do F=0 dimensions really lack faces, or was old data incomplete?

Investigates 7D (Bott echo of 3D), 5D (O_h orbit layer), and 6D (G2 zero-mode)
to determine whether claimed F=0 is structurally forced or merely omitted.

Pure Python, F2 arithmetic, no dependencies.
"""

import sys
from itertools import combinations

sys.stdout.reconfigure(encoding="utf-8")


# ─── F2 Gaussian elimination ───────────────────────────────────────────

def f2_rank(matrix, nrows, ncols):
    """Rank of an nrows x ncols matrix over F2 (bitwise)."""
    if nrows == 0 or ncols == 0:
        return 0
    rows = []
    for r in range(nrows):
        val = 0
        for c in range(ncols):
            if matrix[r][c]:
                val |= (1 << c)
        rows.append(val)

    rank = 0
    for col in range(ncols):
        mask = 1 << col
        pivot = None
        for r in range(rank, len(rows)):
            if rows[r] & mask:
                pivot = r
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and (rows[r] & mask):
                rows[r] ^= rows[rank]
        rank += 1
    return rank


def compute_beta1(vertices, edges, faces):
    """Compute beta_1 over F2 from explicit cell lists.
    vertices: list of vertex labels
    edges: list of (v1, v2) tuples
    faces: list of lists-of-edges (each face = list of edge indices)
    Returns (beta_1, rank_d1, rank_d2, details_dict)
    """
    V = len(vertices)
    E = len(edges)
    F = len(faces)

    # ∂_1: E x V matrix (each edge maps to its two boundary vertices)
    v_index = {v: i for i, v in enumerate(vertices)}
    d1 = [[0] * V for _ in range(E)]
    for ei, (a, b) in enumerate(edges):
        d1[ei][v_index[a]] = 1
        d1[ei][v_index[b]] = 1

    rank_d1 = f2_rank(d1, E, V)

    # ∂_2: F x E matrix (each face maps to its boundary edges)
    if F > 0:
        d2 = [[0] * E for _ in range(F)]
        for fi, face_edges in enumerate(faces):
            for ei in face_edges:
                d2[fi][ei] = 1
        rank_d2 = f2_rank(d2, F, E)
    else:
        rank_d2 = 0

    # beta_1 = dim(ker d1) - dim(im d2) = (E - rank_d1) - rank_d2
    ker_d1 = E - rank_d1
    beta1 = ker_d1 - rank_d2

    return beta1, rank_d1, rank_d2, {
        "V": V, "E": E, "F": F,
        "rank_d1": rank_d1, "rank_d2": rank_d2,
        "ker_d1": ker_d1, "chi": V - E + F
    }


def find_edge_index(edges, a, b):
    """Find index of edge (a,b) or (b,a) in edge list."""
    for i, (u, v) in enumerate(edges):
        if (u == a and v == b) or (u == b and v == a):
            return i
    return None


# ─── Part 1: 7D — Bott echo of 3D ─────────────────────────────────────

def part1_7d_bott_echo():
    print("=" * 72)
    print("PART 1: 7D — Bott echo of 3D cube graph")
    print("=" * 72)
    print()

    # Build the cube graph (same 1-skeleton as 3D SC)
    # Vertices: 8 corners of unit cube
    verts = list(range(8))
    # Edges: pairs differing in exactly one bit
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                edges.append((i, j))

    print(f"Cube graph: V={len(verts)}, E={len(edges)}")
    print(f"Edges: {edges}")
    print()

    # 6 square faces of the cube
    # Each face is defined by fixing one coordinate
    face_defs = [
        # Fix bit 0 = 0: vertices 0,2,4,6
        [0, 2, 6, 4],
        # Fix bit 0 = 1: vertices 1,3,5,7
        [1, 3, 7, 5],
        # Fix bit 1 = 0: vertices 0,1,4,5
        [0, 1, 5, 4],
        # Fix bit 1 = 1: vertices 2,3,6,7
        [2, 3, 7, 6],
        # Fix bit 2 = 0: vertices 0,1,2,3
        [0, 1, 3, 2],
        # Fix bit 2 = 1: vertices 4,5,6,7
        [4, 5, 7, 6],
    ]

    faces = []
    for fverts in face_defs:
        face_edges = []
        n = len(fverts)
        for k in range(n):
            a, b = fverts[k], fverts[(k + 1) % n]
            ei = find_edge_index(edges, a, b)
            assert ei is not None, f"Edge ({a},{b}) not found"
            face_edges.append(ei)
        faces.append(face_edges)

    # Version A: WITH 6 square faces (as in 3D SC)
    print("--- Version A: WITH 6 square faces (3D SC structure) ---")
    b1_a, _, _, info_a = compute_beta1(verts, edges, faces)
    print(f"  V={info_a['V']}, E={info_a['E']}, F={info_a['F']}")
    print(f"  rank(d1)={info_a['rank_d1']}, ker(d1)={info_a['ker_d1']}")
    print(f"  rank(d2)={info_a['rank_d2']}")
    print(f"  beta_1 = {info_a['ker_d1']} - {info_a['rank_d2']} = {b1_a}")
    print(f"  chi = {info_a['chi']}")
    print()

    # Version B: WITHOUT faces (as old data claims for 7D)
    print("--- Version B: WITHOUT faces (old 7D data: F=0) ---")
    b1_b, _, _, info_b = compute_beta1(verts, edges, [])
    print(f"  V={info_b['V']}, E={info_b['E']}, F={info_b['F']}")
    print(f"  rank(d1)={info_b['rank_d1']}, ker(d1)={info_b['ker_d1']}")
    print(f"  rank(d2)={info_b['rank_d2']}")
    print(f"  beta_1 = {info_b['ker_d1']} - {info_b['rank_d2']} = {b1_b}")
    print(f"  chi = {info_b['chi']}")
    print()

    # Analysis
    print("--- Bott Periodicity Analysis ---")
    print()
    print("  Bott periodicity: pi_n(O) ~ pi_{n+8}(O)")
    print("  pi_3(O) = Z,  pi_7(O) = Z  (same homotopy group)")
    print()
    print("  Key distinction:")
    print("    Bott periodicity is about STABLE HOMOTOPY GROUPS of the")
    print("    classifying space, not about specific CW decompositions.")
    print("    The same homotopy type admits many cell structures.")
    print()
    print("    In the d-cascade, cell structure is determined by lattice")
    print("    geometry. The question is: does the 7D lattice geometry")
    print("    reproduce the cube's 2-cells, or only its 1-skeleton?")
    print()
    print(f"  With faces:    beta_1 = {b1_a} (sealed, same as 3D)")
    print(f"  Without faces: beta_1 = {b1_b} (5 open holes)")
    print()
    print("  If Bott echo preserves CW structure -> F=6, beta_1=0 (SUSPICIOUS)")
    print("  If Bott echo preserves only 1-skeleton -> F=0, beta_1=5 (needs proof)")
    print()
    print("  VERDICT: F=0 is SUSPICIOUS.")
    print("    The cube graph inherently supports 6 square faces.")
    print("    Claiming F=0 requires showing that 7D lattice geometry")
    print("    specifically forbids filling these squares.")
    print("    Bott periodicity (same homotopy type) suggests faces SHOULD exist.")
    print()


# ─── Part 2: 5D — O_h orbit layer ─────────────────────────────────────

def part2_5d_orbit():
    print("=" * 72)
    print("PART 2: 5D — O_h orbit layer (V=22, E=22)")
    print("=" * 72)
    print()

    # ── Structural analysis of V=E=22 ──
    print("--- Graph-theoretic analysis of V=22, E=22 ---")
    print()

    V, E = 22, 22
    # Connected graph: components c, then E = V - c + beta_1
    # If connected (c=1): beta_1 = E - V + 1 = 22 - 22 + 1 = 1
    print(f"  For connected graph: beta_1 = E - V + 1 = {E} - {V} + 1 = {E - V + 1}")
    print()

    print("  A connected graph with V=E has exactly ONE independent cycle.")
    print("  Structure: a spanning tree (21 edges) + 1 extra edge creating 1 cycle.")
    print()

    print("--- Can faces exist? ---")
    print()
    print("  The single cycle has some length L (3 <= L <= 22).")
    print("  The remaining 22 - L edges are tree edges (bridges).")
    print()

    # Demonstrate with concrete examples
    print("  Case analysis by cycle length:")
    print()

    # Build a generic graph: cycle of length L + pendant tree edges
    for L in [3, 4, 5, 7, 10, 22]:
        # Cycle on vertices 0..L-1
        cycle_edges = [(i, (i + 1) % L) for i in range(L)]
        # Tree edges: vertices L..21 attached to vertex 0 (for simplicity)
        tree_edges = [(0, i) for i in range(L, 22)]
        all_edges = cycle_edges + tree_edges

        assert len(all_edges) == 22

        verts_22 = list(range(22))

        # Count triangles: need 3 mutual edges
        triangle_count = 0
        edge_set = set()
        for a, b in all_edges:
            edge_set.add((min(a, b), max(a, b)))

        for a, b, c in combinations(range(22), 3):
            ab = (min(a, b), max(a, b))
            ac = (min(a, c), max(a, c))
            bc = (min(b, c), max(b, c))
            if ab in edge_set and ac in edge_set and bc in edge_set:
                triangle_count += 1

        # The cycle itself as a face
        cycle_face_edges = list(range(L))  # first L edges form the cycle

        # Compute beta_1 without any face
        b1_no_face, _, _, info0 = compute_beta1(verts_22, all_edges, [])

        # Compute beta_1 with the cycle as a single face
        b1_with_cycle, _, _, info1 = compute_beta1(verts_22, all_edges, [cycle_face_edges])

        print(f"  L={L:2d}: triangles={triangle_count}, "
              f"beta_1(F=0)={b1_no_face}, "
              f"beta_1(cycle-face)={b1_with_cycle}")

    print()
    print("  KEY INSIGHT: regardless of cycle length L, adding the cycle itself")
    print("  as a single 2-cell always kills beta_1: 1 -> 0.")
    print()
    print("  For L=3: one triangle exists and can serve as a face.")
    print("  For L>=4: no triangles, but the L-gon cycle is still a valid 2-cell.")
    print("  A 2-cell does NOT have to be a triangle. It can be any polygon.")
    print()

    # ── What are the 22 vertices? ──
    print("--- What are the 22 vertices? ---")
    print()
    print("  From cascade description:")
    print("    4D BCC: 9 vertices (8 corners + 1 center)")
    print("    5D adds 13 new vertices for total 22")
    print("    '22 orbit nodes; 13 new from 4D->5D coupling lift'")
    print()
    print("  Possible sources of 13 new vertices:")
    print("    - 12 edge midpoints of cube + 1 additional = 13")
    print("    - O_h orbits: |O_h| = 48, orbit sizes divide 48")
    print("      Possible orbit sizes: 1, 2, 3, 4, 6, 8, 12, 16, 24, 48")
    print("      9 + 13 = 22 could be multiple orbits summing to 22")
    print()

    # ── O_h conjugacy classes and orbit structure ──
    print("  O_h conjugacy classes (10 total):")
    print("    E, 8C3, 6C2, 6C4, 3C2(=C4^2), i, 8S6, 6sigma_d, 6S4, 3sigma_h")
    print()
    print("  O_h orbit sizes on BCC-related points:")
    print("    Corners (8): orbit of size 8 (stabilizer Z3 x Z2, order 6)")
    print("    Center (1): orbit of size 1 (stabilizer O_h, order 48)")
    print("    Edge midpoints (12): orbit of size 12 (stabilizer Z2 x Z2, order 4)")
    print("    Face centers (6): orbit of size 6 (stabilizer D4, order 8)")
    print("    Total: 1 + 6 + 8 + 12 = 27")
    print()
    print("  To get 22: could be 1 + 8 + 12 + 1(?) = 22")
    print("  or some subset selection. The exact identification needs")
    print("  cascade-specific knowledge.")
    print()

    print("--- Conclusion for 5D ---")
    print()
    print("  With V=E=22 (connected), beta_1 = 1.")
    print("  Exactly one independent cycle exists.")
    print("  This cycle (of whatever length) can always be filled by")
    print("  a single 2-cell, giving F=1 and beta_1=0.")
    print()
    print("  F=0 means: the old data claims this one cycle is UNFILLED.")
    print("  This is a meaningful topological statement (the hole is real),")
    print("  but it is NOT forced by the combinatorics.")
    print("  The choice to leave it unfilled needs justification from")
    print("  the lattice/coupling geometry.")
    print()
    print("  VERDICT: F=0 is SUSPICIOUS.")
    print("    Nothing prevents adding a face. The claim F=0 means")
    print("    choosing not to fill the unique cycle, which requires")
    print("    geometric justification, not just combinatorial default.")
    print()


# ─── Part 3: 6D — G2 zero-mode ────────────────────────────────────────

def part3_6d_g2():
    print("=" * 72)
    print("PART 3: 6D — G2 zero-mode (V=7, E=7, F=7)")
    print("=" * 72)
    print()

    V, E, F = 7, 7, 7

    print("--- Basic numerics ---")
    print()
    print(f"  V={V}, E={E}, F={F}")
    print(f"  chi = V - E + F = {V - E + F}")
    print(f"  If connected: beta_1(F=0) = E - V + 1 = {E - V + 1}")
    print()

    # ── Can V=E=F=7 be consistent? ──
    print("--- Consistency check: can a complex with V=E=F=7 exist? ---")
    print()
    print("  Graph with 7 vertices, 7 edges (connected):")
    print("    Spanning tree needs 6 edges. 1 extra edge -> 1 cycle.")
    print("    beta_1(graph) = 1")
    print()

    # Scenario A: C_7 (7-cycle, all vertices degree 2)
    print("  Scenario A: graph = C_7 (7-cycle/heptagon)")
    print("    Every vertex has degree 2. No tree branches.")
    print("    Triangles on C_7 edges: edge (i, i+1 mod 7)")
    print("    For triangle {a,b,c}: need (a,b), (b,c), (a,c) all in C_7")
    print("    Adjacent edges share a vertex, but the diagonal is NOT in C_7")
    print("    -> 0 triangles possible with only C_7 edges")
    print()

    # Verify: build C_7 and count triangles
    c7_edges = [(i, (i + 1) % 7) for i in range(7)]
    c7_set = set((min(a, b), max(a, b)) for a, b in c7_edges)
    c7_tris = 0
    for a, b, c in combinations(range(7), 3):
        if ((min(a, b), max(a, b)) in c7_set and
                (min(a, c), max(a, c)) in c7_set and
                (min(b, c), max(b, c)) in c7_set):
            c7_tris += 1
    print(f"    Verified: triangles in C_7 = {c7_tris}")
    print()

    print("    With 0 triangles and only 7 edges forming a single 7-cycle,")
    print("    the ONLY possible 2-cell is the 7-gon itself (1 face, not 7).")
    print("    F=7 triangular faces on C_7 is IMPOSSIBLE.")
    print()

    # Scenario B: different graph topology (tree + 1 cycle)
    print("  Scenario B: other graph topologies with V=7, E=7")
    print()
    print("    All connected graphs with 7 vertices and 7 edges have")
    print("    exactly 1 cycle. Let's enumerate the triangle count for")
    print("    various cycle lengths:")
    print()

    # Try several topologies
    topologies = {
        "C_3 + 4 pendants": {
            "edges": [(0, 1), (1, 2), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)],
            "cycle_len": 3,
        },
        "C_4 + 3 pendants": {
            "edges": [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (1, 5), (2, 6)],
            "cycle_len": 4,
        },
        "C_5 + 2 pendants": {
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (0, 5), (1, 6)],
            "cycle_len": 5,
        },
        "C_6 + 1 pendant": {
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 6)],
            "cycle_len": 6,
        },
        "C_7 (heptagon)": {
            "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)],
            "cycle_len": 7,
        },
    }

    for name, topo in topologies.items():
        tedges = topo["edges"]
        eset = set((min(a, b), max(a, b)) for a, b in tedges)
        tris = 0
        for a, b, c in combinations(range(7), 3):
            if ((min(a, b), max(a, b)) in eset and
                    (min(a, c), max(a, c)) in eset and
                    (min(b, c), max(b, c)) in eset):
                tris += 1

        verts7 = list(range(7))
        b1_0, _, _, _ = compute_beta1(verts7, tedges, [])

        # If triangle exists, add it as a face
        if tris > 0:
            # Find the triangle
            for a, b, c in combinations(range(7), 3):
                ab = (min(a, b), max(a, b))
                ac = (min(a, c), max(a, c))
                bc = (min(b, c), max(b, c))
                if ab in eset and ac in eset and bc in eset:
                    ei_ab = next(i for i, e in enumerate(tedges)
                                 if (min(e[0], e[1]), max(e[0], e[1])) == ab)
                    ei_ac = next(i for i, e in enumerate(tedges)
                                 if (min(e[0], e[1]), max(e[0], e[1])) == ac)
                    ei_bc = next(i for i, e in enumerate(tedges)
                                 if (min(e[0], e[1]), max(e[0], e[1])) == bc)
                    tri_face = [ei_ab, ei_ac, ei_bc]
                    b1_1, _, _, _ = compute_beta1(verts7, tedges, [tri_face])
                    break
        else:
            # Use the cycle as face
            cycle_len = topo["cycle_len"]
            cycle_face = list(range(cycle_len))
            b1_1, _, _, _ = compute_beta1(verts7, tedges, [cycle_face])

        print(f"    {name:25s}: triangles={tris}, "
              f"beta_1(F=0)={b1_0}, beta_1(+1 face)={b1_1}")

    print()
    print("    In ALL cases: exactly 1 face (triangle or polygon) kills beta_1.")
    print("    But F=7 faces with only 7 edges is problematic:")
    print()

    # ── Can 7 faces exist on 7 edges? ──
    print("  --- Can F=7 faces coexist with E=7? ---")
    print()
    print("    Each face needs >= 3 boundary edges.")
    print("    Total edge-face incidences >= 3 * 7 = 21.")
    print("    With 7 edges, average edge participates in >= 3 faces.")
    print()
    print("    For triangular faces: each uses 3 edges from {0..6}.")
    print("    Maximum triangles on 7 edges (no multi-edges):")

    # Compute: what is the maximum number of triangles on 7 edges?
    # This is equivalent to: given a graph G with 7 edges, max triangles
    # A triangle uses 3 edges; edges can be shared between triangles.
    # The maximum is achieved by K_4 (complete graph on 4 vertices):
    #   K_4 has 6 edges and 4 triangles
    #   Adding 1 more edge (to a 5th vertex) can create at most 1 more triangle
    #   if the new vertex connects to 2 vertices of K_4 that are adjacent
    #   -> at most 5 triangles on 7 edges

    # Actually let's just compute for small cases
    # K_4 = 6 edges, 4 triangles. Add edge (4, 0): no new triangle unless
    # we add (4,1) too. But that's 8 edges. With exactly 7:
    # K_4 + one pendant = 7 edges, 4 triangles
    # K_4 - 1 edge + 2 edges to new vertex = ...

    # Let's enumerate: take 7 edges on up to 7 vertices, maximize triangles
    best_tris = 0
    best_config = None
    # Try: K4 (6 edges, 4 tri) + 1 pendant edge
    k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 4)]
    k4_set = set((min(a, b), max(a, b)) for a, b in k4_edges)
    k4_tris = 0
    for a, b, c in combinations(range(5), 3):
        if ((min(a, b), max(a, b)) in k4_set and
                (min(a, c), max(a, c)) in k4_set and
                (min(b, c), max(b, c)) in k4_set):
            k4_tris += 1
    print(f"      K_4 + pendant: 7 edges, {k4_tris} triangles")

    # K4 - 1 edge + 2 edges to vertex 4 connecting to both ends of removed edge
    # Remove (2,3), add (2,4), (3,4): this creates triangle 2-3-4? No, (2,3) removed
    # Add (0,4), (1,4): creates triangles 0-1-4
    config2_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (0, 4), (1, 4)]
    c2_set = set((min(a, b), max(a, b)) for a, b in config2_edges)
    c2_tris = 0
    for a, b, c in combinations(range(5), 3):
        if ((min(a, b), max(a, b)) in c2_set and
                (min(a, c), max(a, c)) in c2_set and
                (min(b, c), max(b, c)) in c2_set):
            c2_tris += 1
    print(f"      K_4-1+2 (fan): 7 edges, {c2_tris} triangles")

    # K5 has 10 edges. Can't do. Try other dense subgraphs.
    # Wheel W_4: center + C_4 = 4+4 = 8 edges. Too many.
    # Best seems to be K_4 + pendant = 4 triangles.

    print()
    print(f"    Maximum triangles on 7 edges: ~{max(k4_tris, c2_tris)} "
          f"(far short of 7)")
    print()
    print("    Even allowing non-triangular faces (squares, pentagons, etc.),")
    print("    7 faces on 7 edges is only possible if faces share edges heavily.")
    print("    With only 1 independent cycle, rank(d2) <= 1 regardless of F.")
    print()

    # Demonstrate: even with multiple faces, rank(d2) is limited
    print("  --- rank(d2) test with multiple faces ---")
    print()
    # Use K_4+pendant, add all 4 triangles as faces
    k4_verts = list(range(5))
    # Find triangle face edge indices
    k4_faces = []
    for a, b, c in combinations(range(5), 3):
        ab = (min(a, b), max(a, b))
        ac = (min(a, c), max(a, c))
        bc = (min(b, c), max(b, c))
        if ab in k4_set and ac in k4_set and bc in k4_set:
            ei_ab = next(i for i, e in enumerate(k4_edges)
                         if (min(e[0], e[1]), max(e[0], e[1])) == ab)
            ei_ac = next(i for i, e in enumerate(k4_edges)
                         if (min(e[0], e[1]), max(e[0], e[1])) == ac)
            ei_bc = next(i for i, e in enumerate(k4_edges)
                         if (min(e[0], e[1]), max(e[0], e[1])) == bc)
            k4_faces.append([ei_ab, ei_ac, ei_bc])

    b1_k4_0, _, _, info_k4_0 = compute_beta1(k4_verts, k4_edges, [])
    b1_k4_f, _, _, info_k4_f = compute_beta1(k4_verts, k4_edges, k4_faces)
    print(f"    K_4+pendant, F=0: beta_1={b1_k4_0}, rank(d1)={info_k4_0['rank_d1']}")
    print(f"    K_4+pendant, F={len(k4_faces)} triangles: beta_1={b1_k4_f}, "
          f"rank(d2)={info_k4_f['rank_d2']}")
    print()
    print(f"    Even with {len(k4_faces)} faces, rank(d2)={info_k4_f['rank_d2']}.")
    print("    This is because d2 maps into ker(d1), which has dimension")
    print(f"    ker(d1) = E - rank(d1) = {info_k4_f['E']} - {info_k4_f['rank_d1']}"
          f" = {info_k4_f['ker_d1']}.")
    print("    Multiple faces that bound the same cycle are F2-dependent.")
    print()

    # ── G2 / octonion connection ──
    print("  --- G2 / octonion connection ---")
    print()
    print("    The label 'G2 zero-mode' and number 7 suggests octonions.")
    print("    7 imaginary octonion units: e1, e2, ..., e7")
    print("    Octonion multiplication defines the FANO PLANE:")
    print("      7 points, 7 lines, each line through 3 points")
    print()
    print("    Fano plane as simplicial complex:")
    print("      V = 7 (points)")
    print("      E = 21 (all pairs among 7 points)")
    print("      But as a LINE STRUCTURE: 7 lines, each a 3-element set")
    print()
    print("    If '7 edges' means 7 LINES of the Fano plane:")
    print("      Each 'edge' is actually a TRIANGLE (3-element line)")
    print("      This is a hypergraph, not a simple graph")
    print("      The old V=7, E=7 data might be encoding a hypergraph!")
    print()
    print("    Fano plane combinatorics:")
    print("      7 points, 7 lines")
    print("      Each point on 3 lines, each line through 3 points")
    print("      This is a (7,3,1)-design (Steiner triple system S(2,3,7))")
    print()

    # Build Fano plane and compute its homology as a simplicial complex
    # Fano lines (standard labeling):
    fano_lines = [
        (0, 1, 3), (1, 2, 4), (2, 3, 5),
        (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)
    ]

    # As a simplicial complex: vertices = 7, edges = all edges of all triangles
    fano_edge_set = set()
    for line in fano_lines:
        for a, b in combinations(line, 2):
            fano_edge_set.add((min(a, b), max(a, b)))
    fano_edges = sorted(fano_edge_set)
    fano_verts = list(range(7))

    print(f"    Fano as simplicial complex: V={len(fano_verts)}, E={len(fano_edges)}")
    print(f"    (E=21 = C(7,2), complete graph K_7)")
    print()

    # Build face list for Fano triangles
    fano_faces = []
    for line in fano_lines:
        face_edges = []
        for a, b in combinations(line, 2):
            ei = next(i for i, e in enumerate(fano_edges)
                      if e == (min(a, b), max(a, b)))
            face_edges.append(ei)
        fano_faces.append(face_edges)

    b1_fano_0, _, _, info_f0 = compute_beta1(fano_verts, fano_edges, [])
    b1_fano_f, _, _, info_ff = compute_beta1(fano_verts, fano_edges, fano_faces)

    print(f"    Fano, F=0:  V=7, E=21, beta_1={b1_fano_0}")
    print(f"    Fano, F=7:  V=7, E=21, F=7, beta_1={b1_fano_f}, "
          f"rank(d2)={info_ff['rank_d2']}")
    print(f"    chi = {info_ff['chi']}")
    print()

    print("    CRITICAL FINDING:")
    print("    If 6D is the Fano plane (7 triangular faces from 7 octonion lines),")
    print("    then E=21 (not 7). The old data E=7 would be wrong -- it counted")
    print("    LINES (hyperedges) not simple edges.")
    print()
    print("    Corrected: V=7, E=21, F=7")
    print(f"    beta_1 = {b1_fano_f}")
    print(f"    beta_0 = 1 (connected)")
    print(f"    chi = 7 - 21 + 7 = {7 - 21 + 7}")
    print()

    # Alternative: V=7, E=7 as a literal graph (not Fano)
    print("    Alternative: if E=7 is literal (simple graph, 7 edges):")
    print("    Then 7 triangular faces CANNOT exist (max ~4 triangles on 7 edges).")
    print("    And F=7 non-triangular faces are topologically redundant")
    print("    (rank(d2) <= 1 for any connected graph with 1 cycle).")
    print()
    print("  VERDICT: V=7, E=7, F=7 is INCONSISTENT as a simple CW complex.")
    print("    Most likely interpretation: the 6D data encodes the Fano plane")
    print("    with V=7, E(hyper)=7, and the actual edge count is E=21.")
    print("    The corrected data V=7, E=21, F=7 is consistent.")
    print()


# ─── Part 4: Summary Decision Table ───────────────────────────────────

def part4_summary():
    print("=" * 72)
    print("PART 4: Summary Decision Table")
    print("=" * 72)
    print()

    # Recompute key numbers
    # 3D reference
    verts3 = list(range(8))
    edges3 = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                edges3.append((i, j))
    b1_3d_sealed, _, _, _ = compute_beta1(verts3, edges3,
                                           _cube_faces(edges3))
    b1_3d_open, _, _, _ = compute_beta1(verts3, edges3, [])

    # 5D
    # Generic connected graph V=E=22
    b1_5d = 1  # E - V + 1 = 1

    # 6D (Fano interpretation)
    fano_edge_set = set()
    fano_lines = [
        (0, 1, 3), (1, 2, 4), (2, 3, 5),
        (3, 4, 6), (4, 5, 0), (5, 6, 1), (6, 0, 2)
    ]
    for line in fano_lines:
        for a, b in combinations(line, 2):
            fano_edge_set.add((min(a, b), max(a, b)))
    fano_edges = sorted(fano_edge_set)
    fano_faces = []
    for line in fano_lines:
        face_edges = []
        for a, b in combinations(line, 2):
            ei = next(i for i, e in enumerate(fano_edges)
                      if e == (min(a, b), max(a, b)))
            face_edges.append(ei)
        fano_faces.append(face_edges)
    b1_6d_fano, _, _, _ = compute_beta1(list(range(7)), fano_edges, fano_faces)

    # 7D
    b1_7d_sealed = 0  # same as 3D with faces
    b1_7d_open = 5    # same as 3D without faces

    # 8D (already proven impossible)

    print("  +-----+---------+-----------+-----------------+----------------+")
    print("  | Dim | Old F   | b1(F=0)   | b1(with faces)  | Verdict        |")
    print("  +-----+---------+-----------+-----------------+----------------+")

    rows = [
        ("3D",  "6",  f"{b1_3d_open}",  f"{b1_3d_sealed}",
         "REFERENCE (faces exist, sealed)"),
        ("5D",  "0",  "1",  "0 (add 1 face)",
         "SUSPICIOUS: face can exist"),
        ("6D",  "7*", "15 (E=21)",  f"{b1_6d_fano} (Fano, E=21)",
         "WRONG: E=7 is hyperedge count"),
        ("7D",  "0",  f"{b1_7d_open}",  f"{b1_7d_sealed} (6 sq faces)",
         "SUSPICIOUS: Bott echo -> faces"),
        ("8D",  "24", "n/a",  "n/a",
         "WRONG: E=16 impossible (proven)"),
    ]

    for dim, old_f, b1_0, b1_f, verdict in rows:
        print(f"  | {dim:3s} | {old_f:7s} | {b1_0:9s} | {b1_f:15s} | {verdict:14s} |")

    print("  +-----+---------+-----------+-----------------+----------------+")
    print()
    print("  Legend:")
    print("    REFERENCE  = known correct baseline")
    print("    CONFIRMED  = F=0 structurally forced (no possible faces)")
    print("    SUSPICIOUS = faces CAN exist; F=0 needs geometric justification")
    print("    WRONG      = old data has errors (wrong edge count or impossible)")
    print()

    print("  DETAILED VERDICTS:")
    print()
    print("  5D (V=22, E=22, F=0 -> beta_1=1):")
    print("    SUSPICIOUS. The graph has exactly 1 cycle. Adding the cycle")
    print("    as a face gives F=1, beta_1=0. Whether the cycle should be")
    print("    filled depends on the coupling geometry, not combinatorics.")
    print("    F=0 is a CHOICE, not a necessity.")
    print()
    print("  6D (V=7, E=7, F=7 -> 'G2 zero-mode'):")
    print("    WRONG as stated. V=E=F=7 is inconsistent for a simple CW complex.")
    print("    If this is the Fano plane (octonion structure), then:")
    print("      Corrected: V=7, E=21, F=7 (Fano triangles)")
    print(f"      beta_1 = {b1_6d_fano}")
    print("    The '7 edges' likely counted Fano LINES (hyperedges), not graph edges.")
    print()
    print("  7D (V=8, E=12, F=0 -> beta_1=5):")
    print("    SUSPICIOUS. The graph IS the cube graph (same as 3D).")
    print("    The cube graph naturally carries 6 square faces.")
    print("    Bott periodicity (pi_3 ~ pi_7) suggests the 2-cell structure")
    print("    should be preserved. F=0 would require showing that the 7D")
    print("    lattice geometry specifically prevents filling the cube squares.")
    print()
    print("  8D (V=9, E=16, F=24):")
    print("    WRONG (previously proven). E=16 is impossible for the BCC")
    print("    1-skeleton which has E=20.")
    print()


def _cube_faces(edges):
    """Build 6 square face lists for the cube graph."""
    face_defs = [
        [0, 2, 6, 4], [1, 3, 7, 5],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [0, 1, 3, 2], [4, 5, 7, 6],
    ]
    faces = []
    for fverts in face_defs:
        face_edges = []
        n = len(fverts)
        for k in range(n):
            a, b = fverts[k], fverts[(k + 1) % n]
            ei = find_edge_index(edges, a, b)
            face_edges.append(ei)
        faces.append(face_edges)
    return faces


# ─── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    part1_7d_bott_echo()
    print()
    part2_5d_orbit()
    print()
    part3_6d_g2()
    print()
    part4_summary()
