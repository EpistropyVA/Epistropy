"""
rh_betti_scan.py — beta_1 (first Betti number over F2) across the d-cascade 0D..16D

Computes boundary matrices over F2, does Gaussian elimination for rank,
and reports which dimensions are sealed (beta_1=0) vs have open holes (beta_1>0).
"""

import sys
import time
from itertools import combinations

# UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")


# ─── F2 Gaussian elimination ───────────────────────────────────────────

def f2_rank(matrix, nrows, ncols):
    """Rank of an nrows x ncols matrix over F2.
    matrix: list of lists (each row is a list of 0/1 ints).
    Uses bitwise representation for speed.
    """
    if nrows == 0 or ncols == 0:
        return 0

    # Convert rows to integers (bit vectors)
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
        # Find pivot row
        pivot = None
        for r in range(rank, len(rows)):
            if rows[r] & mask:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap pivot to position 'rank'
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        # Eliminate
        for r in range(len(rows)):
            if r != rank and (rows[r] & mask):
                rows[r] ^= rows[rank]
        rank += 1

    return rank


# ─── Complex builders ──────────────────────────────────────────────────

def build_point():
    """0D: single point."""
    return {"name": "Point", "dim": "0D", "V": 1, "E": 0, "F": 0,
            "vertices": [0], "edges": [], "faces": []}


def build_c7():
    """1D: Fano seed — cycle graph C7 (heptagon)."""
    verts = list(range(7))
    edges = [(i, (i + 1) % 7) for i in range(7)]
    return {"name": "C7 (Fano seed)", "dim": "1D", "V": 7, "E": 7, "F": 0,
            "vertices": verts, "edges": edges, "faces": []}


def build_fano():
    """2D: Fano plane — K7 edges + 7 Fano lines as triangular faces."""
    verts = list(range(7))
    # All edges of K7
    edges = list(combinations(range(7), 2))
    # The 7 lines of the Fano plane (standard labeling)
    fano_lines = [
        (0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6),
        (4, 5, 0), (5, 6, 1), (6, 0, 2)
    ]
    return {"name": "Fano plane", "dim": "2D", "V": 7, "E": len(edges), "F": 7,
            "vertices": verts, "edges": edges, "faces": fano_lines}


def build_sc_cube():
    """3D: Simple cubic — cube graph with 6 square faces."""
    # Vertices of the unit cube
    verts = list(range(8))
    # Edges: pairs differing in exactly 1 bit
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                edges.append((i, j))
    # 6 square faces of the cube
    # Each face: fix one coordinate, vary the other two
    faces = []
    for fixed_coord in range(3):
        for fixed_val in range(2):
            # The 4 vertices with coordinate fixed_coord == fixed_val
            face_verts = []
            for v in range(8):
                if (v >> fixed_coord) & 1 == fixed_val:
                    face_verts.append(v)
            # Order them as a cycle: arrange by the two varying coords
            # face_verts has 4 elements — form the square boundary
            faces.append(tuple(sorted(face_verts)))

    return {"name": "SC cube", "dim": "3D", "V": 8, "E": len(edges), "F": 6,
            "vertices": verts, "edges": edges, "faces": faces}


def build_bcc():
    """4D: BCC — cube (8 corners) + center (vertex 8), 18 triangular faces."""
    verts = list(range(9))  # 0..7 = cube corners, 8 = center
    # Cube edges
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                edges.append((i, j))
    # Edges from center to all corners
    for i in range(8):
        edges.append((i, 8))
    # Triangular faces: for each cube edge (i,j), the triangle (i,j,8)
    # That gives 12 triangles. Plus the 6 square faces of the cube.
    # But the spec says F=18: 12 triangles (center-edge) + 6 squares = 18
    tri_faces = [(i, j, 8) for (i, j) in edges[:12]]  # cube edges only
    sq_faces = []
    for fixed_coord in range(3):
        for fixed_val in range(2):
            face_verts = []
            for v in range(8):
                if (v >> fixed_coord) & 1 == fixed_val:
                    face_verts.append(v)
            sq_faces.append(tuple(sorted(face_verts)))

    return {"name": "BCC", "dim": "4D", "V": 9, "E": len(edges), "F": len(tri_faces) + len(sq_faces),
            "vertices": verts, "edges": edges,
            "faces_tri": tri_faces, "faces_sq": sq_faces}


def build_hypercube(n):
    """n-dimensional hypercube: 2^n vertices, n*2^(n-1) edges, C(n,2)*2^(n-2) square faces."""
    V = 1 << n
    verts = list(range(V))

    # Edges: pairs differing in exactly 1 bit
    edges = []
    for i in range(V):
        for bit in range(n):
            j = i ^ (1 << bit)
            if j > i:
                edges.append((i, j))

    # Square faces: choose 2 coordinates (i_coord, j_coord), fix the rest
    faces = []
    for i_coord, j_coord in combinations(range(n), 2):
        mask_i = 1 << i_coord
        mask_j = 1 << j_coord
        # For each fixing of the other n-2 coordinates
        seen = set()
        for base in range(V):
            # Zero out coords i and j to get the "base" pattern
            base_fixed = base & ~mask_i & ~mask_j
            if base_fixed in seen:
                continue
            seen.add(base_fixed)
            # The 4 vertices
            v00 = base_fixed
            v01 = base_fixed | mask_j
            v10 = base_fixed | mask_i
            v11 = base_fixed | mask_i | mask_j
            faces.append(tuple(sorted([v00, v01, v10, v11])))

    return {"name": f"{n}-cube", "dim": f"{n}D-cube",
            "V": V, "E": len(edges), "F": len(faces),
            "vertices": verts, "edges": edges, "faces": faces}


# ─── Boundary matrix construction ──────────────────────────────────────

def edge_index(edges, edge_map, v0, v1):
    """Get index of edge (v0,v1) in the edge list."""
    key = (min(v0, v1), max(v0, v1))
    return edge_map[key]


def build_d1(vertices, edges):
    """Build boundary matrix d1: E x V over F2. d1[e] has 1s at the two endpoints."""
    V = len(vertices)
    E = len(edges)
    v_map = {v: i for i, v in enumerate(vertices)}
    mat = [[0] * V for _ in range(E)]
    for e_idx, (v0, v1) in enumerate(edges):
        mat[e_idx][v_map[v0]] = 1
        mat[e_idx][v_map[v1]] = 1
    return mat, E, V


def build_d2_triangles(edges, tri_faces):
    """Build d2 for triangular faces: F x E over F2."""
    E = len(edges)
    F = len(tri_faces)
    e_map = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(edges)}
    mat = [[0] * E for _ in range(F)]
    for f_idx, (a, b, c) in enumerate(tri_faces):
        for v0, v1 in [(a, b), (b, c), (a, c)]:
            key = (min(v0, v1), max(v0, v1))
            mat[f_idx][e_map[key]] = 1
    return mat, F, E


def build_d2_squares(edges, sq_faces):
    """Build d2 for square faces: F x E over F2.
    Each square face (4 vertices) has 4 boundary edges.
    For a square with vertices {a,b,c,d} (sorted), the edges
    forming the boundary are the 4 edges of the square cycle.
    """
    E = len(edges)
    F = len(sq_faces)
    e_map = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(edges)}

    mat = [[0] * E for _ in range(F)]
    for f_idx, face in enumerate(sq_faces):
        # face is sorted tuple of 4 vertices
        # Find which edges exist among these 4 vertices
        for i in range(len(face)):
            for j in range(i + 1, len(face)):
                key = (face[i], face[j])
                if key in e_map:
                    mat[f_idx][e_map[key]] = 1
    return mat, F, E


def build_d2_mixed(edges, tri_faces, sq_faces):
    """Build d2 for a mix of triangular and square faces."""
    E = len(edges)
    all_faces = list(tri_faces) + list(sq_faces)
    F = len(all_faces)
    e_map = {(min(a, b), max(a, b)): i for i, (a, b) in enumerate(edges)}

    mat = [[0] * E for _ in range(F)]
    for f_idx, face in enumerate(all_faces):
        if len(face) == 3:
            a, b, c = face
            for v0, v1 in [(a, b), (b, c), (a, c)]:
                key = (min(v0, v1), max(v0, v1))
                mat[f_idx][e_map[key]] = 1
        elif len(face) == 4:
            for i in range(len(face)):
                for j in range(i + 1, len(face)):
                    key = (min(face[i], face[j]), max(face[i], face[j]))
                    if key in e_map:
                        mat[f_idx][e_map[key]] = 1
    return mat, F, E


def build_d2_hypercube_squares(edges, faces):
    """Build d2 for hypercube square faces."""
    E = len(edges)
    F = len(faces)
    e_map = {}
    for i, (a, b) in enumerate(edges):
        e_map[(min(a, b), max(a, b))] = i

    mat = [[0] * E for _ in range(F)]
    for f_idx, face in enumerate(faces):
        # Each square face has exactly 4 edges (adjacent vertices in the square)
        # For a hypercube square with vertices v00, v01, v10, v11
        # (differ in coords i,j), the 4 edges are:
        # v00-v01, v00-v10, v01-v11, v10-v11
        # Since face is sorted, we need to identify which pairs are edges
        # (differ in exactly 1 bit)
        for i in range(4):
            for j in range(i + 1, 4):
                v0, v1 = face[i], face[j]
                if bin(v0 ^ v1).count('1') == 1:
                    key = (min(v0, v1), max(v0, v1))
                    mat[f_idx][e_map[key]] = 1
    return mat, F, E


# ─── Compute beta_1 for a complex ─────────────────────────────────────

def compute_beta1(complex_data):
    """Compute beta_1 = dim(ker d1) - dim(im d2) = (E - rank(d1)) - rank(d2).
    For connected complex: dim(ker d1) = E - V + 1, so beta_1 = E - V + 1 - rank(d2).
    """
    V = complex_data["V"]
    E = complex_data["E"]
    F = complex_data["F"]

    if E == 0:
        return {"beta1": 0, "rank_d1": 0, "rank_d2": 0, "graph_beta1": 0}

    # Build and rank d1
    d1, nrows1, ncols1 = build_d1(complex_data["vertices"], complex_data["edges"])
    rank_d1 = f2_rank(d1, nrows1, ncols1)

    # graph_beta1 = E - rank(d1) = E - (V - #components)
    # For connected graph: graph_beta1 = E - V + 1
    graph_beta1 = E - rank_d1

    # Build and rank d2
    rank_d2 = 0
    if F > 0:
        if "faces_tri" in complex_data:
            # Mixed triangular + square faces (BCC)
            d2, nrows2, ncols2 = build_d2_mixed(
                complex_data["edges"],
                complex_data["faces_tri"],
                complex_data["faces_sq"])
            rank_d2 = f2_rank(d2, nrows2, ncols2)
        elif "faces" in complex_data and len(complex_data["faces"]) > 0:
            sample = complex_data["faces"][0]
            if len(sample) == 3:
                d2, nrows2, ncols2 = build_d2_triangles(
                    complex_data["edges"], complex_data["faces"])
                rank_d2 = f2_rank(d2, nrows2, ncols2)
            elif len(sample) == 4:
                d2, nrows2, ncols2 = build_d2_hypercube_squares(
                    complex_data["edges"], complex_data["faces"])
                rank_d2 = f2_rank(d2, nrows2, ncols2)

    beta1 = graph_beta1 - rank_d2
    return {"beta1": beta1, "rank_d1": rank_d1, "rank_d2": rank_d2,
            "graph_beta1": graph_beta1}


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("  beta_1 SCAN across the d-cascade (F2 homology)")
    print("=" * 90)
    print()

    results = []

    # ── Part 1: Build and compute ──────────────────────────────────────

    print("─── Part 1: Constructing complexes and computing beta_1 ───")
    print()

    # 0D
    cx = build_point()
    r = compute_beta1(cx)
    results.append((cx, r))
    print(f"  0D Point: V={cx['V']}, E={cx['E']}, F={cx['F']} → beta_1 = {r['beta1']}")

    # 1D
    cx = build_c7()
    r = compute_beta1(cx)
    results.append((cx, r))
    print(f"  1D C7:    V={cx['V']}, E={cx['E']}, F={cx['F']} → beta_1 = {r['beta1']}")

    # 2D
    cx = build_fano()
    r = compute_beta1(cx)
    results.append((cx, r))
    print(f"  2D Fano:  V={cx['V']}, E={cx['E']}, F={cx['F']} → rank(d2)={r['rank_d2']}, beta_1 = {r['beta1']}")

    # 3D
    cx = build_sc_cube()
    r = compute_beta1(cx)
    results.append((cx, r))
    print(f"  3D SC:    V={cx['V']}, E={cx['E']}, F={cx['F']} → rank(d2)={r['rank_d2']}, beta_1 = {r['beta1']}")

    # 4D
    cx = build_bcc()
    r = compute_beta1(cx)
    results.append((cx, r))
    print(f"  4D BCC:   V={cx['V']}, E={cx['E']}, F={cx['F']} → rank(d2)={r['rank_d2']}, beta_1 = {r['beta1']}")

    # Hypercubes 4..7
    for n in range(4, 8):
        print(f"\n  Building {n}-cube...", end=" ", flush=True)
        t0 = time.time()
        cx = build_hypercube(n)
        t_build = time.time() - t0
        print(f"V={cx['V']}, E={cx['E']}, F={cx['F']} (built in {t_build:.2f}s)")

        print(f"    Computing ranks...", end=" ", flush=True)
        t0 = time.time()

        # Timeout safety for large cubes
        if n >= 7:
            # For n=7: 448 edges, 672 faces — try it
            pass

        r = compute_beta1(cx)
        t_rank = time.time() - t0
        print(f"rank(d1)={r['rank_d1']}, rank(d2)={r['rank_d2']}, "
              f"beta_1 = {r['beta1']} ({t_rank:.2f}s)")
        results.append((cx, r))

    # ── Part 2 & 3: Summary table ──────────────────────────────────────

    print()
    print("─── Part 3: Summary table ───")
    print()
    header = f"{'Dim':<10} {'Complex':<16} {'V':>5} {'E':>5} {'F':>5} {'graph_b1':>9} {'rank(d2)':>9} {'beta_1':>7}  Status"
    print(header)
    print("─" * len(header))

    for cx, r in results:
        status = "sealed" if r["beta1"] == 0 else f"{r['beta1']} HOLE{'S' if r['beta1'] > 1 else ''}"
        print(f"{cx['dim']:<10} {cx['name']:<16} {cx['V']:>5} {cx['E']:>5} {cx['F']:>5} "
              f"{r['graph_beta1']:>9} {r['rank_d2']:>9} {r['beta1']:>7}  {status}")

    # ── Part 4: Efficiency analysis ────────────────────────────────────

    print()
    print("─── Part 4: Efficiency analysis ───")
    print()
    header2 = f"{'Dim':<10} {'Complex':<16} {'F':>5} {'rank(d2)':>9} {'efficiency':>11} {'repair_ratio':>13}"
    print(header2)
    print("─" * len(header2))

    for cx, r in results:
        if cx["F"] > 0 and r["graph_beta1"] > 0:
            eff = r["rank_d2"] / cx["F"]
            repair = r["rank_d2"] / r["graph_beta1"]
            print(f"{cx['dim']:<10} {cx['name']:<16} {cx['F']:>5} {r['rank_d2']:>9} "
                  f"{eff:>11.4f} {repair:>13.4f}"
                  f"{'  <-- ALL SEALED' if repair >= 1.0 - 1e-9 else ''}")
        elif cx["F"] > 0:
            eff = "N/A (no graph cycles)"
            print(f"{cx['dim']:<10} {cx['name']:<16} {cx['F']:>5} {r['rank_d2']:>9} "
                  f"  {eff}")
        else:
            print(f"{cx['dim']:<10} {cx['name']:<16} {cx['F']:>5} {r['rank_d2']:>9} "
                  f"       N/A (no faces)")

    # ── Part 5: The 5D-8D question ─────────────────────────────────────

    print()
    print("─── Part 5: The 5D-8D question (old hand-filled data) ───")
    print()

    old_data = [
        ("5D", "Petersen-like", 22, 22, 0),
        ("6D", "G2 zero-mode", 7, 7, 7),
        ("7D", "Cube echo", 8, 12, 0),
        ("8D", "BCC echo?", 9, 16, 24),
    ]

    print("  Old cascade data (partially known wrong):")
    print(f"  {'Dim':<5} {'Name':<16} {'V':>5} {'E':>5} {'F':>5} {'graph_b1 (if conn.)':>20}")
    print("  " + "─" * 60)

    for dim, name, V, E, F in old_data:
        gb1 = E - V + 1  # assuming connected
        print(f"  {dim:<5} {name:<16} {V:>5} {E:>5} {F:>5} {gb1:>20}")

    print()
    print("  Analysis:")
    print()

    # 5D: V=22, E=22, connected → beta_1 = 22-22+1 = 1, no faces → 1 open hole
    print("  5D (V=22, E=22, F=0, assuming connected):")
    gb1_5d = 22 - 22 + 1
    print(f"    graph_beta_1 = E - V + 1 = {gb1_5d}")
    print(f"    No faces → beta_1 = {gb1_5d}")
    print(f"    → 1 OPEN HOLE (no faces to seal it)")
    print()

    # 6D: The old E=7 is suspicious
    print("  6D (V=7, E=7, F=7 — 'G2 zero-mode' after Bott collapse):")
    gb1_6d = 7 - 7 + 1
    print(f"    graph_beta_1 = E - V + 1 = {gb1_6d} (if connected)")
    print(f"    With F=7 faces and 7 triangular faces: rank(d2) could be up to {gb1_6d}")
    print(f"    NOTE: E=7 for 7 vertices is NOT K7 (which has 21 edges).")
    print(f"    This is a different structure from the 2D Fano plane.")
    print()

    # 7D: Cube graph without faces
    print("  7D (V=8, E=12, F=0 — 'Cube echo' without faces):")
    gb1_7d = 12 - 8 + 1
    print(f"    graph_beta_1 = E - V + 1 = {gb1_7d}")
    print(f"    No faces → beta_1 = {gb1_7d}")
    print(f"    → 5 OPEN HOLES")
    print(f"    KEY INSIGHT: 7D echoes 3D's graph (cube), but WITHOUT the 6 faces")
    print(f"    that sealed it. The 5 cycles that SC's faces killed REAPPEAR at 7D!")
    print()

    # 8D: E=16 impossible
    print("  8D (V=9, E=16, F=24 — old data):")
    print(f"    E=16 is impossible for BCC-like (V=9):")
    print(f"      Cube edges = 12, center-to-corner = 8, total = 20")
    print(f"      No valid subgraph of BCC on 9 vertices has exactly 16 edges.")
    print(f"    → OLD DATA IS WRONG for 8D. Needs reconstruction.")
    print()

    print("  CONCLUSION:")
    print("  If F=0 entries are correct, then 5D and 7D have OPEN topological defects —")
    print("  holes that are NOT sealed. 7D is especially striking: it replicates the")
    print("  cube graph of 3D but strips away all face-fillings, reopening 5 holes.")
    print()

    # ── Final summary ──────────────────────────────────────────────────

    print("=" * 90)
    print("  SEALED vs OPEN across the cascade")
    print("=" * 90)
    print()
    for cx, r in results:
        marker = "SEALED" if r["beta1"] == 0 else f"OPEN ({r['beta1']} hole{'s' if r['beta1'] > 1 else ''})"
        bar = "█" * r["beta1"] if r["beta1"] > 0 else "·"
        print(f"  {cx['dim']:<10} {cx['name']:<16} beta_1={r['beta1']:>3}  {marker:<20} {bar}")

    print()
    print("  Old data (graph-only, unverified):")
    print(f"  {'5D':<10} {'Petersen-like':<16} beta_1={'1':>3}  {'OPEN (1 hole)':<20} █")
    print(f"  {'7D':<10} {'Cube echo':<16} beta_1={'5':>3}  {'OPEN (5 holes)':<20} █████")
    print()


if __name__ == "__main__":
    main()
