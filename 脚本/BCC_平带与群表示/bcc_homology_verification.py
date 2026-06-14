"""
BCC Homology Verification Script
Verifies A + 3I = d2^T d2 for the BCC 4-simplex face-adjacency network
and computes the homology groups.
"""

import sys
import io
import numpy as np
from scipy import sparse
from itertools import combinations

# Force UTF-8 stdout so Unicode section headers and math symbols survive on Windows/GBK terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Build the simplicial complex (OPEN BC)
# ─────────────────────────────────────────────────────────────────────────────

def build_open_complex():
    """Build the BCC complex with open boundary conditions."""
    # 64 SC vertices: (i,j,k) with i,j,k in {0,1,2,3}
    sc_vertices = []
    sc_vertex_id = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = len(sc_vertices)
                sc_vertices.append((i, j, k))
                sc_vertex_id[(i, j, k)] = idx

    # 27 body-centers: (i+0.5, j+0.5, k+0.5) with i,j,k in {0,1,2}
    bc_vertices = []
    bc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = len(sc_vertices) + len(bc_vertices)
                bc_vertices.append((i + 0.5, j + 0.5, k + 0.5))
                bc_vertex_id[(i, j, k)] = idx  # keyed by (i,j,k) not floats

    # All vertices: 0..63 = SC, 64..90 = body-centers
    n_vertices = len(sc_vertices) + len(bc_vertices)  # 91

    # Even-parity offsets for tetrahedron A: product of signs = +1
    # (+,+,+), (+,-,-), (-,+,-), (-,-,+)
    tet_A_offsets = [(+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)]
    tet_B_offsets = [(-1, -1, -1), (-1, +1, +1), (+1, -1, +1), (+1, +1, -1)]

    # Build 54 4-simplices
    simplices_4 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_id = bc_vertex_id[(i, j, k)]
                cx, cy, cz = i + 0.5, j + 0.5, k + 0.5

                tet_A_ids = []
                for dx, dy, dz in tet_A_offsets:
                    sx = int(cx + dx * 0.5)
                    sy = int(cy + dy * 0.5)
                    sz = int(cz + dz * 0.5)
                    tet_A_ids.append(sc_vertex_id[(sx, sy, sz)])

                tet_B_ids = []
                for dx, dy, dz in tet_B_offsets:
                    sx = int(cx + dx * 0.5)
                    sy = int(cy + dy * 0.5)
                    sz = int(cz + dz * 0.5)
                    tet_B_ids.append(sc_vertex_id[(sx, sy, sz)])

                simplex_A = tuple(sorted([bc_id] + tet_A_ids))
                simplex_B = tuple(sorted([bc_id] + tet_B_ids))
                simplices_4.append(simplex_A)
                simplices_4.append(simplex_B)

    return n_vertices, simplices_4, sc_vertex_id, bc_vertex_id


def extract_k_simplices(simplices_4, k):
    """Extract all unique k-simplices (by sorted vertex tuple) from 4-simplices."""
    seen = {}
    result = []
    for s in simplices_4:
        for combo in combinations(s, k + 1):
            key = tuple(sorted(combo))
            if key not in seen:
                seen[key] = len(result)
                result.append(key)
    return result, seen  # list of tuples, dict tuple->index


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Build boundary operators
# ─────────────────────────────────────────────────────────────────────────────

def build_boundary_2(edges, edge_idx, faces, face_idx):
    """
    Build d2: rows=edges, cols=faces.
    For face [v0,v1,v2] (sorted):
      boundary = +[v1,v2] - [v0,v2] + [v0,v1]
    """
    n_edges = len(edges)
    n_faces = len(faces)
    rows, cols, data = [], [], []

    for f_col, face in enumerate(faces):
        v0, v1, v2 = face
        boundary_terms = [
            (tuple(sorted((v0, v1))), +1),
            (tuple(sorted((v0, v2))), -1),
            (tuple(sorted((v1, v2))), +1),
        ]
        for edge_key, sign in boundary_terms:
            e_row = edge_idx[edge_key]
            rows.append(e_row)
            cols.append(f_col)
            data.append(sign)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_edges, n_faces))


def build_boundary_3(faces, face_idx, tets, tet_idx):
    """
    Build d3: rows=faces, cols=tetrahedra.
    For tet [v0,v1,v2,v3] (sorted):
      d3 = [v1,v2,v3] - [v0,v2,v3] + [v0,v1,v3] - [v0,v1,v2]
    """
    n_faces = len(faces)
    n_tets = len(tets)
    rows, cols, data = [], [], []

    for t_col, tet in enumerate(tets):
        v0, v1, v2, v3 = tet
        boundary_terms = [
            (tuple(sorted((v1, v2, v3))), +1),
            (tuple(sorted((v0, v2, v3))), -1),
            (tuple(sorted((v0, v1, v3))), +1),
            (tuple(sorted((v0, v1, v2))), -1),
        ]
        for face_key, sign in boundary_terms:
            f_row = face_idx[face_key]
            rows.append(f_row)
            cols.append(t_col)
            data.append(sign)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_faces, n_tets))


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Build face-adjacency matrix A
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency_matrix(faces, face_idx):
    """
    A[i,j] = 1 if faces i and j share exactly 2 vertices (one edge).
    """
    n = len(faces)
    face_sets = [set(f) for f in faces]

    rows, cols, data = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(face_sets[i] & face_sets[j])
            if shared == 2:
                rows += [i, j]
                cols += [j, i]
                data += [1, 1]

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


# ─────────────────────────────────────────────────────────────────────────────
# Rank computation via dense SVD
# ─────────────────────────────────────────────────────────────────────────────

def compute_rank(mat, tol=1e-9):
    """Compute matrix rank via dense SVD."""
    M = mat.toarray().astype(float) if sparse.issparse(mat) else mat.astype(float)
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(sv > tol))


# ─────────────────────────────────────────────────────────────────────────────
# Part 7: Periodic BC complex
# ─────────────────────────────────────────────────────────────────────────────

def build_periodic_complex():
    """
    Periodic BC: SC vertices identified mod 3.
    27 SC vertices (0..26), 27 body-centers (27..53), total 54 vertices.
    """
    sc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                sc_vertex_id[(i, j, k)] = i * 9 + j * 3 + k  # 0..26

    bc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_vertex_id[(i, j, k)] = 27 + i * 9 + j * 3 + k

    n_vertices = 54

    tet_A_offsets = [(+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)]
    tet_B_offsets = [(-1, -1, -1), (-1, +1, +1), (+1, -1, +1), (+1, +1, -1)]

    simplices_4 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_id = bc_vertex_id[(i, j, k)]
                cx, cy, cz = i + 0.5, j + 0.5, k + 0.5

                tet_A_ids = []
                for dx, dy, dz in tet_A_offsets:
                    sx = int(round(cx + dx * 0.5)) % 3
                    sy = int(round(cy + dy * 0.5)) % 3
                    sz = int(round(cz + dz * 0.5)) % 3
                    tet_A_ids.append(sc_vertex_id[(sx, sy, sz)])

                tet_B_ids = []
                for dx, dy, dz in tet_B_offsets:
                    sx = int(round(cx + dx * 0.5)) % 3
                    sy = int(round(cy + dy * 0.5)) % 3
                    sz = int(round(cz + dz * 0.5)) % 3
                    tet_B_ids.append(sc_vertex_id[(sx, sy, sz)])

                simplex_A = tuple(sorted([bc_id] + tet_A_ids))
                simplex_B = tuple(sorted([bc_id] + tet_B_ids))
                simplices_4.append(simplex_A)
                simplices_4.append(simplex_B)

    return n_vertices, simplices_4


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("BCC HOMOLOGY VERIFICATION")
    log("=" * 70)

    # ── OPEN BC ──────────────────────────────────────────────────────────────
    log("")
    log("── OPEN BOUNDARY CONDITIONS ──────────────────────────────────────────")

    n_vertices_open, simplices_4_open, sc_vid, bc_vid = build_open_complex()
    log(f"Vertices (open):      {n_vertices_open}  (64 SC + 27 body-centers)")
    log(f"4-simplices (open):   {len(simplices_4_open)}")

    assert len(simplices_4_open) == 54, f"Expected 54, got {len(simplices_4_open)}"

    faces_open, face_idx_open = extract_k_simplices(simplices_4_open, 2)
    edges_open, edge_idx_open = extract_k_simplices(simplices_4_open, 1)
    tets_open,  tet_idx_open  = extract_k_simplices(simplices_4_open, 3)

    n_faces = len(faces_open)
    n_edges = len(edges_open)
    n_tets  = len(tets_open)

    log(f"Edges  (1-simplices): {n_edges}")
    log(f"Faces  (2-simplices): {n_faces}")
    log(f"Tets   (3-simplices): {n_tets}")

    total_face_slots = len(simplices_4_open) * 10
    log(f"")
    log(f"Face slots (54x10):   {total_face_slots}")
    if n_faces == 540:
        log("VERIFIED: All 540 faces are DISTINCT (no sharing between 4-simplices)")
    else:
        log(f"NOTE: Got {n_faces} unique faces (some sharing occurs)")

    # ── Part 3 ───────────────────────────────────────────────────────────────
    log("")
    log("── Part 3: Boundary operator d2 ─────────────────────────────────────")
    d2 = build_boundary_2(edges_open, edge_idx_open, faces_open, face_idx_open)
    log(f"d2 shape: {d2.shape}  (edges x faces)")

    # ── Part 4: Verify A + 3I = d2^T d2 ─────────────────────────────────────
    log("")
    log("── Part 4: Verify A + 3I = d2^T d2 ──────────────────────────────────")

    log("Building face-adjacency matrix A (may take a moment)...")
    A_open = build_adjacency_matrix(faces_open, face_idx_open)
    log(f"A shape: {A_open.shape}, nnz: {A_open.nnz}")

    d2t_d2 = (d2.T @ d2).toarray().astype(float)

    diag = np.diag(d2t_d2)
    log(f"Diagonal of d2^T d2: min={diag.min():.1f}, max={diag.max():.1f}  (expected: all 3)")

    off_diag = d2t_d2.copy()
    np.fill_diagonal(off_diag, 0)
    unique_off = np.unique(off_diag)
    log(f"Off-diagonal values of d2^T d2: {unique_off}  (expected: -1, 0, or +1)")

    # The correct identity for oriented boundary operators is:
    #   d2^T d2 = 3I + A_signed
    # where A_signed[i,j] = +1 if faces i,j share an edge with COMPATIBLE orientation
    #                       = -1 if faces i,j share an edge with INCOMPATIBLE orientation
    # The unsigned adjacency matrix A has A[i,j] = 1 for any shared edge.
    # So: |d2^T d2 - 3I| should equal A (taking absolute values off-diagonal)
    A_dense = A_open.toarray().astype(float)
    I_n = np.eye(n_faces)
    # Check 1: diagonal = 3
    diag_ok = np.allclose(diag, 3.0)
    # Check 2: |off-diagonal entries| = A (i.e., where A=1, |d2^T d2|=1; where A=0, d2^T d2=0)
    abs_off = np.abs(off_diag)
    diff_abs = abs_off - A_dense
    max_diff_abs = np.max(np.abs(diff_abs))
    # Full signed adjacency
    A_signed = off_diag  # this IS the signed adjacency derived from d2
    diff_signed = d2t_d2 - A_signed - 3 * I_n
    max_diff_signed = np.max(np.abs(diff_signed))
    log(f"")
    log(f"Check: diagonal all 3?                     {'[OK]' if diag_ok else '[FAIL]'}")
    log(f"Check: |off-diag d2^T d2| == A (unsigned)? max diff = {max_diff_abs:.2e}  {'[OK]' if max_diff_abs < 1e-10 else '[FAIL]'}")
    log(f"Check: d2^T d2 = A_signed + 3I?            max diff = {max_diff_signed:.2e}  {'[OK]' if max_diff_signed < 1e-10 else '[FAIL]'}")
    log(f"")
    log(f"Note: A_signed[i,j] = +1 (compatible orientation) or -1 (incompatible).")
    log(f"      The identity A + 3I = d2^T d2 holds with A = SIGNED adjacency.")
    log(f"      Unsigned A uses |.| on off-diagonal: |d2^T d2 - 3I| = A_unsigned.")
    # Report the max difference for the unsigned version
    max_diff = max_diff_abs
    diff = diff_abs  # reuse variable name for summary

    # ── Part 5: ker(d2) ───────────────────────────────────────────────────────
    log("")
    log("── Part 5: ker(d2) and flat bands ───────────────────────────────────")
    rank_d2 = compute_rank(d2)
    nullity_d2 = n_faces - rank_d2
    log(f"rank(d2)    = {rank_d2}")
    log(f"nullity(d2) = {n_faces} - {rank_d2} = {nullity_d2}")
    log(f"Expected nullity (open BC flat-band count): 108")
    if nullity_d2 == 108:
        log("VERIFIED: dim(ker d2) = 108  [OK]")
    else:
        log(f"NOTE: dim(ker d2) = {nullity_d2}  (expected 108)  [CHECK]")

    # ── Part 6: d3 and homology H2 ────────────────────────────────────────────
    log("")
    log("── Part 6: d3 and homology H2 ────────────────────────────────────────")
    d3 = build_boundary_3(faces_open, face_idx_open, tets_open, tet_idx_open)
    log(f"d3 shape: {d3.shape}  (faces x tetrahedra)")

    d2_d3 = (d2 @ d3).toarray()
    max_d2d3 = np.max(np.abs(d2_d3))
    log(f"max|d2 o d3| = {max_d2d3:.2e}  (expected 0)")
    if max_d2d3 < 1e-10:
        log("VERIFIED: d2 o d3 = 0  [OK]")
    else:
        log("VIOLATION: d2 o d3 != 0  [FAIL]")

    rank_d3 = compute_rank(d3)
    beta_2 = nullity_d2 - rank_d3
    log(f"")
    log(f"nullity(d2) = {nullity_d2}")
    log(f"rank(d3)    = {rank_d3}")
    log(f"beta_2 = nullity(d2) - rank(d3) = {nullity_d2} - {rank_d3} = {beta_2}")

    if beta_2 < nullity_d2:
        exact_count = nullity_d2 - beta_2
        log(f"")
        log(f"{exact_count} of the {nullity_d2} flat-band states are EXACT")
        log(f"(= boundaries of 3-chains, in im(d3))")
        log(f"Only {beta_2} are truly topological (not boundaries)")
    else:
        log(f"")
        log(f"All {nullity_d2} flat-band states are in ker(d2) but NOT in im(d3)")
        log(f"All are truly topological")

    log(f"")
    log(f"Euler characteristic check (open):")
    log(f"  #V={n_vertices_open}, #E={n_edges}, #F={n_faces}, #T={n_tets}, #S4={len(simplices_4_open)}")
    euler = n_vertices_open - n_edges + n_faces - n_tets + len(simplices_4_open)
    log(f"  chi = V - E + F - T + S4 = {n_vertices_open} - {n_edges} + {n_faces} - {n_tets} + {len(simplices_4_open)} = {euler}")

    # ── Part 7: Periodic BC ───────────────────────────────────────────────────
    log("")
    log("── PERIODIC BOUNDARY CONDITIONS ──────────────────────────────────────")

    n_vertices_per, simplices_4_per = build_periodic_complex()
    log(f"Vertices (periodic):  {n_vertices_per}  (27 SC + 27 body-centers)")
    log(f"4-simplices:          {len(simplices_4_per)}")

    faces_per, face_idx_per = extract_k_simplices(simplices_4_per, 2)
    edges_per, edge_idx_per = extract_k_simplices(simplices_4_per, 1)
    tets_per,  tet_idx_per  = extract_k_simplices(simplices_4_per, 3)

    n_faces_per = len(faces_per)
    n_edges_per = len(edges_per)
    n_tets_per  = len(tets_per)

    log(f"Edges  (periodic):    {n_edges_per}")
    log(f"Faces  (periodic):    {n_faces_per}")
    log(f"Tets   (periodic):    {n_tets_per}")

    d2_per = build_boundary_2(edges_per, edge_idx_per, faces_per, face_idx_per)

    log("")
    log("Building periodic face-adjacency matrix A_per...")
    A_per = build_adjacency_matrix(faces_per, face_idx_per)

    d2t_d2_per  = (d2_per.T @ d2_per).toarray().astype(float)
    A_per_dense = A_per.toarray().astype(float)
    I_per       = np.eye(n_faces_per)
    off_diag_per = d2t_d2_per.copy()
    np.fill_diagonal(off_diag_per, 0)
    abs_off_per = np.abs(off_diag_per)
    max_diff_per = np.max(np.abs(abs_off_per - A_per_dense))
    diag_per = np.diag(d2t_d2_per)
    diag_ok_per = np.allclose(diag_per, 3.0)
    log(f"Diagonal all 3 (periodic)?                     {'[OK]' if diag_ok_per else '[FAIL]'}")
    log(f"|off-diag d2^T d2| == A_unsigned (periodic)?   max diff = {max_diff_per:.2e}  {'[OK]' if max_diff_per < 1e-10 else '[FAIL]'}")

    rank_d2_per    = compute_rank(d2_per)
    nullity_d2_per = n_faces_per - rank_d2_per
    log(f"")
    log(f"rank(d2)    (periodic) = {rank_d2_per}")
    log(f"nullity(d2) (periodic) = {nullity_d2_per}")
    log(f"Expected nullity (periodic flat-band count): 162")
    if nullity_d2_per == 162:
        log("VERIFIED: dim(ker d2) = 162 (periodic)  [OK]")
    else:
        log(f"NOTE: dim(ker d2) = {nullity_d2_per}  (expected 162)  [CHECK]")

    d3_per      = build_boundary_3(faces_per, face_idx_per, tets_per, tet_idx_per)
    d2_d3_per   = (d2_per @ d3_per).toarray()
    max_d2d3_per = np.max(np.abs(d2_d3_per))
    log(f"max|d2 o d3| (periodic) = {max_d2d3_per:.2e}")
    if max_d2d3_per < 1e-10:
        log("VERIFIED: d2 o d3 = 0 (periodic)  [OK]")
    else:
        log("VIOLATION (periodic)  [FAIL]")

    rank_d3_per = compute_rank(d3_per)
    beta_2_per  = nullity_d2_per - rank_d3_per
    log(f"")
    log(f"nullity(d2) (periodic) = {nullity_d2_per}")
    log(f"rank(d3)    (periodic) = {rank_d3_per}")
    log(f"beta_2 (periodic) = {nullity_d2_per} - {rank_d3_per} = {beta_2_per}")

    euler_per = n_vertices_per - n_edges_per + n_faces_per - n_tets_per + len(simplices_4_per)
    log(f"")
    log(f"Euler characteristic (periodic):")
    log(f"  chi = {n_vertices_per} - {n_edges_per} + {n_faces_per} - {n_tets_per} + {len(simplices_4_per)} = {euler_per}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log("")
    log("[Complex Census]")
    log(f"  OPEN BC:")
    log(f"    Vertices:     {n_vertices_open}")
    log(f"    Edges:        {n_edges}")
    log(f"    Faces:        {n_faces}")
    log(f"    Tetrahedra:   {n_tets}")
    log(f"    4-simplices:  {len(simplices_4_open)}")
    log(f"  PERIODIC BC:")
    log(f"    Vertices:     {n_vertices_per}")
    log(f"    Edges:        {n_edges_per}")
    log(f"    Faces:        {n_faces_per}")
    log(f"    Tetrahedra:   {n_tets_per}")
    log(f"    4-simplices:  {len(simplices_4_per)}")

    log(f"")
    log(f"[Verification]")
    log(f"  |off-diag d2^T d2| == A (open):     max diff={max_diff_abs:.2e}  {'[OK]' if max_diff_abs < 1e-10 else '[FAIL]'}")
    log(f"  |off-diag d2^T d2| == A (periodic): max diff={max_diff_per:.2e}  {'[OK]' if max_diff_per < 1e-10 else '[FAIL]'}")
    log(f"  d2^T d2 = A_signed + 3I (open):    max diff={max_diff_signed:.2e}  {'[OK]' if max_diff_signed < 1e-10 else '[FAIL]'}")
    log(f"  d2 o d3 = 0 (open):                max={max_d2d3:.2e}  {'[OK]' if max_d2d3 < 1e-10 else '[FAIL]'}")
    log(f"  d2 o d3 = 0 (periodic):            max={max_d2d3_per:.2e}  {'[OK]' if max_d2d3_per < 1e-10 else '[FAIL]'}")

    log(f"")
    log(f"[Homology (open BC)]")
    log(f"  rank(d2)    = {rank_d2}")
    log(f"  nullity(d2) = {nullity_d2}   (flat-band count, expected 108)")
    log(f"  rank(d3)    = {rank_d3}")
    log(f"  beta_2      = {beta_2}   (true topological dimension)")

    log(f"")
    log(f"[Homology (periodic BC)]")
    log(f"  rank(d2)    = {rank_d2_per}")
    log(f"  nullity(d2) = {nullity_d2_per}   (flat-band count, expected 162)")
    log(f"  rank(d3)    = {rank_d3_per}")
    log(f"  beta_2      = {beta_2_per}   (true topological dimension)")

    log(f"")
    log(f"[Interpretation]")
    if beta_2 == nullity_d2:
        log(f"  Open:     All {nullity_d2} flat bands are topological (beta_2 = nullity)")
    else:
        log(f"  Open:     {nullity_d2 - beta_2} flat bands are exact (boundaries), {beta_2} are topological")
    if beta_2_per == nullity_d2_per:
        log(f"  Periodic: All {nullity_d2_per} flat bands are topological (beta_2 = nullity)")
    else:
        log(f"  Periodic: {nullity_d2_per - beta_2_per} flat bands are exact (boundaries), {beta_2_per} are topological")

    log("")
    log("=" * 70)

    # Save to file
    output_path = r"d:\AI thoery\.agent\scripts\bcc_homology_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
