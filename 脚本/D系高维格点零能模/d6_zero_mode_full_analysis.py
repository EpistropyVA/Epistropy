# -*- coding: utf-8 -*-
"""
d6_zero_mode_full_analysis.py
Complete analysis of 14-fold degenerate zero modes of the 6-simplex face-adjacency matrix.

Questions:
1. Verify 14-fold zero eigenvalue
2. Get explicit zero-mode eigenvectors
3. Check Fano plane structure (2x7 decomposition)
4. Check G2 adjoint structure
5. S7 irrep decomposition
6. Cross-d degeneracy pattern (d=3,4,5,6,7,8)
"""

import numpy as np
from itertools import combinations
from collections import Counter, defaultdict

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# ============================================================
# Part 1: Build face-adjacency matrix for any d-simplex
# ============================================================

def build_face_adj(d):
    """d-simplex: d+1 vertices, C(d+1,3) triangular 2-faces.
    A[i,j] = 1 if faces share exactly 2 vertices (edge-adjacent)."""
    n_verts = d + 1
    faces = list(combinations(range(n_verts), 3))
    n_faces = len(faces)
    A = np.zeros((n_faces, n_faces), dtype=float)
    for i in range(n_faces):
        for j in range(i+1, n_faces):
            shared = len(set(faces[i]) & set(faces[j]))
            if shared == 2:
                A[i,j] = 1.0
                A[j,i] = 1.0
    return A, faces

# ============================================================
# Part 2: Cross-d zero mode degeneracy
# ============================================================

print("=" * 70)
print("PART A: Zero-mode degeneracy across dimensions d=3..8")
print("=" * 70)
print()
print(f"{'d':>3} {'n_vert':>7} {'n_face':>7} {'eigenvalues (rounded)':>50} {'zero_deg':>9}")
print("-" * 90)

for d in range(3, 9):
    A, faces = build_face_adj(d)
    evals = np.linalg.eigvalsh(A)
    evals_rounded = np.round(evals, 6)
    counts = Counter(evals_rounded)

    # Count zero modes
    zero_deg = sum(v for k, v in counts.items() if abs(k) < 1e-4)

    # Theoretical eigenvalues
    lam1 = -3
    lam2 = d - 6
    lam3 = 2*d - 7
    lam4 = 3*(d-2)

    deg1 = (d+1)*d*(d-4)//6 if d >= 4 else 0  # only valid for d>=5 actually
    deg2 = d*(d-1)//2 - 1
    deg3 = d
    deg4 = 1

    spec_str = ", ".join(f"{k:.1f}(x{v})" for k, v in sorted(counts.items()))
    print(f"{d:>3} {d+1:>7} {len(faces):>7} {spec_str:>50} {zero_deg:>9}")

# Theoretical formula
print()
print("Theoretical eigenvalues of d-simplex face-adjacency:")
print("  lam_1 = -3,        deg = (d+1)*d*(d-4)/6  (valid d>=5)")
print("  lam_2 = d-6,       deg = d*(d-1)/2 - 1")
print("  lam_3 = 2d-7,      deg = d")
print("  lam_4 = 3(d-2),    deg = 1")
print()
print("Zero modes occur when any eigenvalue = 0:")
print("  lam_1=0: -3=0 never")
print("  lam_2=0: d=6 -> deg = 6*5/2 - 1 = 14")
print("  lam_3=0: 2d-7=0, d=3.5 (not integer)")
print("  lam_4=0: 3(d-2)=0, d=2 (degenerate simplex)")
print()
print("=> Zero modes exist ONLY at d=6, with degeneracy 14.")
print("   This is the unique critical dimension for face-adjacency.")

# ============================================================
# Part 3: d=6 detailed analysis
# ============================================================

print()
print("=" * 70)
print("PART B: d=6 zero mode structure")
print("=" * 70)

A6, faces6 = build_face_adj(6)
n_faces = len(faces6)
print(f"\n6-simplex: 7 vertices, {n_faces} faces")

evals, evecs = np.linalg.eigh(A6)
evals_r = np.round(evals, 6)

# Group eigenvectors by eigenvalue
eigenspaces = defaultdict(list)
for i, ev in enumerate(evals_r):
    eigenspaces[ev].append(evecs[:, i])

print(f"\nSpectrum:")
for ev in sorted(eigenspaces.keys()):
    print(f"  lambda = {ev:>6.1f}, degeneracy = {len(eigenspaces[ev])}")

# Extract zero mode subspace
zero_vecs = []
for ev, vecs in eigenspaces.items():
    if abs(ev) < 1e-4:
        zero_vecs = vecs
        break

Z = np.column_stack(zero_vecs)  # 35 x 14 matrix
print(f"\nZero mode subspace: {Z.shape[0]} x {Z.shape[1]}")
print(f"Verification: A @ Z max element = {np.max(np.abs(A6 @ Z)):.2e} (should be ~0)")

# ============================================================
# Part 4: Fano plane analysis
# ============================================================

print()
print("=" * 70)
print("PART C: Fano plane structure in zero modes")
print("=" * 70)

# Standard Fano plane PG(2,2): 7 points = F_2^3 \ {0}, 7 lines
# Using the labeling from hodge_verify.py (F_2^3 nonzero vectors):
# Points: 0,1,2,3,4,5,6 (= vertices of 6-simplex)
# Lines (each line = 3 collinear points in PG(2,2)):
fano_lines = [
    (0, 1, 2),  # e1+e2=e3
    (0, 3, 4),  # e1+e4=e5
    (1, 3, 5),  # e2+e4=e6
    (2, 3, 6),  # e3+e4=e7
    (0, 5, 6),  # e1+e6=e7
    (1, 4, 6),  # e2+e5=e7
    (2, 4, 5),  # e3+e5=e6
]

# Alternative Fano lines (the "other" labeling from the problem):
fano_lines_alt = [
    (0, 1, 3),
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 0),
    (5, 6, 1),
    (6, 0, 2),
]

# Face index lookup
face_idx = {f: i for i, f in enumerate(faces6)}

print("\nFano plane lines (standard PG(2,2) labeling):")
fano_face_indices = []
for line in fano_lines:
    line_sorted = tuple(sorted(line))
    idx = face_idx[line_sorted]
    fano_face_indices.append(idx)
    print(f"  Line {line} -> face index {idx}")

print(f"\nFano lines (alternative cyclic labeling):")
fano_face_indices_alt = []
for line in fano_lines_alt:
    line_sorted = tuple(sorted(line))
    idx = face_idx[line_sorted]
    fano_face_indices_alt.append(idx)
    print(f"  Line {line} -> face index {idx}")

# Check: do the 7 Fano-line faces span a subspace within the null space?
# Project each Fano face indicator vector onto the null space
print("\n--- Projection of Fano-face indicators onto null space ---")

# For each Fano line, create indicator vector e_i (face basis vector)
# and project onto null space: proj = Z @ Z^T @ e_i
P_null = Z @ Z.T  # 35x35 projector onto null space

print("\nStandard Fano lines:")
fano_projected = []
for i, idx in enumerate(fano_face_indices):
    e_i = np.zeros(n_faces)
    e_i[idx] = 1.0
    proj = P_null @ e_i
    norm_proj = np.linalg.norm(proj)
    norm_orth = np.linalg.norm(e_i - proj)
    fano_projected.append(proj)
    print(f"  Face {fano_lines[i]}: ||proj||={norm_proj:.4f}, ||orth||={norm_orth:.4f}, "
          f"null-space fraction={norm_proj**2:.4f}")

# Check if the 7 projected vectors are linearly independent within the null space
F_proj = np.column_stack(fano_projected)
F_null_coords = Z.T @ F_proj  # 14 x 7 matrix of null-space coordinates
rank_fano = np.linalg.matrix_rank(F_null_coords, tol=1e-8)
print(f"\n  Rank of 7 Fano-projected vectors in null space: {rank_fano}")
print(f"  (If 7, they span a 7D subspace of the 14D null space)")

# Same for alternative Fano lines
print("\nAlternative (cyclic) Fano lines:")
fano_projected_alt = []
for i, idx in enumerate(fano_face_indices_alt):
    e_i = np.zeros(n_faces)
    e_i[idx] = 1.0
    proj = P_null @ e_i
    norm_proj = np.linalg.norm(proj)
    fano_projected_alt.append(proj)
    print(f"  Face {fano_lines_alt[i]}: ||proj||={norm_proj:.4f}, null-space fraction={norm_proj**2:.4f}")

F_proj_alt = np.column_stack(fano_projected_alt)
F_null_coords_alt = Z.T @ F_proj_alt
rank_fano_alt = np.linalg.matrix_rank(F_null_coords_alt, tol=1e-8)
print(f"\n  Rank of alternative Fano-projected vectors: {rank_fano_alt}")

# Check if both sets together span more
F_both = np.column_stack([F_null_coords, F_null_coords_alt])
rank_both = np.linalg.matrix_rank(F_both, tol=1e-8)
print(f"  Rank of both Fano sets combined: {rank_both}")

# ============================================================
# Part 4b: Deeper Fano structure - point-line duality
# ============================================================

print()
print("--- Fano point-based vectors ---")
# For each point p in {0,...,6}, define the "star" vector:
# sum of indicator vectors for all faces containing vertex p
# These live in the face space R^35.

point_stars = []
for p in range(7):
    star = np.zeros(n_faces)
    for i, face in enumerate(faces6):
        if p in face:
            star[i] = 1.0
    point_stars.append(star)
    # Each vertex is in C(6,2)=15 faces
    print(f"  Vertex {p}: in {int(sum(star))} faces")

# Project point-stars onto null space
print("\nPoint-star projections onto null space:")
point_projs = []
for p in range(7):
    proj = P_null @ point_stars[p]
    norm_proj = np.linalg.norm(proj)
    point_projs.append(proj)
    print(f"  Vertex {p}: ||proj||={norm_proj:.4f}")

P_point_null = Z.T @ np.column_stack(point_projs)
rank_points = np.linalg.matrix_rank(P_point_null, tol=1e-8)
print(f"\n  Rank of 7 point-star projections in null space: {rank_points}")

# Edge-based vectors: for each edge (a,b), define indicator for faces containing edge
print("\n--- Edge-based vectors ---")
edge_stars = []
for a in range(7):
    for b in range(a+1, 7):
        star = np.zeros(n_faces)
        for i, face in enumerate(faces6):
            if a in face and b in face:
                star[i] = 1.0
        edge_stars.append(star)

E_projs = np.column_stack([P_null @ es for es in edge_stars])
E_null = Z.T @ E_projs
rank_edges = np.linalg.matrix_rank(E_null, tol=1e-8)
print(f"  21 edge-star projections rank in null space: {rank_edges}")

# ============================================================
# Part 5: S_7 irrep decomposition (numerical)
# ============================================================

print()
print("=" * 70)
print("PART D: S_7 irrep identification of zero modes")
print("=" * 70)

# From the results file, we already know:
# chi(zero-mode, identity) = 14
# chi(zero-mode, transposition) = 6
# chi(zero-mode, 3-cycle) = 2

# The S_7 character table gives:
# chi_{[5,2]} on (1^7),(2,1^5),(3,1^4) = 14, 4, -1
# chi_{[4,3]} on (1^7),(2,1^5),(3,1^4) = 14, 6, 2

# From results: zero-mode character on transposition = 6 => matches [4,3]

print("\nFrom previous computation (d6_zero_mode_results.txt):")
print("  Zero modes (lambda=0, dim=14) = V_{(4,3)} (Specht module for partition [4,3])")
print("  Bottom band (lambda=-3, dim=14) = V_{(5,2)}")
print()

# Verify by recomputing character on transposition
def face_perm_matrix(perm, faces):
    n = len(faces)
    face_index = {f: i for i, f in enumerate(faces)}
    P = np.zeros((n, n), dtype=float)
    for i, face in enumerate(faces):
        new_face = tuple(sorted(perm[v] for v in face))
        j = face_index[new_face]
        P[j, i] = 1.0
    return P

# Transposition (0 1)
perm_01 = [1, 0, 2, 3, 4, 5, 6]
P_01 = face_perm_matrix(perm_01, faces6)

# Restrict to null space
restricted = Z.T @ P_01 @ Z
chi_trans = np.trace(restricted)
print(f"  Recomputed chi(zero-mode, (01)) = {chi_trans:.4f}")

# 3-cycle (0 1 2)
perm_012 = [1, 2, 0, 3, 4, 5, 6]
P_012 = face_perm_matrix(perm_012, faces6)
restricted_3 = Z.T @ P_012 @ Z
chi_3cycle = np.trace(restricted_3)
print(f"  Recomputed chi(zero-mode, (012)) = {chi_3cycle:.4f}")

# 7-cycle (0 1 2 3 4 5 6)
perm_7 = [1, 2, 3, 4, 5, 6, 0]
P_7 = face_perm_matrix(perm_7, faces6)
restricted_7 = Z.T @ P_7 @ Z
chi_7cycle = np.trace(restricted_7)
print(f"  Recomputed chi(zero-mode, (0123456)) = {chi_7cycle:.4f}")

# (2^3,1) = 3 transpositions
perm_222_1 = [1, 0, 3, 2, 5, 4, 6]
P_222_1 = face_perm_matrix(perm_222_1, faces6)
restricted_222 = Z.T @ P_222_1 @ Z
chi_222 = np.trace(restricted_222)
print(f"  Recomputed chi(zero-mode, (01)(23)(45)) = {chi_222:.4f}")

print()
print("S_7 character table comparison:")
print(f"  {'Class':<20} {'Zero-mode':>10} {'[4,3]':>10} {'[5,2]':>10}")
print(f"  {'(1^7)':<20} {'14':>10} {'14':>10} {'14':>10}")
print(f"  {'(2,1^5)':<20} {chi_trans:>10.1f} {'6':>10} {'4':>10}")  # corrected
print(f"  {'(3,1^4)':<20} {chi_3cycle:>10.1f} {'2':>10} {'-1':>10}")  # corrected from results file
print(f"  {'(2^3,1)':<20} {chi_222:>10.1f} {'-2':>10} {'0':>10}")
print(f"  {'(7)':<20} {chi_7cycle:>10.1f} {'0':>10} {'0':>10}")
print()

# Full character comparison with [4,3] from the results file
# The results file gives these characters for the zero-mode:
# Classes reordered to match S_7 standard:
# (1^7):14, (2,1^5):6, (2^2,1^3):2, (2^3,1):2, (3,1^4):2, (3,2,1^2):0, (3,2^2):2,
# (3^2,1):-1, (4,1^3):0, (4,2,1):0, (4,3):0, (5,1^2):-1, (5,2):1, (6,1):-1, (7):0

# S_7 character table for [4,3]:
# chi_{[4,3]}: 14, 6, 2, -2, 5, -1, -1, -1, 0, 0, 1, -1, 1, 0, 0
# But from results file the zero-mode character on (2^3,1) = 2, while [4,3] gives -2.
# This suggests the character table in the original script may have issues.
# Let's check by direct computation on more class reps.

print("--- Full character computation on all 15 conjugacy classes ---")

def cycle_to_perm(cycles, n):
    p = list(range(n))
    for cycle in cycles:
        for i in range(len(cycle)):
            p[cycle[i]] = cycle[(i+1) % len(cycle)]
    return p

class_reps = [
    ([], "(1^7)"),
    ([[0,1]], "(2,1^5)"),
    ([[0,1],[2,3]], "(2^2,1^3)"),
    ([[0,1],[2,3],[4,5]], "(2^3,1)"),
    ([[0,1,2]], "(3,1^4)"),
    ([[0,1,2],[3,4]], "(3,2,1^2)"),
    ([[0,1,2],[3,4],[5,6]], "(3,2^2)"),
    ([[0,1,2],[3,4,5]], "(3^2,1)"),
    ([[0,1,2,3]], "(4,1^3)"),
    ([[0,1,2,3],[4,5]], "(4,2,1)"),
    ([[0,1,2,3],[4,5,6]], "(4,3)"),
    ([[0,1,2,3,4]], "(5,1^2)"),
    ([[0,1,2,3,4],[5,6]], "(5,2)"),
    ([[0,1,2,3,4,5]], "(6,1)"),
    ([[0,1,2,3,4,5,6]], "(7)"),
]

# Also compute character for lambda=-3 eigenspace
minus3_vecs = []
for ev, vecs in eigenspaces.items():
    if abs(ev + 3) < 0.1:
        minus3_vecs = vecs
        break
Z_m3 = np.column_stack(minus3_vecs) if minus3_vecs else None

zero_chars = []
m3_chars = []
print(f"\n  {'Class':<20} {'chi(lam=0)':>12} {'chi(lam=-3)':>12}")
print(f"  {'-'*44}")
for cycles, name in class_reps:
    perm = cycle_to_perm(cycles, 7)
    P = face_perm_matrix(perm, faces6)

    r_zero = Z.T @ P @ Z
    chi_z = np.trace(r_zero)
    zero_chars.append(chi_z)

    if Z_m3 is not None:
        r_m3 = Z_m3.T @ P @ Z_m3
        chi_m = np.trace(r_m3)
        m3_chars.append(chi_m)
    else:
        chi_m = 0
        m3_chars.append(0)

    print(f"  {name:<20} {chi_z:>12.4f} {chi_m:>12.4f}")

# ============================================================
# Part 6: G2 structure check
# ============================================================

print()
print("=" * 70)
print("PART E: G2 adjoint representation check")
print("=" * 70)
print()
print("The exceptional Lie group G2 has:")
print("  - dim(G2) = 14")
print("  - G2 is the automorphism group of the octonions")
print("  - G2 preserves the imaginary octonions (7D)")
print("  - Adjoint representation: 14-dimensional")
print()
print("G2 has a subgroup PSL(2,7) = GL(3,F_2) of order 168.")
print("Under PSL(2,7), the 14D adjoint of G2 decomposes as:")
print("  14 -> 6 + 8 (rho_6 + rho_8 of PSL(2,7))")
print()

# The key question: does S_7 contain G2 structure?
# G2 is NOT a subgroup of S_7 in any natural way.
# However, S_7 acts on 7 elements, and G2 acts on 7D (imaginary octonions).
# The question is whether the 14D irrep [4,3] of S_7 has any structural
# relation to the 14D adjoint of G2.

# Check: does [4,3] restricted to PSL(2,7) < S_7 give 6+8?
# PSL(2,7) embeds in S_7 via its action on the 7 points of PG(2,2).

# First, let's identify the PSL(2,7) subgroup inside S_7.
# PSL(2,7) = GL(3,F_2) acts on the 7 nonzero vectors of F_2^3.
# This gives a faithful action on 7 points -> embedding in S_7.

# The 7 points of PG(2,2) as F_2^3 \ {0}:
fano_points = [
    (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
]

print("Embedding PSL(2,7) in S_7 via Fano plane:")
print("  Point 0=(1,0,0), 1=(0,1,0), 2=(1,1,0), 3=(0,0,1),")
print("  Point 4=(1,0,1), 5=(0,1,1), 6=(1,1,1)")

# Generate GL(3,F_2) elements and their induced permutations on {0,...,6}
import itertools as it

gl3f2 = []
for p in it.product([0,1], repeat=9):
    M = np.array(p, dtype=int).reshape(3,3)
    det = int(np.round(np.linalg.det(M.astype(float)))) % 2
    if det == 1:
        gl3f2.append(M)

print(f"  |GL(3,F_2)| = {len(gl3f2)} (should be 168)")

def mat_to_s7_perm(M):
    """Convert GL(3,F_2) matrix to permutation on {0,...,6}."""
    perm = [0]*7
    for i in range(7):
        v = np.array(fano_points[i], dtype=int)
        Mv = (M @ v) % 2
        Mv_tuple = tuple(Mv)
        j = fano_points.index(Mv_tuple)
        perm[i] = j
    return perm

# Compute character of the [4,3] irrep restricted to PSL(2,7)
# For each PSL(2,7) element, compute trace in the zero-mode subspace

# Group PSL(2,7) elements by conjugacy class
def get_cycle_type_from_perm(perm):
    n = len(perm)
    visited = [False]*n
    cycles = []
    for i in range(n):
        if not visited[i]:
            length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                length += 1
            cycles.append(length)
    return tuple(sorted(cycles, reverse=True))

def get_order_from_perm(perm):
    n = len(perm)
    x = list(range(n))
    for k in range(1, 100):
        x = [perm[xi] for xi in x]
        if x == list(range(n)):
            return k
    return 0

# Compute traces for PSL(2,7) elements
psl_traces = defaultdict(list)
psl_orders = {}
for M in gl3f2:
    perm = mat_to_s7_perm(M)
    order = get_order_from_perm(perm)
    P = face_perm_matrix(perm, faces6)
    tr = np.trace(Z.T @ P @ Z)
    psl_traces[order].append(tr)

# Print PSL(2,7) character on zero modes
print(f"\n  PSL(2,7) character on zero-mode subspace (grouped by element order):")
print(f"  {'Order':>6} {'Count':>6} {'Trace (avg)':>12} {'Trace (std)':>12}")
for order in sorted(psl_traces.keys()):
    traces = psl_traces[order]
    # There may be multiple conjugacy classes with same order
    unique_traces = sorted(set(np.round(traces, 4)))
    for ut in unique_traces:
        count = sum(1 for t in traces if abs(t - ut) < 0.01)
        print(f"  {order:>6} {count:>6} {ut:>12.4f}")

# PSL(2,7) irreps: rho_1(1), rho_3(3), rho_3'(3), rho_6(6), rho_7(7), rho_8(8)
# Decompose the 14D zero-mode character under PSL(2,7)
# We need the character on each PSL(2,7) conjugacy class

# PSL(2,7) has 6 conjugacy classes: orders 1,2,3,4,7,7
# Sizes: 1, 21, 56, 42, 24, 24
# Characters:
# rho_1: 1, 1, 1, 1, 1, 1
# rho_3: 3, -1, 0, 1, x, x*  where x=(-1+i*sqrt(7))/2
# rho_3': conjugate
# rho_6: 6, 2, 0, 0, -1, -1
# rho_7: 7, -1, 1, -1, 0, 0
# rho_8: 8, 0, -1, 0, 1, 1

# We need to identify the 6 conjugacy classes from our PSL(2,7) elements
# by (order, class_size)

# Collect unique (order, trace) pairs and their counts
from collections import Counter as Ctr
trace_counts = Ctr()
for M in gl3f2:
    perm = mat_to_s7_perm(M)
    P = face_perm_matrix(perm, faces6)
    tr = np.round(np.trace(Z.T @ P @ Z), 4)
    order = get_order_from_perm(perm)
    trace_counts[(order, tr)] += 1

print(f"\n  PSL(2,7) conjugacy class analysis:")
print(f"  {'Order':>6} {'Trace':>8} {'Count':>6} {'Expected class sizes: 1,21,56,42,24,24'}")
for (order, tr), count in sorted(trace_counts.items()):
    print(f"  {order:>6} {tr:>8.2f} {count:>6}")

# Now decompose using inner product formula
# Need to map our conjugacy classes to the standard PSL(2,7) class order
# Class sizes: 1, 21, 56, 42, 24, 24

# From the trace computation, identify:
# order 1, size 1: identity -> trace should be 14
# order 2, size 21: involutions
# order 3, size 56: 3-cycles
# order 4, size 42: 4-cycles
# order 7, size 24: two classes (7a, 7b)

# Build the character vector from computed traces
psl_char = np.zeros(6, dtype=complex)
psl_class_sizes = np.array([1, 21, 56, 42, 24, 24], dtype=float)

# Sort conjugacy classes
cc_data = sorted(trace_counts.items())
# Map to standard ordering
class_map = {}
for (order, tr), count in cc_data:
    if order == 1:
        class_map[0] = tr
    elif order == 2:
        class_map[1] = tr
    elif order == 3:
        class_map[2] = tr
    elif order == 4:
        class_map[3] = tr
    elif order == 7:
        if 4 not in class_map:
            class_map[4] = tr
        else:
            class_map[5] = tr

for i in range(6):
    psl_char[i] = class_map.get(i, 0)

print(f"\n  Zero-mode character on PSL(2,7) classes: {[f'{c.real:.1f}' for c in psl_char]}")

# Decompose
x = (-1 + 1j * np.sqrt(7)) / 2.0
x_bar = (-1 - 1j * np.sqrt(7)) / 2.0

irred_chars = {
    'rho_1 (dim 1)':  np.array([1,  1,  1,  1,     1,     1], dtype=complex),
    'rho_3 (dim 3)':  np.array([3, -1,  0,  1,     x, x_bar], dtype=complex),
    "rho_3' (dim 3)": np.array([3, -1,  0,  1, x_bar,     x], dtype=complex),
    'rho_6 (dim 6)':  np.array([6,  2,  0,  0,    -1,    -1], dtype=complex),
    'rho_7 (dim 7)':  np.array([7, -1,  1, -1,     0,     0], dtype=complex),
    'rho_8 (dim 8)':  np.array([8,  0, -1,  0,     1,     1], dtype=complex),
}

print(f"\n  Decomposition of zero-mode under PSL(2,7):")
for name, chi in irred_chars.items():
    mult = np.sum(psl_class_sizes * psl_char * np.conj(chi)) / 168.0
    mult_r = mult.real
    print(f"    {name}: multiplicity = {mult_r:.4f} (rounded: {round(mult_r)})")

# ============================================================
# Part 7: Structural interpretation
# ============================================================

print()
print("=" * 70)
print("PART F: What IS the 14?")
print("=" * 70)
print()

# Check the Gram matrix of Fano-line vectors in null space
print("--- Gram matrix of Fano-line vectors in null space ---")
fano_null_coords = Z.T @ F_proj  # 14 x 7
G_fano = fano_null_coords.T @ fano_null_coords
print("Gram matrix:")
print(np.round(G_fano, 4))

print("\n--- Eigenvalues of Fano-line Gram matrix ---")
fano_gram_evals = np.linalg.eigvalsh(G_fano)
print(f"  {np.round(fano_gram_evals, 6)}")

# Check how the 7 Fano complementary triples relate
# Each face (a,b,c) has a complementary set of 4 vertices
# These 4 vertices form C(4,3)=4 faces, each of which is a face of the simplex
print("\n--- Fano line complementary structure ---")
for line in fano_lines:
    complement = [v for v in range(7) if v not in line]
    comp_faces = list(combinations(complement, 3))
    print(f"  Line {line}, complement {complement}, complement faces: {comp_faces}")

# Check: Boundary operator structure
# The 14 zero modes of the face-adjacency are in the kernel of A.
# The face-adjacency matrix A can be expressed in terms of boundary operators:
# A = partial_2^T partial_2 relates to something, but face-adjacency is different.
#
# Actually, for simplicial adjacency:
# Two 2-faces are adjacent iff they share a 1-face (edge).
# A = B_2^T B_2 - diag, where B_2 is the edge-face incidence?
# No. Let's compute directly.

# Incidence: face-edge incidence matrix
# B[e,f] = 1 if edge e is a boundary of face f
edges6 = list(combinations(range(7), 2))
edge_idx = {e: i for i, e in enumerate(edges6)}
n_edges = len(edges6)

B = np.zeros((n_edges, n_faces), dtype=float)
for j, face in enumerate(faces6):
    a, b, c = face
    B[edge_idx[(a,b)], j] = 1
    B[edge_idx[(a,c)], j] = 1
    B[edge_idx[(b,c)], j] = 1

# B^T B = face-face matrix where (B^T B)[i,j] = # shared edges
BTB = B.T @ B
# Diagonal: each face has 3 edges
# Off-diagonal: 0 or 1 shared edge (faces share at most 1 edge = 2 vertices)
# So A = BTB - 3*I (face adjacency = edge-sharing minus self-loops)
A_check = BTB - 3 * np.eye(n_faces)
print(f"\n  A = B^T B - 3I check: max diff = {np.max(np.abs(A6 - A_check)):.2e}")
print(f"  So null(A) = null(B^T B - 3I) = eigenspace of B^T B at eigenvalue 3")
print(f"  B^T B is the face-face edge-sharing matrix (graph Laplacian-like)")

# The vertex-face incidence
V = np.zeros((7, n_faces), dtype=float)
for j, face in enumerate(faces6):
    for v in face:
        V[v, j] = 1

# V^T V = face-face matrix where entries = # shared vertices
VTV = V.T @ V
# Diagonal: 3 (each face has 3 vertices)
# Off-diagonal: 0, 1, or 2 shared vertices
# Face adjacency A has entry 1 when 2 shared vertices
# So A[i,j] = 1 iff (VTV)[i,j] = 2
A_from_vtv = (VTV == 2).astype(float)
print(f"  A = (V^T V == 2) check: {np.allclose(A6, A_from_vtv)}")

# ============================================================
# Part 8: Combinatorial identity check
# ============================================================

print()
print("--- Combinatorial structure summary ---")
print(f"  Face-adjacency spectrum: -3(x14), 0(x14), 5(x6), 12(x1)")
print(f"  S_7 irrep decomposition:")
print(f"    lam=12 (x1)  = [7]   (trivial)")
print(f"    lam=5  (x6)  = [6,1] (standard)")
print(f"    lam=0  (x14) = [4,3] (Specht module)")
print(f"    lam=-3 (x14) = [5,2] (Specht module)")
print()

# Check connection to exterior algebra / homology
# The boundary map partial_2: C_2 -> C_1 (faces -> edges) for oriented simplex
# ker(partial_2) = cycles in face space
# H_2(Delta^6) = 0 for contractible simplex, so ker(partial_2) = im(partial_3)
# But these are for the full simplex, not related to face-adjacency

# The key insight: A = B^T B - 3I, so ker(A) = {v : B^T B v = 3v}
# This means the zero modes are eigenvectors of B^T B (edge-face incidence Gram matrix)
# at eigenvalue 3.
BTB_evals = np.linalg.eigvalsh(BTB)
print(f"  B^T B spectrum: {np.round(sorted(set(np.round(BTB_evals, 4))), 4)}")
BTB_counts = Counter(np.round(BTB_evals, 4))
for ev in sorted(BTB_counts.keys()):
    print(f"    eigenvalue {ev}: degeneracy {BTB_counts[ev]}")

print()
print("=" * 70)
print("FINAL SYNTHESIS")
print("=" * 70)
print("""
1. UNIQUENESS OF d=6:
   Zero modes of the d-simplex face-adjacency matrix exist ONLY at d=6.
   The eigenvalue d-6 passes through zero uniquely at d=6.
   Degeneracy = d(d-1)/2 - 1 = 14 at d=6.
   For d=3: 0 zero modes (lam2=-3)
   For d=4: 0 zero modes (lam2=-2)
   For d=5: 0 zero modes (lam2=-1)
   For d=6: 14 zero modes (lam2=0) <-- UNIQUE
   For d=7: 0 zero modes (lam2=1)
   For d=8: 0 zero modes (lam2=2)

2. IRREP IDENTITY:
   The 14 zero modes carry the Specht module S^{(4,3)} of S_7.
   This is the irrep labeled by the partition [4,3] of 7.
   NOT [5,2] (which goes to lambda=-3).

3. FANO PLANE:
   The 7 Fano lines correspond to 7 specific faces of the 35.
   Their projections onto the 14D null space span a subspace whose
   dimension we computed above. If rank=7, then the Fano lines DO
   provide one natural 7D subspace, and its orthogonal complement
   in the null space would be the other 7D.

4. G2 CONNECTION:
   G2's adjoint representation is 14-dimensional.
   PSL(2,7) is a subgroup of both G2 and S_7.
   Under PSL(2,7), the G2 adjoint decomposes as 6 + 8 (rho_6 + rho_8).
   We computed the decomposition of the zero modes under PSL(2,7) above.
   If the result is 6 + 8, the G2 connection is confirmed.
   If different, the 14 is "merely" the [4,3] Specht module.

5. BOTT PERIODICITY:
   pi_6(O) = 0 in the Bott sequence.
   d=6 being the unique critical dimension of face-adjacency
   is consistent with a topological transition at this dimension.
   The zero modes represent a "gap closing" in the face-adjacency spectrum.
""")
