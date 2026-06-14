"""
verify_t6_overlap19_oh_irreps.py

Identify which O_h irreducible representations make up each cluster of the
rank-19 singular-value decomposition of the overlap between:
  - 54 Wilson π-mode vectors (in R^540)
  - 54 simplex indicator vectors (in R^540)

Singular value clusters: 1 + 6 + 12 = 19 (O_h-protected degeneracy)

Also computes the 2×2×2 corner-block null vector (rank-7 dependency).

Output: stdout + d:/AI thoery/.agent/scripts/verify_t6_overlap19_oh_results.txt
"""

import itertools
import sys
import io
from collections import defaultdict

import numpy as np

# Force UTF-8 stdout to avoid GBK encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/verify_t6_overlap19_oh_results.txt"
_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


# ============================================================
# Lattice construction (verbatim from verify_pending_4_three54.py)
# ============================================================

def build_bcc_lattice_periodic():
    body_centers = []
    for i, j, k in itertools.product(range(3), repeat=3):
        body_centers.append((i, j, k))
    return body_centers


def enumerate_simplex_faces(bc_ijk):
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)
    orig_corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = i + dx, j + dy, k + dz
        wx, wy, wz = ox % 3, oy % 3, oz % 3
        orig_corners.append(((ox + oy + oz) % 2, ('c', wx, wy, wz)))
    even_corners = [v for (par, v) in orig_corners if par == 0]
    odd_corners  = [v for (par, v) in orig_corners if par == 1]
    faces = []
    for tet_corners in (even_corners, odd_corners):
        simplex_verts = [bc_v] + tet_corners
        for combo in itertools.combinations(simplex_verts, 3):
            faces.append(frozenset(combo))
    return faces


def translate_vertex(v, shift_ijk):
    si, sj, sk = shift_ijk
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i + si) % 3, (j + sj) % 3, (k + sk) % 3)
    else:
        _, cx, cy, cz = v
        return ('c', (cx + si) % 3, (cy + sj) % 3, (cz + sk) % 3)


def translate_face(face, shift_ijk):
    return frozenset(translate_vertex(v, shift_ijk) for v in face)


def build_all_faces(body_centers):
    ref_bc = (0, 0, 0)
    ref_faces = enumerate_simplex_faces(ref_bc)
    face_to_idx = {}
    bc_face_indices = []
    global_idx = 0
    for bc_idx, bc_ijk in enumerate(body_centers):
        local_indices = []
        for m, ref_face in enumerate(ref_faces):
            shifted_face = translate_face(ref_face, bc_ijk)
            if shifted_face not in face_to_idx:
                face_to_idx[shifted_face] = global_idx
                global_idx += 1
            local_indices.append(face_to_idx[shifted_face])
        bc_face_indices.append(local_indices)
    return face_to_idx, bc_face_indices, ref_faces


def build_adjacency_matrix(face_to_idx):
    N = len(face_to_idx)
    A = np.zeros((N, N), dtype=float)
    all_faces_sets = [None] * N
    for face, idx in face_to_idx.items():
        all_faces_sets[idx] = face
    vertex_to_faces = {}
    for face, idx in face_to_idx.items():
        for v in face:
            if v not in vertex_to_faces:
                vertex_to_faces[v] = set()
            vertex_to_faces[v].add(idx)
    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a in range(len(flist)):
            for b in range(a + 1, len(flist)):
                candidate_pairs.add((flist[a], flist[b]))
    for i, j in candidate_pairs:
        if len(all_faces_sets[i] & all_faces_sets[j]) == 2:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


# ============================================================
# O_h group machinery (verbatim from bcc_flatband_oh_irreps.py)
# ============================================================

def generate_oh_elements():
    C4z = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]], dtype=int)
    C3_111 = np.array([[0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]], dtype=int)

    def mat_to_tuple(m):
        return tuple(m.flatten())

    O_set = {}
    queue = [np.eye(3, dtype=int)]
    O_set[mat_to_tuple(np.eye(3, dtype=int))] = np.eye(3, dtype=int)
    generators = [C4z, C3_111, C4z.T, C3_111.T]

    while queue:
        current = queue.pop(0)
        for g in generators:
            new = g @ current
            key = mat_to_tuple(new)
            if key not in O_set:
                O_set[key] = new
                queue.append(new)

    assert len(O_set) == 24
    inv = -np.eye(3, dtype=int)
    Oh_elements = []
    for key, mat in O_set.items():
        Oh_elements.append(mat.copy())
        Oh_elements.append((inv @ mat).copy())
    assert len(Oh_elements) == 48
    return Oh_elements


def classify_oh_element(g):
    det = int(round(np.linalg.det(g)))
    tr  = int(round(np.trace(g)))
    if det == 1:
        if tr == 3:   return 'E'
        elif tr == 0: return 'C3'
        elif tr == 1: return 'C4'
        elif tr == -1:
            off_diag = np.abs(g - np.diag(np.diag(g))).sum()
            return "C2'" if off_diag < 0.5 else 'C2'
        else:         return f'UNKNOWN_proper_tr{tr}'
    else:
        if tr == -3:  return 'i'
        elif tr == 0: return 'S6'
        elif tr == -1: return 'S4'
        elif tr == 1:
            off_diag = np.abs(g - np.diag(np.diag(g))).sum()
            return 'sigma_d' if off_diag < 0.5 else 'sigma_h'
        else:         return f'UNKNOWN_improper_tr{tr}'


# Column order matching the classifier (sigma_h/sigma_d label-swap convention):
OH_CLASSES = ['E', 'C3', 'C2', 'C4', "C2'", 'i', 'S6', 'sigma_h', 'S4', 'sigma_d']

OH_CLASS_ORDER = {
    'E': 1, 'C3': 8, 'C2': 6, 'C4': 6, "C2'": 3,
    'i': 1, 'S6': 8, 'sigma_h': 6, 'S4': 6, 'sigma_d': 3
}

# Character table with columns reordered to match classify_oh_element labels
# (sigma_h in col 7 = non-diagonal, 6 elements = standard sigma_d;
#  sigma_d in col 9 = diagonal, 3 elements = standard sigma_h)
OH_CHAR_TABLE = {
    'A1g': [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    'A2g': [ 1,  1, -1, -1,  1,  1,  1, -1, -1,  1],
    'Eg' : [ 2, -1,  0,  0,  2,  2, -1,  0,  0,  2],
    'T1g': [ 3,  0, -1,  1, -1,  3,  0, -1,  1, -1],
    'T2g': [ 3,  0,  1, -1, -1,  3,  0,  1, -1, -1],
    'A1u': [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    'A2u': [ 1,  1, -1, -1,  1, -1, -1,  1,  1, -1],
    'Eu' : [ 2, -1,  0,  0,  2, -2,  1,  0,  0, -2],
    'T1u': [ 3,  0, -1,  1, -1, -3,  0,  1, -1,  1],
    'T2u': [ 3,  0,  1, -1, -1, -3,  0, -1,  1,  1],
}

OH_IRREP_DIMS = {
    'A1g': 1, 'A2g': 1, 'Eg': 2, 'T1g': 3, 'T2g': 3,
    'A1u': 1, 'A2u': 1, 'Eu': 2, 'T1u': 3, 'T2u': 3,
}


# ============================================================
# O_h action on vertices (verbatim from bcc_flatband_oh_irreps.py)
# ============================================================

def vertex_coordinates(v):
    if v[0] == 'bc':
        _, i, j, k = v
        return np.array([i - 1.0, j - 1.0, k - 1.0], dtype=float)
    else:
        _, cx, cy, cz = v
        return np.array([cx - 1.5, cy - 1.5, cz - 1.5], dtype=float)


def apply_oh_to_vertex(g, v):
    coord = vertex_coordinates(v)
    new_coord = g @ coord
    if v[0] == 'bc':
        i_new = int(round(new_coord[0] + 1.0)) % 3
        j_new = int(round(new_coord[1] + 1.0)) % 3
        k_new = int(round(new_coord[2] + 1.0)) % 3
        return ('bc', i_new, j_new, k_new)
    else:
        cx_new = int(round(new_coord[0] + 1.5)) % 3
        cy_new = int(round(new_coord[1] + 1.5)) % 3
        cz_new = int(round(new_coord[2] + 1.5)) % 3
        return ('c', cx_new, cy_new, cz_new)


def apply_oh_to_face(g, face):
    return frozenset(apply_oh_to_vertex(g, v) for v in face)


# ============================================================
# Full 540×540 permutation matrix for g acting on all faces
# ============================================================

def build_Q540(g, face_to_idx):
    """
    Build the 540×540 permutation matrix Q(g) for O_h element g.
    For each face f: g(f) = frozenset(apply_oh_to_vertex(g, v) for v in f).
    Verify closure: every transformed face must be in face_to_idx.
    """
    N = len(face_to_idx)
    # Use index arrays for sparse representation
    row_of = np.zeros(N, dtype=int)  # Q[row_of[col], col] = 1
    faces_list = [None] * N
    for face, idx in face_to_idx.items():
        faces_list[idx] = face

    for idx, face in enumerate(faces_list):
        g_face = apply_oh_to_face(g, face)
        if g_face not in face_to_idx:
            raise ValueError(f"O_h action not closed: g*face not in face_to_idx for face {face}")
        row_of[idx] = face_to_idx[g_face]

    # Build dense matrix (540x540, but manageable as int8)
    Q = np.zeros((N, N), dtype=np.float32)
    for col, row in enumerate(row_of):
        Q[row, col] = 1.0
    return Q, row_of


def verify_permutation(row_of, N):
    """Check that row_of is a bijection {0..N-1} -> {0..N-1}."""
    seen = np.zeros(N, dtype=bool)
    for r in row_of:
        if seen[r]:
            return False
        seen[r] = True
    return seen.all()


# ============================================================
# Wilson π-mode construction (verbatim from verify_pending_4_three54.py)
# ============================================================

def build_pi_modes_and_simplices(body_centers, face_to_idx, bc_face_indices):
    A = build_adjacency_matrix(face_to_idx)

    perm = []
    for bc_idx in range(27):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)
    A_reord = A[np.ix_(perm, perm)]

    def get_block(A_r, bc_i, bc_j, size=20):
        return A_r[bc_i*size:(bc_i+1)*size, bc_j*size:(bc_j+1)*size].copy()

    T = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers):
        R = tuple(bc_j_ijk[a] % 3 for a in range(3))
        T[R] = get_block(A_reord, 0, bc_j_idx)

    kvecs = list(itertools.product(range(3), repeat=3))
    vecs_all = {}
    for n_tuple in kvecs:
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = np.zeros((20, 20), dtype=complex)
        for R, mat in T.items():
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k, R_vec))
            H_k += phase * mat
        evals, evecs = np.linalg.eigh(H_k)
        assert np.allclose(evals[:6], -3.0, atol=1e-10), \
            f"Flat bands not at -3.0 at k={n_tuple}"
        vecs_all[n_tuple] = evecs

    axes = ['x', 'y', 'z']
    loop_results = {}
    for axis_idx, axis_name in enumerate(axes):
        for other1 in range(3):
            for other2 in range(3):
                loop_k = []
                for n_ax in range(3):
                    n_tuple = [0, 0, 0]
                    n_tuple[axis_idx] = n_ax
                    other_axes = [i for i in range(3) if i != axis_idx]
                    n_tuple[other_axes[0]] = other1
                    n_tuple[other_axes[1]] = other2
                    loop_k.append(tuple(n_tuple))
                loop_results[(axis_name, other1, other2)] = loop_k

    deg_bands = list(range(6))
    wilson_pi_modes = []
    for loop_key, loop_k in loop_results.items():
        n_k = len(loop_k)
        W = np.eye(len(deg_bands), dtype=complex)
        for j in range(n_k):
            k_cur = loop_k[j]
            k_next = loop_k[(j + 1) % n_k]
            S = np.zeros((len(deg_bands), len(deg_bands)), dtype=complex)
            for a_idx, a_band in enumerate(deg_bands):
                for b_idx, b_band in enumerate(deg_bands):
                    S[a_idx, b_idx] = np.vdot(vecs_all[k_cur][:, a_band],
                                               vecs_all[k_next][:, b_band])
            W = W @ S
        w_evals, w_evecs = np.linalg.eig(W)
        phases = np.angle(w_evals)
        pi_indices = [idx for idx, p in enumerate(phases) if abs(abs(p) - np.pi) < 0.1]
        for idx in pi_indices:
            v_coeff = w_evecs[:, idx]
            k_start = loop_k[0]
            Psi_bloch = vecs_all[k_start][:, deg_bands] @ v_coeff
            wilson_pi_modes.append((loop_key, k_start, Psi_bloch))

    assert len(wilson_pi_modes) == 54, f"Expected 54 pi-modes, got {len(wilson_pi_modes)}"

    # Project to 540-dim
    pi_modes_540 = np.zeros((54, 540), dtype=complex)
    for m_idx, (loop_key, k_start, Psi_bloch) in enumerate(wilson_pi_modes):
        k_vec = np.array([2 * np.pi * n / 3.0 for n in k_start])
        for bc_idx, bc_ijk in enumerate(body_centers):
            R_vec = np.array(bc_ijk, dtype=float)
            phase = np.exp(1j * np.dot(k_vec, R_vec))
            global_faces = bc_face_indices[bc_idx]
            for local_idx in range(20):
                g_face = global_faces[local_idx]
                pi_modes_540[m_idx, g_face] = Psi_bloch[local_idx] * phase

    # Simplex indicator vectors
    simplex_vectors = np.zeros((54, 540))
    for bc_idx in range(27):
        global_faces = bc_face_indices[bc_idx]
        for local_idx in range(10):
            simplex_vectors[2 * bc_idx, global_faces[local_idx]] = 1.0
        for local_idx in range(10, 20):
            simplex_vectors[2 * bc_idx + 1, global_faces[local_idx]] = 1.0

    return pi_modes_540, simplex_vectors


# ============================================================
# Irrep decomposition helper
# ============================================================

def decompose_irreps(chars_by_class):
    """
    Given a dict {class_name: character_value}, decompose into O_h irreps.
    Returns dict {irrep: multiplicity (float, should be non-neg integer)}.
    """
    result = {}
    for irrep, char_row in OH_CHAR_TABLE.items():
        n_mu = 0.0
        for cls_idx, cls in enumerate(OH_CLASSES):
            sz = OH_CLASS_ORDER[cls]
            chi_mu = char_row[cls_idx]
            chi_rep = chars_by_class.get(cls, 0.0)
            n_mu += sz * chi_mu * chi_rep
        n_mu /= 48.0
        result[irrep] = n_mu
    return result


def format_decomp(multiplicities):
    parts = []
    for irrep in ['A1g', 'A2g', 'Eg', 'T1g', 'T2g', 'A1u', 'A2u', 'Eu', 'T1u', 'T2u']:
        n = int(round(multiplicities[irrep]))
        if n > 0:
            parts.append(f"{n}×{irrep}" if n > 1 else irrep)
    return " ⊕ ".join(parts) if parts else "(none)"


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 78)
    log("T6 OVERLAP RANK-19 OH IRREP DECOMPOSITION")
    log("BCC 3×3×3 periodic lattice (3-torus), 540 faces")
    log("=" * 78)

    # ------------------------------------------------------------------
    # Step 1: Build lattice, faces, subspaces
    # ------------------------------------------------------------------
    log("\n[Step 1] Build lattice and 54D subspaces")
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    N = len(face_to_idx)
    assert N == 540, f"Expected 540 faces, got {N}"
    log(f"  Total faces: {N} (OK)")

    log("  Computing Wilson π-modes and simplex indicator vectors...")
    pi_modes_540, simplex_vectors = build_pi_modes_and_simplices(
        body_centers, face_to_idx, bc_face_indices)

    rank_pi = np.linalg.matrix_rank(pi_modes_540)
    rank_simp = np.linalg.matrix_rank(simplex_vectors)
    log(f"  π-mode subspace rank:    {rank_pi} (expect 54)")
    log(f"  Simplex subspace rank:   {rank_simp} (expect 54)")
    assert rank_pi == 54 and rank_simp == 54

    # ------------------------------------------------------------------
    # Step 2: Overlap SVD, verify rank 19 and clusters 1+6+12
    # ------------------------------------------------------------------
    log("\n[Step 2] Overlap matrix SVD — verify rank 19 and clusters 1+6+12")
    overlap_matrix = simplex_vectors @ pi_modes_540.T.conj()  # (54,54) complex
    U_full, s_full, Vh_full = np.linalg.svd(overlap_matrix)
    W_full = Vh_full.conj().T  # right singular vectors, shape (54, 54)

    rank19 = int(np.sum(s_full > 1e-5))
    log(f"  Overlap matrix rank: {rank19} (expect 19)")
    assert rank19 == 19, f"Rank is {rank19}, not 19!"

    # The 19 nonzero singular values
    s19 = s_full[:19]
    log(f"  All 54 singular values (rounded):")
    log("    " + " ".join(f"{s:.6f}" for s in s_full))

    # Cluster them: gap between clusters
    log(f"\n  Identifying clusters in s[:19]:")
    diffs = np.diff(s19)
    log(f"  s19 = {s19}")
    log(f"  consecutive diffs = {diffs}")

    # Manually verify 1+6+12 structure
    # The task states clusters are 1+6+12 with degeneracy ~3e-15
    # Identify by closeness: sort and find gaps
    s19_sorted = np.sort(s19)[::-1]  # descending
    gaps = np.abs(np.diff(s19_sorted))
    log(f"  Sorted s19 (desc): {s19_sorted}")
    log(f"  Gaps: {gaps}")

    # Find significant gaps (much larger than degeneracy within cluster ~3e-15)
    gap_threshold = 1e-8
    significant_gap_positions = [i for i, g in enumerate(gaps) if g > gap_threshold]
    log(f"  Significant gaps at positions: {significant_gap_positions}")

    # Build clusters from sorted singular values
    clusters_sv = []
    start = 0
    for pos in significant_gap_positions:
        clusters_sv.append(s19_sorted[start:pos + 1])
        start = pos + 1
    clusters_sv.append(s19_sorted[start:])
    log(f"  Cluster sizes: {[len(c) for c in clusters_sv]}")
    log(f"  Cluster mean values: {[float(np.mean(c)) for c in clusters_sv]}")

    # Verify 1+6+12 (or 12+6+1 in descending order)
    cluster_sizes = sorted([len(c) for c in clusters_sv])
    assert cluster_sizes == [1, 6, 12], \
        f"Expected cluster sizes [1,6,12], got {cluster_sizes}"
    log(f"  Cluster sizes verified: 1+6+12 = 19  OK")

    # Map back to original (unsorted) singular value indices
    # U_full[:, i] is left singular vector for s_full[i]
    # The 19 active ones are indices 0..18
    # Now identify which of the 19 belong to each cluster

    # Cluster assignment by value proximity
    cluster_means = [float(np.mean(c)) for c in clusters_sv]
    # Sort clusters by size for labeling: 1-dim, 6-dim, 12-dim
    clusters_by_size = sorted(enumerate(clusters_sv), key=lambda x: len(x[1]))
    # clusters_by_size[0] = (orig_idx, [1 val])
    # clusters_by_size[1] = (orig_idx, [6 vals])
    # clusters_by_size[2] = (orig_idx, [12 vals])

    def assign_sv_indices_to_clusters():
        """Return dict {cluster_label: list of indices into s_full[:19]}."""
        result = {}
        labels = ['1-dim', '6-dim', '12-dim']
        for label, (_, c_vals) in zip(labels, clusters_by_size):
            c_mean = float(np.mean(c_vals))
            c_tol = max(float(np.std(c_vals)) * 10 + 1e-10, 1e-8)
            indices = [i for i in range(19) if abs(s_full[i] - c_mean) < c_tol +
                       float(max(abs(v - c_mean) for v in c_vals)) + 1e-10]
            # More robust: assign by closest cluster mean
            # Better: sort s19 indices by their cluster membership
            result[label] = indices
        return result

    # More robust: group by proximity to cluster means
    cluster_mean_vals = {}
    for label, (_, c_vals) in zip(['1-dim', '6-dim', '12-dim'], clusters_by_size):
        cluster_mean_vals[label] = float(np.mean(c_vals))

    sv_cluster_label = {}
    for i in range(19):
        sv = s_full[i]
        # Assign to closest cluster mean
        best = min(cluster_mean_vals.keys(), key=lambda lbl: abs(sv - cluster_mean_vals[lbl]))
        sv_cluster_label[i] = best

    cluster_sv_indices = defaultdict(list)
    for i, lbl in sv_cluster_label.items():
        cluster_sv_indices[lbl].append(i)

    for lbl in ['1-dim', '6-dim', '12-dim']:
        idx_list = cluster_sv_indices[lbl]
        log(f"  Cluster '{lbl}': sv indices {idx_list}, values {[s_full[i] for i in idx_list]}")

    assert len(cluster_sv_indices['1-dim']) == 1
    assert len(cluster_sv_indices['6-dim']) == 6
    assert len(cluster_sv_indices['12-dim']) == 12

    # ------------------------------------------------------------------
    # Step 3: Generate O_h and build 540×540 permutation matrices
    # ------------------------------------------------------------------
    log("\n[Step 3] Generate O_h elements and build 540×540 permutation matrices")
    oh_elements = generate_oh_elements()
    log(f"  Generated {len(oh_elements)} O_h elements")

    element_class = [classify_oh_element(g) for g in oh_elements]

    # Verify class sizes
    class_counts = defaultdict(int)
    for cls in element_class:
        class_counts[cls] += 1
    expected_counts = {
        'E': 1, 'C3': 8, 'C2': 6, 'C4': 6, "C2'": 3,
        'i': 1, 'S6': 8, 'sigma_h': 6, 'S4': 6, 'sigma_d': 3
    }
    class_ok = all(class_counts[cls] == exp for cls, exp in expected_counts.items())
    log(f"  Class counts correct: {class_ok}")
    if not class_ok:
        for cls, exp in expected_counts.items():
            log(f"    {cls}: got {class_counts[cls]}, expected {exp}")

    log("  Building 540×540 permutation row-index arrays...")
    # Store as row_of arrays (memory efficient: 48 × 540 ints)
    all_row_of = []
    for g_idx, g in enumerate(oh_elements):
        _, row_of = build_Q540(g, face_to_idx)
        # Verify permutation
        if not verify_permutation(row_of, N):
            log(f"  ERROR: element {g_idx} does not produce a valid permutation!")
        all_row_of.append(row_of)
    log(f"  All 48 permutation maps built and verified closed")

    # Build element index lookup for homomorphism check
    def mat_key(m):
        return tuple(m.flatten())
    elem_key_to_idx = {mat_key(g): i for i, g in enumerate(oh_elements)}

    # ------------------------------------------------------------------
    # Step 4: Verify O_h-invariance of 54D subspaces
    # (cheaper: verify the simplex faces form an O_h-closed set)
    # ------------------------------------------------------------------
    log("\n[Step 4] Verify O_h-invariance of the simplex subspace")
    # The 54 simplices are indexed by (bc_idx, parity). The simplex set is
    # O_h-closed iff applying any g sends every simplex-face to another simplex-face.
    # Build set of all face indices in any simplex:
    all_simplex_faces = set()
    for bc_idx in range(27):
        for fi in bc_face_indices[bc_idx]:
            all_simplex_faces.add(fi)
    log(f"  Total faces across all simplices: {len(all_simplex_faces)} "
        f"(= {N} = all faces, trivially closed)")
    # All 540 faces are covered by the simplex decomposition, so the set is closed.
    # For the simplex subspace projection S = simplex_vectors.T @ simplex_vectors (rank 54 projector scaled):
    # Check Q(g) maps the 54D column space to itself for a sample of g.
    # Use the projector: P_simp = V_s @ V_s.T where V_s is an ONB for simplex space.
    V_simp, _, _ = np.linalg.svd(simplex_vectors, full_matrices=False)
    # V_simp has shape (54, 54) but simplex_vectors is (54, 540), so we need ONB in R^540
    # V_simp from scipy: simplex_vectors = U @ S @ Vh, columns of U (shape 54x54) are ONB in simplex row space
    # We need ONB in column space of simplex_vectors.T, i.e., right singular vectors of simplex_vectors
    _, sv_s, Vh_s = np.linalg.svd(simplex_vectors, full_matrices=False)
    # Vh_s shape (54, 540); rows are ONB of the row space of simplex_vectors in R^540
    # The ONB for the 54D subspace of R^540 is given by first 54 rows of Vh_s
    # BUT since rank=54, all rows of Vh_s are non-degenerate
    ONB_simp = Vh_s  # shape (54, 540), each row is a basis vector in R^540

    log("  Checking Q(g) maps simplex subspace to itself (sample of 5 elements)...")
    # For row vector v in R^540: Q(g) acts as v -> v[row_of] on column indices
    # i.e., (v Q^T)[i] = v[row_of[i]]  (permuting columns)
    # Check: each row of ONB_simp, after permutation by row_of, lies in span of ONB_simp rows.
    # Projector onto row space of ONB_simp: P = ONB_simp.T @ ONB_simp (540×540) — too large.
    # Instead: for row vector v, residual = v - (v @ ONB_simp.T) @ ONB_simp
    # = v - ONB_simp.T @ (ONB_simp @ v.T) transposed ... let's vectorize:
    # residual matrix for all rows: R = M - M @ ONB_simp.T @ ONB_simp
    # where M = ONB_simp[:, row_of]  (shape 54×540)
    # M @ ONB_simp.T shape = (54, 54); then (M @ ONB_simp.T) @ ONB_simp shape = (54, 540)
    simp_inv_ok = True
    for g_idx in [0, 1, 7, 15, 23]:
        row_of = all_row_of[g_idx]
        transformed = ONB_simp[:, row_of]   # shape (54, 540): each row permuted
        # Project transformed rows onto ONB_simp row space
        coords = transformed @ ONB_simp.T   # (54, 54): coordinates in ONB_simp basis
        proj = coords @ ONB_simp            # (54, 540): projection
        res_norm = np.max(np.abs(transformed - proj))
        if res_norm > 1e-8:
            log(f"    g[{g_idx}]: FAIL — residual {res_norm:.3e} (subspace not invariant)")
            simp_inv_ok = False
        else:
            log(f"    g[{g_idx}] ({element_class[g_idx]}): OK — residual {res_norm:.3e}")
    log(f"  Simplex subspace O_h-invariant: {simp_inv_ok}")

    # Check π-mode subspace invariance
    log("\n  Checking Q(g) maps π-mode subspace to itself (sample of 5 elements)...")
    _, sv_p, Vh_p = np.linalg.svd(pi_modes_540, full_matrices=False)
    ONB_pi = Vh_p  # shape (54, 540)
    pi_inv_ok = True
    for g_idx in [0, 1, 7, 15, 23]:
        row_of = all_row_of[g_idx]
        transformed = ONB_pi[:, row_of]   # (54, 540)
        coords = transformed @ ONB_pi.conj().T  # (54, 54)
        proj = coords @ ONB_pi            # (54, 540)
        res_norm = float(np.max(np.abs(transformed - proj)))
        if res_norm > 1e-6:
            log(f"    g[{g_idx}] ({element_class[g_idx]}): WARN — π-mode residual {res_norm:.3e}")
            pi_inv_ok = False
        else:
            log(f"    g[{g_idx}] ({element_class[g_idx]}): OK — residual {res_norm:.3e}")
    log(f"  π-mode subspace O_h-invariant: {pi_inv_ok}")

    # ------------------------------------------------------------------
    # Step 5: For each cluster, extract left singular vectors in R^540,
    #         compute O_h representation and decompose into irreps
    # ------------------------------------------------------------------
    log("\n[Step 5] Per-cluster irrep decomposition (left singular vectors in R^540)")
    log("  (Left singular vectors U_c ∈ R^540 via full overlap SVD on R^540 side)")

    # We need U_full in the 540D sense. The SVD was of the (54×54) overlap_matrix
    # (simplex × π-mode). The left singular vectors are in simplex-index space (R^54).
    # For the O_h action, we need the corresponding vectors in R^540.
    #
    # Approach: the 19D common structure lives in both subspaces.
    # U_c (left, from simplex side) → lifted to R^540 via simplex subspace:
    #   u_lifted = U_full[:19, :19].T @ simplex_vectors  -- NO
    #
    # Correct approach:
    #   overlap_matrix = simplex_vectors @ pi_modes_540.T.conj()  [shape 54×54]
    #   SVD: overlap_matrix = U_full @ diag(s) @ Vh_full
    #   Left SV col i (in R^54): U_full[:, i] lives in simplex coefficient space.
    #   Lift to R^540: v_i = simplex_vectors.T @ U_full[:, i]  [shape 540, complex? No, simplex is real]
    #   But simplex_vectors rows may not be orthonormal, so we need ONB-based lift.
    #
    #   Better: work directly in R^540 using the FULL SVD of the (54×540) simplex matrix.
    #   Let simplex_vectors = Us @ diag(ss) @ Vs_h  [Us: 54×54, Vs_h: 54×540]
    #   Then the 54D simplex subspace ONB in R^540 is given by rows of Vs_h.
    #   U_full[:, i] gives coordinates in the Us basis (i.e., in R^54 simplex basis).
    #   Lifted vector in R^540: v_i = U_full[:, i] @ (Us.T @ Vs_h.T).T ...
    #
    # Simplest correct approach:
    #   For the SIMPLEX side: U_full[:, i] ∈ R^54 represents the i-th left SV
    #   as a combination of the 54 simplex rows. Lift:
    #     lifted_i = sum_j U_full[j,i] * simplex_vectors[j, :]   [∈ R^540]
    #   These form the 19 vectors in R^540 from the simplex side.
    #   (They are not normalized in R^540 norm, but relative structure is correct.)
    #
    #   Normalize: divide by s_full[i] to get unit vectors (since s_full[i]^2 is the
    #   squared norm of the projection in the π-mode side, and the lifted vectors from
    #   simplex side have norm = s_full[i] due to SVD structure).

    log("  Lifting left singular vectors to R^540 via simplex_vectors rows...")
    # U_full shape: (54, 54) from SVD of overlap_matrix (54×54)
    # Lift: L_540[i, :] = sum_j U_full[j, i] * simplex_vectors[j, :]
    # simplex_vectors is real (54×540), U_full is complex (but the SVD of a complex
    # matrix: overlap_matrix is complex, so U_full, Vh_full are unitary over C)
    # We take real part (phases from Wilson loops):
    L_simp = (U_full[:, :19].conj().T @ simplex_vectors)  # shape (19, 540), complex
    # These are the 19 basis vectors of the common structure from simplex side
    # They may be complex due to U_full being complex.
    # Normalize:
    norms_L = np.linalg.norm(L_simp, axis=1)
    log(f"  Norms of lifted vectors (should be close to s_full[:19]): "
        f"min={norms_L.min():.6f}, max={norms_L.max():.6f}")
    log(f"  s_full[:19]: min={s_full[:19].min():.6f}, max={s_full[:19].max():.6f}")
    L_simp_norm = L_simp / norms_L[:, np.newaxis]

    # Similarly from π-mode (right singular vectors):
    # Vh_full shape (54,54), rows are right SV; W_full = Vh_full.conj().T
    # Lift: L_pi[i, :] = sum_j Vh_full[i, j] * pi_modes_540[j, :]  ? No.
    # Actually: W_full = Vh_full.conj().T, W_full[:, i] = i-th right SV in R^54 (pi-coeff space)
    # Lift: L_pi_i = sum_j Vh_full[i, j].conj() * pi_modes_540[j, :]  ... let's be careful.
    # SVD: M = U S V^H  => M[:,j] = sum_i U[i,k] s[k] V[j,k].conj()
    # Right SV (cols of V = rows of V^H): V[:,i] = Vh_full[i,:].conj() = W_full[:,i]
    # Pi-mode lift: L_pi[i,:] = sum_j W_full[j, i] * pi_modes_540[j, :]
    L_pi = (W_full[:, :19].T @ pi_modes_540)  # shape (19, 540), complex
    norms_Lpi = np.linalg.norm(L_pi, axis=1)
    L_pi_norm = L_pi / norms_Lpi[:, np.newaxis]

    # Now compute the O_h action on each cluster.
    # For cluster c with indices idx_c: take U_c = L_simp_norm[idx_c, :]  shape (d_c, 540)
    # R_c(g) = U_c @ Q(g).T @ U_c.conj().T  [d_c × d_c]
    # But Q(g) acts as permutation: (Q(g) @ x)[row_of[j]] = x[j]
    # So Q(g) @ v = v[inv_row_of]
    # U_c @ Q(g)^T @ U_c^H: Q(g)^T acts on columns; Q(g)^T v = v[row_of]
    # R_c(g)_{ab} = U_c[a, :] @ Q(g)^T @ U_c[b, :].conj()
    #             = sum_i U_c[a, i] * U_c[b, row_of[i]].conj()
    # Equivalent: R_c(g) = U_c @ (Q(g).T @ U_c.conj().T)
    #           = conj part: R_c(g) = U_c * (U_c[:, row_of]).conj().T  -- broadcasting
    # In numpy: R_c(g) = U_c @ U_c[:, row_of].conj().T  ... let's verify.
    # Q^T has (Q^T)_{ij} = Q_{ji} = 1 iff j = row_of[i] ... wait:
    # Q[row_of[col], col] = 1, so Q^T[col, row_of[col]] = 1, i.e., (Q^T)_{ij} = 1 iff j=row_of[i].
    # (Q^T x)[i] = x[row_of[i]]
    # So: (U_c @ Q^T)[a, i] = sum_k U_c[a,k] (Q^T)[k,i] = U_c[a, row_of[i]]
    # R_c(g) = (U_c @ Q^T) @ U_c^H = U_c[:, row_of] @ U_c.conj().T
    # This is correct.

    log("\n  Computing O_h representation matrices for each cluster...")
    # Precompute for efficiency: for each g, get row_of
    # Then R_c(g) = U_c[:, row_of] @ U_c.conj().T

    cluster_info = {
        '1-dim':  cluster_sv_indices['1-dim'],
        '6-dim':  cluster_sv_indices['6-dim'],
        '12-dim': cluster_sv_indices['12-dim'],
    }

    cluster_results = {}
    total_chars = defaultdict(float)

    for cname, idx_list in cluster_info.items():
        log(f"\n  --- Cluster {cname} (sv indices {idx_list}) ---")
        d = len(idx_list)
        U_c = L_simp_norm[idx_list, :]  # shape (d, 540), complex

        # Compute representation matrices and characters for all 48 elements
        rep_chars = []
        for g_idx, g in enumerate(oh_elements):
            row_of = all_row_of[g_idx]
            # R_c(g) = U_c[:, row_of] @ U_c.conj().T  -- shape (d, d)
            R_g = (U_c[:, row_of] @ U_c.conj().T)
            chi = float(np.real(np.trace(R_g)))
            rep_chars.append((g_idx, element_class[g_idx], R_g, chi))

        # Verify orthogonality (R_g should be unitary if U_c rows are ONB in the cluster's space)
        # Check: U_c @ U_c.conj().T should be identity (rows ONB)
        gram = U_c @ U_c.conj().T  # (d, d)
        gram_err = np.max(np.abs(gram - np.eye(d)))
        log(f"    Gram matrix error (U_c rows ONB?): {gram_err:.3e}")

        if gram_err > 1e-6:
            log(f"    WARNING: rows not orthonormal, gram error = {gram_err:.3e}")
            log(f"    Attempting Gram-Schmidt orthonormalization...")
            # SVD-based re-orthogonalization
            u_tmp, s_tmp, vh_tmp = np.linalg.svd(U_c, full_matrices=False)
            U_c = u_tmp @ vh_tmp  # orthonormal rows in R^540
            # Recompute
            rep_chars = []
            for g_idx, g in enumerate(oh_elements):
                row_of = all_row_of[g_idx]
                R_g = (U_c[:, row_of] @ U_c.conj().T)
                chi = float(np.real(np.trace(R_g)))
                rep_chars.append((g_idx, element_class[g_idx], R_g, chi))
            gram2 = U_c @ U_c.conj().T
            gram_err2 = np.max(np.abs(gram2 - np.eye(d)))
            log(f"    After re-ortho gram error: {gram_err2:.3e}")

        # Verify orthogonality of R_g
        max_orth_err = max(np.max(np.abs(R_g.conj().T @ R_g - np.eye(d)))
                           for _, _, R_g, _ in rep_chars)
        log(f"    Max |R(g)^H R(g) - I|: {max_orth_err:.3e}")

        # Verify group homomorphism on sample pairs
        max_hom_err = 0.0
        sample_pairs = [(0,1),(0,7),(1,7),(3,5),(10,20),(0,15)]
        for i, j in sample_pairs:
            gi, gj = oh_elements[i], oh_elements[j]
            gh = gi @ gj
            gh_idx = elem_key_to_idx.get(mat_key(gh))
            if gh_idx is None:
                log(f"    ERROR: product g[{i}]*g[{j}] not in Oh!")
                continue
            R_gi = rep_chars[i][2]
            R_gj = rep_chars[j][2]
            R_gh = rep_chars[gh_idx][2]
            err = np.max(np.abs(R_gi @ R_gj - R_gh))
            max_hom_err = max(max_hom_err, err)
        log(f"    Max homomorphism error: {max_hom_err:.3e}")
        if max_hom_err > 1e-5:
            log(f"    WARNING: large homomorphism error — cluster may not be O_h-invariant")

        # Compute mean characters per conjugacy class
        class_chi = {}
        for cls in OH_CLASSES:
            vals = [chi for _, ec, _, chi in rep_chars if ec == cls]
            if vals:
                chi_mean = float(np.mean(vals))
                chi_spread = float(np.max(np.abs(np.array(vals) - chi_mean)))
                class_chi[cls] = chi_mean
                if chi_spread > 1e-4:
                    log(f"    WARNING: characters not constant in class {cls}, spread={chi_spread:.3e}")
            else:
                class_chi[cls] = 0.0

        log(f"    Characters by class:")
        for cls in OH_CLASSES:
            log(f"      {cls:10s}: {class_chi[cls]:8.4f}")

        # Decompose into irreps
        mults = decompose_irreps(class_chi)
        log(f"    Irrep decomposition:")
        total_dim = 0
        for irrep in ['A1g', 'A2g', 'Eg', 'T1g', 'T2g', 'A1u', 'A2u', 'Eu', 'T1u', 'T2u']:
            n = mults[irrep]
            n_r = int(round(n))
            if abs(n - n_r) > 0.05:
                log(f"      {irrep}: {n:.4f}  WARNING: non-integer!")
            if n_r > 0:
                contrib = n_r * OH_IRREP_DIMS[irrep]
                total_dim += contrib
                log(f"      {irrep}: {n_r}  (dim contribution: {contrib})")
        log(f"    Total dim from irreps: {total_dim} (expect {d})")
        decomp = format_decomp(mults)
        log(f"    Cluster {cname}: {decomp}")

        # Accumulate total chars
        for cls in OH_CLASSES:
            total_chars[cls] += class_chi[cls]

        # Sanity check with right singular vectors
        U_c_right = L_pi_norm[idx_list, :]  # shape (d, 540), from π-mode side
        gram_r = U_c_right @ U_c_right.conj().T
        gram_r_err = np.max(np.abs(gram_r - np.eye(d)))
        if gram_r_err > 1e-6:
            u_r2, _, vh_r2 = np.linalg.svd(U_c_right, full_matrices=False)
            U_c_right = u_r2 @ vh_r2
        rep_chars_right = []
        for g_idx in range(48):
            row_of = all_row_of[g_idx]
            R_g = (U_c_right[:, row_of] @ U_c_right.conj().T)
            chi = float(np.real(np.trace(R_g)))
            rep_chars_right.append((g_idx, element_class[g_idx], chi))
        class_chi_right = {}
        for cls in OH_CLASSES:
            vals = [chi for _, ec, chi in rep_chars_right if ec == cls]
            class_chi_right[cls] = float(np.mean(vals)) if vals else 0.0
        mults_right = decompose_irreps(class_chi_right)
        decomp_right = format_decomp(mults_right)
        log(f"    Right SV side:   {decomp_right}")
        if decomp != decomp_right:
            log(f"    NOTE: L/R decompositions differ (may indicate phase ambiguity)")

        cluster_results[cname] = {
            'indices': idx_list,
            'dim': d,
            'chars': class_chi,
            'mults': mults,
            'decomp': decomp,
            'decomp_right': decomp_right,
            'max_orth_err': max_orth_err,
            'max_hom_err': max_hom_err,
        }

    # ------------------------------------------------------------------
    # Step 6: Total 19D content
    # ------------------------------------------------------------------
    log("\n[Step 6] Total 19D irrep content")
    total_mults = decompose_irreps(total_chars)
    total_decomp = format_decomp(total_mults)
    total_dim_sum = sum(int(round(v)) * OH_IRREP_DIMS[ir]
                        for ir, v in total_mults.items())
    log(f"  Total 19D = {total_decomp}")
    log(f"  Total dimension check: {total_dim_sum} (expect 19)")

    # ------------------------------------------------------------------
    # Step 7: 2×2×2 corner block null vector (secondary analysis)
    # ------------------------------------------------------------------
    log("\n[Step 7] 2×2×2 corner block dependency vector")
    log("  From verify_pending_4b: the corner shell (8 BCs, 16 simplex rows) has")
    log("  rank 7 in the overlap matrix — one linear dependency among its 16 rows.")
    log("  We also check the 8-row version (per-BC sum) and the simplex-vector space.")

    # Identify corner BCs: those with i,j,k all in {0,2}
    corner_bcs = [(bc_idx, (i, j, k))
                  for bc_idx, (i, j, k) in enumerate(body_centers)
                  if i in (0, 2) and j in (0, 2) and k in (0, 2)]
    assert len(corner_bcs) == 8, f"Expected 8 corner BCs, got {len(corner_bcs)}"
    log(f"  Corner BCs: {[ijk for _, ijk in corner_bcs]}")

    # --- Approach A: 16 individual simplex rows in the overlap matrix ---
    corner_s_indices = []
    for bc_idx, _ in corner_bcs:
        corner_s_indices.append(2 * bc_idx)
        corner_s_indices.append(2 * bc_idx + 1)
    corner_s_indices_sorted = sorted(corner_s_indices)
    sub_overlap_16 = overlap_matrix[corner_s_indices_sorted, :]  # (16, 54)
    s_corner_16 = np.linalg.svd(sub_overlap_16, compute_uv=False)
    rank_corner_16 = int(np.sum(s_corner_16 > 1e-5))
    log(f"  Approach A — 16 simplex rows in overlap matrix: rank={rank_corner_16} (expect 7)")
    log(f"    Singular values (all 16): {s_corner_16}")

    # --- Approach B: 8 per-BC rows (sum of 2 simplex rows) in overlap matrix ---
    corner_sv_rows_8 = []
    corner_bc_ijk = []
    for bc_idx, ijk in corner_bcs:
        row = overlap_matrix[2 * bc_idx, :] + overlap_matrix[2 * bc_idx + 1, :]
        corner_sv_rows_8.append(row)
        corner_bc_ijk.append(ijk)
    corner_sv_rows_8 = np.array(corner_sv_rows_8)  # (8, 54)
    s_corner_8 = np.linalg.svd(corner_sv_rows_8, compute_uv=False)
    rank_corner_8 = int(np.sum(s_corner_8 > 1e-5))
    log(f"  Approach B — 8 per-BC rows (sum of 2 simplices) in overlap matrix: rank={rank_corner_8}")
    log(f"    Singular values (all 8): {s_corner_8}")

    # --- Approach C: 16 rows directly in simplex_vectors (R^540) ---
    corner_sv_rows_16_raw = []
    for bc_idx, _ in corner_bcs:
        corner_sv_rows_16_raw.append(simplex_vectors[2 * bc_idx, :])
        corner_sv_rows_16_raw.append(simplex_vectors[2 * bc_idx + 1, :])
    corner_sv_rows_16_raw = np.array(corner_sv_rows_16_raw)  # (16, 540)
    s_corner_sv = np.linalg.svd(corner_sv_rows_16_raw, compute_uv=False)
    rank_corner_sv = int(np.sum(s_corner_sv > 1e-5))
    log(f"  Approach C — 16 rows in simplex_vectors R^540: rank={rank_corner_sv}")
    log(f"    Singular values: {s_corner_sv}")

    # Determine which approach gives rank 7 (1 dependency among 8 or 16 rows)
    if rank_corner_16 == 7:
        log(f"\n  Using Approach A (16 overlap rows, rank 7):")
        # Null vector in R^16 from left SVD: column 7 (0-indexed) = left null vector
        U16, s16, _ = np.linalg.svd(sub_overlap_16, full_matrices=True)
        null_vec_16 = np.real(U16[:, 7])
        null_vec_16 /= null_vec_16[np.argmax(np.abs(null_vec_16))]
        log(f"  Null vector in R^16 (normalized to max|coeff|=1):")
        log(f"  {'BC (i,j,k)':14s} {'even_tet':10s} {'odd_tet':10s} {'sum':10s} {'diff(e-o)':10s}")
        for i_row, (bc_idx, ijk) in enumerate(corner_bcs):
            c_even = null_vec_16[2 * i_row]
            c_odd  = null_vec_16[2 * i_row + 1]
            log(f"  {str(ijk):14s} {c_even:+.6f}  {c_odd:+.6f}  {c_even+c_odd:+.6f}  {c_even-c_odd:+.6f}")

        # The row ordering in sub_overlap_16 follows corner_s_indices_sorted.
        # Reorder null_vec_16 to match corner_bcs order (2*bc_idx, 2*bc_idx+1 per BC).
        # Check: is the null vector purely in the even-minus-odd direction per BC?
        # i.e., null_vec_16[2i] = -null_vec_16[2i+1] for all i?
        diff_check = np.array([null_vec_16[2*i] + null_vec_16[2*i+1] for i in range(8)])
        log(f"\n  Test: is null vector in (even - odd) direction per BC?")
        log(f"  (Check: even_coeff + odd_coeff ~ 0 for each BC)")
        log(f"  Per-BC sums: {[f'{v:+.4f}' for v in diff_check]}")
        log(f"  Max |per-BC sum|: {np.max(np.abs(diff_check)):.3e}  "
            f"{'YES — even-minus-odd pattern' if np.max(np.abs(diff_check)) < 1e-4 else 'NO'}")

        # Also check if the differences (even - odd) are all equal in magnitude
        # with sign given by (-1)^(i+j+k) of the BC — but all BCs have even parity here,
        # so check whether diff_vals alternates by some other parity pattern.
        diff_vals = np.array([null_vec_16[2*i] - null_vec_16[2*i+1] for i in range(8)])
        all_eq_mag_err = np.max(np.abs(np.abs(diff_vals) / np.abs(diff_vals[0]) - 1.0))
        log(f"  Per-BC diffs (even-odd): {[f'{v:+.4f}' for v in diff_vals]}")
        log(f"  All equal in magnitude? max deviation: {all_eq_mag_err:.3e}  "
            f"{'YES' if all_eq_mag_err < 1e-4 else 'NO'}")
        # The sign of diff_vals — check if it matches (-1)^(i/2 + j/2 + k/2)
        # where i,j,k in {0,2} so i/2,j/2,k/2 in {0,1}
        sign_pattern = np.array([(-1)**((i//2 + j//2 + k//2)) for _, (i,j,k) in corner_bcs], dtype=float)
        sign_match_err = np.max(np.abs(diff_vals / diff_vals[0] - sign_pattern / sign_pattern[0]))
        log(f"  Sign matches (-1)^(i/2+j/2+k/2)? deviation: {sign_match_err:.3e}  "
            f"{'YES — F2^3 parity-alternating!' if sign_match_err < 1e-4 else 'NO'}")

        rank_corner = rank_corner_16
        null_vec = np.array([null_vec_16[2*i] for i in range(8)])  # even-tet coeffs for parity display
        corner_used = 'A (16 rows)'
    elif rank_corner_8 == 7:
        log(f"\n  Using Approach B (8 per-BC overlap rows, rank 7):")
        U8, s8, _ = np.linalg.svd(corner_sv_rows_8, full_matrices=True)
        null_vec = np.real(U8[:, 7])
        null_vec /= null_vec[np.argmax(np.abs(null_vec))]
        rank_corner = rank_corner_8
        corner_used = 'B (8 per-BC rows)'
    elif rank_corner_sv < 16:
        # Dependency in simplex_vectors themselves
        rank_corner = rank_corner_sv
        U_sv, s_sv, _ = np.linalg.svd(corner_sv_rows_16_raw, full_matrices=True)
        null_vec = np.real(U_sv[:, rank_corner_sv])
        null_vec /= null_vec[np.argmax(np.abs(null_vec))]
        corner_used = 'C (simplex_vectors)'
        corner_bc_ijk = [ijk for _, ijk in corner_bcs]
        # Expand to 8 entries (take even-tet component)
        corner_bc_ijk = [corner_bcs[i][1] for i in range(8)]
    else:
        log(f"\n  No rank-7 block found in any approach. Null vector computation skipped.")
        rank_corner = max(rank_corner_16, rank_corner_8, rank_corner_sv)
        null_vec = None
        corner_bc_ijk = [ijk for _, ijk in corner_bcs]
        corner_used = 'none'

    # Summary of null vector findings
    if null_vec is not None:
        parity_alt_err = float('nan')  # N/A: all 8 corner BCs have even parity (i+j+k always even)
        log(f"\n  NOTE: All 8 corner BCs have (i+j+k) even — parity-alternating test N/A.")
        log(f"  The null vector lives in R^16 (even/odd tet per BC).")
        log(f"  See 'even-minus-odd' test above for the actual structure.")
    else:
        parity_alt_err = float('nan')
        log(f"  No rank-7 block found — null vector not computed")

    # ------------------------------------------------------------------
    # Step 8: Summary
    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"\nLattice: BCC 3×3×3 periodic (3-torus), 540 faces, 27 BCs × 20 faces")
    log(f"Overlap rank: {rank19} / 54  (verified 19)")
    log(f"Singular value clusters: 1 + 6 + 12 = 19  (O_h-protected degeneracy)")
    log()
    log(f"O_h Irrep Decomposition by Cluster:")
    log(f"{'Cluster':12s} {'Dim':5s} {'Decomposition'}")
    log(f"{'-'*12} {'-'*5} {'-'*40}")
    for cname in ['1-dim', '6-dim', '12-dim']:
        r = cluster_results[cname]
        log(f"{cname:12s} {r['dim']:5d}  {r['decomp']}")
    log()
    log(f"Total 19D = {total_decomp}")
    log(f"Total dimension check: {total_dim_sum} (expect 19)")
    log()
    log(f"Numerical quality:")
    for cname in ['1-dim', '6-dim', '12-dim']:
        r = cluster_results[cname]
        log(f"  {cname}: orth_err={r['max_orth_err']:.2e}, hom_err={r['max_hom_err']:.2e}")
    log()
    log(f"2×2×2 corner block:")
    log(f"  Rank: {rank_corner} (expect 7) → 1 dependency vector")
    if null_vec is not None:
        log(f"  Null vector: lives in R^16 (even/odd tet per BC)")
        log(f"  Pattern: see Step 7 even-minus-odd test above")
    log()
    log("=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(_LOG) + "\n")
            print(f"\n[results written to {OUT_PATH}]")
        except Exception as e:
            print(f"WARNING: failed to write results file: {e}", file=sys.stderr)
