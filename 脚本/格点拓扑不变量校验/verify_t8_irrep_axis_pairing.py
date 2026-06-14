"""
verify_t8_irrep_axis_pairing.py

Item 8: Irrep-axis pairing for the BCC 3x3x3 flat-band 6D eigenspace.

Goal: For each axis a in {x, y, z}, determine which 2 dimensions of the 6D
flat band the per-axis hopping C_a = B.T H_a B "sees" (the rank-2 support),
expressed in irrep terms (A2u / Eu / T1u components).

Context:
- BCC 3x3x3 periodic lattice, 540 faces = 27 body-centers x 20 faces.
- H(Gamma) has a 6D eigenspace at eigenvalue -3.
- 6D space decomposes under O_h as A2u (1) + Eu (2) + T1u (3).
- Per-axis hopping C_a = V.T H_a V has spectrum {-8/3, -2, 0, 0, 0, 0} -- rank 2 per axis.
"""

import itertools
import sys
from collections import defaultdict

import numpy as np

OUT_PATH = "d:/AI thoery/.agent/scripts/verify_t8_irrep_axis_pairing_results.txt"
_LOG_LINES = []


def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))


# ============================================================
# Lattice construction (from verify_pending_4b_rank19_shell.py)
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
    vertex_to_faces = defaultdict(set)
    for face, idx in face_to_idx.items():
        for v in face:
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
# Oh group generation (from bcc_flatband_oh_irreps.py)
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
    tr = int(round(np.trace(g)))
    if det == 1:
        if tr == 3:   return 'E'
        elif tr == 0: return 'C3'
        elif tr == 1: return 'C4'
        elif tr == -1:
            off_diag = np.abs(g - np.diag(np.diag(g))).sum()
            return "C2'" if off_diag < 0.5 else 'C2'
        else: return f'UNKNOWN_proper_tr{tr}'
    else:
        if tr == -3:   return 'i'
        elif tr == 0:  return 'S6'
        elif tr == -1: return 'S4'
        elif tr == 1:
            off_diag = np.abs(g - np.diag(np.diag(g))).sum()
            return 'sigma_d' if off_diag < 0.5 else 'sigma_h'
        else: return f'UNKNOWN_improper_tr{tr}'


# OH character table with sigma_h/sigma_d SWAPPED relative to standard
# (matching the classifier above -- see bcc_flatband_oh_irreps.py for full explanation)
OH_CLASSES = ['E', 'C3', 'C2', 'C4', "C2'", 'i', 'S6', 'sigma_h', 'S4', 'sigma_d']
OH_CLASS_ORDER = {
    'E': 1, 'C3': 8, 'C2': 6, 'C4': 6, "C2'": 3,
    'i': 1, 'S6': 8, 'sigma_h': 6, 'S4': 6, 'sigma_d': 3
}
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


def which_bc_owns_face(face):
    for v in face:
        if v[0] == 'bc':
            return (v[1], v[2], v[3])
    return None


def build_oh_permutation(g, ref_faces, ref_face_to_local_idx,
                          face_to_idx, bc_face_indices, body_centers):
    P = np.zeros((20, 20), dtype=float)
    for m, face in enumerate(ref_faces):
        g_face = apply_oh_to_face(g, face)
        bc_owner = which_bc_owns_face(g_face)
        if bc_owner is None:
            if g_face in face_to_idx:
                global_idx = face_to_idx[g_face]
                for bc_idx, bc_ijk in enumerate(body_centers):
                    if global_idx in bc_face_indices[bc_idx]:
                        bc_owner = bc_ijk
                        break
            if bc_owner is None:
                raise ValueError(f"Cannot find BC owner for face {g_face}")
        inv_shift = tuple((-s) % 3 for s in bc_owner)
        translated_face = translate_face(g_face, inv_shift)
        if translated_face not in ref_face_to_local_idx:
            raise ValueError(f"Translated face {translated_face} not in reference face set.")
        n = ref_face_to_local_idx[translated_face]
        P[n, m] = 1.0
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    assert np.allclose(row_sums, 1.0)
    assert np.allclose(col_sums, 1.0)
    return P


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 78)
    log("Item 8: Irrep-axis pairing for BCC flat-band 6D eigenspace")
    log("=" * 78)

    # ----------------------------------------------------------
    # Step 1: Build lattice, faces, adjacency matrix
    # ----------------------------------------------------------
    log("\n[Step 1] Build lattice and faces")
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    N = len(face_to_idx)
    assert N == 540, f"Expected 540 faces, got {N}"
    log(f"  Faces: {N} (OK)")

    ref_face_to_local_idx = {face: m for m, face in enumerate(ref_faces)}

    # ----------------------------------------------------------
    # Step 2: Build adjacency matrix and extract T(R)
    # ----------------------------------------------------------
    log("\n[Step 2] Build adjacency matrix and extract T(R) blocks")
    A = build_adjacency_matrix(face_to_idx)

    # Reorder into BC-block form
    perm = []
    for bc_idx in range(27):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)
    A_reord = A[np.ix_(perm, perm)]

    # Extract T(R) for all centered displacements R in {-1,0,1}^3
    T = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers):
        Rc = tuple((r + 1) % 3 - 1 for r in bc_j_ijk)
        T[Rc] = A_reord[0:20, bc_j_idx * 20:(bc_j_idx + 1) * 20].copy()

    nonzero_Rs = sorted([R for R, mat in T.items() if np.linalg.norm(mat, 'fro') > 1e-10])
    log(f"  Non-zero T(R) at R: {nonzero_Rs}")

    # ----------------------------------------------------------
    # Step 3: Build H(Gamma) and extract 6D flat-band eigenvectors V
    # ----------------------------------------------------------
    log("\n[Step 3] Build H(Gamma) = sum_R T(R) and extract 6D flat-band eigenvectors")
    H_gamma = np.zeros((20, 20), dtype=float)
    for Rc, mat in T.items():
        H_gamma += mat

    log(f"  H(Gamma) symmetric: {np.allclose(H_gamma, H_gamma.T)}")

    eigvals, eigvecs = np.linalg.eigh(H_gamma)
    # Group eigenvalues
    tol = 1e-6
    log(f"  H(Gamma) eigenvalues:")
    prev = None
    count = 0
    for ev in eigvals:
        if prev is None or abs(ev - prev) > tol:
            if prev is not None:
                marker = " <--- TARGET" if abs(prev - (-3.0)) < 1e-4 else ""
                log(f"    lambda = {prev:10.6f}  (deg {count}){marker}")
            prev = ev
            count = 1
        else:
            count += 1
    if prev is not None:
        marker = " <--- TARGET" if abs(prev - (-3.0)) < 1e-4 else ""
        log(f"    lambda = {prev:10.6f}  (deg {count}){marker}")

    # Extract the 6D eigenspace at -3
    mask = np.abs(eigvals - (-3.0)) < 1e-10
    assert np.sum(mask) == 6, f"Expected 6 eigenvectors at -3, got {np.sum(mask)}"
    V = eigvecs[:, mask]  # 20 x 6
    log(f"  V shape: {V.shape} (20 x 6, OK)")

    # Verify
    HV_err = np.max(np.abs(H_gamma @ V - (-3.0) * V))
    log(f"  H(Gamma) V = -3 V: max err = {HV_err:.2e}  OK={HV_err < 1e-10}")

    VTV_err = np.max(np.abs(V.T @ V - np.eye(6)))
    log(f"  V^T V orthonormality error: {VTV_err:.3e}")

    # ----------------------------------------------------------
    # Step 4: Build O_h projectors and construct irrep-adapted basis B
    # ----------------------------------------------------------
    log("\n[Step 4] Build O_h projectors on 20D face space and project onto 6D flat band")

    oh_elements = generate_oh_elements()
    element_class = [classify_oh_element(g) for g in oh_elements]

    log("  Building 48 permutation matrices P(g) on 20D space...")
    perm_matrices = []
    for g in oh_elements:
        P = build_oh_permutation(g, ref_faces, ref_face_to_local_idx,
                                 face_to_idx, bc_face_indices, body_centers)
        perm_matrices.append(P)
    log(f"  Done.")

    # Build irrep projectors: P_mu = (d_mu / 48) * sum_g chi_mu(g)* P(g)
    # Only need A2u (dim 1), Eu (dim 2), T1u (dim 3)
    target_irreps = ['A2u', 'Eu', 'T1u']
    target_dims   = {'A2u': 1, 'Eu': 2, 'T1u': 3}

    log("  Building irrep projectors for A2u, Eu, T1u...")
    projectors = {}
    for irrep in target_irreps:
        d_mu = OH_IRREP_DIMS[irrep]
        char_row = OH_CHAR_TABLE[irrep]
        P_mu = np.zeros((20, 20), dtype=float)
        for g_idx, g in enumerate(oh_elements):
            cls = element_class[g_idx]
            cls_idx = OH_CLASSES.index(cls)
            chi = char_row[cls_idx]  # real characters
            P_mu += chi * perm_matrices[g_idx]
        P_mu *= d_mu / 48.0
        projectors[irrep] = P_mu
        # Verify projector is idempotent: P^2 = P
        idem_err = np.max(np.abs(P_mu @ P_mu - P_mu))
        log(f"    P_{irrep}: idempotent error = {idem_err:.3e}")

    # Project V columns into each irrep sector
    log("\n  Projecting 6D flat-band space into irrep sectors...")
    irrep_bases = {}
    for irrep in target_irreps:
        P_mu = projectors[irrep]
        # Project: W_mu = P_mu @ V  (20x6), then restrict to the 6D flat-band subspace
        W_mu = P_mu @ V  # 20x6
        # The columns of W_mu that lie in V's span: project W_mu onto V
        # W_in_V = V (V^T W_mu) gives coordinates in V's basis
        coords = V.T @ W_mu  # 6x6
        # Orthonormalize these coordinate vectors to get the irrep basis in V-space
        # SVD-based orthonormalization
        U, s, Vt = np.linalg.svd(coords, full_matrices=True)
        rank = np.sum(s > 1e-8)
        log(f"    {irrep}: rank in flat-band space = {rank} (expected {target_dims[irrep]})")
        assert rank == target_dims[irrep], \
            f"Irrep {irrep}: expected rank {target_dims[irrep]}, got {rank}"
        # Orthonormal basis vectors for this irrep in the 6D V-coordinate space
        # The right singular vectors corresponding to nonzero singular values
        basis_coords = Vt[:rank, :].T  # 6 x rank -- columns are basis vectors in V-space
        irrep_bases[irrep] = basis_coords  # 6 x d_mu matrix

    # Verify dimensions sum to 6
    total = sum(irrep_bases[ir].shape[1] for ir in target_irreps)
    log(f"\n  Total irrep dimensions: {total} (expect 6, OK={total == 6})")

    # Build irrep-adapted orthonormal basis B of flat band (20 x 6)
    # Order: [A2u(1), Eu(2), T1u(3)]
    # B columns are V @ b_i for each irrep basis vector b_i
    B_coords = np.hstack([irrep_bases[ir] for ir in target_irreps])  # 6 x 6
    # Verify B_coords is orthogonal
    orth_err = np.max(np.abs(B_coords.T @ B_coords - np.eye(6)))
    log(f"  B_coords orthogonality error: {orth_err:.3e}")

    # The irrep-adapted basis in 20D space
    B = V @ B_coords  # 20 x 6
    log(f"  B = V @ B_coords: shape {B.shape}")
    B_orth_err = np.max(np.abs(B.T @ B - np.eye(6)))
    log(f"  B orthogonality error: {B_orth_err:.3e}")

    # Verify B spans same space as V
    # Project V onto B's column space
    span_err = np.max(np.abs(B @ B.T @ V - V))
    log(f"  B spans same 6D space as V: max err = {span_err:.3e}  OK={span_err < 1e-8}")

    # Slot ranges: A2u = cols 0, Eu = cols 1:3, T1u = cols 3:6
    slot_ranges = {'A2u': slice(0, 1), 'Eu': slice(1, 3), 'T1u': slice(3, 6)}

    # ----------------------------------------------------------
    # Step 5: Per-axis analysis
    # ----------------------------------------------------------
    log("\n[Step 5] Per-axis C_a = B.T @ H_a @ B analysis")

    # Cross-check: B.T H(Gamma) B should be -3 * I
    HG_in_B = B.T @ H_gamma @ B
    HG_minus3I_err = np.max(np.abs(HG_in_B - (-3.0) * np.eye(6)))
    log(f"  Cross-check: B.T H(Gamma) B = -3*I, max err = {HG_minus3I_err:.3e}  OK={HG_minus3I_err < 1e-8}")

    # On-site T(0) in B basis
    T0_in_B = B.T @ T[(0, 0, 0)] @ B
    T0_trace = np.trace(T0_in_B)
    log(f"  B.T T(0) B trace = {T0_trace:.6f} (expect -4)")

    axes_info = {
        'x': ((1, 0, 0), (-1, 0, 0)),
        'y': ((0, 1, 0), (0, -1, 0)),
        'z': ((0, 0, 1), (0, 0, -1)),
    }

    axis_results = {}
    sum_Ca_trace = 0.0

    for axis_name in ['x', 'y', 'z']:
        ep, em = axes_info[axis_name]
        H_a = T[ep] + T[em]  # 20 x 20
        C_a = B.T @ H_a @ B  # 6 x 6

        log(f"\n  --- Axis {axis_name} ---")
        log(f"  C_{axis_name} in irrep-adapted basis (rounded to 4 decimals):")
        for row in range(6):
            row_str = "  " + "  ".join(f"{C_a[row, col]:8.4f}" for col in range(6))
            log(row_str)

        # Add block labels
        log(f"  [rows/cols: 0=A2u, 1-2=Eu, 3-5=T1u]")

        # Diagonalize C_a
        evals, evecs = np.linalg.eigh(C_a)
        # Sort by eigenvalue
        idx_sort = np.argsort(evals)
        evals = evals[idx_sort]
        evecs = evecs[:, idx_sort]

        log(f"  C_{axis_name} eigenvalues: {np.array2string(evals, precision=6)}")

        # Verify spectrum: expect -8/3, -2, 0, 0, 0, 0
        expected_evals = np.array([-8/3, -2.0, 0.0, 0.0, 0.0, 0.0])
        spec_err = np.max(np.abs(evals - expected_evals))
        log(f"  Spectrum check (expect {np.array2string(expected_evals, precision=6)}): max err = {spec_err:.4e}  OK={spec_err < 1e-6}")

        sum_Ca_trace += np.trace(C_a)

        # For the two active eigenvectors (evals -8/3 and -2):
        results_axis = []
        log(f"  Irrep weight decomposition for rank-2 eigenvectors:")
        log(f"  {'Eigenvalue':>12}  {'A2u weight':>12}  {'Eu weight':>12}  {'T1u weight':>12}  {'dominant'}")

        for ev_idx in range(2):
            ev_val = evals[ev_idx]
            evec = evecs[:, ev_idx]  # 6-component vector in irrep-adapted basis

            # Weights = |component|^2 summed over each irrep's slots
            w_A2u = float(np.sum(np.abs(evec[slot_ranges['A2u']])**2))
            w_Eu  = float(np.sum(np.abs(evec[slot_ranges['Eu']])**2))
            w_T1u = float(np.sum(np.abs(evec[slot_ranges['T1u']])**2))
            total_w = w_A2u + w_Eu + w_T1u

            # Dominant irrep
            weights = {'A2u': w_A2u, 'Eu': w_Eu, 'T1u': w_T1u}
            dominant = max(weights, key=weights.get)

            log(f"  {ev_val:12.6f}  {w_A2u:12.6f}  {w_Eu:12.6f}  {w_T1u:12.6f}  {dominant} (sum={total_w:.6f})")
            results_axis.append({
                'eval': ev_val,
                'w_A2u': w_A2u,
                'w_Eu': w_Eu,
                'w_T1u': w_T1u,
                'dominant': dominant,
            })

        # Also report the null space (eigenvalue ~0) to complete the picture
        log(f"  Null space (eigenvalue ~0) eigenvectors:")
        for ev_idx in range(2, 6):
            ev_val = evals[ev_idx]
            evec = evecs[:, ev_idx]
            w_A2u = float(np.sum(np.abs(evec[slot_ranges['A2u']])**2))
            w_Eu  = float(np.sum(np.abs(evec[slot_ranges['Eu']])**2))
            w_T1u = float(np.sum(np.abs(evec[slot_ranges['T1u']])**2))
            log(f"  {ev_val:12.6f}  {w_A2u:12.6f}  {w_Eu:12.6f}  {w_T1u:12.6f}")

        axis_results[axis_name] = results_axis

    # ----------------------------------------------------------
    # Step 6: Cross-checks
    # ----------------------------------------------------------
    log("\n[Step 6] Cross-checks")

    # Verify sum of three C_a traces + T(0) trace = -3 * 6
    T0_trace_check = np.trace(B.T @ T[(0, 0, 0)] @ B)
    total_trace = sum_Ca_trace + T0_trace_check
    log(f"  sum_a Tr(C_a) = {sum_Ca_trace:.6f}")
    log(f"  Tr(B.T T(0) B) = {T0_trace_check:.6f}  (expect -4)")
    log(f"  Total = {total_trace:.6f}  (expect {6 * (-3):.1f} = -18)")
    log(f"  Total = -18? {abs(total_trace - (-18.0)) < 1e-8}")

    # Verify sum of per-axis A2u weights (over both active eigenvectors, summed over 3 axes)
    log("\n  Per-axis A2u/Eu/T1u weight sums (over active eigenvectors, summed across 3 axes):")
    for irrep_name in ['A2u', 'Eu', 'T1u']:
        key = f'w_{irrep_name}'
        total = 0.0
        for axis_name in ['x', 'y', 'z']:
            for r in axis_results[axis_name]:
                total += r[key]
        log(f"  Sum over 3 axes x 2 eigvecs of {irrep_name} weight: {total:.6f}")
        # Theoretical: A2u(1) + Eu(2) + T1u(3) = 6 total slots, 3 axes x 2 active = 6 active dims
        # If the active dims tile the 6D space exactly, sum should equal the irrep dimension.
    log("  (If active dims tile 6D space: A2u->1, Eu->2, T1u->3)")

    # ----------------------------------------------------------
    # Step 7: Final summary table
    # ----------------------------------------------------------
    log("\n" + "=" * 78)
    log("FINAL SUMMARY TABLE: Irrep-axis pairing")
    log("=" * 78)
    log(f"\n  {'Axis':6}  {'Eigenvalue':>12}  {'A2u wt':>10}  {'Eu wt':>10}  {'T1u wt':>10}  {'dominant'}")
    log(f"  {'------':6}  {'----------':>12}  {'------':>10}  {'------':>10}  {'-------':>10}  {'---------'}")

    for axis_name in ['x', 'y', 'z']:
        for r in axis_results[axis_name]:
            log(f"  {axis_name:6}  {r['eval']:12.6f}  {r['w_A2u']:10.6f}  {r['w_Eu']:10.6f}  {r['w_T1u']:10.6f}  {r['dominant']}")

    log("\n  Interpretation:")
    log("  Each axis's C_a has rank 2 (two non-zero eigenvalues: -8/3 and -2).")
    log("  The irrep weights tell us which part of the 6D space each axis 'sees'.")

    # Summarize per-axis total support across irreps
    log("\n  Per-axis total irrep support (sum of weights over both active eigenvectors):")
    log(f"  {'Axis':6}  {'A2u total':>12}  {'Eu total':>12}  {'T1u total':>12}")
    for axis_name in ['x', 'y', 'z']:
        totA = sum(r['w_A2u'] for r in axis_results[axis_name])
        totE = sum(r['w_Eu']  for r in axis_results[axis_name])
        totT = sum(r['w_T1u'] for r in axis_results[axis_name])
        log(f"  {axis_name:6}  {totA:12.6f}  {totE:12.6f}  {totT:12.6f}")

    log("\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(_LOG_LINES) + "\n")
            print(f"\n[results written to {OUT_PATH}]")
        except Exception as e:
            print(f"WARNING: failed to write results file: {e}", file=sys.stderr)
