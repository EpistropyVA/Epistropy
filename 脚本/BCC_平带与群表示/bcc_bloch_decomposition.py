"""
bcc_bloch_decomposition.py

Irrep (Bloch wave) decomposition of the 540x540 periodic-BC face-adjacency
matrix for the 4-simplex network of a 3x3x3 BCC lattice on the 3-torus.

Translation group: Z3 x Z3 x Z3 (order 27).
Each BC is at grid position (i,j,k) with i,j,k in {0,1,2}.
The 540 faces split into 27 BCs x 20 faces per BC.

Key insight for translation-equivariant labeling:
  Each face belongs to exactly 1 simplex (all 540 face-instances are unique
  after mod-3 identification).  Each simplex is owned by exactly 1 BC.
  So we label the 20 faces of BC(i,j,k) by their simplex-local index within
  that BC's two tetrahedra.  Translation maps BC(0,0,0) -> BC(i,j,k) by
  shifting all vertex coordinates by (i,j,k) mod 3, and the same simplex-local
  ordering applies.  This gives a canonical translation-equivariant basis.

Bloch decomposition:
  For wavevector k = 2*pi/3*(n1,n2,n3), define the 20x20 Bloch Hamiltonian:
    H(k) = sum_R exp(i k.R) T(R)
  where T(R) is the 20x20 inter-BC hopping matrix.
  The 27x20=540 eigenvalues of all H(k) reproduce the full spectrum.

Results written to: d:/AI thoery/.agent/scripts/bcc_bloch_results.txt
"""

import itertools
import sys
from collections import defaultdict

import numpy as np

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_bloch_results.txt"

_LOG_LINES = []


def log(msg=""):
    """Print to stdout AND buffer for results file."""
    print(msg)
    _LOG_LINES.append(str(msg))


# ============================================================
# Lattice construction
# ============================================================

def build_bcc_lattice_periodic():
    """27 body centers on the 3-torus; cube corners reduced mod 3."""
    body_centers = []
    for i, j, k in itertools.product(range(3), repeat=3):
        body_centers.append((i, j, k))  # store as integer grid coords
    return body_centers


def enumerate_simplex_faces(bc_ijk):
    """
    For BC at grid position bc_ijk = (i,j,k), enumerate its 20 triangular faces
    in a CANONICAL ORDER that is the same for every BC (just shifted mod 3).

    Returns list of 20 frozensets, each containing 3 vertices.
    Vertices:
      - The BC itself, encoded as ('bc', i, j, k)
      - Corner vertices encoded as ('c', cx, cy, cz) with cx,cy,cz in {0,1,2}

    The 20 faces arise from 2 tetrahedra (each has C(5,3)=10 triangular faces):
      Simplex = {bc} union {4 tet corners}  =>  5 vertices, C(5,3)=10 faces per simplex
      2 simplices (even-parity and odd-parity tet) -> 20 faces total.
    """
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)

    # Build even and odd tet corners (original parity, wrapped coords)
    orig_corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = i + dx, j + dy, k + dz
        wx, wy, wz = ox % 3, oy % 3, oz % 3
        orig_corners.append(((ox + oy + oz) % 2, ('c', wx, wy, wz)))

    even_corners = [v for (par, v) in orig_corners if par == 0]
    odd_corners  = [v for (par, v) in orig_corners if par == 1]
    assert len(even_corners) == 4 and len(odd_corners) == 4

    faces = []
    for tet_corners in (even_corners, odd_corners):
        simplex_verts = [bc_v] + tet_corners  # 5 vertices
        for combo in itertools.combinations(simplex_verts, 3):
            faces.append(frozenset(combo))

    assert len(faces) == 20
    return faces


def translate_vertex(v, shift_ijk):
    """Apply translation by shift_ijk (mod 3) to a vertex."""
    si, sj, sk = shift_ijk
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i + si) % 3, (j + sj) % 3, (k + sk) % 3)
    else:
        _, cx, cy, cz = v
        return ('c', (cx + si) % 3, (cy + sj) % 3, (cz + sk) % 3)


def translate_face(face, shift_ijk):
    """Translate all vertices in a face by shift_ijk (mod 3)."""
    return frozenset(translate_vertex(v, shift_ijk) for v in face)


def build_all_faces(body_centers):
    """
    Build the complete face list with translation-equivariant labeling.

    Returns:
        face_to_idx:  dict frozenset -> global index in [0, 540)
        bc_face_indices: list of 27 lists, each of length 20.
          bc_face_indices[bc_flat][m] = global face index of internal face m of BC bc_flat.
          The ordering within each BC's 20 faces is IDENTICAL (translation-equivariant).
    """
    # Enumerate reference faces at BC (0,0,0)
    ref_bc = (0, 0, 0)
    ref_faces = enumerate_simplex_faces(ref_bc)  # 20 faces in canonical order

    face_to_idx = {}
    bc_face_indices = []

    global_idx = 0
    for bc_idx, bc_ijk in enumerate(body_centers):
        local_indices = []
        for m, ref_face in enumerate(ref_faces):
            # Translate reference face by bc_ijk
            shifted_face = translate_face(ref_face, bc_ijk)
            if shifted_face not in face_to_idx:
                face_to_idx[shifted_face] = global_idx
                global_idx += 1
            local_indices.append(face_to_idx[shifted_face])
        bc_face_indices.append(local_indices)

    return face_to_idx, bc_face_indices, ref_faces


def build_adjacency_matrix(face_to_idx, body_centers, bc_face_indices):
    """
    Two faces are adjacent iff they share exactly 2 vertices.
    Build the N x N adjacency matrix in terms of the global face indices.
    """
    N = len(face_to_idx)
    A = np.zeros((N, N), dtype=float)

    # Collect all faces as sorted vertex lists for intersection
    all_faces_sets = [None] * N
    for face, idx in face_to_idx.items():
        all_faces_sets[idx] = face

    # Build vertex -> face index mapping
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


def bc_flat_index(bc_ijk):
    return bc_ijk[0] * 9 + bc_ijk[1] * 3 + bc_ijk[2]


def group_eigenvalues(eigvals, tol=1e-6):
    groups = []
    cur_vals = [eigvals[0]]
    for ev in eigvals[1:]:
        if abs(ev - cur_vals[-1]) <= tol:
            cur_vals.append(ev)
        else:
            groups.append(cur_vals)
            cur_vals = [ev]
    groups.append(cur_vals)

    out = []
    cumulative = 0
    for g in groups:
        rep = float(np.mean(g))
        deg = len(g)
        cumulative += deg
        out.append((rep, deg, cumulative))
    return out


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 78)
    log("BCC 3x3x3 Bloch (irrep) decomposition of the 540x540 periodic-BC")
    log("face-adjacency matrix on the 3-torus Z3 x Z3 x Z3")
    log("=" * 78)

    # --------------------------------------------------------
    # Step 1: Build faces with translation-equivariant labeling
    # --------------------------------------------------------
    log("\n[Step 1] Build faces with canonical translation-equivariant labeling")
    body_centers = build_bcc_lattice_periodic()  # list of 27 (i,j,k)
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)

    N = len(face_to_idx)
    log(f"  Total unique faces: {N} (expect 540)")
    assert N == 540, f"Expected 540, got {N}"

    # Verify each BC has exactly 20 distinct faces
    sizes = [len(set(idxs)) for idxs in bc_face_indices]
    log(f"  Faces per BC: min={min(sizes)}, max={max(sizes)}, all==20? {all(s==20 for s in sizes)}")
    assert all(s == 20 for s in sizes)

    # --------------------------------------------------------
    # Step 2: Build 540x540 adjacency matrix
    # --------------------------------------------------------
    log("\n[Step 2] Build 540x540 adjacency matrix")
    A = build_adjacency_matrix(face_to_idx, body_centers, bc_face_indices)
    log(f"  Shape: {A.shape}, symmetric: {np.allclose(A, A.T)}, trace: {np.trace(A):.0f}")

    # --------------------------------------------------------
    # Step 3: Build BC-reordered version: 27 blocks of 20
    # --------------------------------------------------------
    log("\n[Step 3] Reorder adjacency matrix into BC-block form (27 x 20)")
    perm = []
    for bc_idx in range(27):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)
    # Check perm is a valid permutation
    assert sorted(perm) == list(range(N)), "perm is not a valid permutation!"
    A_reord = A[np.ix_(perm, perm)]
    log(f"  Permutation valid: True")
    log(f"  A_reord shape: {A_reord.shape}, symmetric: {np.allclose(A_reord, A_reord.T)}")

    # --------------------------------------------------------
    # Step 4: Direct diagonalization (reference)
    # --------------------------------------------------------
    log("\n[Step 4] Direct diagonalization of 540x540 A (reference spectrum)")
    eigvals_full = np.sort(np.linalg.eigvalsh(A))
    grouped_full = group_eigenvalues(eigvals_full)
    log(f"  {'eigenvalue':>14} | {'deg':>6} | {'cumul':>6}")
    log(f"  {'-'*14}-+-{'-'*6}-+-{'-'*6}")
    for rep, deg, cum in grouped_full:
        log(f"  {rep:14.6f} | {deg:6d} | {cum:6d}")

    # --------------------------------------------------------
    # Step 5: Build hopping matrices T(R) and verify translation symmetry
    # --------------------------------------------------------
    log("\n[Step 5] Build hopping matrices T(R) from BC-block structure")

    def get_block(A_r, bc_i, bc_j, size=20):
        return A_r[bc_i*size:(bc_i+1)*size, bc_j*size:(bc_j+1)*size].copy()

    # T(R) from reference BC (0,0,0) = flat index 0
    ref_flat = bc_flat_index((0, 0, 0))  # = 0

    T = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers):
        # displacement R = bc_j_ijk - (0,0,0) mod 3
        R = tuple(bc_j_ijk[a] % 3 for a in range(3))
        T[R] = get_block(A_reord, ref_flat, bc_j_idx)

    # Verify translation symmetry: T(R) from any origin must match T(R) from origin (0,0,0)
    log("\n  Verifying translation symmetry: T(R) is origin-independent")
    max_trans_dev = 0.0
    worst_case = None
    for bc_i_idx, bc_i_ijk in enumerate(body_centers):
        for bc_j_idx, bc_j_ijk in enumerate(body_centers):
            R = tuple((bc_j_ijk[a] - bc_i_ijk[a]) % 3 for a in range(3))
            T_here = get_block(A_reord, bc_i_idx, bc_j_idx)
            dev = np.max(np.abs(T_here - T[R]))
            if dev > max_trans_dev:
                max_trans_dev = dev
                worst_case = (bc_i_ijk, bc_j_ijk, R)
    log(f"  Max deviation across all BC pairs: {max_trans_dev:.3e}")
    if worst_case:
        log(f"  Worst case: origin={worst_case[0]}, target={worst_case[1]}, R={worst_case[2]}")
    log(f"  Translation symmetry exact? {max_trans_dev < 1e-10}")

    # --------------------------------------------------------
    # Step 6: Characterize nonzero T(R) by torus distance
    # --------------------------------------------------------
    log("\n[Step 6] Characterize nonzero T(R) matrices")

    nonzero_T = {}
    for R, mat in T.items():
        nrm = np.linalg.norm(mat, 'fro')
        if nrm > 1e-10:
            nonzero_T[R] = mat

    dist_groups = defaultdict(list)
    for R in nonzero_T:
        d = sum(min(r, 3-r) for r in R)
        dist_groups[d].append(R)

    log(f"  Nonzero T(R) matrices: {len(nonzero_T)} out of 27")
    log(f"  Breakdown by torus distance:")
    for d in sorted(dist_groups):
        label = "self" if d == 0 else ("NN" if d == 1 else f"NNN(d={d})")
        log(f"    dist={d} ({label}): {len(dist_groups[d])} matrices")
        for R in sorted(dist_groups[d]):
            nrm = np.linalg.norm(T[R], 'fro')
            log(f"      R={R}  ||T||_F={nrm:.6f}")

    # Fraction of ||A||^2 from each distance
    A_norm_sq = np.linalg.norm(A_reord, 'fro')**2
    log(f"\n  ||A||^2 = {A_norm_sq:.4f}")

    tier_sq = defaultdict(float)
    for bc_i_idx, bc_i_ijk in enumerate(body_centers):
        for bc_j_idx, bc_j_ijk in enumerate(body_centers):
            R = tuple((bc_j_ijk[a] - bc_i_ijk[a]) % 3 for a in range(3))
            d = sum(min(r, 3-r) for r in R)
            blk = get_block(A_reord, bc_i_idx, bc_j_idx)
            tier_sq[d] += np.linalg.norm(blk, 'fro')**2

    for d in sorted(tier_sq):
        label = "self" if d == 0 else ("NN" if d == 1 else f"NNN(d={d})")
        frac = tier_sq[d] / A_norm_sq
        log(f"  dist={d} ({label}): {tier_sq[d]:.4f} ({100*frac:.2f}%)")

    # --------------------------------------------------------
    # Step 7: Construct H(k) for all 27 k-points and diagonalize
    # --------------------------------------------------------
    log("\n[Step 7] Construct H(k) = sum_R exp(i k.R) T(R) for all 27 wavevectors")

    kvecs = list(itertools.product(range(3), repeat=3))  # (n1,n2,n3)

    all_bloch_eigs = []
    H_all = {}

    for n_tuple in kvecs:
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = np.zeros((20, 20), dtype=complex)
        for R, mat in T.items():
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k, R_vec))
            H_k += phase * mat
        H_all[n_tuple] = H_k
        eigs_k = np.linalg.eigvalsh(H_k)
        all_bloch_eigs.extend(eigs_k.tolist())

    all_bloch_eigs = np.sort(np.array(all_bloch_eigs))
    log(f"  Total Bloch eigenvalues: {len(all_bloch_eigs)} (expect 540)")

    # --------------------------------------------------------
    # Step 8: Verify spectrum match
    # --------------------------------------------------------
    log("\n[Step 8] Verify Bloch spectrum matches direct diagonalization")
    max_diff = float(np.max(np.abs(all_bloch_eigs - eigvals_full)))
    log(f"  Max |lambda_Bloch - lambda_direct| = {max_diff:.3e}")

    # --------------------------------------------------------
    # Step 9: Hermiticity check
    # --------------------------------------------------------
    log("\n[Step 9] Hermiticity check for each H(k)")
    max_herm_dev = 0.0
    for n_tuple, H_k in H_all.items():
        dev = np.max(np.abs(H_k - H_k.conj().T))
        if dev > max_herm_dev:
            max_herm_dev = dev
    log(f"  Max ||H(k) - H(k)^H|| over all k: {max_herm_dev:.3e}")
    log(f"  All H(k) Hermitian? {max_herm_dev < 1e-10}")

    # --------------------------------------------------------
    # Step 10: Dispersion relation — eigenvalues vs k
    # --------------------------------------------------------
    log("\n[Step 10] Dispersion relation: H(k) eigenvalues for each k-point")

    def bz_label(n_tuple):
        nonzero = sum(1 for n in n_tuple if n != 0)
        return ["Gamma", "edge", "face", "corner"][nonzero]

    # 27x20 eigenvalue matrix (sorted within each H(k))
    eigs_by_k = {}
    for n_tuple in kvecs:
        eigs_by_k[n_tuple] = np.sort(np.linalg.eigvalsh(H_all[n_tuple]))

    log(f"\n  {'k = 2pi/3*n':>15} | {'BZ':>6} | eigenvalues (sorted, 20 values)")
    log(f"  {'-'*15}-+-{'-'*6}-+-{'-'*60}")
    for n_tuple in sorted(kvecs):
        ev = eigs_by_k[n_tuple]
        label = bz_label(n_tuple)
        ev_str = np.array2string(ev, precision=4, separator=',',
                                  suppress_small=True, max_line_width=400)
        log(f"  {str(n_tuple):>15} | {label:>6} | {ev_str}")

    # Band structure: for each of 20 bands (ranked within H(k)), collect across k
    eigs_matrix = np.array([eigs_by_k[n] for n in sorted(kvecs)])  # (27, 20)

    log(f"\n[Step 10b] Band bandwidths (rank within H(k) = band index)")
    log(f"  {'band':>5} | {'min':>10} | {'max':>10} | {'BW':>10} | flat?")
    log(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*5}")
    n_flat = 0
    flat_bands = []
    for band_idx in range(20):
        band_vals = eigs_matrix[:, band_idx]
        bmin, bmax = float(np.min(band_vals)), float(np.max(band_vals))
        bw = bmax - bmin
        flat = bw < 1e-6
        if flat:
            n_flat += 1
            flat_bands.append((band_idx, bmin))
        log(f"  {band_idx:5d} | {bmin:10.6f} | {bmax:10.6f} | {bw:10.6f} | {flat}")

    # --------------------------------------------------------
    # Step 11: Where does lambda=-3 appear?
    # --------------------------------------------------------
    log("\n[Step 11] At which k-points does lambda=-3 appear?")
    tol_3 = 1e-4
    lam3_kpoints = []
    for n_tuple in kvecs:
        ev = eigs_by_k[n_tuple]
        hits = [e for e in ev if abs(e - (-3.0)) < tol_3]
        if hits:
            lam3_kpoints.append((n_tuple, bz_label(n_tuple), len(hits)))

    if lam3_kpoints:
        for n_tuple, label, cnt in sorted(lam3_kpoints):
            log(f"  k=2pi/3*{n_tuple}  BZ={label}  multiplicity={cnt}")
        total = sum(c for _, _, c in lam3_kpoints)
        log(f"  Total lambda=-3 in Bloch decomp: {total}")
        full_count = sum(1 for e in eigvals_full if abs(e - (-3.0)) < tol_3)
        log(f"  Total lambda=-3 in direct diag: {full_count}")
    else:
        log(f"  No k-point has H(k) eigenvalue = -3 (tol={tol_3})")
        nearest = sorted([(abs(e - (-3.0)), e, n_tuple)
                          for n_tuple in kvecs
                          for e in eigs_by_k[n_tuple]])[:5]
        log(f"  Nearest to -3: {[(f'{e:.6f}', str(n)) for _, e, n in nearest]}")

    # --------------------------------------------------------
    # Step 12: Gamma-point analysis
    # --------------------------------------------------------
    log("\n[Step 12] Gamma-point H(0,0,0) analysis")
    H_gamma = H_all[(0, 0, 0)]
    imag_max = float(np.max(np.abs(np.imag(H_gamma))))
    log(f"  H(0,0,0) imaginary part max: {imag_max:.3e} (should be 0)")
    H_gamma_real = np.real(H_gamma)
    eigs_gamma = np.sort(np.linalg.eigvalsh(H_gamma_real))
    log(f"  H(0,0,0) eigenvalues:")
    grouped_g = group_eigenvalues(eigs_gamma)
    for rep, deg, cum in grouped_g:
        log(f"    lambda={rep:10.6f}  deg={deg}")
    log(f"  H(0,0,0) = sum_R T(R) is the row-sum of all inter-BC couplings")
    log(f"  (This equals the restriction of A to any single BC's 20 faces plus the")
    log(f"   sum of all outgoing hoppings, integrated over k via Bloch sum at k=0)")

    # --------------------------------------------------------
    # Step 13: Why Kronecker product fails
    # --------------------------------------------------------
    log("\n[Step 13] Why the Kronecker product decomposition fails")
    log("  Kronecker ansatz: A = kron(I27, L_local) + kron(L_x,M_x) + kron(L_y,M_y) + kron(L_z,M_z)")
    log("  where L_local = T(0,0,0) and M_axis = NN hopping matrix along that axis.")
    log("")

    nn_x = [(1,0,0), (2,0,0)]
    nn_y = [(0,1,0), (0,2,0)]
    nn_z = [(0,0,1), (0,0,2)]

    def avg_T(disp_list):
        return np.mean([T[R] for R in disp_list], axis=0)

    M_x = avg_T(nn_x)
    M_y = avg_T(nn_y)
    M_z = avg_T(nn_z)

    log(f"  Are M_x, M_y, M_z identical to each other?")
    log(f"    ||M_x - M_y||_F = {np.linalg.norm(M_x - M_y,'fro'):.6f}")
    log(f"    ||M_x - M_z||_F = {np.linalg.norm(M_x - M_z,'fro'):.6f}")
    log(f"    ||M_y - M_z||_F = {np.linalg.norm(M_y - M_z,'fro'):.6f}")

    log(f"  Are +x and -x NN matrices identical?")
    log(f"    ||T(1,0,0) - T(2,0,0)||_F = {np.linalg.norm(T[(1,0,0)] - T[(2,0,0)],'fro'):.6f}")
    log(f"    ||T(0,1,0) - T(0,2,0)||_F = {np.linalg.norm(T[(0,1,0)] - T[(0,2,0)],'fro'):.6f}")
    log(f"    ||T(0,0,1) - T(0,0,2)||_F = {np.linalg.norm(T[(0,0,1)] - T[(0,0,2)],'fro'):.6f}")

    # Build Kronecker approximation
    def build_L_shift(axis, delta):
        L = np.zeros((27, 27))
        for flat_i, bc_ijk in enumerate(body_centers):
            ijk_j = list(bc_ijk)
            ijk_j[axis] = (bc_ijk[axis] + delta) % 3
            flat_j = bc_flat_index(tuple(ijk_j))
            L[flat_i, flat_j] = 1.0
        return L

    I27 = np.eye(27)
    L_local = T[(0, 0, 0)]
    A_Kron = np.kron(I27, L_local)
    for axis, (nn_disp, M_ax) in enumerate(zip([nn_x, nn_y, nn_z], [M_x, M_y, M_z])):
        dp = nn_disp[0][axis]
        L_p = build_L_shift(axis, dp)
        L_m = build_L_shift(axis, 3 - dp)
        A_Kron += np.kron(L_p, M_ax) + np.kron(L_m, M_ax)

    A_norm = np.linalg.norm(A_reord, 'fro')
    kron_res = np.linalg.norm(A_reord - A_Kron, 'fro') / A_norm
    log(f"\n  Kronecker (NN-only, isotropic) residual: {kron_res:.4f} ({100*kron_res:.1f}%)")

    # Check if Kronecker ansatz would work in k-space:
    # H_Kron(k) = L_local + sum_axis (e^{i k_alpha} + e^{-i k_alpha}) M_axis
    # = L_local + 2*sum_axis cos(k_alpha) * M_axis
    # Compare to actual H(k)
    log(f"\n  Testing Kronecker ansatz H_Kron(k) = L_local + 2*sum_axis cos(k_alpha)*M_axis:")
    log(f"  {'k=2pi/3*n':>15} | {'||H(k)-H_Kron(k)||':>20} | {'match?':>6}")
    log(f"  {'-'*15}-+-{'-'*20}-+-{'-'*6}")
    max_kron_k = 0.0
    for n_tuple in sorted(kvecs):
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_kron_k = (L_local.astype(complex)
                    + 2*np.cos(k[0])*M_x
                    + 2*np.cos(k[1])*M_y
                    + 2*np.cos(k[2])*M_z)
        dev = np.linalg.norm(H_all[n_tuple] - H_kron_k, 'fro')
        max_kron_k = max(max_kron_k, dev)
        log(f"  {str(n_tuple):>15} | {dev:20.6f} | {dev < 1e-8}")
    log(f"  Max Kronecker error in k-space: {max_kron_k:.6f}")
    log(f"  Root cause: if M_x=M_y=M_z and T(+R)=T(-R) the Kronecker ansatz would be")
    log(f"  exact. It fails because the actual hopping matrices break the assumed symmetry.")

    # --------------------------------------------------------
    # Step 14: Summary
    # --------------------------------------------------------
    log("\n" + "=" * 78)
    log("[KEY QUESTIONS SUMMARY]")
    log("=" * 78)

    q1 = max_diff < 1e-8
    log(f"\nQ1: Does Bloch decomposition perfectly reconstruct the full spectrum?")
    log(f"    Max eigenvalue discrepancy: {max_diff:.3e}")
    log(f"    -> {'YES (tol 1e-8 satisfied)' if q1 else 'NO'}")

    log(f"\nQ2: How many distinct T(R) hopping matrices are nonzero?")
    log(f"    -> {len(nonzero_T)} nonzero (out of 27 possible R vectors)")
    for d in sorted(dist_groups):
        label = "self" if d == 0 else ("nearest-neighbor" if d == 1 else f"dist={d}")
        log(f"       {label}: {len(dist_groups[d])} matrices")
    nn_only = all(sum(min(r, 3-r) for r in R) <= 1 for R in nonzero_T)
    log(f"    Nearest-neighbor only? {'YES' if nn_only else 'NO'}")

    log(f"\nQ3: Are there flat bands? How many?")
    log(f"    -> {n_flat} flat band(s) out of 20")
    for bi, bval in flat_bands:
        log(f"       Band {bi}: eigenvalue = {bval:.6f}")

    log(f"\nQ4: What fraction of coupling is nearest-neighbor vs beyond?")
    total_sq = sum(tier_sq.values())
    for d in sorted(tier_sq):
        label = "self" if d == 0 else ("NN" if d == 1 else f"dist={d}")
        log(f"    {label}: {100*tier_sq[d]/total_sq:.2f}%")
    nn_frac = tier_sq.get(1, 0) / total_sq
    beyond_frac = sum(v for d,v in tier_sq.items() if d > 1) / total_sq
    log(f"    -> NN fraction: {100*nn_frac:.2f}%, beyond-NN: {100*beyond_frac:.2f}%")

    log(f"\nQ5: At which k-points does lambda=-3 appear?")
    if lam3_kpoints:
        for n_tuple, label, cnt in sorted(lam3_kpoints):
            log(f"    k=2pi/3*{n_tuple} ({label}): multiplicity {cnt}")
        log(f"    -> lambda=-3 appears at {len(lam3_kpoints)} k-points, total {sum(c for _,_,c in lam3_kpoints)} eigenvalues")
    else:
        log(f"    -> lambda=-3 does not appear as exact H(k) eigenvalue")

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
