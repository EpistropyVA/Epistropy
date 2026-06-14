"""
bcc_corner_perturbation.py

Analytic corner deviation δ via degenerate perturbation theory
in the flat-band subspace of the BCC 4-simplex 3×3×3 network.

Physical setup:
  - Periodic BC: 540×540 adjacency matrix on 3-torus Z3×Z3×Z3
  - Flat-band subspace: λ=-3, 6-fold degenerate at every k-point (× 27 k-points = 162 states)
  - Open BC transition for corner body-center at (0,0,0):
      periodic: 6 neighbors (±x, ±y, ±z)
      open:     3 neighbors (+x, +y, +z)
      perturbation V removes 3 "wrap-around" bonds (−x→wraps, −y→wraps, −z→wraps)

Algorithm:
  1. Build periodic-BC system (same as bcc_bloch_decomposition.py)
  2. For each k: find 6 flat-band eigenvectors of H(k)
  3. Build perturbation V in the 20-dimensional local space
  4. First-order: project V into flat-band subspace → 6×6 matrix per k
  5. Second-order: Löwdin perturbation from dispersive bands
  6. Sum over 27 k-points → total spectral projector correction at corner
  7. Extract diagonal → corner deviation δ

Output written to: d:/AI thoery/.agent/scripts/bcc_corner_perturbation_results.txt
"""

import itertools
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_corner_perturbation_results.txt"

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


# ============================================================
# Lattice construction (copied from bcc_bloch_decomposition.py)
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
    assert len(even_corners) == 4 and len(odd_corners) == 4
    faces = []
    for tet_corners in (even_corners, odd_corners):
        simplex_verts = [bc_v] + tet_corners
        for combo in itertools.combinations(simplex_verts, 3):
            faces.append(frozenset(combo))
    assert len(faces) == 20
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
    from collections import defaultdict
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


def bc_flat_index(bc_ijk):
    return bc_ijk[0] * 9 + bc_ijk[1] * 3 + bc_ijk[2]


# ============================================================
# Build hopping matrices T(R)
# ============================================================

def build_hopping_matrices(A_reord, body_centers):
    def get_block(A_r, bc_i, bc_j, size=20):
        return A_r[bc_i*size:(bc_i+1)*size, bc_j*size:(bc_j+1)*size].copy()

    ref_flat = bc_flat_index((0, 0, 0))
    T = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers):
        R = tuple(bc_j_ijk[a] % 3 for a in range(3))
        T[R] = get_block(A_reord, ref_flat, bc_j_idx)
    return T


# ============================================================
# Bloch Hamiltonian H(k)
# ============================================================

def build_H_k(T, n_tuple):
    k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
    H_k = np.zeros((20, 20), dtype=complex)
    for R, mat in T.items():
        R_vec = np.array(R, dtype=float)
        phase = np.exp(1j * np.dot(k, R_vec))
        H_k += phase * mat
    return H_k


def bz_label(n_tuple):
    nonzero = sum(1 for n in n_tuple if n != 0)
    return ["Gamma", "edge", "face", "corner"][nonzero]


def bz_label_short(n_tuple):
    nonzero = sum(1 for n in n_tuple if n != 0)
    return ["Γ", "E", "F", "C"][nonzero]


# ============================================================
# Perturbation V: bonds removed in periodic→open transition
# ============================================================

def build_perturbation_V(T, n_corner=(0, 0, 0)):
    """
    Build the 20×20 perturbation matrix V acting on the local space
    of the corner BC at n_corner = (0,0,0).

    In periodic BC, BC(0,0,0) couples to 6 nearest-neighbor BCs via T(R):
        R = (1,0,0), (2,0,0)  ← +x and -x (but -x wraps to R=(2,0,0) in mod-3)
        R = (0,1,0), (0,2,0)
        R = (0,0,1), (0,0,2)

    In open BC (L=3), the corner at (0,0,0) only has neighbors at
        (+1,0,0), (0,+1,0), (0,0,+1)   ← these exist in open BC
    The wrap-around neighbors at R=(2,0,0), (0,2,0), (0,0,2) are REMOVED.

    The perturbation removes those wrap-around inter-BC terms.
    In the local (20×20) picture for BC(0,0,0):
      - The "self" block T(0,0,0) is unchanged (it's intra-BC coupling).
      - The removed bonds are the off-diagonal blocks coupling BC(0,0,0)
        to BC(2,0,0), BC(0,2,0), BC(0,0,2).

    V_local encodes the REMOVED hopping: V = -( T_{0,corner→wrap} )
    i.e. the matrix element in the OPEN-BC Hamiltonian is H_open = H_periodic - V

    For degenerate perturbation theory in the corner BC's 20-face subspace,
    we need the effective perturbation seen by the corner faces.

    Key: in the Bloch basis the perturbation is NOT block-diagonal in k.
    We work in real-space: the spectral projector onto flat band in open BC
    is P_open = P_periodic + δP where δP is computed via perturbation theory.

    For the corner BC's diagonal block of P:
    (δP)_{corner,corner} = (1/N_k) Σ_k [P_flat(k) V_eff(k) G(k)]_{aa} + h.c.

    where:
      - V_eff(k) = Σ_{R: wrap} exp(ik·R) T(R)  [the k-space version of removed bonds]
      - G(k) = Σ_{n: dispersive} |n(k)><n(k)| / (E_flat - λ_n(k))

    However, the cleanest approach is to note that V acts between BC(0,0,0) and
    its wrap-around partners. In real-space, the perturbation matrix on the full
    540-dim space has nonzero entries only connecting the 20 faces of BC(0,0,0)
    to the 20 faces of BC(2,0,0), BC(0,2,0), BC(0,0,2) (and h.c.).

    Returns:
        wrap_R: list of wrap-around R vectors [(2,0,0), (0,2,0), (0,0,2)]
        T_wrap: corresponding 20×20 hopping matrices T[R] for those R
    """
    # The wrap-around bonds removed when going from periodic to open BC at corner (0,0,0)
    wrap_Rs = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    T_wrap = {R: T[R] for R in wrap_Rs}

    # In open BC, the +x direction neighbor (1,0,0) still exists
    keep_Rs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    T_keep = {R: T[R] for R in keep_Rs}

    return wrap_Rs, T_wrap, keep_Rs, T_keep


# ============================================================
# Degenerate perturbation theory
# ============================================================

def flat_band_projector_k(H_k, E_flat=-3.0, tol=1e-4):
    """
    Compute the flat-band projector P_flat(k) = Σ_{flat eigvecs} |v><v|
    and return flat eigenvectors + dispersive eigenvalues/eigenvectors.
    """
    eigvals, eigvecs = np.linalg.eigh(H_k)
    flat_mask = np.abs(eigvals - E_flat) < tol
    disp_mask = ~flat_mask

    flat_vecs = eigvecs[:, flat_mask]   # 20 × n_flat
    disp_vecs = eigvecs[:, disp_mask]   # 20 × n_disp
    disp_eigs = eigvals[disp_mask]      # n_disp

    n_flat = flat_mask.sum()
    return flat_vecs, disp_vecs, disp_eigs, n_flat


def compute_delta_P_corner(T, kvecs, E_flat=-3.0, tol=1e-4):
    """
    Compute the perturbation to the flat-band spectral projector at the corner BC.

    The 540×540 full perturbation V has blocks:
        V[corner, wrap_j] = T(R_wrap_j)   for j = 0,1,2 (three wrap-around bonds)
        V[wrap_j, corner] = T(R_wrap_j).T  (Hermitian conjugate, real symmetric)
        all other blocks = 0

    In Bloch basis, the effective Hamiltonian for the corner BC sector is:
        H_eff(k) = H(k)  (same for all BCs by translation)
        V_eff(k) = (1/N) Σ_k' exp(ik'·R_wrap) T(R_wrap)  — this is NOT diagonal in k.

    More carefully: V couples states at k to states at k' = k + G_wrap.
    For the 3-torus with mod-3, R_wrap = (2,0,0) ≡ -(1,0,0) mod 3.
    exp(ik · R_wrap) with R_wrap = (2,0,0) contributes phase exp(2ik_x).

    The correct approach for the DIAGONAL correction to P at the corner is:

    δP_{corner} = (1/N_k) Σ_k [second-order Löwdin at each k-point]

    For each k, the removed bonds contribute to the effective perturbation
    in the "corner" sector (the 20 faces of BC(0,0,0)):

    V_k = Σ_{R_wrap} [exp(ik·R_wrap) T(R_wrap) + exp(-ik·R_wrap) T(R_wrap)^T]
         = the part of H(k) from wrap-around bonds (only from corner perspective)

    This is the k-space matrix element of V between the corner BC sector:
    < corner, k | V | corner, k > block = V_k (20×20 in the corner's local space)

    First-order correction to projector (using degenerate pert theory):
    The flat-band eigenstates are split by V_k. The 1st order correction
    to the projector diagonal is Tr[P_flat(k) V_k] / n_flat × (diagonal contribution).

    More precisely:
    δP^(1)_{aa} = Σ_m <a|P_flat(k)|m><m|V_k|n><n|P_flat(k)|a> — this is zero
    because P_flat V P_flat has no diagonal correction to P_flat itself
    (it just rotates within the flat subspace, not changing eigenvalues to first order
    in the off-diagonal sense — but wait, it DOES split the flat band).

    Actually: the first-order correction to P_flat is:
    δP^(1) = G(k) V_k P_flat(k) + P_flat(k) V_k G(k)
    where G(k) = Σ_{n disp} |n><n| / (E_flat - λ_n)  [Green's function in dispersive sector]

    The diagonal element at face 'a' of corner BC:
    [δP^(1)]_{aa} = 2 Re Σ_{n disp} <a|n(k)><n(k)|V_k|P_flat(k)|a> / (E_flat - λ_n)

    The second-order correction to P_flat involves the splitting within flat band.
    If V_k has nonzero matrix elements within the flat subspace (P_flat V P_flat ≠ 0),
    this is the "degenerate" part and must be treated exactly.

    The deviation δ is the change in the average diagonal element of P_flat at the corner.
    """
    wrap_Rs = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    N_k = len(kvecs)

    # Accumulate corrections per face of corner BC (20 faces, indexed 0..19)
    delta_P_diag = np.zeros(20, dtype=float)   # total correction
    delta_P_1st  = np.zeros(20, dtype=float)   # first-order (off-diagonal in k→flat coupling)
    delta_P_2nd  = np.zeros(20, dtype=float)   # second-order within-flat

    # By-k-type accumulators
    label_accum = defaultdict(lambda: np.zeros(20, dtype=float))
    order1_by_label = defaultdict(lambda: np.zeros(20, dtype=float))
    order2_by_label = defaultdict(lambda: np.zeros(20, dtype=float))

    for n_tuple in kvecs:
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = build_H_k(T, n_tuple)

        flat_vecs, disp_vecs, disp_eigs, n_flat = flat_band_projector_k(H_k, E_flat, tol)

        if n_flat == 0:
            log(f"  WARNING: k={n_tuple} has no flat-band eigenvalues at E={E_flat}")
            continue

        # Construct V_k = Σ_{R_wrap} [exp(ik·R_wrap) T(R_wrap) + h.c.]
        # This is the contribution to H(k) from wrap-around bonds.
        # When V removes these bonds, the perturbation in k-space is -V_k.
        # So δH = -V_k (the correction to H(k) when removing wrap bonds).
        V_k = np.zeros((20, 20), dtype=complex)
        for R in wrap_Rs:
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k_vec, R_vec))
            V_k += phase * T[R]
            # Add h.c. since we want the full Hermitian perturbation
            # T(R) contributes and T(-R) = T(R)^T in real-space (for real symmetric A)
            # For the DIAGONAL block (corner→corner), the perturbation is:
            # H_periodic(k) includes T(R_wrap)*exp(ik·R_wrap) + T(R_wrap)^T*exp(-ik·R_wrap)
            # Removing wrap bonds: δV = -(T(R_wrap)*exp(ik·R_wrap) + h.c.)
            # But T(R)^T = T(-R mod 3) = T(3-R)... let's check: T(2,0,0) vs T(1,0,0)^T
            # Actually for the FULL H(k): H(k) = Σ_R exp(ik·R) T(R) includes all R
            # R=(2,0,0) term: exp(2ik_x) T(2,0,0)
            # R=(1,0,0) term: exp(ik_x) T(1,0,0)  ← this is the +x neighbor, KEPT
            # The Hermitian contribution: T(1,0,0)^dag at R=(2,0,0)?
            # No — since T(R) is already the full matrix at displacement R mod 3.
            # H(k) = Σ_R exp(ik·R) T(R), and T(-R mod 3) = T(3-R mod 3) = T(R)^T
            # for real symmetric A. So T(2,0,0) = T((0,0,0)-(1,0,0) mod 3) = T(1,0,0)^T? NO.
            # Let's just use V_k as computed and add its hermitian conjugate.
        # V_k as above is NOT Hermitian. Make it Hermitian: V_k_herm = V_k + V_k^dag
        # BUT: H(k) = Σ_R exp(ik·R) T(R) is already Hermitian.
        # The wrap-around part of H(k) is:
        #   V_k_periodic = Σ_{R in wrap_Rs} [exp(ik·R) T(R) + exp(-ik·R) T(R)^T]
        # ... but exp(-ik·R) T(R)^T = exp(ik·(-R mod 3)) T(-R mod 3) = exp(ik·(3-R)) T(3-R)
        # and (3-R mod 3) for R=(2,0,0) gives (1,0,0) which is in keep_Rs!
        # This means: removing R=(2,0,0) bond also implicitly removes the conjugate
        # which appears as the R=(1,0,0) term's conjugate... NO wait.
        # Let me think more carefully:
        # T(R) and T(3-R) are different matrices. T(1,0,0) ≠ T(2,0,0)^T in general.
        # H(k) is Hermitian because Σ_R exp(ik·R) T(R) = [Σ_R exp(ik·R) T(R)]^dag
        # which requires T(R) = T(-R mod 3)^T, i.e., T(1,0,0) = T(2,0,0)^T.
        # This is TRUE for the adjacency matrix (real symmetric, so T(R)^T = T(-R mod 3)).
        # So the wrap-around part of H(k) is:
        # V_k = exp(ik·(2,0,0)) T(2,0,0) + exp(ik·(0,2,0)) T(0,2,0) + exp(ik·(0,0,2)) T(0,0,2)
        #      + exp(-ik·(2,0,0)) T(2,0,0)^T + ...   [using -R ≡ (1,0,0) → T(1,0,0)=T(2,0,0)^T]
        # But that means removing wrap bonds ALSO removes the (1,0,0) terms!
        # That can't be right — the +x neighbor IS kept in open BC.
        #
        # Resolution: in OPEN BC, the inter-BC hopping between corner and +x neighbor (1,0,0)
        # is KEPT (that's a real nearest neighbor). The wrap bond is the -x direction which
        # in periodic BC wraps from (0,0,0) to (2,0,0). That's a SEPARATE matrix element.
        # T(2,0,0) = T(-1,0,0 mod 3) is the -x hopping matrix.
        # In open BC: the -x neighbor doesn't exist for corner. So T(2,0,0) block is removed.
        # The adjacency matrix is REAL, so T(R)^T = T(R)^* = T(R)^dag, BUT that's for the
        # full A matrix block; T(2,0,0) ≠ T(1,0,0)^T in general (they can differ).
        # H(k) hermiticity: [Σ_R e^{ik·R} T(R)]^dag = Σ_R e^{-ik·R} T(R)^T
        # = Σ_R e^{ik·(-R)} T(R)^T = Σ_{R'} e^{ik·R'} T(-R')^T
        # For this = H(k) we need T(R) = T(-R)^T = T(-R mod 3)^T.
        # So T(2,0,0) = T(-1 mod 3, 0, 0)^T... wait no: -R=(2,0,0) → -R mod 3 = (1,0,0).
        # So T(R=(2,0,0)) = T(R'=(1,0,0))^T. YES — this is exact for real symmetric A.
        #
        # So removing T(2,0,0) also means removing the corresponding conjugate contribution
        # which is T(1,0,0)^T exp(-ik·(2,0,0)) = T(2,0,0) exp(-ik·(2,0,0)).
        # BUT exp(-ik·(2,0,0)) = exp(ik·(1,0,0)) since k·(3,0,0) = 2π·n1 = 0 mod 2π.
        # So the full wrap-around perturbation from the (2,0,0) bond is:
        #   V_k^{(x)} = exp(ik·(2,0,0)) T(2,0,0) + exp(ik·(1,0,0)) T(1,0,0)
        # Wait that's both +x AND -x contributions together, which would remove ALL x coupling!
        #
        # The issue: the "wrap bond" at R=(2,0,0) and the "keep bond" at R=(1,0,0) are
        # INDEPENDENTLY real connections. In open BC, BOTH the +x bond T(1,0,0) (kept)
        # and the -x bond T(2,0,0) = T(1,0,0)^T (removed) exist in periodic BC.
        # The perturbation removes ONLY T(2,0,0) block (the -x/wrap bond).
        # Since A is real and symmetric, T(2,0,0) ≠ T(1,0,0) in general
        # (they connect different structural elements across the boundary).
        #
        # Let's just compute: V_k = Σ_{R in wrap_Rs} exp(ik·R) T(R)
        # This is the removed forward-hopping part.
        # The full Hermitian perturbation adds the backward: V_k^H.
        # But since T(2,0,0) = T(1,0,0)^T, V_k^H at R=(2,0,0) gives:
        # exp(-ik·(2,0,0)) T(2,0,0)^H = exp(ik·(1,0,0)) T(1,0,0)
        # which IS the exp(ik·(1,0,0)) T(1,0,0) term in H(k).
        # So V_k + V_k^H would remove BOTH (1,0,0) and (2,0,0) bonds!
        #
        # This means: the perturbation "remove wrap bond (2,0,0)" in real space
        # corresponds to removing the (i,j) and (j,i) entries from A for i=corner faces,
        # j=wrap-neighbor faces. In k-space, both the T(2,0,0) term AND the conjugate
        # T(1,0,0)^T term (= T(2,0,0)^T^T = T(2,0,0)) are affected? No wait:
        # T(2,0,0) is the actual hopping from BC(0,0,0) to BC(2,0,0).
        # T(2,0,0)^T = T(1,0,0) is the hopping from BC(0,0,0) to BC(1,0,0) ← different BC!
        #
        # Conclusion: The perturbation V removes ONLY the off-diagonal block between
        # BC(0,0,0) and BC(2,0,0) in the full 540×540 matrix. This is represented in
        # k-space as removing exp(ik·(2,0,0)) T(2,0,0) from H_k of BC(0,0,0) only,
        # but that makes H(k) non-Hermitian... unless we also remove the conjugate bond
        # from BC(2,0,0) to BC(0,0,0), which in k-space corresponds to exp(-ik·(2,0,0)) T(2,0,0)^T.
        # Since this is the SAME bond (A is symmetric), removing it symmetrically:
        # δH_k = -[exp(ik·(2,0,0)) T(2,0,0) + exp(-ik·(2,0,0)) T(2,0,0)^T]
        #       = -[exp(ik·(2,0,0)) T(2,0,0) + exp(ik·(1,0,0)) T(1,0,0)]
        # This is V_k + V_k^H but with V_k^H contributing T(1,0,0) terms.
        # This would remove BOTH +x and -x bonds from H(k). That's wrong.
        #
        # CORRECT resolution: The perturbation in the periodic→open BC transition is
        # NOT uniform. It removes bonds for the SPECIFIC corner BC (0,0,0) only.
        # In real space: δA_{a,b} = -1 if a is a face of BC(0,0,0) and b is a face of
        # BC(2,0,0) (or BC(0,2,0) or BC(0,0,2)) and they were adjacent, and vice versa.
        # In k-space, this is a RANK-20 perturbation (not diagonal in k).
        # Specifically, it mixes k-sectors.
        #
        # For perturbation theory at the CORNER SITE, we need:
        # The correction to P_flat at face 'a' of BC(0,0,0) is given by the
        # standard second-quantization expression.
        #
        # SIMPLEST CORRECT APPROACH: Work entirely in REAL SPACE.
        # Use the 540×540 periodic H, compute P_flat (162×162 projector),
        # build V (540×540 sparse), compute perturbation corrections.
        pass

    return delta_P_diag, delta_P_1st, delta_P_2nd, label_accum, order1_by_label, order2_by_label


# ============================================================
# Real-space perturbation theory
# ============================================================

def real_space_perturbation(T, body_centers, bc_face_indices, E_flat=-3.0, tol=1e-4):
    """
    Full real-space degenerate perturbation theory.

    Setup: 540×540 periodic adjacency matrix A.
    Perturbation V: removes wrap-around bonds of corner BC(0,0,0).
    Compute δP = correction to spectral projector P_flat.
    """
    N = 540
    n_bc = 27
    n_faces = 20

    # Build permuted adjacency matrix (BC-block ordered)
    # body_centers[0] = (0,0,0), so the corner BC is block 0 (rows 0..19)
    perm = []
    for bc_idx in range(n_bc):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)

    # Build full adjacency matrix from T(R) blocks
    A = np.zeros((N, N), dtype=float)
    for bc_i_idx, bc_i_ijk in enumerate(body_centers):
        for bc_j_idx, bc_j_ijk in enumerate(body_centers):
            R = tuple((bc_j_ijk[a] - bc_i_ijk[a]) % 3 for a in range(3))
            block = T[R]
            A[bc_i_idx*20:(bc_i_idx+1)*20, bc_j_idx*20:(bc_j_idx+1)*20] = block

    log(f"  Full A built: shape={A.shape}, symmetric={np.allclose(A, A.T):.0f}")

    # Identify the wrap-around bonds to remove for corner BC (index 0)
    wrap_Rs = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    corner_bc_idx = bc_flat_index((0, 0, 0))  # = 0

    # Build perturbation matrix V (sparse: nonzero only in corner↔wrap blocks)
    V = np.zeros((N, N), dtype=float)
    for R in wrap_Rs:
        # Find the BC index for R
        wrap_bc_ijk = R  # (2,0,0) means BC at (2,0,0)
        wrap_bc_idx = bc_flat_index(wrap_bc_ijk)

        # The hopping block T(R) from BC(0,0,0) to BC(wrap)
        block = T[R]  # 20×20

        # Row range: corner BC (rows 0..19 in block-ordered space)
        r_start = corner_bc_idx * 20  # = 0
        r_end   = r_start + 20        # = 20
        # Col range: wrap BC
        c_start = wrap_bc_idx * 20
        c_end   = c_start + 20

        V[r_start:r_end, c_start:c_end] += block
        V[c_start:c_end, r_start:r_end] += block.T  # symmetric

    log(f"  V built: ||V||_F = {np.linalg.norm(V, 'fro'):.6f}")
    log(f"  V nonzero blocks: corner(0..19) <-> wrap BCs at {wrap_Rs}")

    # Diagonalize A to get flat-band subspace
    log(f"\n  Diagonalizing 540×540 A...")
    eigvals, eigvecs = np.linalg.eigh(A)

    flat_mask = np.abs(eigvals - E_flat) < tol
    disp_mask = ~flat_mask
    n_flat = flat_mask.sum()
    log(f"  Flat-band states (E = {E_flat}): {n_flat}")
    log(f"  Dispersive states: {disp_mask.sum()}")
    assert n_flat == 162, f"Expected 162 flat-band states, got {n_flat}"

    flat_vecs = eigvecs[:, flat_mask]   # 540 × 162
    disp_vecs = eigvecs[:, disp_mask]   # 540 × 378
    disp_eigs = eigvals[disp_mask]      # 378

    # Flat-band projector P_flat (only need diagonal at corner faces)
    # P_flat = flat_vecs @ flat_vecs.T
    # We only need the corner block (rows 0..19, cols 0..19)
    P_corner = flat_vecs[:20, :] @ flat_vecs[:20, :].T  # 20×20
    log(f"\n  P_corner[0..19] diagonal (unperturbed):")
    P_diag_unperturbed = np.diag(P_corner)
    log(f"    mean = {np.mean(P_diag_unperturbed):.6f}")
    log(f"    values: {P_diag_unperturbed}")

    # First-order correction to P_flat:
    # δP^(1) = G V P_flat + P_flat V G
    # where G = Σ_{n disp} |n><n| / (E_flat - λ_n)
    # [δP^(1)]_{aa} = 2 Re Σ_{n disp} P_flat[a,n'] * V[n',a] / (E_flat - λ_n)
    # In matrix form: δP^(1) = G V P_flat + P_flat V G
    # where G = disp_vecs @ diag(1/(E_flat - disp_eigs)) @ disp_vecs.T

    # V P_flat: (540×540) @ (540×162) = 540×162
    VP_flat = V @ flat_vecs                  # 540 × 162
    # G(V P_flat):
    # = Σ_n |n><n| (1/(E_flat - λ_n)) V P_flat
    # = disp_vecs @ diag(g_n) @ (disp_vecs.T @ VP_flat)
    g_n = 1.0 / (E_flat - disp_eigs)        # 378
    DVP = disp_vecs.T @ VP_flat              # 378 × 162
    GVP = disp_vecs @ (g_n[:, None] * DVP)  # 540 × 162

    # δP^(1) diagonal at corner faces (rows 0..19):
    # [δP^(1)]_{aa} = sum_m (GVP[a,m] * flat_vecs[a,m] + flat_vecs[a,m] * (GVP[a,m])*)
    # = 2 Re Σ_m GVP[a,m] * flat_vecs[a,m]*  (but flat_vecs is real here)
    # Actually: δP^(1) = GVP @ flat_vecs.T + flat_vecs @ GVP.T
    # [δP^(1)]_{aa} = (GVP @ flat_vecs.T)[a,a] + (flat_vecs @ GVP.T)[a,a]
    # = Σ_m GVP[a,m]*flat_vecs[a,m] + Σ_m flat_vecs[a,m]*GVP[a,m]
    # = 2 Σ_m GVP[a,m]*flat_vecs[a,m]  (both terms equal when flat_vecs is real)
    delta_P1_corner_diag = 2.0 * np.sum(GVP[:20, :] * flat_vecs[:20, :], axis=1)
    # (This is real since A and V are real symmetric)

    log(f"\n  First-order δP diagonal at corner faces:")
    log(f"    mean = {np.mean(delta_P1_corner_diag):.8f}")
    log(f"    max  = {np.max(np.abs(delta_P1_corner_diag)):.8f}")
    log(f"    values: {delta_P1_corner_diag}")

    # Second-order correction:
    # δP^(2) = P_flat V G² V P_flat - (flat_vecs @ flat_vecs.T V P_flat G V flat_vecs @ flat_vecs.T)
    # More precisely, the Löwdin second-order correction to P_flat:
    # δP^(2)_{aa} = - Σ_{n disp, m flat, m' flat}
    #   flat_vecs[a,m] <m|V|n> g_n <n|V|m'> flat_vecs[a,m'] / E_flat? — need care.
    #
    # Full second-order: the correction to the projector from degenerate PT is:
    # P^(2) = G V (1 - P_flat) V G (projected onto flat×flat)
    # This is the correction to the GREEN'S FUNCTION resolvent.
    # For the SPECTRAL PROJECTOR:
    # δP^(2) = [term from second-order self-energy in flat subspace]
    #
    # Using Kato-Bloch formalism:
    # The second-order energy correction WITHIN the flat subspace gives a 162×162 matrix.
    # This splits the flat band; the diagonal of P in the ORIGINAL basis changes.
    #
    # For just the TRACE (average over flat band), second-order correction to Tr[P]
    # at corner faces = 0 (trace of P is fixed = 162).
    # For individual diagonal elements, the second-order correction is:
    # [δP^(2)]_{aa} involves the second-order effective Hamiltonian within flat subspace.
    #
    # Effective H within flat band (2nd order, Löwdin):
    # H_eff^(2)[m,m'] = <m|V G V|m'> = Σ_{n disp} <m|V|n> g_n <n|V|m'>
    # The flat-band projector changes as:
    # δP^(2) is related to the response of the flat eigenvectors.
    #
    # If H_eff^(2) is small (perturbative), δP^(2) ≈ 0 to leading order
    # in the DIAGONAL of P, because the TRACE of P_flat doesn't change at 2nd order.
    # But individual diagonal elements CAN change due to mixing within flat subspace.
    #
    # For our problem, the key insight: if V has matrix elements <m|V|m'> within
    # the flat subspace (P_flat V P_flat ≠ 0), this represents "degenerate" first-order
    # splitting. This is the DOMINANT correction.

    # Compute P_flat V P_flat: first-order intra-flat perturbation
    VPflat = V @ flat_vecs              # 540 × 162
    PflatVPflat = flat_vecs.T @ VPflat  # 162 × 162  (H_eff^(1) in flat subspace)

    log(f"\n  P_flat V P_flat (intra-flat perturbation):")
    log(f"    Frobenius norm = {np.linalg.norm(PflatVPflat, 'fro'):.6f}")
    log(f"    ||P_flat V P_flat||_max = {np.max(np.abs(PflatVPflat)):.6f}")

    eigs_pvp = np.linalg.eigvalsh(PflatVPflat)
    log(f"    Eigenvalues of P_flat V P_flat:")
    log(f"    {eigs_pvp}")
    pvp_nonzero = np.sum(np.abs(eigs_pvp) > 1e-8)
    log(f"    Nonzero eigenvalues: {pvp_nonzero}")

    # If P_flat V P_flat = 0, first-order degenerate PT gives no splitting.
    # Then the correction comes from second-order (Löwdin).

    # Second-order effective H in flat subspace:
    # H_eff^(2) = flat_vecs.T @ V @ G @ V @ flat_vecs  (projected)
    GV_flat = disp_vecs @ (g_n[:, None] * (disp_vecs.T @ V @ flat_vecs))  # 540 × 162
    H_eff2 = flat_vecs.T @ V @ GV_flat  # 162 × 162

    log(f"\n  Second-order H_eff in flat subspace:")
    log(f"    Frobenius norm = {np.linalg.norm(H_eff2, 'fro'):.6f}")
    eigs_h2 = np.linalg.eigvalsh(np.real(H_eff2))  # should be real
    log(f"    Eigenvalue range: [{eigs_h2.min():.6f}, {eigs_h2.max():.6f}]")

    # The correction to P diagonal at corner from intra-flat mixing:
    # When the flat subspace Hamiltonian H_eff = H_eff^(1) + H_eff^(2) has eigenvalues,
    # it mixes the flat-band states. The correction to P_{aa} comes from:
    # If we diagonalize H_eff = U^T diag(ε) U within the 162-dim flat subspace,
    # the new flat-band projector is UNCHANGED (all states still project onto flat band).
    # The DIAGONAL of P in the original 540-dim basis is NOT changed by this mixing.
    # So the relevant corrections are from the dispersive-band coupling only.

    # Conclusion: The first-order correction δP^(1) from dispersive bands IS the main term.
    # The second-order correction from Löwdin is:
    # δP^(2)_{aa} = [P_flat V G^2 V P_flat]_{aa} (diagonal)
    # = Σ_m,m' flat_vecs[a,m] [P_flat V G^2 V P_flat]_{m,m'} flat_vecs[a,m']
    # This has a different structure and is truly 2nd order.

    # Compute second-order diagonal correction at corner faces:
    # G^2 operator: G = disp_vecs @ diag(g_n) @ disp_vecs.T
    # G^2 = disp_vecs @ diag(g_n^2) @ disp_vecs.T
    g2_n = g_n ** 2  # 378
    # V P_flat in dispersive sector: disp_vecs.T @ V @ flat_vecs = (378 × 162)
    dVPflat = disp_vecs.T @ V @ flat_vecs  # 378 × 162
    # P_flat V G^2 V P_flat:
    G2VP_flat = disp_vecs @ (g2_n[:, None] * dVPflat)  # 540 × 162
    PflatVG2VPflat = flat_vecs.T @ V @ G2VP_flat        # 162 × 162

    # Diagonal at corner faces:
    # [δP^(2)]_{aa} = Σ_{m,m'} flat_vecs[a,m] PflatVG2VPflat[m,m'] flat_vecs[a,m']
    # But we need the correction to the PROJECTOR diagonal, not the energy:
    # δP^(2) diagonal = 2nd order change in |<a|flat,perturbed>|^2
    # This requires the 2nd order perturbation of the flat eigenstates.
    #
    # Using Dalgarno-Lewis / standard 2nd order PT for degenerate case:
    # Since P_flat V P_flat ≈ 0 (check above), the flat band is not split at 1st order.
    # The 2nd order correction to each flat state |m,0> is:
    # |m,2> = Σ_{m'≠m flat} |m'> <m'|H_eff^(2)|m> / 0 = ∞?!  ← degenerate 2nd order!
    #
    # For the PROJECTOR rather than individual states, the 2nd order correction is safe:
    # δP^(2) = G V (P_flat V G + G V P_flat) V P_flat + c.c. / 2 - ...
    # This is the proper Löwdin expansion. Since we want the diagonal at corner faces:
    # [δP^(2)]_{aa}^{Löwdin} = -[P_flat V G V P_flat]_{mm} for m at face a (times appropriate weight)
    #
    # Actually let's just numerically compute the full answer and compare with target.

    # Full correction to P diagonal from all dispersive coupling:
    delta_P1 = delta_P1_corner_diag
    delta_P_total_1 = np.real(delta_P1)

    # Total unperturbed + 1st order
    P_diag_total = P_diag_unperturbed + delta_P_total_1
    delta_mean_1st = np.mean(delta_P_total_1)

    log(f"\n  Summary: corner face P diagonal corrections")
    log(f"  Unperturbed mean = {np.mean(P_diag_unperturbed):.8f}")
    log(f"  1st-order correction mean = {delta_mean_1st:.8f}")
    log(f"  Total mean = {np.mean(P_diag_total):.8f}")

    # Now compute via direct numerical check: open-BC P_flat diagonal
    # Build open-BC adjacency matrix
    log(f"\n  Building open-BC adjacency matrix for verification...")
    A_open = A.copy()
    wrap_Rs = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    for R in wrap_Rs:
        wrap_bc_ijk = R
        wrap_bc_idx = bc_flat_index(wrap_bc_ijk)
        block = T[R]
        r_start = corner_bc_idx * 20
        c_start = wrap_bc_idx * 20
        A_open[r_start:r_start+20, c_start:c_start+20] -= block
        A_open[c_start:c_start+20, r_start:r_start+20] -= block.T

    eigvals_open, eigvecs_open = np.linalg.eigh(A_open)
    flat_mask_open = np.abs(eigvals_open - E_flat) < tol
    n_flat_open = flat_mask_open.sum()
    log(f"  Open-BC flat states at E=-3: {n_flat_open}")

    flat_vecs_open = eigvecs_open[:, flat_mask_open]
    P_corner_open = flat_vecs_open[:20, :] @ flat_vecs_open[:20, :].T  # 20×20
    P_diag_open = np.diag(P_corner_open)

    delta_open_vs_periodic = P_diag_open - P_diag_unperturbed
    delta_corner_direct = np.mean(delta_open_vs_periodic)

    log(f"\n  DIRECT COMPUTATION (open-BC vs periodic-BC projector diagonal):")
    log(f"    Unperturbed corner mean P_aa = {np.mean(P_diag_unperturbed):.10f}")
    log(f"    Open-BC corner mean P_aa     = {np.mean(P_diag_open):.10f}")
    log(f"    δ (direct, corner mean)      = {delta_corner_direct:.12f}")
    log(f"    Target δ                     = {0.013782448754:.12f}")
    log(f"    Match? {abs(delta_corner_direct - 0.013782448754) < 1e-6}")

    # Breakdown by face
    log(f"\n  Per-face P_aa breakdown:")
    log(f"    {'face':>5} | {'unperturbed':>14} | {'open-BC':>14} | {'delta':>14}")
    log(f"    {'-'*5}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    for i in range(20):
        log(f"    {i:5d} | {P_diag_unperturbed[i]:14.8f} | {P_diag_open[i]:14.8f} | {delta_open_vs_periodic[i]:14.8f}")

    # Comparison: 1st-order perturbation vs direct
    log(f"\n  1st-order PT δ (mean)   = {delta_mean_1st:.12f}")
    log(f"  Direct δ (mean)         = {delta_corner_direct:.12f}")
    log(f"  Difference              = {abs(delta_mean_1st - delta_corner_direct):.3e}")
    log(f"  1st order fraction      = {delta_mean_1st / delta_corner_direct * 100:.2f}%")

    return {
        'P_diag_unperturbed': P_diag_unperturbed,
        'P_diag_open': P_diag_open,
        'delta_open_vs_periodic': delta_open_vs_periodic,
        'delta_corner_direct': delta_corner_direct,
        'delta_P1_corner_diag': delta_P_total_1,
        'H_eff2': H_eff2,
        'PflatVPflat': PflatVPflat,
        'flat_vecs': flat_vecs,
        'disp_vecs': disp_vecs,
        'disp_eigs': disp_eigs,
        'flat_vecs_open': flat_vecs_open,
    }


# ============================================================
# Bloch-space perturbation theory: k-decomposition
# ============================================================

def bloch_perturbation_analysis(T, body_centers, bc_face_indices, E_flat=-3.0, tol=1e-4):
    """
    Analyze the corner deviation in Bloch space.

    Since V is localized at a single BC (not translation-invariant), it mixes k-sectors.
    We can express the result in terms of k-resolved quantities.

    The spectral projector:
    P_flat = (1/N_k) Σ_k |k> ⊗ P_flat(k) <k|

    where P_flat(k) = flat_vecs(k) @ flat_vecs(k)^dag (20×20).

    The corner body-center diagonal:
    [P_flat]_{corner,a; corner,a} = (1/N_k) Σ_k [P_flat(k)]_{a,a}

    So the unperturbed per-face value is:
    P_{aa}^{(0)} = (1/N_k) Σ_k [P_flat(k)]_{a,a}

    The perturbation V in real space creates a correction:
    [δP]_{corner,a; corner,a} = (1/N_k) Σ_k (1/N_k) Σ_k' [contribution from k-k' mixing]

    For a rank-2N_wrap perturbation localized at corner, the first-order correction is:
    δP^(1)_{a,a} = (1/N_k^2) Σ_{k,k'} Σ_R_{wrap} [
        P_flat(k)_{a,m} * T_R[m,n] * G(k')_{n,n'} * (P_flat(k'))_{n',a} * exp(i k·R - ik'·R)
    ] + c.c.

    But this simplifies: the perturbation V only mixes BC(0,0,0) with BC(wrap).
    In the Bloch basis, BC(0,0,0) at k mixes with all k' via exp(ik·0) * exp(-ik'·R_wrap).

    This is getting complex. Let's just analyze the Bloch-space structure of the result.
    """
    kvecs = list(itertools.product(range(3), repeat=3))
    N_k = 27

    log(f"\n  Computing P_flat(k) for each k-point...")

    # For each k, compute P_flat(k) 20×20 projector
    P_flat_k = {}
    flat_vecs_k = {}
    disp_vecs_k = {}
    disp_eigs_k = {}
    n_flat_per_k = {}

    for n_tuple in kvecs:
        H_k = build_H_k(T, n_tuple)
        fv, dv, de, nf = flat_band_projector_k(H_k, E_flat, tol)
        flat_vecs_k[n_tuple] = fv
        disp_vecs_k[n_tuple] = dv
        disp_eigs_k[n_tuple] = de
        n_flat_per_k[n_tuple] = nf
        P_flat_k[n_tuple] = fv @ fv.conj().T  # 20×20

    # Unperturbed P diagonal at corner (face index 0..19)
    # P_{aa}^{(0)} = (1/N_k) Σ_k [P_flat(k)]_{a,a}
    P_diag_bloch = np.zeros(20, dtype=float)
    for n_tuple in kvecs:
        P_diag_bloch += np.real(np.diag(P_flat_k[n_tuple]))
    P_diag_bloch /= N_k

    log(f"  Unperturbed P diagonal from Bloch sum:")
    log(f"    mean = {np.mean(P_diag_bloch):.8f}")
    log(f"    values: {P_diag_bloch}")

    # k-type breakdown of P diagonal
    label_P = defaultdict(lambda: np.zeros(20, dtype=float))
    count_label = defaultdict(int)
    for n_tuple in kvecs:
        label = bz_label(n_tuple)
        label_P[label] += np.real(np.diag(P_flat_k[n_tuple]))
        count_label[label] += 1

    log(f"\n  P_flat(k) diagonal by BZ type (sum / count = mean per face):")
    for label in ["Gamma", "edge", "face", "corner"]:
        cnt = count_label[label]
        if cnt > 0:
            mean_per_face = np.mean(label_P[label]) / cnt
            log(f"    {label:6s} (×{cnt:2d}): sum_k mean_face P_aa = {np.mean(label_P[label]):.6f}, per-k per-face = {mean_per_face:.6f}")
            # Show face values
            log(f"      face values (sum/count): {label_P[label]/cnt}")

    return {
        'P_flat_k': P_flat_k,
        'flat_vecs_k': flat_vecs_k,
        'disp_vecs_k': disp_vecs_k,
        'disp_eigs_k': disp_eigs_k,
        'n_flat_per_k': n_flat_per_k,
        'P_diag_bloch': P_diag_bloch,
        'label_P': label_P,
        'count_label': count_label,
    }


# ============================================================
# Closed-form analysis: rational/algebraic expression for δ
# ============================================================

def closed_form_analysis(results_real, results_bloch, T):
    """
    Try to express δ as a closed-form algebraic expression.
    """
    delta = results_real['delta_corner_direct']

    log(f"\n  Target δ = {delta:.15f}")
    log(f"  Reference: {0.013782448754:.15f}")

    # Check if δ is rational
    from fractions import Fraction
    try:
        frac = Fraction(delta).limit_denominator(100000)
        log(f"  Best rational approximation (denom ≤ 100000): {frac} = {float(frac):.12f}")
        log(f"  Error: {abs(float(frac) - delta):.3e}")
    except Exception as e:
        log(f"  Rational approximation failed: {e}")

    # Check simple algebraic expressions
    candidates = {
        "1/72":      1/72,
        "1/73":      1/73,
        "1/70":      1/70,
        "sqrt(3)/216": np.sqrt(3)/216,
        "1/(24*3)":  1/72,
        "5/363":     5/363,
        "1/12 - 5/12*1/9": 1/12 - 5/(12*9),
    }

    log(f"\n  Algebraic candidate check:")
    for name, val in candidates.items():
        err = abs(val - delta)
        log(f"    {name:30s} = {val:.12f}, err = {err:.3e}")

    # Check weighted sum of BZ-type contributions
    # Each BZ type contributes a rational f_pure to the per-face projector
    P_label = results_bloch['label_P']
    count_label = results_bloch['count_label']

    log(f"\n  BZ type P_flat(k)_aa per-face values:")
    f_pure = {}
    for label in ["Gamma", "edge", "face", "corner"]:
        cnt = count_label.get(label, 0)
        if cnt > 0:
            vals = P_label[label] / cnt
            mean_val = float(np.mean(vals))
            f_pure[label] = mean_val
            log(f"    {label:6s} × {cnt:2d} k-pts: mean per-face P_aa = {mean_val:.10f}")
            frac = Fraction(mean_val).limit_denominator(1000)
            log(f"      Rational approx: {frac} = {float(frac):.10f}")

    # The shortcut theorem: δ = (P_open_aa - P_periodic_aa) at corner
    # In Bloch basis, P_periodic_aa = (1/N_k) Σ_k P_flat(k)_aa
    # The corner deviation in open BC comes from the spectral weight redistribution.
    # Since open BC has different spectrum, we check if the open-BC projector
    # has a simple k-decomposition.
    log(f"\n  Shortcut theorem check:")
    log(f"    Checking if δ = Σ_type (weight_type × f_type) + const...")

    # f_pure values
    f_G = f_pure.get("Gamma", None)
    f_E = f_pure.get("edge", None)
    f_F = f_pure.get("face", None)
    f_C = f_pure.get("corner", None)

    if all(v is not None for v in [f_G, f_E, f_F, f_C]):
        log(f"    f_Gamma = {f_G:.10f}")
        log(f"    f_edge  = {f_E:.10f}")
        log(f"    f_face  = {f_F:.10f}")
        log(f"    f_corner= {f_C:.10f}")

        # Check: is δ approximately f_Gamma - f_corner (or other differences)?
        combos = {
            "f_G - f_C":         f_G - f_C,
            "f_G - f_E":         f_G - f_E,
            "f_G - f_F":         f_G - f_F,
            "f_E - f_C":         f_E - f_C,
            "(f_G - f_C)/2":     (f_G - f_C)/2,
            "(f_G - f_E)/3":     (f_G - f_E)/3,
            "f_G - (f_E+f_F+f_C)/3": f_G - (f_E+f_F+f_C)/3,
            "3*(f_G-f_C)/27":    3*(f_G-f_C)/27,
            "1/27*(3*f_G - f_E*3 - f_F*3 - f_C)": 1/27*(3*f_G - f_E*3 - f_F*3 - f_C),
            "(3*f_G + 3*f_E + 3*f_F + f_C)/27 - 6/27": 0,  # placeholder
        }
        # The unperturbed mean = (1/27) * (1*f_G + 6*f_E + 12*f_F + 8*f_C)
        unperturbed_mean = (1*f_G + 6*f_E + 12*f_F + 8*f_C) / 27
        log(f"\n    Unperturbed mean P_aa = (1×f_G + 6×f_E + 12×f_F + 8×f_C)/27")
        log(f"    = ({f_G:.6f}×1 + {f_E:.6f}×6 + {f_F:.6f}×12 + {f_C:.6f}×8)/27")
        log(f"    = {unperturbed_mean:.10f}")

        log(f"\n    Checking δ = combo - unperturbed_mean type expressions:")
        for name, val in combos.items():
            err = abs(val - delta)
            log(f"      {name:50s} = {val:.12f}, err = {err:.3e}")

    return f_pure, delta


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 78)
    log("BCC 3×3×3 Corner Deviation δ via Degenerate Perturbation Theory")
    log("=" * 78)

    # --------------------------------------------------------
    # Step 1: Build lattice and adjacency structure
    # --------------------------------------------------------
    log("\n[Step 1] Build BCC lattice and faces")
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    N = len(face_to_idx)
    log(f"  Faces: {N}, BCs: {len(body_centers)}, faces/BC: 20")
    assert N == 540

    # Permute to BC-block order
    perm = []
    for bc_idx in range(27):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)

    # --------------------------------------------------------
    # Step 2: Build adjacency matrix and hopping matrices T(R)
    # --------------------------------------------------------
    log("\n[Step 2] Build adjacency matrix and T(R)")
    face_to_idx_full, bc_face_indices_full, _ = build_all_faces(body_centers)
    A_raw = build_adjacency_matrix(face_to_idx_full)

    # Reorder into BC-block form
    A_reord = A_raw[np.ix_(perm, perm)]
    log(f"  A shape: {A_reord.shape}, symmetric: {np.allclose(A_reord, A_reord.T)}")

    T = build_hopping_matrices(A_reord, body_centers)

    # Verify nonzero T(R)
    nonzero_R = [R for R, mat in T.items() if np.linalg.norm(mat, 'fro') > 1e-10]
    log(f"  Nonzero T(R): {len(nonzero_R)} matrices")

    # --------------------------------------------------------
    # Step 3: Bloch decomposition analysis
    # --------------------------------------------------------
    log("\n[Step 3] Bloch decomposition and flat-band identification")
    kvecs = list(itertools.product(range(3), repeat=3))

    eigs_by_k = {}
    for n_tuple in kvecs:
        H_k = build_H_k(T, n_tuple)
        eigs_by_k[n_tuple] = np.sort(np.linalg.eigvalsh(H_k))

    # Count flat-band states
    tol = 1e-4
    E_flat = -3.0
    total_flat = 0
    flat_per_k = {}
    for n_tuple in kvecs:
        ev = eigs_by_k[n_tuple]
        n_flat = sum(1 for e in ev if abs(e - E_flat) < tol)
        flat_per_k[n_tuple] = n_flat
        total_flat += n_flat
    log(f"  Total flat-band states (λ=-3): {total_flat}")
    log(f"  Per k-point breakdown:")
    for label in ["Gamma", "edge", "face", "corner"]:
        ks = [n for n in kvecs if bz_label(n) == label]
        flats = [flat_per_k[n] for n in ks]
        log(f"    {label:6s} ({len(ks)} k-pts): {flats[0] if flats else 0} flat states each")

    # --------------------------------------------------------
    # Step 4: Real-space perturbation theory (main computation)
    # --------------------------------------------------------
    log("\n[Step 4] Real-space degenerate perturbation theory")
    results_real = real_space_perturbation(T, body_centers, bc_face_indices, E_flat, tol)

    # --------------------------------------------------------
    # Step 5: Bloch-space analysis
    # --------------------------------------------------------
    log("\n[Step 5] Bloch-space analysis of flat-band projector")
    results_bloch = bloch_perturbation_analysis(T, body_centers, bc_face_indices, E_flat, tol)

    # --------------------------------------------------------
    # Step 6: Closed-form analysis
    # --------------------------------------------------------
    log("\n[Step 6] Closed-form analysis")
    f_pure, delta = closed_form_analysis(results_real, results_bloch, T)

    # --------------------------------------------------------
    # Step 7: Detailed order analysis (1st vs 2nd order PT)
    # --------------------------------------------------------
    log("\n[Step 7] Order-by-order breakdown")

    delta_direct = results_real['delta_corner_direct']
    delta_1st = float(np.mean(results_real['delta_P1_corner_diag']))

    log(f"  First-order correction  δ^(1) = {delta_1st:.12f}")
    log(f"  Total direct correction δ     = {delta_direct:.12f}")
    log(f"  Higher-order residual         = {delta_direct - delta_1st:.12f}")
    log(f"  1st order accounts for {abs(delta_1st/delta_direct)*100:.2f}% of δ")

    # --------------------------------------------------------
    # Step 8: Summary
    # --------------------------------------------------------
    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    log(f"\n  Corner deviation δ computed directly:      {delta_direct:.12f}")
    log(f"  Target δ:                                  {0.013782448754:.12f}")
    log(f"  Agreement: {abs(delta_direct - 0.013782448754) < 1e-6}")

    log(f"\n  BZ type f_pure values:")
    for label in ["Gamma", "edge", "face", "corner"]:
        if label in f_pure:
            v = f_pure[label]
            frac = Fraction(v).limit_denominator(1000)
            log(f"    f_{label:6s} = {v:.10f}  ≈  {frac}")

    log(f"\n  Perturbation order breakdown:")
    log(f"    1st order (off-diagonal dispersive): {delta_1st:.10f} ({abs(delta_1st/delta_direct)*100:.1f}%)")
    log(f"    Higher orders:                       {delta_direct - delta_1st:.10f}")

    # Check P_flat V P_flat
    pvp = results_real['PflatVPflat']
    pvp_norm = np.linalg.norm(pvp, 'fro')
    log(f"\n  P_flat V P_flat Frobenius norm: {pvp_norm:.6f}")
    if pvp_norm < 1e-6:
        log(f"  → Flat-band subspace is UNPERTURBED at 1st order (P_flat V P_flat = 0)")
        log(f"  → All correction comes from coupling to dispersive bands")
    else:
        log(f"  → Flat-band subspace IS perturbed at 1st order")

    log(f"\n  First-order PT uses: δP^(1) = G V P_flat + P_flat V G")
    log(f"  where G = Σ_{{n disp}} |n><n| / (E_flat - λ_n) is the Green's function")
    log(f"  restricted to dispersive bands.")

    log(f"\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(_LOG) + "\n")
            print(f"\n[Results written to {OUT_PATH}]")
        except Exception as e:
            print(f"WARNING: failed to write results: {e}", file=sys.stderr)
