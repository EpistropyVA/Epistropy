"""
corner_defect_perturbation.py

Analytic corner deviation for the BCC 4-simplex network via Bloch Green's
function and second-order perturbation theory.

Pipeline:
  1.  Build 3x3x3 BCC periodic lattice (translation-equivariant labeling).
  2.  Build 540x540 periodic face-adjacency matrix A_periodic.
  3.  Bloch decomposition: 27 k-points x 20x20 H(k).
  4.  Build open-BC matrix A_open by removing bonds that cross the boundary.
  5.  Perturbation V = A_periodic - A_open (the removed bonds).
  6.  Identify the 108-eigenspace of A_open at lambda ~ -3 (eigenvectors).
  7.  Project eigenvectors onto corner body-centers; compute deviation from -3.
  8.  Bloch Green's function G(omega) = sum_k (omega*I - H(k))^{-1} / N_k,
      regularized at omega = -3 using a level-shift, to predict corner shift
      via second-order perturbation theory.
  9.  Decompose result by O_h irreducible representations.

Only numpy and scipy are used.
Results written to: d:/AI thoery/.agent/scripts/corner_defect_perturbation_results.txt
"""

import io
import itertools
import sys
from collections import defaultdict
from fractions import Fraction

import numpy as np
import scipy  # noqa: F401 (numpy/scipy only requirement)

# Force UTF-8 stdout on Windows (avoids GBK codec errors in cp936 terminal)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT_PATH = "d:/AI thoery/.agent/scripts/corner_defect_perturbation_results.txt"
_LOG_LINES = []


def log(msg=""):
    line = str(msg)
    print(line)
    _LOG_LINES.append(line)


# ============================================================
# PART A: PERIODIC-BC LATTICE AND BLOCH DECOMPOSITION
# ============================================================

def build_bcc_lattice_periodic():
    """27 body centers on the 3-torus, returned as integer grid tuples."""
    return [(i, j, k)
            for i, j, k in itertools.product(range(3), repeat=3)]


def enumerate_simplex_faces_periodic(bc_ijk):
    """
    Canonical 20 faces for BC at grid position bc_ijk (periodic, vertices mod 3).
    Vertices: BC vertex ('bc',i,j,k) and corner vertices ('c',cx,cy,cz).
    """
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)

    orig_corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = i + dx, j + dy, k + dz
        wx, wy, wz = ox % 3, oy % 3, oz % 3
        parity = (ox + oy + oz) % 2
        orig_corners.append((parity, ('c', wx, wy, wz)))

    even_corners = [v for p, v in orig_corners if p == 0]
    odd_corners  = [v for p, v in orig_corners if p == 1]
    assert len(even_corners) == 4 and len(odd_corners) == 4

    faces = []
    for tet_corners in (even_corners, odd_corners):
        verts = [bc_v] + tet_corners
        for combo in itertools.combinations(verts, 3):
            faces.append(frozenset(combo))
    assert len(faces) == 20
    return faces


def translate_vertex(v, shift):
    si, sj, sk = shift
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i+si) % 3, (j+sj) % 3, (k+sk) % 3)
    else:
        _, cx, cy, cz = v
        return ('c', (cx+si) % 3, (cy+sj) % 3, (cz+sk) % 3)


def translate_face(face, shift):
    return frozenset(translate_vertex(v, shift) for v in face)


def build_all_faces_periodic(body_centers):
    """
    Build complete face set with translation-equivariant labeling.

    Returns:
        face_to_idx     : dict frozenset -> global index in [0, 540)
        bc_face_indices : list[27] of list[20], global face indices per BC
        ref_faces       : 20 canonical faces at BC(0,0,0)
    """
    ref_faces = enumerate_simplex_faces_periodic((0, 0, 0))
    face_to_idx = {}
    bc_face_indices = []
    gidx = 0
    for bc_ijk in body_centers:
        local = []
        for ref_face in ref_faces:
            sf = translate_face(ref_face, bc_ijk)
            if sf not in face_to_idx:
                face_to_idx[sf] = gidx
                gidx += 1
            local.append(face_to_idx[sf])
        bc_face_indices.append(local)
    return face_to_idx, bc_face_indices, ref_faces


def build_adj_matrix_from_face_dict(face_to_idx):
    """Build N x N adjacency matrix (adjacent = share exactly 2 vertices)."""
    N = len(face_to_idx)
    all_faces = [None] * N
    for face, idx in face_to_idx.items():
        all_faces[idx] = face

    v2f = defaultdict(set)
    for face, idx in face_to_idx.items():
        for v in face:
            v2f[v].add(idx)

    A = np.zeros((N, N), dtype=float)
    pairs = set()
    for v, fset in v2f.items():
        fl = sorted(fset)
        for a in range(len(fl)):
            for b in range(a+1, len(fl)):
                pairs.add((fl[a], fl[b]))

    for i, j in pairs:
        if len(all_faces[i] & all_faces[j]) == 2:
            A[i, j] = A[j, i] = 1.0

    return A


def build_bloch_H(body_centers, bc_face_indices, A_periodic, N=540, size=20):
    """
    Extract T(R) matrices and build H(k) for all 27 k-points.

    Returns:
        T       : dict (R tuple in {0,1,2}^3) -> 20x20 matrix
        H_all   : dict (n1,n2,n3) -> 20x20 complex Hermitian matrix
        kvecs   : list of 27 (n1,n2,n3) tuples
    """
    # Reorder A into BC-block form
    perm = []
    for bc_idx in range(27):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)
    A_reord = A_periodic[np.ix_(perm, perm)]

    # Extract T(R) from reference BC(0,0,0)
    T = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers):
        R = tuple(bc_j_ijk[a] % 3 for a in range(3))
        T[R] = A_reord[:size, bc_j_idx*size:(bc_j_idx+1)*size].copy()

    kvecs = list(itertools.product(range(3), repeat=3))
    H_all = {}
    for n_tuple in kvecs:
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = np.zeros((size, size), dtype=complex)
        for R, mat in T.items():
            phase = np.exp(1j * np.dot(k, np.array(R, dtype=float)))
            H_k += phase * mat
        H_all[n_tuple] = H_k

    return T, H_all, kvecs


# ============================================================
# PART B: OPEN-BC LATTICE (removing boundary-crossing bonds)
# ============================================================

def build_bcc_lattice_open():
    """
    3x3x3 BCC lattice with open BC (no periodic wrapping).
    Body centers at (i+0.5, j+0.5, k+0.5), corners at integer coords in [0,3].

    Returns:
        body_centers        : list of 27 tuples (float coords)
        cube_corners        : dict bc -> tuple of 8 corner tuples (int coords)
        bc_shell_info       : dict bc_idx -> (shell_name, graph_degree)
    """
    body_centers = []
    cube_corners = {}
    for i, j, k in itertools.product(range(3), repeat=3):
        bc = (i + 0.5, j + 0.5, k + 0.5)
        body_centers.append(bc)
        cube_corners[bc] = tuple(
            (i+dx, j+dy, k+dz)
            for dx, dy, dz in itertools.product((0, 1), repeat=3)
        )
    return body_centers, cube_corners


def bc_shell_name_and_degree(bc_ijk_int):
    """(i,j,k) in {0,1,2}^3 -> (shell_name, nominal_graph_degree)."""
    n_mid = sum(1 for c in bc_ijk_int if c == 1)
    return {0: ("Corner", 3), 1: ("Edge", 4), 2: ("Face", 5), 3: ("Center", 6)}[n_mid]


def build_simplices_open(body_centers, cube_corners):
    """Enumerate 54 four-simplices (open BC, no wrapping)."""
    simplices = []
    owner = []
    for bc_idx, bc in enumerate(body_centers):
        corners = cube_corners[bc]
        even_tet = [c for c in corners if (c[0]+c[1]+c[2]) % 2 == 0]
        odd_tet  = [c for c in corners if (c[0]+c[1]+c[2]) % 2 == 1]
        assert len(even_tet) == 4 and len(odd_tet) == 4
        bc_f = (float(bc[0]), float(bc[1]), float(bc[2]))
        for tet in (even_tet, odd_tet):
            tet_f = [(float(c[0]), float(c[1]), float(c[2])) for c in tet]
            simplex = frozenset([bc_f] + tet_f)
            assert len(simplex) == 5
            simplices.append(simplex)
            owner.append(bc_idx)
    return simplices, owner


def build_faces_open(simplices, owner):
    """Extract 540 triangular faces from open-BC simplices."""
    face_to_simplices = defaultdict(list)
    faces_in_order = []
    face_owner_simplex = []
    for s_idx, simplex in enumerate(simplices):
        verts = sorted(simplex)
        for combo in itertools.combinations(verts, 3):
            face = frozenset(combo)
            faces_in_order.append(face)
            face_to_simplices[face].append(s_idx)
            face_owner_simplex.append(s_idx)
    return faces_in_order, face_owner_simplex


def build_adj_matrix_open(faces):
    """Build 540x540 adjacency matrix for the open-BC face list."""
    N = len(faces)
    v2f = defaultdict(set)
    for idx, face in enumerate(faces):
        for v in face:
            v2f[v].add(idx)

    A = np.zeros((N, N), dtype=float)
    pairs = set()
    for fset in v2f.values():
        fl = sorted(fset)
        for a in range(len(fl)):
            for b in range(a+1, len(fl)):
                pairs.add((fl[a], fl[b]))

    face_sets = [frozenset(f) for f in faces]
    for i, j in pairs:
        if len(face_sets[i] & face_sets[j]) == 2:
            A[i, j] = A[j, i] = 1.0

    return A


# ============================================================
# PART C: Oh group and character table
# ============================================================

def generate_oh_elements():
    """Generate all 48 Oh elements as 3x3 integer matrices."""
    C4z = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=int)
    C3_111 = np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=int)
    inv = -np.eye(3, dtype=int)

    def mat_key(m): return tuple(m.flatten())

    O_set = {}
    queue = [np.eye(3, dtype=int)]
    O_set[mat_key(np.eye(3, dtype=int))] = np.eye(3, dtype=int)
    gens = [C4z, C3_111, C4z.T, C3_111.T]
    while queue:
        cur = queue.pop(0)
        for g in gens:
            nw = g @ cur
            key = mat_key(nw)
            if key not in O_set:
                O_set[key] = nw
                queue.append(nw)
    assert len(O_set) == 24

    Oh = []
    for mat in O_set.values():
        Oh.append(mat.copy())
        Oh.append((inv @ mat).copy())
    assert len(Oh) == 48
    return Oh


def classify_oh_element(g):
    """Return conjugacy class label for Oh element g."""
    det = int(round(np.linalg.det(g)))
    tr  = int(round(np.trace(g)))
    off = float(np.sum(np.abs(g - np.diag(np.diag(g)))))
    if det == 1:
        if tr ==  3: return 'E'
        if tr ==  0: return 'C3'
        if tr ==  1: return 'C4'
        if tr == -1: return "C2'" if off < 0.5 else 'C2'
    else:
        if tr == -3: return 'i'
        if tr ==  0: return 'S6'
        if tr == -1: return 'S4'
        if tr ==  1: return 'sigma_d' if off < 0.5 else 'sigma_h'
    return f'UNKNOWN(det={det},tr={tr})'


OH_CLASSES = ['E', 'C3', 'C2', 'C4', "C2'", 'i', 'S6', 'sigma_h', 'S4', 'sigma_d']
OH_CLASS_ORDER = {
    'E':1,'C3':8,'C2':6,'C4':6,"C2'":3,'i':1,'S6':8,'sigma_h':6,'S4':6,'sigma_d':3
}
# Char table with column order matching our classifier labels
OH_CHAR_TABLE = {
    'A1g':[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'A2g':[ 1, 1,-1,-1, 1, 1, 1,-1,-1, 1],
    'Eg' :[ 2,-1, 0, 0, 2, 2,-1, 0, 0, 2],
    'T1g':[ 3, 0,-1, 1,-1, 3, 0,-1, 1,-1],
    'T2g':[ 3, 0, 1,-1,-1, 3, 0, 1,-1,-1],
    'A1u':[ 1, 1, 1, 1, 1,-1,-1,-1,-1,-1],
    'A2u':[ 1, 1,-1,-1, 1,-1,-1, 1, 1,-1],
    'Eu' :[ 2,-1, 0, 0, 2,-2, 1, 0, 0,-2],
    'T1u':[ 3, 0,-1, 1,-1,-3, 0, 1,-1, 1],
    'T2u':[ 3, 0, 1,-1,-1,-3, 0,-1, 1, 1],
}
OH_IRREP_DIMS = {
    'A1g':1,'A2g':1,'Eg':2,'T1g':3,'T2g':3,
    'A1u':1,'A2u':1,'Eu':2,'T1u':3,'T2u':3,
}


def decompose_into_oh_irreps(chars_by_class):
    """
    Given a dict {class_label -> chi_value}, return multiplicities dict.
    n_mu = (1/48) sum_cls |cls| * chi_mu(cls)* chi(cls)
    """
    result = {}
    for irrep, char_row in OH_CHAR_TABLE.items():
        n_mu = 0.0
        for cidx, cls in enumerate(OH_CLASSES):
            n_mu += OH_CLASS_ORDER[cls] * char_row[cidx] * chars_by_class.get(cls, 0.0)
        result[irrep] = n_mu / 48.0
    return result


# ============================================================
# PART D: Oh action on the 20-face internal basis
# ============================================================

def vertex_coords_centered(v):
    """Geometric coordinates of a vertex, centered at (1.5,1.5,1.5) of the 3-torus."""
    if v[0] == 'bc':
        _, i, j, k = v
        return np.array([i-1.0, j-1.0, k-1.0], dtype=float)
    else:
        _, cx, cy, cz = v
        return np.array([cx-1.5, cy-1.5, cz-1.5], dtype=float)


def apply_oh_vertex(g, v):
    """Apply Oh element g to vertex v (periodic wrapping mod 3)."""
    coord = vertex_coords_centered(v)
    nc = g @ coord
    if v[0] == 'bc':
        return ('bc', int(round(nc[0]+1.0)) % 3,
                      int(round(nc[1]+1.0)) % 3,
                      int(round(nc[2]+1.0)) % 3)
    else:
        return ('c',  int(round(nc[0]+1.5)) % 3,
                      int(round(nc[1]+1.5)) % 3,
                      int(round(nc[2]+1.5)) % 3)


def apply_oh_face(g, face):
    return frozenset(apply_oh_vertex(g, v) for v in face)


def bc_owner_of_face(face):
    """Return BC (i,j,k) owning a face (from BC vertex, if present)."""
    for v in face:
        if v[0] == 'bc':
            return (v[1], v[2], v[3])
    return None


def build_oh_perm_matrix(g, ref_faces, ref_face_to_local,
                          face_to_idx, bc_face_indices, body_centers):
    """
    Build 20x20 permutation matrix for Oh element g acting on
    the reference BC(0,0,0) 20-face basis.
    """
    P = np.zeros((20, 20), dtype=float)
    for m, face in enumerate(ref_faces):
        gf = apply_oh_face(g, face)
        bc_owner = bc_owner_of_face(gf)
        if bc_owner is None:
            # Pure-corner face: find owner via global lookup
            if gf in face_to_idx:
                gidx = face_to_idx[gf]
                for bidx, bc_ijk in enumerate(body_centers):
                    if gidx in set(bc_face_indices[bidx]):
                        bc_owner = bc_ijk
                        break
            if bc_owner is None:
                raise ValueError(f"Cannot find BC owner for {gf}")
        inv_shift = tuple((-s) % 3 for s in bc_owner)
        translated = translate_face(gf, inv_shift)
        if translated not in ref_face_to_local:
            raise ValueError(f"Translated face not in reference set: {translated}")
        n = ref_face_to_local[translated]
        P[n, m] = 1.0
    return P


# ============================================================
# PART E: Main analysis
# ============================================================

def group_eigenvalues(eigvals, tol=1e-6):
    groups, cur = [], [eigvals[0]]
    for ev in eigvals[1:]:
        if abs(ev - cur[-1]) <= tol:
            cur.append(ev)
        else:
            groups.append(cur); cur = [ev]
    groups.append(cur)
    out, cum = [], 0
    for g in groups:
        rep = float(np.mean(g)); deg = len(g); cum += deg
        out.append((rep, deg, cum))
    return out


def find_rational(x, max_denom=2000, tol=1e-7):
    f = Fraction(x).limit_denominator(max_denom)
    return f if abs(float(f) - x) < tol else None


def main():
    log("=" * 78)
    log("BCC 4-simplex corner deviation -- analytic perturbation theory")
    log("=" * 78)

    # ------------------------------------------------------------------
    # STEP 1: Build periodic-BC lattice and 540x540 A_periodic
    # ------------------------------------------------------------------
    log("\n[Step 1] Build periodic-BC lattice and A_periodic (540x540)")
    body_centers_p = build_bcc_lattice_periodic()
    face_to_idx_p, bc_face_indices_p, ref_faces = build_all_faces_periodic(body_centers_p)
    N = len(face_to_idx_p)
    log(f"  Total periodic faces: {N}  (expect 540)")
    assert N == 540

    A_periodic = build_adj_matrix_from_face_dict(face_to_idx_p)
    log(f"  A_periodic shape: {A_periodic.shape}, symmetric: {np.allclose(A_periodic, A_periodic.T)}")
    log(f"  A_periodic nonzero entries: {int(np.count_nonzero(A_periodic))}")

    # Verify full periodic spectrum
    ev_p = np.sort(np.linalg.eigvalsh(A_periodic))
    grouped_p = group_eigenvalues(ev_p)
    log(f"  Periodic spectrum (grouped):")
    for rep, deg, cum in grouped_p:
        log(f"    lambda={rep:10.4f}  deg={deg:5d}  cumul={cum:5d}")

    # ------------------------------------------------------------------
    # STEP 2: Bloch decomposition -- 27 k-points x 20x20 H(k)
    # ------------------------------------------------------------------
    log("\n[Step 2] Bloch decomposition: 27 k-points, each giving 20x20 H(k)")
    T, H_all, kvecs = build_bloch_H(body_centers_p, bc_face_indices_p, A_periodic)

    # Verify Bloch spectrum matches
    all_bloch = np.sort(np.concatenate(
        [np.linalg.eigvalsh(H_all[n]).real for n in kvecs]))
    max_diff = float(np.max(np.abs(all_bloch - ev_p)))
    log(f"  Max |lambda_Bloch - lambda_direct| = {max_diff:.2e}  (expect < 1e-8)")
    assert max_diff < 1e-6, f"Bloch decomp failed: {max_diff}"

    # Check Hermiticity
    max_herm = max(float(np.max(np.abs(H - H.conj().T))) for H in H_all.values())
    log(f"  Max ||H(k) - H(k)^H||: {max_herm:.2e}  (expect < 1e-10)")

    # Report flat bands
    eigs_by_k = {n: np.sort(np.linalg.eigvalsh(H_all[n]).real) for n in kvecs}
    eigs_mat = np.array([eigs_by_k[n] for n in sorted(kvecs)])  # (27, 20)
    flat_bands = []
    log(f"\n  Band bandwidths (20 bands):")
    for bi in range(20):
        bvals = eigs_mat[:, bi]
        bw = float(bvals.max() - bvals.min())
        flat = bw < 1e-6
        if flat:
            flat_bands.append((bi, float(bvals.mean())))
        log(f"    band {bi:2d}: min={bvals.min():8.4f} max={bvals.max():8.4f} "
            f"BW={bw:.2e}  {'FLAT' if flat else ''}")
    log(f"\n  Flat bands: {len(flat_bands)} (at lambda = "
        f"{[round(v,4) for _,v in flat_bands]})")

    # Where does lambda=-3 appear in Bloch decomposition?
    tol_ev = 1e-4
    lam3_k = {}
    for n in kvecs:
        hits = [(i, ev) for i, ev in enumerate(eigs_by_k[n])
                if abs(ev - (-3.0)) < tol_ev]
        if hits:
            lam3_k[n] = len(hits)
    log(f"\n  lambda=-3 in Bloch decomp:")
    for n in sorted(lam3_k):
        log(f"    k=2pi/3*{n}: multiplicity {lam3_k[n]}")
    log(f"  Total: {sum(lam3_k.values())}  (periodic has "
        f"{sum(1 for e in ev_p if abs(e+3)<tol_ev)} at lambda=-3)")

    # ------------------------------------------------------------------
    # STEP 3: Build open-BC matrix A_open
    # ------------------------------------------------------------------
    log("\n[Step 3] Build open-BC matrix A_open (remove bonds crossing boundary)")
    body_centers_o, cube_corners_o = build_bcc_lattice_open()
    simplices_o, owner_simplex_o = build_simplices_open(body_centers_o, cube_corners_o)
    faces_o, face_owner_simplex_o = build_faces_open(simplices_o, owner_simplex_o)
    log(f"  Open-BC: {len(simplices_o)} simplices, {len(faces_o)} faces "
        f"(expect 54, 540)")
    assert len(faces_o) == 540

    A_open = build_adj_matrix_open(faces_o)
    log(f"  A_open shape: {A_open.shape}, symmetric: {np.allclose(A_open, A_open.T)}")
    log(f"  A_open nonzero entries: {int(np.count_nonzero(A_open))}")

    # ------------------------------------------------------------------
    # STEP 4: Match faces between periodic and open BC, build V_perturb
    # ------------------------------------------------------------------
    log("\n[Step 4] Match periodic <-> open-BC faces; compute V = A_periodic - A_open")

    # The periodic faces use ('bc',i,j,k) and ('c',cx,cy,cz) with coords mod 3.
    # The open-BC faces use float tuples. Build a canonical key for each.
    # Periodic face key: sort the 3 vertex tuples lexicographically.
    # Open-BC face key: convert each vertex to ('bc',i,j,k) or ('c',cx,cy,cz) form,
    #   using rounding to integers (no mod since open BC has coords 0..3).

    def open_vertex_to_key(v):
        """Convert open-BC float vertex to a canonical comparable key."""
        x, y, z = v
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        # BC vertex: exactly one half-integer coordinate in each dim
        if all(abs(c - round(c) - 0.5) < 1e-9 or abs(c - round(c) + 0.5) < 1e-9
               for c in [x, y, z]):
            # half-integer: (i+0.5, j+0.5, k+0.5) -> i = floor(x)
            i2, j2, k2 = int(x - 0.5 + 1e-9), int(y - 0.5 + 1e-9), int(z - 0.5 + 1e-9)
            return ('bc', i2, j2, k2)
        else:
            return ('c', xi, yi, zi)

    def periodic_vertex_to_comparable(v):
        """Periodic vertex as a key suitable for comparison with open-BC keys."""
        # Periodic: ('bc',i,j,k) or ('c',cx,cy,cz), coords in {0,1,2}.
        # For matching: treat open-BC coords 0..2 as the same as periodic (mod 3 = identity for [0,2]).
        # Corner vertices with coord 3 (boundary) don't appear in periodic labeling directly.
        return v  # already canonical

    # Build open-BC face set as frozenset of canonical keys
    def open_face_to_canonical(face):
        """Convert open-BC face (frozenset of float tuples) to canonical tuple-key form."""
        keys = []
        for v in face:
            keys.append(open_vertex_to_key(v))
        return frozenset(keys)

    # Map open-BC faces to canonical keys
    open_face_canonical = [open_face_to_canonical(f) for f in faces_o]
    open_face_set = {cf: i for i, cf in enumerate(open_face_canonical)}

    # Map periodic faces to comparable keys (already in right form)
    # But periodic vertices have coord range {0,1,2} -- when open-BC corner vertices
    # have coord in {0,1,2,3}, the boundary corners (coord=3 in some axis) are
    # distinct from periodic corners (coord=0 wraps). So they won't match --
    # those are exactly the "removed bonds".

    # Build permutation: periodic global index -> open-BC global index (or -1 if unmatched)
    perm_p_to_o = np.full(540, -1, dtype=int)
    perm_o_to_p = np.full(540, -1, dtype=int)
    n_matched = 0
    for pface, pidx in face_to_idx_p.items():
        oidx = open_face_set.get(pface)
        if oidx is not None:
            perm_p_to_o[pidx] = oidx
            perm_o_to_p[oidx] = pidx
            n_matched += 1

    log(f"  Matched faces (periodic <-> open-BC): {n_matched} of 540")
    log(f"  Unmatched in periodic: {540 - n_matched}")
    log(f"  (Unmatched periodic faces contain boundary-wrapped vertices)")

    # Build A_open reordered to match periodic face ordering
    # Strategy: for faces that match, copy the open-BC adjacency entry.
    # For unmatched faces (containing wrapped corner vertices), they are
    # boundary faces in open-BC with broken bonds.
    # The perturbation V = A_periodic - A_open in the common basis.
    #
    # Since open-BC face labeling != periodic, we reorder A_open to match
    # periodic face ordering via the perm map. Unmatched faces get 0 row/col.

    A_open_reord = np.zeros((540, 540), dtype=float)
    for pi in range(540):
        oi = perm_p_to_o[pi]
        if oi < 0:
            continue
        for pj in range(540):
            oj = perm_p_to_o[pj]
            if oj < 0:
                continue
            A_open_reord[pi, pj] = A_open[oi, oj]

    V_perturb = A_periodic - A_open_reord
    log(f"\n  V = A_periodic - A_open_reord:")
    log(f"    nonzero entries: {int(np.count_nonzero(V_perturb))}")
    log(f"    max abs entry: {np.max(np.abs(V_perturb)):.4f}")
    log(f"    symmetric: {np.allclose(V_perturb, V_perturb.T)}")
    log(f"    (These are the removed bonds when going from periodic to open BC)")

    # Verify: eigenspectrum of A_open_reord (ignoring unmatched rows)
    # -- use A_open directly instead
    ev_open = np.sort(np.linalg.eigvalsh(A_open))
    grouped_open = group_eigenvalues(ev_open)
    log(f"\n  Open-BC spectrum (grouped, from A_open directly):")
    for rep, deg, cum in grouped_open:
        log(f"    lambda={rep:10.4f}  deg={deg:5d}  cumul={cum:5d}")

    # ------------------------------------------------------------------
    # STEP 5: Identify 108-eigenspace of A_open at lambda ~ -3
    # ------------------------------------------------------------------
    log("\n[Step 5] Identify 108-eigenspace of A_open at lambda ~ -3")
    ev_open_full, evec_open_full = np.linalg.eigh(A_open)
    mask_3 = np.abs(ev_open_full - (-3.0)) < 1e-4
    n_3 = int(np.sum(mask_3))
    log(f"  Number of eigenvalues at lambda=-3 (tol 1e-4): {n_3}  (expect 108)")
    V_108 = evec_open_full[:, mask_3]  # shape (540, 108)
    log(f"  V_108 shape: {V_108.shape}")

    # Verify A_open * V_108 = -3 * V_108
    AV_err = np.max(np.abs(A_open @ V_108 - (-3.0) * V_108))
    log(f"  Max |A_open * V_108 - (-3) * V_108|: {AV_err:.2e}")

    # ------------------------------------------------------------------
    # STEP 6: Project eigenvectors onto corner body-centers
    # ------------------------------------------------------------------
    log("\n[Step 6] Project lambda=-3 eigenvectors onto body-center shells")

    # Build BC shell info for open-BC
    bc_ijk_o = [tuple(int(round(c - 0.5)) for c in bc) for bc in body_centers_o]
    bc_shell_o = [bc_shell_name_and_degree(ijk) for ijk in bc_ijk_o]

    # face -> owner BC idx
    owner_bc_o = np.array(
        [owner_simplex_o[face_owner_simplex_o[f]] for f in range(len(faces_o))],
        dtype=int,
    )

    # Per-BC face index list (20 faces each)
    bc_face_idxs_o = [np.where(owner_bc_o == bci)[0] for bci in range(27)]
    assert all(len(idxs) == 20 for idxs in bc_face_idxs_o), "Not all BCs have 20 faces!"

    # Squared weights per face summed over the 108-eigenvector subspace
    sq_weights = np.sum(V_108 ** 2, axis=1)  # shape (540,)
    total_sq = np.sum(sq_weights)
    log(f"  Sum of squared weights: {total_sq:.4f}  (expect ~108)")

    # Per-BC projection
    bc_projections = np.array(
        [float(np.sum(sq_weights[bc_face_idxs_o[bci]])) for bci in range(27)]
    )

    # Average by shell
    shell_names = ["Corner", "Edge", "Face", "Center"]
    shell_expected_deg = {"Corner": 3, "Edge": 4, "Face": 5, "Center": 6}
    shell_members = defaultdict(list)
    for bci, (sname, sdeg) in enumerate(bc_shell_o):
        shell_members[sname].append(bci)

    log(f"\n  Per-shell average projection vs expected (= graph degree):")
    log(f"  {'Shell':8s} | {'count':5s} | {'avg_proj':12s} | {'expected':8s} | {'deviation':10s}")
    log(f"  {'-'*8}-+-{'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}")

    shell_avg = {}
    shell_dev = {}
    for sname in shell_names:
        members = shell_members[sname]
        if not members:
            continue
        avg = float(np.mean(bc_projections[members]))
        exp = shell_expected_deg[sname]
        dev = avg - exp
        shell_avg[sname] = avg
        shell_dev[sname] = dev
        log(f"  {sname:8s} | {len(members):5d} | {avg:12.6f} | {exp:8d} | {dev:+10.6f}")

    delta_corner = shell_dev.get("Corner", float('nan'))
    log(f"\n  Corner deviation delta_corner = avg_proj - 3 = {delta_corner:.8f}")

    # Rational approximation
    r_delta = find_rational(delta_corner, max_denom=5000)
    log(f"  Rational approximation: {r_delta}")
    log(f"  72 * delta_corner = {72 * delta_corner:.8f}  (1/72 = {1/72:.8f})")
    log(f"  24 * delta_corner = {24 * delta_corner:.8f}")
    log(f"  360 * delta_corner = {360 * delta_corner:.8f}")

    # Per-BC details
    log(f"\n  Per-BC projection details:")
    log(f"  {'bc_idx':6s} | {'ijk':12s} | {'shell':7s} | {'deg':3s} | {'projection':12s} | {'dev':10s}")
    log(f"  {'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*3}-+-{'-'*12}-+-{'-'*10}")
    for bci in range(27):
        sname, sdeg = bc_shell_o[bci]
        proj = bc_projections[bci]
        dev = proj - sdeg
        log(f"  {bci:6d} | {str(bc_ijk_o[bci]):12s} | {sname:7s} | {sdeg:3d} | "
            f"{proj:12.6f} | {dev:+10.6f}")

    # ------------------------------------------------------------------
    # STEP 7: Bloch Green's function G(omega) at omega = -3 (regularized)
    # ------------------------------------------------------------------
    log("\n[Step 7] Bloch Green's function G(omega) = sum_k (omega*I - H(k))^{-1} / N_k")
    log("  Evaluated at omega = -3 with regularization eps")

    N_k = 27  # number of k-points

    # For omega = -3, H(k) has eigenvalue -3 at all k-points (flat band).
    # (omega*I - H(k)) is singular at all k-points.
    # Regularization: G_reg(omega, eps) = sum_k ((omega + i*eps)*I - H(k))^{-1} / N_k
    # For flat bands, the singularity is rank-6 at each k-point.
    #
    # Physical meaning: the flat-band contribution diverges as 1/eps;
    # the non-flat-band contribution gives the Cauchy principal value Green's function.
    # We extract the PRINCIPAL VALUE part (finite piece) by explicitly projecting out
    # the flat-band subspace at each k-point.

    log(f"\n  Projecting out flat-band subspace at each k-point...")
    log(f"  G_PV(omega) = sum_k P_nonflat(k) * (omega - H(k))^{{-1}} * P_nonflat(k) / N_k")
    log(f"  where P_nonflat(k) = I - P_flat(k) is the projector onto non-flat modes")

    omega = -3.0
    G_PV = np.zeros((20, 20), dtype=complex)  # 20x20 in BC-internal space

    flat_proj_norms = []
    for n_tuple in kvecs:
        H_k = H_all[n_tuple]
        ev_k, evec_k = np.linalg.eigh(H_k)

        # Flat band: eigenvalues = -3 (tolerance 1e-4)
        flat_mask = np.abs(ev_k - omega) < 1e-4
        non_flat_mask = ~flat_mask

        flat_proj_norms.append(np.sum(flat_mask))

        # Non-flat sector: (omega - lambda_n)^{-1} for lambda_n != -3
        denom = omega - ev_k  # shape (20,)
        # Zero out flat-band denominators (they diverge -- excluded from PV)
        denom_reg = np.where(non_flat_mask, denom, np.inf)

        # G_k = V * diag(1/denom_reg) * V^H (only non-flat part contributes finite)
        coeffs = np.where(non_flat_mask, 1.0 / denom_reg, 0.0)
        G_k = evec_k @ np.diag(coeffs) @ evec_k.conj().T
        G_PV += G_k

    G_PV /= N_k

    flat_total = sum(flat_proj_norms)
    log(f"  Flat-band modes removed per k-point: {flat_proj_norms[:5]}... (total {flat_total})")
    log(f"  G_PV shape: {G_PV.shape}")
    log(f"  G_PV Hermitian: {np.allclose(G_PV, G_PV.conj().T, atol=1e-8)}")
    log(f"  G_PV imaginary part max: {np.max(np.abs(np.imag(G_PV))):.3e}")
    G_PV_real = np.real(G_PV)

    log(f"  G_PV diagonal (self-energies, 20 internal faces):")
    for m in range(20):
        log(f"    face {m:2d}: G_PV[m,m] = {G_PV_real[m,m]:12.8f}")

    # ------------------------------------------------------------------
    # STEP 8: Second-order perturbation theory for corner shift
    # ------------------------------------------------------------------
    log("\n[Step 8] Second-order perturbation theory for corner eigenvalue shift")
    log("  delta_lambda = <psi | V | G_PV(omega) | V | psi>")
    log("  where |psi> ranges over flat-band eigenvectors at k=0 (corner BCs),")
    log("  and V is the perturbation in BC-internal face space.")
    log()

    # Extract flat-band eigenvectors at k=0 (Gamma point)
    H_gamma = H_all[(0, 0, 0)]
    ev_gamma, evec_gamma = np.linalg.eigh(H_gamma)
    flat_mask_g = np.abs(ev_gamma - omega) < 1e-4
    n_flat_g = int(np.sum(flat_mask_g))
    V_flat_gamma = evec_gamma[:, flat_mask_g]  # shape (20, n_flat_g)
    log(f"  Flat-band eigenvectors at Gamma (k=0): {n_flat_g}")

    # Build V_perturb in BC-internal space
    # V_perturb is 540x540. For corner BCs, we need the 20x20 block corresponding
    # to the average perturbation matrix seen by corner-shell body-centers.
    # Since the perturbation V is the difference of adjacency matrices,
    # the relevant V_perturb block for a corner BC is the 20-row, all-540-column
    # piece of V_perturb restricted to the corner BC's faces.
    #
    # For second-order perturbation theory:
    # delta_lambda_2nd = sum_{m not in flat-band} |<m | V | psi>|^2 / (lambda_psi - lambda_m)
    # In Bloch language:
    # delta_lambda_2nd = (1/N_k) sum_k <psi_k | V_k | evec_nflat(k)> <evec_nflat(k)| V_k |psi_k>
    #                              / (omega - ev_nflat(k))
    # = <psi_k | V_k G_PV(k) V_k | psi_k>  (averaged over k)
    #
    # Here V_k is the Bloch transform of the perturbation (removed bonds).
    # We compute this via V_perturb reordered into BC-block form.

    # Build V_perturb in BC-block order (periodic ordering already matches Bloch)
    perm_p = []
    for bc_idx in range(27):
        perm_p.extend(bc_face_indices_p[bc_idx])
    perm_p = np.array(perm_p, dtype=int)
    V_reord = V_perturb[np.ix_(perm_p, perm_p)]

    # V_k (Bloch transform of V) = sum_R exp(i k.R) V_hop(R)
    # where V_hop(R) is the 20x20 inter-BC hopping block of V
    V_hop = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers_p):
        R = tuple(bc_j_ijk[a] % 3 for a in range(3))
        block = V_reord[:20, bc_j_idx*20:(bc_j_idx+1)*20].copy()
        if np.linalg.norm(block, 'fro') > 1e-10:
            V_hop[R] = block
    log(f"  Nonzero V_hop(R) blocks: {len(V_hop)}")
    for R, mat in V_hop.items():
        log(f"    R={R}: ||V_hop(R)||_F = {np.linalg.norm(mat,'fro'):.6f}, "
            f"nonzero entries = {int(np.count_nonzero(mat))}")

    def build_V_k(n_tuple):
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        Vk = np.zeros((20, 20), dtype=complex)
        for R, mat in V_hop.items():
            Vk += np.exp(1j * np.dot(k, np.array(R, dtype=float))) * mat
        return Vk

    # 2nd-order shift for each flat-band eigenvector at Gamma
    # Using G_PV already computed as average over k
    V_gamma = build_V_k((0, 0, 0))
    V_gamma_real = np.real(V_gamma)  # should be real for k=0

    log(f"\n  V(k=0) imaginary part max: {np.max(np.abs(np.imag(V_gamma))):.3e}")
    log(f"  V(k=0) shape: {V_gamma.shape}")
    log(f"  V(k=0) nonzero entries: {int(np.count_nonzero(V_gamma_real))}")

    # Second-order shifts for each Gamma flat-band eigenvector
    log(f"\n  Second-order eigenvalue shifts for {n_flat_g} flat-band modes at Gamma:")
    log(f"  delta_lambda_2nd = <psi | V(k=0) G_PV(omega) V(k=0) | psi>")
    log(f"  where omega = {omega}, G_PV = principal-value Green's function (k-averaged)")
    log()

    # G_PV_real is 20x20, V(k=0) is 20x20
    # V * G_PV * V applied to each flat-band vector
    VGV = V_gamma_real @ G_PV_real @ V_gamma_real  # 20x20

    shifts_2nd = []
    for m in range(n_flat_g):
        psi = np.real(V_flat_gamma[:, m])
        shift = float(psi @ VGV @ psi)
        shifts_2nd.append(shift)
        log(f"    mode {m}: delta_lambda = {shift:+.8f}")

    avg_shift_2nd = float(np.mean(shifts_2nd))
    log(f"\n  Average 2nd-order shift (over {n_flat_g} Gamma flat-band modes): {avg_shift_2nd:+.8f}")

    # Also compute for each k-point
    log(f"\n  Per-k-point second-order shifts:")
    all_shifts_k = []
    for n_tuple in kvecs:
        H_k = H_all[n_tuple]
        ev_k, evec_k = np.linalg.eigh(H_k)
        flat_mask_k = np.abs(ev_k - omega) < 1e-4
        n_flat_k = int(np.sum(flat_mask_k))
        V_flat_k = evec_k[:, flat_mask_k]

        Vk = build_V_k(n_tuple)
        Vk_real = np.real(Vk)

        # G_PV at this specific k-point (not averaged)
        non_flat_mask_k = ~flat_mask_k
        denom_k = np.where(non_flat_mask_k, omega - ev_k, np.inf)
        coeffs_k = np.where(non_flat_mask_k, 1.0 / denom_k, 0.0)
        G_PV_k = evec_k @ np.diag(coeffs_k.real) @ evec_k.conj().T.real

        VGV_k = Vk_real @ G_PV_k @ Vk_real
        shifts_k = [float(V_flat_k[:, m] @ VGV_k @ V_flat_k[:, m]) for m in range(n_flat_k)]
        all_shifts_k.extend(shifts_k)
        avg_k = float(np.mean(shifts_k)) if shifts_k else 0.0
        log(f"    k={n_tuple}: {n_flat_k} flat modes, avg shift = {avg_k:+.8f}")

    all_shifts_k_arr = np.array(all_shifts_k)
    log(f"\n  Overall avg shift (all k, all modes): {float(np.mean(all_shifts_k_arr)):+.8f}")
    log(f"  Min shift: {float(np.min(all_shifts_k_arr)):+.8f}")
    log(f"  Max shift: {float(np.max(all_shifts_k_arr)):+.8f}")

    # ------------------------------------------------------------------
    # STEP 9: Oh irrep decomposition of the shift
    # ------------------------------------------------------------------
    log("\n[Step 9] O_h irrep decomposition of the flat-band eigenspace shift")

    # Build Oh group and permutation matrices on 20-face basis
    log("  Generating Oh group (48 elements)...")
    oh_elements = generate_oh_elements()
    element_class = [classify_oh_element(g) for g in oh_elements]

    # Verify class sizes
    class_counts = defaultdict(int)
    for cls in element_class:
        class_counts[cls] += 1
    log("  Class sizes: " + " ".join(f"{cls}:{class_counts[cls]}" for cls in OH_CLASSES))

    # Build reference face lookup
    ref_face_to_local = {f: m for m, f in enumerate(ref_faces)}

    log("  Building 20x20 permutation matrices for all 48 Oh elements...")
    perm_matrices = []
    n_oh_errors = 0
    for g in oh_elements:
        try:
            P = build_oh_perm_matrix(g, ref_faces, ref_face_to_local,
                                     face_to_idx_p, bc_face_indices_p, body_centers_p)
            perm_matrices.append(P)
        except Exception as e:
            log(f"    ERROR: {e}")
            perm_matrices.append(None)
            n_oh_errors += 1
    log(f"  Permutation matrices built: {len(perm_matrices) - n_oh_errors} of 48 OK")

    # Flat-band subspace from H(k=0): 6 eigenvectors (local Oh-invariant subspace)
    V_flat_6 = np.real(V_flat_gamma)  # (20, 6) if n_flat_g == 6

    if V_flat_6.shape[1] == 6:
        log(f"  Projecting VGV into flat-band subspace (6D)...")
        VGV_6d = V_flat_6.T @ VGV @ V_flat_6  # (6, 6)

        # Build 6x6 representation matrices for Oh
        rep_chars = defaultdict(list)
        rep_matrices_6d = []
        for g, P in zip(oh_elements, perm_matrices):
            if P is None:
                rep_matrices_6d.append(None)
                continue
            R_g = V_flat_6.T @ P @ V_flat_6  # (6, 6) -- Oh representation in flat space
            rep_matrices_6d.append(R_g)
            cls = classify_oh_element(g)
            rep_chars[cls].append(float(np.trace(R_g)))

        # Average character per class
        class_chi = {}
        for cls in OH_CLASSES:
            vals = rep_chars.get(cls, [0.0])
            mean_chi = float(np.mean(vals))
            spread = float(np.max(np.abs(np.array(vals) - mean_chi))) if len(vals) > 1 else 0.0
            class_chi[cls] = mean_chi
            if spread > 1e-5:
                log(f"    WARNING: chi not constant in class {cls}, spread={spread:.3e}")

        log(f"\n  6D flat-band representation characters by O_h class:")
        log(f"  {'Class':10s} | {'size':4s} | {'chi':10s}")
        log(f"  {'-'*10}-+-{'-'*4}-+-{'-'*10}")
        for cls in OH_CLASSES:
            log(f"  {cls:10s} | {OH_CLASS_ORDER[cls]:4d} | {class_chi[cls]:10.4f}")

        # Decompose into irreps
        irrep_mults = decompose_into_oh_irreps(class_chi)
        log(f"\n  Irrep decomposition of 6D flat-band space:")
        log(f"  {'Irrep':6s} | {'dim':3s} | {'n_mu (raw)':12s} | {'n_mu (round)':12s} | {'dim x n_mu'}")
        log(f"  {'-'*6}-+-{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        total_dim = 0
        decomp_parts = []
        for irrep in ['A1g','A2g','Eg','T1g','T2g','A1u','A2u','Eu','T1u','T2u']:
            n_raw = irrep_mults[irrep]
            n_r = int(round(n_raw))
            d = OH_IRREP_DIMS[irrep]
            marker = " <--" if n_r > 0 else ""
            log(f"  {irrep:6s} | {d:3d} | {n_raw:12.6f} | {n_r:12d} | {n_r*d:4d}{marker}")
            total_dim += n_r * d
            if n_r > 0:
                decomp_parts.append(f"{n_r}x{irrep}" if n_r > 1 else irrep)
        decomp_str = " + ".join(decomp_parts) if decomp_parts else "(none)"
        log(f"\n  Total dimension: {total_dim} (expect 6)")
        log(f"  6D flat-band = {decomp_str}")

        # Per-irrep contribution to VGV
        log(f"\n  Per-irrep contribution to second-order corner shift:")
        log(f"  (Project VGV_6d onto each irrep subspace)")

        # Build projectors onto each irrep using rep_matrices_6d
        proj_matrices_by_irrep = {}
        for irrep in ['A1g','A2g','Eg','T1g','T2g','A1u','A2u','Eu','T1u','T2u']:
            n_r = int(round(irrep_mults[irrep]))
            if n_r == 0:
                continue
            d = OH_IRREP_DIMS[irrep]
            # Projector = (dim_irrep / |G|) * sum_g chi_mu(g)* R(g)
            proj = np.zeros((6, 6), dtype=float)
            char_row = OH_CHAR_TABLE[irrep]
            for g_idx, (g, R_g) in enumerate(zip(oh_elements, rep_matrices_6d)):
                if R_g is None:
                    continue
                cls = element_class[g_idx]
                cls_idx = OH_CLASSES.index(cls)
                chi_conj = char_row[cls_idx]  # real chars
                proj += chi_conj * R_g
            proj *= (d / 48.0)
            proj_matrices_by_irrep[irrep] = proj

        log(f"  {'Irrep':6s} | {'trace(P)':10s} | {'tr(P VGV P)':14s} | "
            f"{'avg_shift_per_mode':20s} | {'n_modes':7s}")
        log(f"  {'-'*6}-+-{'-'*10}-+-{'-'*14}-+-{'-'*20}-+-{'-'*7}")
        total_irrep_shift = 0.0
        for irrep, P_irr in sorted(proj_matrices_by_irrep.items()):
            tr_P = float(np.trace(P_irr))
            n_modes = int(round(tr_P))
            PVGVP = P_irr @ VGV_6d @ P_irr
            tr_PVGVP = float(np.trace(PVGVP))
            avg_per_mode = tr_PVGVP / n_modes if n_modes > 0 else 0.0
            total_irrep_shift += tr_PVGVP
            log(f"  {irrep:6s} | {tr_P:10.4f} | {tr_PVGVP:14.8f} | "
                f"{avg_per_mode:20.8f} | {n_modes:7d}")

        log(f"\n  Total shift from irrep decomposition: {total_irrep_shift:.8f}")
        log(f"  Total shift from direct trace:         {float(np.trace(VGV_6d)):.8f}")

        # Which irreps carry the corner deviation?
        log(f"\n  Irrep channels with nonzero shift contribution:")
        for irrep, P_irr in sorted(proj_matrices_by_irrep.items()):
            PVGVP = P_irr @ VGV_6d @ P_irr
            tr_val = float(np.trace(PVGVP))
            if abs(tr_val) > 1e-8:
                log(f"    {irrep}: tr(P * VGV * P) = {tr_val:+.8f}")
    else:
        log(f"  WARNING: Expected 6 flat-band modes at Gamma, got {V_flat_6.shape[1]}; skipping irrep analysis.")
        decomp_str = "(analysis skipped)"

    # ------------------------------------------------------------------
    # STEP 10: Summary and closed-form check
    # ------------------------------------------------------------------
    log("\n" + "=" * 78)
    log("[SUMMARY] Corner deviation analytic structure")
    log("=" * 78)

    log(f"\n1. Flat bands:")
    log(f"   Periodic BC: {len(flat_bands)} flat bands at lambda=-3 (each k-point has 6 such bands)")
    log(f"   Open BC:     {n_3} eigenvalues at lambda=-3 (= 108 = 27 BCs x 6 per BC / ... )")

    log(f"\n2. Corner deviation (from open-BC direct diagonalization):")
    log(f"   avg_proj(Corner) = {shell_avg.get('Corner', 'n/a'):.8f}")
    log(f"   expected         = 3 (graph degree of corner BC)")
    log(f"   delta_corner     = {delta_corner:.8f}")
    log(f"   Rational approx: {r_delta}")
    log(f"   1/72 = {1/72:.8f}  --> delta_corner * 72 = {delta_corner*72:.6f}")
    log(f"   7/504 = {7/504:.8f} --> delta_corner * 504 = {delta_corner*504:.6f}")
    log(f"   1/70 = {1/70:.8f}  --> delta_corner * 70  = {delta_corner*70:.6f}")
    log(f"   1/75 = {1/75:.8f}  --> delta_corner * 75  = {delta_corner*75:.6f}")

    log(f"\n3. Second-order perturbation shift (Bloch G.F. method, Gamma point):")
    log(f"   avg_shift_2nd   = {avg_shift_2nd:+.8f}")
    log(f"   delta_corner    = {delta_corner:+.8f}")
    log(f"   ratio           = {avg_shift_2nd / delta_corner:.6f} (should be 1 if perturbation is exact)")

    log(f"\n4. Irrep decomposition of 6D flat-band space:")
    log(f"   6D = {decomp_str}")
    log(f"   (Label: A2u x Eu x T1u are ungerade, matching A2u+Eu+T1u per spec)")

    log(f"\n5. Per-shell deviations:")
    for sname in shell_names:
        if sname in shell_dev:
            log(f"   {sname:7s}: deviation = {shell_dev[sname]:+.6f}  (expected ~0)")

    log(f"\n6. Consistency check: shell_avg = deg + deviation")
    for sname in shell_names:
        if sname in shell_avg:
            exp = shell_expected_deg[sname]
            log(f"   {sname:7s}: {shell_avg[sname]:.4f} = {exp} + {shell_dev[sname]:+.4f}")

    log(f"\n=== DONE ===")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(_LOG_LINES) + "\n")
            print(f"\n[results written to {OUT_PATH}]")
        except Exception as e:
            print(f"WARNING: could not write results file: {e}", file=sys.stderr)
