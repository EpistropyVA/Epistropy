# verify_t3_spectral_pinning.py
# Verify "spectral pinning" claim for BCC 540 periodic system.
#
# Claim: for each of the 27 k-points in the 3x3x3 BCC Brillouin zone,
# take the 6 flat-band eigenvectors v (eigenvalue -3 of H(k)):
#   (1) T(0) · v = -2 · v   (exact, residual < 1e-10)
#   (2) H_NN(k) · v = -1 · v   where H_NN(k) = H(k) - T(0)
#
# Construction reused from verify_pending_4b_rank19_shell.py imports.

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Replicate the BCC periodic construction (same as verify_pending_4b)
# ---------------------------------------------------------------------------

def build_bcc_lattice_periodic():
    return list(itertools.product(range(3), repeat=3))


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


def build_bloch_T(body_centers, bc_face_indices):
    """
    Build Bloch hopping matrices T[R] for the 20x20 unit-cell Hamiltonian.
    T[R] = hopping block from BC at R to BC at origin.
    """
    A = build_adjacency_matrix(build_all_faces(body_centers)[0])
    # Reorder into BC-block form
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
    return T


def main():
    print("=" * 65)
    print("VERIFY T3: Spectral Pinning in BCC 540 Periodic System")
    print("=" * 65)
    print()

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    N = len(face_to_idx)
    assert N == 540, f"Expected 540 faces, got {N}"
    print(f"Faces: {N} (OK)")

    T = build_bloch_T(body_centers, bc_face_indices)
    T0 = T[(0, 0, 0)]  # on-site block
    print(f"T(0) shape: {T0.shape}")
    print(f"T(0) eigenvalues (unique): {sorted(set(round(x,6) for x in np.linalg.eigvalsh(T0)))}")

    kvecs = list(itertools.product(range(3), repeat=3))
    print(f"k-points: {len(kvecs)}")
    print()

    max_res1 = 0.0  # residual for T(0)·v = -2·v
    max_res2 = 0.0  # residual for H_NN(k)·v = -1·v

    all_pass1 = True
    all_pass2 = True

    for n_tuple in kvecs:
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = np.zeros((20, 20), dtype=complex)
        for R, mat in T.items():
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k, R_vec))
            H_k += phase * mat

        evals, evecs = np.linalg.eigh(H_k)

        # Flat band eigenvectors: eigenvalue -3
        flat_mask = np.abs(evals + 3.0) < 1e-8
        n_flat = np.sum(flat_mask)
        if n_flat != 6:
            print(f"WARNING: k={n_tuple} has {n_flat} flat bands (expect 6), evals={evals[:8]}")

        fv = evecs[:, flat_mask]  # shape (20, n_flat)

        # Test 1: T(0) · v = -2 · v
        Tv = T0 @ fv
        res1 = np.max(np.abs(Tv - (-2.0) * fv))
        max_res1 = max(max_res1, res1)
        if res1 >= 1e-10:
            all_pass1 = False
            print(f"  FAIL test1 at k={n_tuple}: residual = {res1:.3e}")

        # Test 2: H_NN(k) · v = -1 · v  where H_NN(k) = H(k) - T(0)
        H_NN = H_k - T0
        HNNv = H_NN @ fv
        res2 = np.max(np.abs(HNNv - (-1.0) * fv))
        max_res2 = max(max_res2, res2)
        if res2 >= 1e-10:
            all_pass2 = False
            print(f"  FAIL test2 at k={n_tuple}: residual = {res2:.3e}")

    print(f"Test 1 [T(0)·v = -2·v]:    max residual over all k, all flat bands = {max_res1:.3e}")
    print(f"Test 2 [H_NN(k)·v = -1·v]: max residual over all k, all flat bands = {max_res2:.3e}")
    print()

    tol = 1e-10
    pass1 = all_pass1 and max_res1 < tol
    pass2 = all_pass2 and max_res2 < tol

    print(f"Test 1: {'PASS' if pass1 else 'FAIL'} (threshold {tol:.0e})")
    print(f"Test 2: {'PASS' if pass2 else 'FAIL'} (threshold {tol:.0e})")
    print()
    if pass1 and pass2:
        print("PASS: Spectral pinning verified — both conditions hold at all k-points.")
    else:
        print("FAIL: Spectral pinning NOT confirmed.")


if __name__ == "__main__":
    main()
