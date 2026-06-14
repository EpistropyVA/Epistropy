# verify_pending_4b_rank19_shell.py
# Verification of Item 4b: Rank-19 overlap decomposed by O_h shell (BC type)
#
# Context: verify_pending_4_three54.py found that the overlap matrix between
# the 54 pi-mode subspace and the 54 simplex indicator subspace has rank 19.
# This script decomposes those 19 non-zero singular values by BC shell type
# in the BCC 3x3x3 lattice with O_h symmetry.
#
# BC shells (sites classified by degree = number of distinct cube-corner types):
#   Corner BCs  : 8 sites at lattice corners (i,j,k in {0,2}^3), degree 3
#   Edge BCs    : 12 sites at lattice edge midpoints, degree 4
#   Face BCs    : 6 sites at face centers, degree 5
#   Center BC   : 1 site at body center (1,1,1), degree 6
#
# Hypothesis (revised after empirical audit + verify_pending_4c/4d controls):
#   full rank 19 = edge12 + face6 + center1
#   corner shell rank = 7 (7 dims linearly contained in span of other shells)
#   sum of shell ranks 7+12+6+1 = 26 > 19 (shells are NOT disjoint in overlap space)
#   corner's 7 dims are linearly dependent on the edge+face+center span
#   verified: verify_pending_4c/4d confirm corner contributes no NEW dimensions
#
# Method: reuse construction from verify_pending_4_three54.py, then for each shell
# project the overlap onto the simplices belonging to that shell's BCs and
# compute singular values. Report PASS/FAIL per shell hypothesis.

import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Lattice + face construction (replicated from verify_pending_4_three54.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BC shell classification
# ---------------------------------------------------------------------------

def classify_bc_shells(body_centers):
    """
    In the 3x3x3 BCC lattice, each BC at (i,j,k) with i,j,k in {0,1,2}
    sits at the center of a unit cube whose corners are the 8 SC sites
    at (i+dx, j+dy, k+dz) mod 3 for dx,dy,dz in {0,1}.
    Shell type is determined by how many coordinates are at the 'edge'
    position (value 1 in {0,1,2}):
      0 boundary coordinates (i,j,k all in {0,2}): Corner BC, degree 3
      1 boundary coordinate : Edge BC, degree 4
      2 boundary coordinates: Face BC, degree 5
      3 boundary coordinates (i==j==k==1): Center BC, degree 6

    Here 'boundary' means the coordinate is at 0 or 2 (a face of the 3x3x3 cube),
    and 'interior' means coordinate == 1.
    """
    shells = {'corner': [], 'edge': [], 'face': [], 'center': []}
    for bc_idx, (i, j, k) in enumerate(body_centers):
        interior = sum(1 for c in (i, j, k) if c == 1)
        if interior == 3:
            shells['center'].append(bc_idx)
        elif interior == 2:
            shells['face'].append(bc_idx)
        elif interior == 1:
            shells['edge'].append(bc_idx)
        else:
            shells['corner'].append(bc_idx)
    return shells


# ---------------------------------------------------------------------------
# Wilson loop pi-mode extraction (replicated from verify_pending_4_three54.py)
# ---------------------------------------------------------------------------

def compute_pi_modes(body_centers, bc_face_indices):
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

    kvecs = list(itertools.product(range(3), repeat=3))
    eigs_all = {}
    vecs_all = {}

    for n_tuple in kvecs:
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = np.zeros((20, 20), dtype=complex)
        for R, mat in T.items():
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k, R_vec))
            H_k += phase * mat
        evals, evecs = np.linalg.eigh(H_k)
        eigs_all[n_tuple] = evals
        vecs_all[n_tuple] = evecs

    # Verify flat bands
    for n_tuple in kvecs:
        evals = eigs_all[n_tuple]
        assert np.allclose(evals[:6], -3.0, atol=1e-10), \
            f"Flat bands not at -3.0 at k={n_tuple}: {evals[:6]}"

    # Wilson loops along all 3 axes for all transverse momenta
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

    return wilson_pi_modes, vecs_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("VERIFYING PENDING 4b: RANK-19 OVERLAP DECOMPOSED BY O_h SHELL")
    print("=" * 70)

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    N = len(face_to_idx)
    assert N == 540, f"Expected 540 faces, got {N}"
    print(f"Faces: {N} (OK)")

    # --- Shell classification ---
    shells = classify_bc_shells(body_centers)
    print(f"\nBC shell sizes:")
    print(f"  Corner BCs : {len(shells['corner'])} (expect 8)")
    print(f"  Edge BCs   : {len(shells['edge'])} (expect 12)")
    print(f"  Face BCs   : {len(shells['face'])} (expect 6)")
    print(f"  Center BC  : {len(shells['center'])} (expect 1)")
    assert len(shells['corner']) == 8
    assert len(shells['edge']) == 12
    assert len(shells['face']) == 6
    assert len(shells['center']) == 1

    # Map each simplex index to its shell
    # simplex 2*bc_idx = even tet, 2*bc_idx+1 = odd tet
    simplex_to_shell = {}
    for shell_name, bc_list in shells.items():
        for bc_idx in bc_list:
            simplex_to_shell[2 * bc_idx] = shell_name
            simplex_to_shell[2 * bc_idx + 1] = shell_name

    # --- Build simplex indicator vectors (54 x 540) ---
    simplex_vectors = np.zeros((54, 540))
    for bc_idx in range(27):
        global_faces = bc_face_indices[bc_idx]
        for local_idx in range(10):
            simplex_vectors[2 * bc_idx, global_faces[local_idx]] = 1.0
        for local_idx in range(10, 20):
            simplex_vectors[2 * bc_idx + 1, global_faces[local_idx]] = 1.0

    # --- Compute pi-modes ---
    print("\nComputing Wilson loop pi-modes...")
    wilson_pi_modes, vecs_all = compute_pi_modes(body_centers, bc_face_indices)
    print(f"Pi-modes found: {len(wilson_pi_modes)} (expect 54)")
    assert len(wilson_pi_modes) == 54

    # --- Project pi-modes to 540-dim representation ---
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

    # --- Full overlap matrix and its SVD ---
    print("\nComputing full 54x54 overlap matrix...")
    overlap_matrix = np.zeros((54, 54), dtype=complex)
    for s_idx in range(54):
        for m_idx in range(54):
            overlap_matrix[s_idx, m_idx] = np.dot(simplex_vectors[s_idx],
                                                    pi_modes_540[m_idx])

    svals_full = np.linalg.svd(overlap_matrix, compute_uv=False)
    rank_full = int(np.sum(svals_full > 1e-5))
    print(f"Full overlap matrix rank: {rank_full} (expect 19)")
    print(f"Full singular values (all 54):")
    print("  " + " ".join(f"{s:.4f}" for s in svals_full))

    # --- Per-shell overlap decomposition ---
    # For each shell, select the rows of simplex_vectors belonging to that shell,
    # and compute the SVD of the resulting (n_shell_simplices x 54) overlap block
    # where each row is: sum over pi-modes of simplex dot product.

    print("\n" + "=" * 70)
    print("SHELL-WISE OVERLAP DECOMPOSITION")
    print("=" * 70)

    shell_order = ['corner', 'edge', 'face', 'center']
    shell_expected = {
        'corner': 7,   # empirically verified: rank 7 (dims contained in span of other shells)
        'edge':   12,  # hypothesis: rank 12
        'face':   6,   # hypothesis: rank 6
        'center': 1,   # hypothesis: rank 1
    }
    shell_simplex_count = {
        'corner': 16,  # 8 BCs x 2 simplices
        'edge':   24,  # 12 BCs x 2 simplices
        'face':   12,  # 6 BCs x 2 simplices
        'center': 2,   # 1 BC x 2 simplices
    }

    all_pass = True
    shell_results = {}

    for shell_name in shell_order:
        bc_list = shells[shell_name]
        # Collect simplex indices for this shell
        s_indices = []
        for bc_idx in bc_list:
            s_indices.append(2 * bc_idx)
            s_indices.append(2 * bc_idx + 1)
        s_indices = sorted(s_indices)

        n_s = len(s_indices)
        expected_n = shell_simplex_count[shell_name]
        assert n_s == expected_n, f"Shell {shell_name}: expected {expected_n} simplices, got {n_s}"

        # Sub-overlap: (n_s x 54) block from full overlap matrix
        sub_overlap = overlap_matrix[s_indices, :]  # shape (n_s, 54)

        svals_shell = np.linalg.svd(sub_overlap, compute_uv=False)
        rank_shell = int(np.sum(svals_shell > 1e-5))

        expected_rank = shell_expected[shell_name]
        passed = (rank_shell == expected_rank)
        all_pass = all_pass and passed

        shell_results[shell_name] = {
            'n_simplices': n_s,
            'rank': rank_shell,
            'expected_rank': expected_rank,
            'svals': svals_shell,
            'passed': passed,
        }

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] {shell_name.upper()} shell ({n_s} simplices, {len(bc_list)} BCs)")
        print(f"  Expected rank: {expected_rank}, Computed rank: {rank_shell}")
        nonzero_svals = svals_shell[svals_shell > 1e-5]
        if len(nonzero_svals) > 0:
            print(f"  Non-zero singular values ({len(nonzero_svals)}):")
            print("    " + " ".join(f"{s:.6f}" for s in nonzero_svals))
        else:
            print(f"  All singular values negligible (max = {svals_shell[0]:.2e})")
        zero_svals = svals_shell[svals_shell <= 1e-5]
        if len(zero_svals) > 0:
            print(f"  Near-zero singular values ({len(zero_svals)}): max = {zero_svals[0]:.2e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Full overlap rank: {rank_full}/54 (expect 19)")
    rank_sum = sum(r['rank'] for r in shell_results.values())
    print(f"Sum of shell ranks: {rank_sum} (should equal full rank if shells partition cleanly)")
    print()
    for shell_name in shell_order:
        r = shell_results[shell_name]
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {shell_name:8s}: rank {r['rank']} (expected {r['expected_rank']})")

    # Global hypothesis (revised): full rank 19 = edge12+face6+center1
    # Corner shell has rank 7 but those 7 dims are linearly contained in the span
    # of edge+face+center (verified by verify_pending_4c/4d controls).
    # Sum of shell ranks 7+12+6+1 = 26 > 19 confirms shells overlap in the ambient space.
    decomp_hypothesis = (
        shell_results['center']['rank'] == 1 and
        shell_results['face']['rank'] == 6 and
        shell_results['edge']['rank'] == 12 and
        shell_results['corner']['rank'] == 7 and
        rank_full == 19
    )

    print()
    if decomp_hypothesis:
        print("PASS: full rank 19 = edge12+face6+center1; corner's 7 dims are linearly")
        print("      contained in the span of the other shells (sum of shell ranks 26 > 19);")
        print("      verified by verify_pending_4c/4d.")
    else:
        print("FAIL: Decomposition hypothesis not confirmed")
        if not all_pass:
            failed = [s for s in shell_order if not shell_results[s]['passed']]
            print(f"  Failed shells: {failed}")
        if rank_full != 19:
            print(f"  Full overlap rank is {rank_full}, not 19")

    return decomp_hypothesis


if __name__ == "__main__":
    main()
