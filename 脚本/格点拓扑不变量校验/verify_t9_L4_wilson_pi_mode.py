# verify_t9_L4_wilson_pi_mode.py
#
# Test prediction T9: Wilson loop / Berry phase pi-modes depend on L-parity.
#
# PREDICTION:
#   L=3 (odd):  cross-parity coupling opens -> pi-modes present in Wilson spectrum
#   L=4 (even): cross-parity channel closes -> pi-modes absent or qualitatively reduced
#
# Method:
#   1. Build BCC L×L×L periodic lattice with triangular simplex faces (generalized to arbitrary L)
#   2. Construct Bloch Hamiltonian H(k) at each k-point on L×L×L BZ grid
#   3. Extract flat-band eigenspace (eigenvalue -3)
#   4. Compute Wilson loops along each axis for all transverse momentum pairs
#   5. Count pi-phase eigenvalues in Wilson spectrum
#   6. Compare L=3 vs L=4
#
# Additionally: verify parity decomposition (L=3 should couple, L=4 should not)

import itertools
import sys
import io
from collections import defaultdict

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ============================================================
# BCC lattice construction — generalized to arbitrary L
# ============================================================

def build_bcc_lattice_periodic(L):
    """Body centers at (i,j,k) for i,j,k in range(L)."""
    body_centers = []
    for i, j, k in itertools.product(range(L), repeat=3):
        body_centers.append((i, j, k))
    return body_centers


def enumerate_simplex_faces(bc_ijk, L):
    """
    For a body center at (i,j,k), enumerate all 20 triangular faces
    from its two tetrahedra (even-parity and odd-parity corners).
    Vertices are wrapped mod L for periodic boundary conditions.
    Returns list of 20 frozensets, each a triangle (3 vertices).
    """
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)
    orig_corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = i + dx, j + dy, k + dz
        wx, wy, wz = ox % L, oy % L, oz % L
        # Parity from UNWRAPPED coordinates
        parity = (ox + oy + oz) % 2
        orig_corners.append((parity, ('c', wx, wy, wz)))
    even_corners = [v for (par, v) in orig_corners if par == 0]
    odd_corners = [v for (par, v) in orig_corners if par == 1]
    faces = []
    for tet_corners in (even_corners, odd_corners):
        simplex_verts = [bc_v] + tet_corners  # 5 vertices
        for combo in itertools.combinations(simplex_verts, 3):
            face = frozenset(combo)
            if len(face) == 3:  # guard against degenerate faces
                faces.append(face)
    return faces


def translate_vertex(v, shift_ijk, L):
    si, sj, sk = shift_ijk
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i + si) % L, (j + sj) % L, (k + sk) % L)
    else:
        _, cx, cy, cz = v
        return ('c', (cx + si) % L, (cy + sj) % L, (cz + sk) % L)


def translate_face(face, shift_ijk, L):
    return frozenset(translate_vertex(v, shift_ijk, L) for v in face)


def build_all_faces(body_centers, L):
    """
    Build all distinct triangular faces by translating the reference BC's faces
    to all BCs. Returns face_to_idx mapping and per-BC face index lists.
    """
    ref_bc = (0, 0, 0)
    ref_faces = enumerate_simplex_faces(ref_bc, L)
    face_to_idx = {}
    bc_face_indices = []
    global_idx = 0
    for bc_idx, bc_ijk in enumerate(body_centers):
        local_indices = []
        for m, ref_face in enumerate(ref_faces):
            shifted_face = translate_face(ref_face, bc_ijk, L)
            if shifted_face not in face_to_idx:
                face_to_idx[shifted_face] = global_idx
                global_idx += 1
            local_indices.append(face_to_idx[shifted_face])
        bc_face_indices.append(local_indices)
    return face_to_idx, bc_face_indices, ref_faces


def build_adjacency_matrix(face_to_idx):
    """
    Build face-adjacency matrix: two faces are adjacent iff they share exactly 2 vertices.
    """
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
# Parity decomposition
# ============================================================

def check_parity_components(L):
    """
    Check whether even-parity and odd-parity faces couple under periodic BCs.
    Returns (n_components, component_sizes, is_mixed).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    face_to_parity = {}
    for i, j, k in itertools.product(range(L), repeat=3):
        bc_v = ('bc', i % L, j % L, k % L)
        orig_corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            ox, oy, oz = i + dx, j + dy, k + dz
            parity = (ox + oy + oz) % 2
            cv = ('c', ox % L, oy % L, oz % L)
            orig_corners.append((parity, cv))
        even_corners = [v for (p, v) in orig_corners if p == 0]
        odd_corners = [v for (p, v) in orig_corners if p == 1]
        for tet_parity, tet_corners in enumerate([even_corners, odd_corners]):
            simplex_verts = [bc_v] + tet_corners
            for combo in itertools.combinations(simplex_verts, 3):
                face_set = frozenset(combo)
                if len(face_set) < 3:
                    continue
                if face_set not in face_to_parity:
                    face_to_parity[face_set] = tet_parity

    face_list = list(face_to_parity.keys())
    face_parity = [face_to_parity[f] for f in face_list]
    N = len(face_list)

    vertex_to_faces = defaultdict(set)
    for idx, face in enumerate(face_list):
        for v in face:
            vertex_to_faces[v].add(idx)

    rows, cols = [], []
    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a in range(len(flist)):
            for b in range(a + 1, len(flist)):
                candidate_pairs.add((flist[a], flist[b]))
    for i, j in candidate_pairs:
        if len(face_list[i] & face_list[j]) == 2:
            rows.extend([i, j])
            cols.extend([j, i])

    data = [1] * len(rows)
    A_sp = csr_matrix((data, (rows, cols)), shape=(N, N))
    n_comp, labels = connected_components(A_sp, directed=False, return_labels=True)

    from collections import Counter
    comp_info = defaultdict(Counter)
    for idx, (par, comp) in enumerate(zip(face_parity, labels)):
        comp_info[comp][par] += 1

    comp_sizes = []
    is_mixed = False
    for c in range(n_comp):
        ev = comp_info[c].get(0, 0)
        od = comp_info[c].get(1, 0)
        comp_sizes.append((ev, od))
        if ev > 0 and od > 0:
            is_mixed = True

    return n_comp, comp_sizes, is_mixed


# ============================================================
# Bloch Hamiltonian and flat-band extraction
# ============================================================

def build_bloch_and_flatband(L):
    """
    Build the Bloch Hamiltonian H(k) for BCC L×L×L, diagonalize at each k-point,
    and extract the flat-band eigenspace at eigenvalue -3.

    Returns:
        kvecs: list of k-point tuples (n1,n2,n3)
        eigs_all: dict {k_tuple: eigenvalues}
        vecs_all: dict {k_tuple: eigenvectors}
        flat_band_dims: dict {k_tuple: number of eigenvalues at -3}
        n_faces: total number of faces
    """
    body_centers = build_bcc_lattice_periodic(L)
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers, L)
    N = len(face_to_idx)
    n_bc = len(body_centers)
    n_faces_per_bc = len(ref_faces)  # should be 20

    A = build_adjacency_matrix(face_to_idx)

    # Reorder into BC-block form: BC0's faces, BC1's faces, ...
    perm = []
    for bc_idx in range(n_bc):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)
    A_reord = A[np.ix_(perm, perm)]

    # Extract hopping matrices T(R) = block(0, bc_j) of reordered adjacency
    def get_block(A_r, bc_i, bc_j, size=n_faces_per_bc):
        return A_r[bc_i * size:(bc_i + 1) * size, bc_j * size:(bc_j + 1) * size].copy()

    T = {}
    for bc_j_idx, bc_j_ijk in enumerate(body_centers):
        R = tuple(bc_j_ijk[a] % L for a in range(3))
        if R not in T:
            T[R] = get_block(A_reord, 0, bc_j_idx)
        else:
            # Multiple BCs may map to the same R; accumulate
            # Actually each R is unique since bc_j_ijk are distinct in range(L)
            pass

    # Build H(k) at each k-point
    kvecs = list(itertools.product(range(L), repeat=3))
    eigs_all = {}
    vecs_all = {}
    flat_band_dims = {}

    for n_tuple in kvecs:
        k = np.array([2 * np.pi * n / L for n in n_tuple])
        H_k = np.zeros((n_faces_per_bc, n_faces_per_bc), dtype=complex)
        for R, mat in T.items():
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k, R_vec))
            H_k += phase * mat
        evals, evecs = np.linalg.eigh(H_k)
        eigs_all[n_tuple] = evals
        vecs_all[n_tuple] = evecs

        # Count flat-band states at -3
        n_flat = np.sum(np.abs(evals - (-3.0)) < 0.01)
        flat_band_dims[n_tuple] = int(n_flat)

    return kvecs, eigs_all, vecs_all, flat_band_dims, N


# ============================================================
# Wilson loop computation
# ============================================================

def compute_wilson_loops(L, kvecs, eigs_all, vecs_all, flat_band_dims, pi_tol=0.01):
    """
    Compute Wilson loops for the flat-band bundle along each axis direction.

    For each axis (x, y, z) and each fixed pair of transverse momenta,
    compute the Wilson loop W = prod S(k_j -> k_{j+1}) where
    S_{ab} = <psi_a(k_j)|psi_b(k_{j+1})> restricted to flat-band states.

    Returns:
        wilson_results: list of dicts with loop info and phases
        n_pi_total: total number of Wilson eigenvalues at phase pi
        n_nonpi_total: total number not at pi
    """
    axes = ['x', 'y', 'z']
    wilson_results = []
    n_pi_total = 0
    n_nonpi_total = 0

    for axis_idx, axis_name in enumerate(axes):
        other_axes = [i for i in range(3) if i != axis_idx]
        for other1 in range(L):
            for other2 in range(L):
                # Build the loop along this axis
                loop_k = []
                for n_ax in range(L):
                    n_tuple = [0, 0, 0]
                    n_tuple[axis_idx] = n_ax
                    n_tuple[other_axes[0]] = other1
                    n_tuple[other_axes[1]] = other2
                    loop_k.append(tuple(n_tuple))

                # Determine flat-band dimension at each k-point on the loop
                dims = [flat_band_dims[kp] for kp in loop_k]

                # Check uniform degeneracy along the loop
                if len(set(dims)) != 1:
                    # Non-uniform flat-band dimension: skip or handle
                    wilson_results.append({
                        'axis': axis_name,
                        'fixed': (other1, other2),
                        'loop_k': loop_k,
                        'flat_dims': dims,
                        'uniform': False,
                        'phases': None,
                        'n_pi': 0,
                        'n_nonpi': 0,
                        'note': f'Non-uniform flat-band dim: {dims}'
                    })
                    continue

                n_flat = dims[0]
                if n_flat == 0:
                    wilson_results.append({
                        'axis': axis_name,
                        'fixed': (other1, other2),
                        'loop_k': loop_k,
                        'flat_dims': dims,
                        'uniform': True,
                        'phases': np.array([]),
                        'n_pi': 0,
                        'n_nonpi': 0,
                        'note': 'No flat bands at this k-point'
                    })
                    continue

                deg_bands = list(range(n_flat))

                # Compute Wilson loop matrix
                W = np.eye(n_flat, dtype=complex)
                for j in range(L):
                    k_cur = loop_k[j]
                    k_next = loop_k[(j + 1) % L]
                    # Overlap matrix
                    S = np.zeros((n_flat, n_flat), dtype=complex)
                    for a_idx in range(n_flat):
                        for b_idx in range(n_flat):
                            S[a_idx, b_idx] = np.vdot(
                                vecs_all[k_cur][:, a_idx],
                                vecs_all[k_next][:, b_idx]
                            )
                    W = W @ S

                # Eigenvalues of Wilson loop
                w_evals = np.linalg.eigvals(W)
                phases = np.angle(w_evals)

                # Count pi-phases
                n_pi = sum(1 for p in phases if abs(abs(p) - np.pi) < pi_tol)
                n_nonpi = len(phases) - n_pi

                n_pi_total += n_pi
                n_nonpi_total += n_nonpi

                wilson_results.append({
                    'axis': axis_name,
                    'fixed': (other1, other2),
                    'loop_k': loop_k,
                    'flat_dims': dims,
                    'uniform': True,
                    'phases': phases,
                    'n_pi': n_pi,
                    'n_nonpi': n_nonpi,
                    'note': ''
                })

    return wilson_results, n_pi_total, n_nonpi_total


# ============================================================
# Main
# ============================================================

def analyze_L(L):
    """Full analysis for a given L. Returns summary dict."""
    print(f"\n{'=' * 72}")
    print(f"  ANALYZING L = {L}")
    print(f"{'=' * 72}")

    # --- Step 1: Parity decomposition ---
    print(f"\n  [Step 1] Parity component analysis (L={L})")
    n_comp, comp_sizes, is_mixed = check_parity_components(L)
    print(f"    Components: {n_comp}")
    for c, (ev, od) in enumerate(comp_sizes):
        mixed_tag = " [MIXED]" if (ev > 0 and od > 0) else ""
        print(f"    Component {c}: even={ev}, odd={od}, total={ev + od}{mixed_tag}")
    print(f"    Cross-parity coupling: {'YES' if is_mixed else 'NO'}")

    # --- Step 2: Bloch Hamiltonian and flat bands ---
    print(f"\n  [Step 2] Bloch Hamiltonian and flat-band extraction")
    kvecs, eigs_all, vecs_all, flat_band_dims, n_faces = build_bloch_and_flatband(L)
    print(f"    Total faces: {n_faces}")
    print(f"    k-points: {len(kvecs)} ({L}x{L}x{L} grid)")

    # Check flat-band existence
    flat_dims_unique = sorted(set(flat_band_dims.values()))
    total_flat = sum(flat_band_dims.values())
    print(f"    Flat-band dimensions across k-points: {flat_dims_unique}")
    print(f"    Total flat-band states (sum over BZ): {total_flat}")

    # Verify eigenvalue -3 exists
    has_flat = all(d > 0 for d in flat_band_dims.values())
    if has_flat:
        print(f"    Flat band at -3: PRESENT at all k-points")
    else:
        missing = [k for k, d in flat_band_dims.items() if d == 0]
        print(f"    Flat band at -3: MISSING at {len(missing)} k-points")
        if len(missing) <= 10:
            for k in missing:
                print(f"      k={k}: eigenvalues = {eigs_all[k][:5]}...")

    # Sample: print eigenvalues at Gamma point
    gamma = tuple([0] * 3)
    print(f"    Eigenvalues at Gamma: {eigs_all[gamma][:8]}... (first 8)")

    # --- Step 3: Wilson loops ---
    print(f"\n  [Step 3] Wilson loop computation")
    pi_tol = 0.01
    wilson_results, n_pi_total, n_nonpi_total = compute_wilson_loops(
        L, kvecs, eigs_all, vecs_all, flat_band_dims, pi_tol=pi_tol
    )
    n_loops = len(wilson_results)
    n_uniform = sum(1 for r in wilson_results if r['uniform'])
    n_nonuniform = n_loops - n_uniform

    print(f"    Total Wilson loops: {n_loops} (3 axes x {L**2} transverse pairs)")
    print(f"    Uniform flat-band dim: {n_uniform}, Non-uniform: {n_nonuniform}")
    print(f"    Total Wilson eigenvalues with |phase| ~ pi: {n_pi_total}")
    print(f"    Total Wilson eigenvalues with |phase| != pi: {n_nonpi_total}")

    # --- Step 4: Detailed Wilson phase report ---
    print(f"\n  [Step 4] Wilson phase details (pi-tolerance = {pi_tol})")
    axes = ['x', 'y', 'z']
    for axis_name in axes:
        print(f"\n    --- Loops along k_{axis_name} ---")
        for r in wilson_results:
            if r['axis'] != axis_name:
                continue
            if r['phases'] is None:
                print(f"      fixed={r['fixed']}: {r['note']}")
                continue
            if len(r['phases']) == 0:
                print(f"      fixed={r['fixed']}: no flat bands")
                continue
            sorted_phases = np.sort(r['phases'])
            phase_strs = [f"{p / np.pi:+.4f}pi" for p in sorted_phases]
            pi_mark = f"  <-- {r['n_pi']} pi-mode(s)" if r['n_pi'] > 0 else ""
            print(f"      fixed={r['fixed']}: [{', '.join(phase_strs)}]{pi_mark}")

    # --- Collect phase histogram ---
    all_phases = []
    for r in wilson_results:
        if r['phases'] is not None and len(r['phases']) > 0:
            all_phases.extend(r['phases'])
    all_phases = np.array(all_phases)

    if len(all_phases) > 0:
        print(f"\n    Phase histogram (all Wilson eigenvalues):")
        # Cluster phases
        sorted_p = np.sort(all_phases)
        clusters = []
        cur = [sorted_p[0]]
        for p in sorted_p[1:]:
            if abs(p - cur[-1]) < 0.05:
                cur.append(p)
            else:
                clusters.append((np.mean(cur), len(cur)))
                cur = [p]
        clusters.append((np.mean(cur), len(cur)))

        print(f"    {'Phase':>10} | {'Phase/pi':>10} | {'Count':>6}")
        print(f"    {'-' * 10}-+-{'-' * 10}-+-{'-' * 6}")
        for val, count in clusters:
            print(f"    {val:10.6f} | {val / np.pi:10.6f} | {count:6d}")

    return {
        'L': L,
        'n_faces': n_faces,
        'n_components': n_comp,
        'is_mixed': is_mixed,
        'flat_band_dim': flat_dims_unique[0] if len(flat_dims_unique) == 1 else flat_dims_unique,
        'total_flat': total_flat,
        'n_loops': n_loops,
        'n_pi': n_pi_total,
        'n_nonpi': n_nonpi_total,
        'n_wilson_evals': len(all_phases),
    }


def main():
    print("=" * 72)
    print("VERIFY T9: WILSON LOOP PI-MODE L-PARITY DEPENDENCE")
    print("=" * 72)
    print()
    print("PREDICTION:")
    print("  L=3 (odd):  cross-parity coupling OPEN  -> pi-modes PRESENT")
    print("  L=4 (even): cross-parity coupling CLOSED -> pi-modes ABSENT/REDUCED")
    print()
    print("  The e7 parity mechanism: 8 cube corners split by F2^3 into")
    print("  even={000,011,101,110} and odd={001,010,100,111}.")
    print("  L odd:  torus wrap mixes parity -> cross-coupling -> Berry phase pi-modes")
    print("  L even: torus wrap preserves parity -> 2 pure components -> no pi-modes")

    results = {}
    for L in [3, 4]:
        results[L] = analyze_L(L)

    # ====== SUMMARY TABLE ======
    print("\n\n" + "=" * 72)
    print("SUMMARY COMPARISON: L=3 vs L=4")
    print("=" * 72)

    header = (f"{'L':>3} | {'Faces':>6} | {'Flat dim':>9} | {'Components':>10} | "
              f"{'Mixed?':>7} | {'Wilson pi':>10} | {'Wilson non-pi':>13} | "
              f"{'Total evals':>11}")
    print(header)
    print("-" * len(header))
    for L in [3, 4]:
        r = results[L]
        mixed_str = "YES" if r['is_mixed'] else "NO"
        print(f"{r['L']:>3} | {r['n_faces']:>6} | {str(r['flat_band_dim']):>9} | "
              f"{r['n_components']:>10} | {mixed_str:>7} | {r['n_pi']:>10} | "
              f"{r['n_nonpi']:>13} | {r['n_wilson_evals']:>11}")

    # ====== PASS/FAIL ======
    print("\n" + "=" * 72)
    print("PASS/FAIL EVALUATION")
    print("=" * 72)

    r3 = results[3]
    r4 = results[4]

    # Check 1: Parity components
    parity_ok_3 = (r3['n_components'] == 1 and r3['is_mixed'])
    parity_ok_4 = (r4['n_components'] == 2 and not r4['is_mixed'])
    parity_status_3 = "PASS" if parity_ok_3 else "FAIL"
    parity_status_4 = "PASS" if parity_ok_4 else "FAIL"
    print(f"\n[{parity_status_3}] L=3 parity: {r3['n_components']} component(s), "
          f"mixed={'YES' if r3['is_mixed'] else 'NO'} (expected: 1 component, mixed=YES)")
    print(f"[{parity_status_4}] L=4 parity: {r4['n_components']} component(s), "
          f"mixed={'YES' if r4['is_mixed'] else 'NO'} (expected: 2 components, mixed=NO)")

    # Check 2: Flat bands exist
    flat_ok_3 = (r3['total_flat'] > 0)
    flat_ok_4 = (r4['total_flat'] > 0)
    flat_status_3 = "PASS" if flat_ok_3 else "FAIL"
    flat_status_4 = "PASS" if flat_ok_4 else "FAIL"
    print(f"\n[{flat_status_3}] L=3 flat band at -3: dim={r3['flat_band_dim']}, "
          f"total={r3['total_flat']}")
    print(f"[{flat_status_4}] L=4 flat band at -3: dim={r4['flat_band_dim']}, "
          f"total={r4['total_flat']}")

    # Check 3: Pi-mode comparison
    has_pi_3 = (r3['n_pi'] > 0)
    no_pi_4 = (r4['n_pi'] == 0)
    fewer_pi_4 = (r4['n_pi'] < r3['n_pi'])

    pi_status_3 = "PASS" if has_pi_3 else "FAIL"
    print(f"\n[{pi_status_3}] L=3 has pi-modes: {r3['n_pi']} "
          f"(expected: > 0)")

    if no_pi_4:
        pi_status_4 = "PASS"
        print(f"[{pi_status_4}] L=4 has NO pi-modes: {r4['n_pi']} "
              f"(expected: 0 or significantly fewer)")
    elif fewer_pi_4:
        pi_status_4 = "PASS"
        print(f"[{pi_status_4}] L=4 has FEWER pi-modes: {r4['n_pi']} vs L=3's {r3['n_pi']} "
              f"(qualitative reduction confirmed)")
    else:
        pi_status_4 = "FAIL"
        print(f"[{pi_status_4}] L=4 pi-modes: {r4['n_pi']} vs L=3's {r3['n_pi']} "
              f"(expected reduction not observed)")

    # Overall
    all_pass = all([
        parity_ok_3, parity_ok_4,
        flat_ok_3, flat_ok_4,
        has_pi_3,
        no_pi_4 or fewer_pi_4,
    ])

    print(f"\n{'=' * 72}")
    if all_pass:
        print("OVERALL: PASS")
        print("  T9 prediction confirmed: L-parity controls Wilson loop pi-mode structure.")
        print("  L=3 (odd) has cross-parity coupling and pi-modes in the Wilson spectrum.")
        print("  L=4 (even) has decoupled parity components and reduced/absent pi-modes.")
    else:
        print("OVERALL: FAIL (or partial)")
        print("  T9 prediction not fully confirmed. See individual checks above.")
    print("=" * 72)

    return all_pass


if __name__ == "__main__":
    main()
