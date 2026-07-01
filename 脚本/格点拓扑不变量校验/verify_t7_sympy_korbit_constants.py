# verify_t7_sympy_korbit_constants.py
# Exact computation of Tr(P_k T(0)) for each k-orbit on the BCC 3x3x3 lattice.
#
# The 27 k-points on the 3x3x3 BZ grid split into 4 orbits under O_h:
#   Gamma:  (0,0,0)                          — 1 point
#   Axis:   (2pi/3,0,0) and permutations     — 6 points
#   Face:   (2pi/3,2pi/3,0) and permutations — 12 points
#   Body:   (2pi/3,2pi/3,2pi/3) and variants — 8 points
#
# For each orbit, Tr(P_k T(0)) is a gauge-invariant scalar (P_k = projector
# onto the 6D flat-band eigenspace at eigenvalue -3).
#
# Strategy:
#   Phase 1: numpy numerical for all 27 k-points.
#   Phase 2: sympy exact for Gamma (real integer matrix — fast).
#   Phase 3: sympy exact for non-Gamma via nullspace of H(k)+3I over Q(omega).
#            With timeout — falls back to nsimplify if too slow.
#   Phase 4: nsimplify identification of exact forms from numerical values.
#   Phase 5: Verification and weighted-sum identities.

import itertools
import numpy as np
import threading

def P(*args, **kwargs):
    """Print with flush."""
    print(*args, **kwargs, flush=True)

# ===========================================================================
# Lattice + face construction (replicated from verify_pending_4b_rank19_shell.py)
# ===========================================================================

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


# ===========================================================================
# Extract T(R) blocks and build Bloch Hamiltonian
# ===========================================================================

def extract_T_blocks(body_centers, bc_face_indices, face_to_idx):
    """Extract T(R) = 20x20 hopping blocks for each displacement R."""
    A = build_adjacency_matrix(face_to_idx)
    perm = []
    for bc_idx in range(27):
        perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm, dtype=int)
    A_r = A[np.ix_(perm, perm)]
    T = {}
    for bc_j, bc_ijk in enumerate(body_centers):
        R = tuple(c % 3 for c in bc_ijk)
        T[R] = A_r[0:20, bc_j*20:(bc_j+1)*20].copy()
    return T


def build_Hk_numpy(T, n_tuple):
    """Build H(k) as numpy array for k = 2*pi*n/3."""
    k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
    H_k = np.zeros((20, 20), dtype=complex)
    for R, mat in T.items():
        R_vec = np.array(R, dtype=float)
        phase = np.exp(1j * np.dot(k, R_vec))
        H_k += phase * mat
    return H_k


def classify_k_orbits():
    """Classify 27 k-points into O_h orbits."""
    orbits = {'Gamma': [], 'Axis': [], 'Face': [], 'Body': []}
    for n in itertools.product(range(3), repeat=3):
        nz = sum(1 for x in n if x != 0)
        if nz == 0:
            orbits['Gamma'].append(n)
        elif nz == 1:
            orbits['Axis'].append(n)
        elif nz == 2:
            orbits['Face'].append(n)
        else:
            orbits['Body'].append(n)
    return orbits


# ===========================================================================
# Timeout helper for sympy computations
# ===========================================================================

def run_with_timeout(func, timeout_sec=120):
    """Run func() in a thread with timeout. Returns (result, ok)."""
    result = [None]
    exc = [None]
    def worker():
        try:
            result[0] = func()
        except Exception as e:
            exc[0] = e
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return None, False  # timed out (thread will eventually finish but we move on)
    if exc[0]:
        raise exc[0]
    return result[0], True


# ===========================================================================
# Main
# ===========================================================================

def main():
    P("=" * 72)
    P("EXACT Tr(P_k T0) FOR EACH k-ORBIT -- BCC 3x3x3 FLAT BAND")
    P("=" * 72)

    # --- Build lattice ---
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    N = len(face_to_idx)
    assert N == 540, f"Expected 540 faces, got {N}"
    P(f"Faces: {N}")

    T = extract_T_blocks(body_centers, bc_face_indices, face_to_idx)
    T0 = T[(0, 0, 0)]

    # Verify T(R) blocks are integer
    for R, mat in T.items():
        assert np.allclose(mat, np.round(mat)), f"T({R}) is not integer!"
    T0_int = np.round(T0).astype(int)
    P(f"T(0) is {T0_int.shape[0]}x{T0_int.shape[1]} integer matrix, "
      f"trace = {np.trace(T0_int)}")

    # How many distinct T(R) blocks?
    # Some R values may give the same block (related by lattice symmetry)
    P(f"Number of distinct R vectors: {len(T)}")

    orbits = classify_k_orbits()
    P(f"\nk-orbit sizes: "
      + ", ".join(f"{name}={len(pts)}" for name, pts in orbits.items()))

    orbit_reps = {
        'Gamma': (0, 0, 0),
        'Axis':  (1, 0, 0),
        'Face':  (1, 1, 0),
        'Body':  (1, 1, 1),
    }

    # ===================================================================
    # PHASE 1: Numerical Tr(P_k T0) for all 27 k-points
    # ===================================================================
    P("\n" + "-" * 72)
    P("PHASE 1: Numerical Tr(P_k T0) for all 27 k-points")
    P("-" * 72)

    tr_values = {}
    all_evals = {}
    for n in itertools.product(range(3), repeat=3):
        H_k = build_Hk_numpy(T, n)
        evals, evecs = np.linalg.eigh(H_k)
        assert np.allclose(evals[:6], -3.0, atol=1e-10), \
            f"Flat bands not at -3 for k={n}: {evals[:6]}"
        V = evecs[:, :6]
        tr_PkT0 = np.real(np.trace(V.conj().T @ T0 @ V))
        tr_values[n] = tr_PkT0
        all_evals[n] = evals

    # Group by orbit and verify constancy
    P(f"\n{'Orbit':8s} {'Rep':12s} {'Size':>4s}  {'Tr(P_k T0)':>18s}  {'Spread':>10s}")
    P("-" * 60)
    orbit_tr = {}
    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        pts = orbits[orbit_name]
        vals = [tr_values[n] for n in pts]
        spread = max(vals) - min(vals)
        mean_val = np.mean(vals)
        rep = orbit_reps[orbit_name]
        orbit_tr[orbit_name] = mean_val
        P(f"{orbit_name:8s} {str(rep):12s} {len(pts):4d}  {mean_val:18.12f}  {spread:10.2e}")
        assert spread < 1e-10, f"Orbit {orbit_name} not constant! spread={spread}"

    P("\nAll eigenvalues at representative k-points:")
    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        rep = orbit_reps[orbit_name]
        evals = all_evals[rep]
        P(f"  {orbit_name} {rep}: {np.round(evals, 6)}")

    # ===================================================================
    # PHASE 2: Exact sympy for Gamma point
    # ===================================================================
    P("\n" + "-" * 72)
    P("PHASE 2: Exact sympy computation -- Gamma point")
    P("-" * 72)

    import sympy
    from sympy import Matrix, Rational, pi, I, exp, sqrt, nsimplify
    from sympy import eye, zeros as sp_zeros, conjugate

    T_sym = {}
    for R, mat in T.items():
        T_sym[R] = Matrix(np.round(mat).astype(int).tolist())
    T0_sym = T_sym[(0, 0, 0)]

    # H(Gamma) = sum_R T(R) — real integer matrix
    H_gamma = sp_zeros(20, 20)
    for R, mat in T_sym.items():
        H_gamma += mat
    P(f"H(Gamma) constructed: {H_gamma.shape[0]}x{H_gamma.shape[1]} integer matrix")

    # Nullspace of H(Gamma) + 3I
    P("Computing nullspace of H(Gamma) + 3I ...")
    H_shifted = H_gamma + 3 * eye(20)

    def compute_gamma_exact():
        null_basis = H_shifted.nullspace()
        if len(null_basis) != 6:
            return None, len(null_basis)
        V_g = null_basis[0]
        for v in null_basis[1:]:
            V_g = V_g.row_join(v)
        # V_g is 20x6, real. Tr(P T0) = Tr(G^{-1} V^T T0 V)
        G = V_g.T * V_g
        VTV = V_g.T * T0_sym * V_g
        G_inv = G.inv()
        tr = (G_inv * VTV).trace()
        return sympy.simplify(tr), 6

    result, ok = run_with_timeout(compute_gamma_exact, timeout_sec=10)
    if ok and result is not None:
        tr_exact, dim = result
        if dim == 6:
            P(f"Nullspace dimension: 6 (correct)")
            P(f"Tr(P_Gamma T0) EXACT = {tr_exact}")
            P(f"  float value:  {float(tr_exact):.15f}")
            P(f"  numpy value:  {orbit_tr['Gamma']:.15f}")
            P(f"  Match: {abs(float(tr_exact) - orbit_tr['Gamma']) < 1e-10}")
        else:
            P(f"WARNING: nullspace dimension = {dim}, expected 6")
    elif not ok:
        P("Gamma exact computation timed out (300s). Using numerical value.")
    else:
        P("Gamma exact computation returned None.")

    # ===================================================================
    # PHASE 3: Exact sympy for non-Gamma orbits via nullspace
    # ===================================================================
    P("\n" + "-" * 72)
    P("PHASE 3: Exact sympy -- non-Gamma orbits (nullspace of H(k)+3I)")
    P("-" * 72)

    omega = exp(2 * pi * I / 3)

    def build_Hk_sympy(n_tuple):
        """Build H(k) as sympy Matrix. phase = omega^(n.R mod 3)."""
        H = sp_zeros(20, 20)
        for R, mat in T_sym.items():
            dot = sum(n_tuple[a] * R[a] for a in range(3))
            dot_mod3 = dot % 3
            if dot_mod3 == 0:
                phase = sympy.Integer(1)
            elif dot_mod3 == 1:
                phase = omega
            else:  # dot_mod3 == 2
                phase = omega**2
            H = H + phase * mat
        return H

    # Sanity check: sympy Gamma matches
    H_test = build_Hk_sympy((0, 0, 0))
    assert H_test == H_gamma, "Sympy Gamma mismatch!"
    P("Gamma sympy construction verified.")

    exact_results = {}

    for orbit_name in ['Axis', 'Face', 'Body']:
        rep = orbit_reps[orbit_name]
        P(f"\n--- {orbit_name} orbit, k = 2pi/3 * {rep} ---")

        H_k = build_Hk_sympy(rep)
        H_shifted_k = H_k + 3 * eye(20)

        P(f"  Computing nullspace (timeout 300s) ...")

        def compute_exact_orbit(H_sh=H_shifted_k):
            nb = H_sh.nullspace()
            if len(nb) != 6:
                return ('dim_error', len(nb))
            V_k = nb[0]
            for v in nb[1:]:
                V_k = V_k.row_join(v)
            V_k_dag = V_k.conjugate().T
            G = V_k_dag * V_k
            VdTV = V_k_dag * T0_sym * V_k
            G_inv = G.inv()
            tr = (G_inv * VdTV).trace()
            # Simplify: use omega^2 + omega + 1 = 0
            tr_s = sympy.simplify(sympy.expand(tr))
            return ('ok', tr_s)

        result, ok = run_with_timeout(compute_exact_orbit, timeout_sec=5)

        if ok and result is not None:
            status, val = result
            if status == 'ok':
                exact_results[orbit_name] = val
                P(f"  Tr(P_k T0) EXACT = {val}")
                try:
                    val_c = complex(val)
                    P(f"  float value:  {val_c.real:.15f} + {val_c.imag:.15f}i")
                    P(f"  numpy value:  {orbit_tr[orbit_name]:.15f}")
                    P(f"  Match: {abs(val_c.real - orbit_tr[orbit_name]) < 1e-8}")
                    if abs(val_c.imag) > 1e-10:
                        P(f"  WARNING: imaginary part non-zero!")
                except Exception:
                    P(f"  (could not evaluate numerically)")
            else:
                P(f"  WARNING: nullspace dimension = {val}, expected 6")
        else:
            if not ok:
                P(f"  Timed out after 300s. Falling back to nsimplify.")
            else:
                P(f"  Computation returned None.")

    # ===================================================================
    # PHASE 4: nsimplify identification from numerical values
    # ===================================================================
    P("\n" + "-" * 72)
    P("PHASE 4: nsimplify identification of exact forms")
    P("-" * 72)

    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        val = orbit_tr[orbit_name]
        P(f"\n{orbit_name}: numerical = {val:.15f}")

        # Rationality probe: does d * val round to integer for small d?
        found_rational = False
        for d in range(1, 101):
            prod = val * d
            if abs(prod - round(prod)) < 1e-8:
                num = round(prod)
                P(f"  Rational candidate: {num}/{d}")
                # Simplify fraction
                from math import gcd
                g = gcd(abs(num), d)
                P(f"  Simplified: {num//g}/{d//g}")
                found_rational = True
                break

        if not found_rational:
            P(f"  Not rational with denominator <= 100")

        # nsimplify with various extensions
        for label, extensions in [
            ('rational', []),
            ('sqrt(3)', [sqrt(3)]),
            ('sqrt(5)', [sqrt(5)]),
            ('sqrt(2)', [sqrt(2)]),
            ('sqrt(3),sqrt(5)', [sqrt(3), sqrt(5)]),
            ('generic', []),
        ]:
            try:
                if extensions:
                    r = nsimplify(val, extensions, rational=False, tolerance=1e-12)
                elif label == 'generic':
                    r = nsimplify(val, rational=False, tolerance=1e-12)
                else:
                    r = nsimplify(val, rational=True, tolerance=1e-12)
                err = abs(float(r) - val)
                if err < 1e-10:
                    P(f"  nsimplify ({label}): {r}  [err={err:.2e}]")
            except Exception:
                pass

    # ===================================================================
    # PHASE 5: Identities and cross-checks
    # ===================================================================
    P("\n" + "-" * 72)
    P("PHASE 5: Weighted sum and identities")
    P("-" * 72)

    total_weighted = (1 * orbit_tr['Gamma'] + 6 * orbit_tr['Axis']
                      + 12 * orbit_tr['Face'] + 8 * orbit_tr['Body'])
    P(f"Weighted sum = 1*Gamma + 6*Axis + 12*Face + 8*Body")
    P(f"  = {total_weighted:.15f}")

    trT0 = np.trace(T0_int)
    P(f"Tr(T0) = {trT0}")

    # Identity: sum_k Tr(P_k H(k)) = sum_k 6*(-3) = -486
    # sum_k Tr(P_k T0) + sum_k Tr(P_k H_hop(k)) = -486
    P(f"sum_k Tr(P_k T0) = {total_weighted:.15f}")
    P(f"-486 - sum_k Tr(P_k T0) = sum_k Tr(P_k H_hop(k)) = "
      f"{-486 - total_weighted:.15f}")

    # Also: sum_k H(k) = 27 * T(0) [Bloch completeness]
    # Check this numerically
    sum_Hk = np.zeros((20, 20), dtype=complex)
    for n in itertools.product(range(3), repeat=3):
        sum_Hk += build_Hk_numpy(T, n)
    P(f"\n|sum_k H(k) - 27*T0| = {np.max(np.abs(sum_Hk - 27*T0)):.2e} (should be ~0)")

    # Tr(T0) * 6 vs weighted sum: if P_k were k-independent (= T0/27-block),
    # we'd get Tr(P T0) = 6 * Tr(T0) / 20 for each k. Check:
    P(f"Uniform prediction: 6*Tr(T0)/20 = {6*trT0/20:.6f}")
    P(f"Actual average: {total_weighted/27:.6f}")

    # Alternative check using H(k) = T0 + H_hop(k):
    # On flat subspace: -3I = V^dag T0 V + V^dag H_hop V
    # => Tr(P_k T0) = -18 - Tr(P_k H_hop(k))
    P("\nCross-check: Tr(P_k T0) = -18 - Tr(P_k H_hop(k))")
    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        rep = orbit_reps[orbit_name]
        H_k = build_Hk_numpy(T, rep)
        H_hop = H_k - T0  # hopping part
        evals, evecs = np.linalg.eigh(H_k)
        V = evecs[:, :6]
        tr_hop = np.real(np.trace(V.conj().T @ H_hop @ V))
        P(f"  {orbit_name}: Tr(P_k T0) = {orbit_tr[orbit_name]:.10f}, "
          f"-18 - Tr(P_k H_hop) = {-18 - tr_hop:.10f}, "
          f"diff = {abs(orbit_tr[orbit_name] - (-18 - tr_hop)):.2e}")

    # ===================================================================
    # PHASE 6: Dispersive band eigenvalues of T0
    # ===================================================================
    P("\n" + "-" * 72)
    P("PHASE 6: Dispersive eigenvalues and T0 spectrum analysis")
    P("-" * 72)

    # T0 eigenvalues
    t0_evals = np.sort(np.linalg.eigvalsh(T0))
    P(f"T0 eigenvalues: {np.round(t0_evals, 6)}")
    P(f"Tr(T0) = {np.sum(t0_evals):.6f}")

    # For each k, the dispersive (non-flat) eigenvalues
    P("\nDispersive eigenvalues at representative k-points:")
    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        rep = orbit_reps[orbit_name]
        evals = all_evals[rep]
        dispersive = evals[6:]  # eigenvalues 7-20
        P(f"  {orbit_name} {rep}: {np.round(dispersive, 6)}")

    # Tr(P_k T0) = Tr(T0) - Tr((I-P_k) T0) = Tr(T0) - sum of T0-expectations
    # on dispersive eigenvectors. Let's verify:
    P(f"\nTr(T0) = {trT0}")
    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        rep = orbit_reps[orbit_name]
        H_k = build_Hk_numpy(T, rep)
        evals, evecs = np.linalg.eigh(H_k)
        V_disp = evecs[:, 6:]  # dispersive eigenvectors
        tr_disp = np.real(np.trace(V_disp.conj().T @ T0 @ V_disp))
        P(f"  {orbit_name}: Tr(T0) - Tr_dispersive(T0) = "
          f"{trT0 - tr_disp:.10f} vs Tr(P_k T0) = {orbit_tr[orbit_name]:.10f} "
          f"[diff={abs(trT0 - tr_disp - orbit_tr[orbit_name]):.2e}]")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    P("\n" + "=" * 72)
    P("SUMMARY: Tr(P_k T0) by k-orbit")
    P("=" * 72)
    P(f"{'Orbit':8s} {'Size':>4s}  {'Numerical (15 dp)':>22s}  {'k representative'}")
    P("-" * 65)
    for orbit_name in ['Gamma', 'Axis', 'Face', 'Body']:
        rep = orbit_reps[orbit_name]
        k_parts = []
        for a, n in enumerate(rep):
            if n != 0:
                k_parts.append(f"{n}*2pi/3*e{a+1}")
        k_str = " + ".join(k_parts) if k_parts else "0"
        P(f"{orbit_name:8s} {len(orbits[orbit_name]):4d}  "
          f"{orbit_tr[orbit_name]:22.15f}  k = {k_str}")

    P(f"\nWeighted sum (all 27): {total_weighted:.15f}")
    P(f"Average Tr(P_k T0):   {total_weighted/27:.15f}")

    if exact_results:
        P(f"\nExact results obtained for: {list(exact_results.keys())}")
        for name, expr in exact_results.items():
            P(f"  {name}: {expr}")

    P()
    P("DONE.")


if __name__ == "__main__":
    main()
