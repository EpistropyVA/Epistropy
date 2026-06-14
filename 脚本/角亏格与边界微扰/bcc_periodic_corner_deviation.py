"""
bcc_periodic_corner_deviation.py

Investigates the "corner deviation" analog for the BCC 3x3x3 periodic-BC lattice.

Because periodic BC has no boundary, there is no corner/edge/face/bulk distinction.
Instead we compute:

  A) Per-face projection weight of lambda=-3 flat-band eigenstates at each k-point.
     Face types within a single BC (using reference BC at origin):
       - Pure-corner faces: the 4 faces of the all-corner tetrahedron (no BC vertex)
       - BC-containing faces: the remaining 16 faces (contain the BC vertex)

  B) Average over all 27 k-points: what fraction lands on pure-corner vs BC-containing?

  C) Check rationality at Gamma and across k-points. Multiply by small integers
     to identify exact fractions.

  D) Check whether transcendental structure appears (pi, ln2, sqrt(3), etc.)

Reference: bcc_bloch_decomposition.py for lattice construction.
"""

import sys
import io
import itertools
from collections import defaultdict

import numpy as np

OUT_PATH = "d:/AI thoery/.agent/scripts/bcc_periodic_deviation_results.txt"

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_LOG_LINES = []

def log(msg=""):
    print(msg)
    _LOG_LINES.append(str(msg))


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


def build_adjacency_matrix(face_to_idx, body_centers, bc_face_indices):
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
# Face classification
# ============================================================

def classify_ref_faces(ref_faces):
    """
    Classify the 20 reference faces (of BC at origin) into:
      - pure_corner: faces that do NOT contain the BC vertex ('bc',0,0,0)
        These come from the all-corner tetrahedra (faces among the 4 corners only).
        But wait: each simplex is {bc} + 4 corners. C(5,3)=10 faces.
          - 1 face uses 3 corners (no bc): C(4,3)=4 per simplex -> 8 total pure-corner
          - 6 faces use bc + 2 corners: C(4,2)=6 per simplex -> 12 total bc-containing

    Returns:
        pure_corner_indices: list of local face indices (0-19) that are pure-corner
        bc_containing_indices: list of local face indices (0-19) with BC vertex
    """
    bc_v = ('bc', 0, 0, 0)
    pure_corner = []
    bc_containing = []
    for m, face in enumerate(ref_faces):
        if bc_v in face:
            bc_containing.append(m)
        else:
            pure_corner.append(m)
    return pure_corner, bc_containing


# ============================================================
# Rational approximation helper
# ============================================================

def find_rational(x, max_denom=200, tol=1e-7):
    """Try to find p/q with q<=max_denom such that |x - p/q| < tol."""
    from fractions import Fraction
    f = Fraction(x).limit_denominator(max_denom)
    if abs(float(f) - x) < tol:
        return f
    return None


def check_algebraic_simple(x, tol=1e-8):
    """
    Check if x is a simple algebraic number:
    rational, or involves sqrt(2), sqrt(3), sqrt(5), pi, ln(2).
    Returns a string description or None.
    """
    import math
    candidates = {
        "rational": None,  # handled separately
        "sqrt(2)": math.sqrt(2),
        "sqrt(3)": math.sqrt(3),
        "sqrt(5)": math.sqrt(5),
        "sqrt(6)": math.sqrt(6),
        "pi": math.pi,
        "pi^2": math.pi**2,
        "ln(2)": math.log(2),
        "1/ln(2)": 1.0/math.log(2),
        "1/pi": 1.0/math.pi,
        "1/pi^2": 1.0/(math.pi**2),
    }
    # Check rational first
    r = find_rational(x, max_denom=500)
    if r is not None:
        return f"rational: {r}"

    for name, base in candidates.items():
        if base is None:
            continue
        # Check x/base and x*base for rationals
        for val, form in [(x / base, f"x/{name}"), (x * base, f"x*{name}")]:
            r2 = find_rational(val, max_denom=500)
            if r2 is not None:
                if form.startswith("x/"):
                    return f"{r2} * {name}"
                else:
                    return f"{r2} / {name}"
    return None


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 78)
    log("BCC 3x3x3 Periodic-BC: Corner Deviation Analysis via Bloch Decomposition")
    log("=" * 78)

    # --------------------------------------------------------
    # Step 1: Build lattice and faces
    # --------------------------------------------------------
    log("\n[Step 1] Build lattice and face labeling")
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    N = len(face_to_idx)
    log(f"  Total faces: {N} (expect 540)")
    assert N == 540

    # Classify reference faces
    pure_corner_idx, bc_containing_idx = classify_ref_faces(ref_faces)
    log(f"  Reference BC face classification:")
    log(f"    Pure-corner faces (no BC vertex): {len(pure_corner_idx)} (indices: {pure_corner_idx})")
    log(f"    BC-containing faces: {len(bc_containing_idx)} (indices: {bc_containing_idx[:6]}...)")
    assert len(pure_corner_idx) == 8, f"Expected 8 pure-corner faces, got {len(pure_corner_idx)}"
    assert len(bc_containing_idx) == 12, f"Expected 12 BC-containing faces, got {len(bc_containing_idx)}"

    # --------------------------------------------------------
    # Step 2: Build 540x540 adjacency matrix
    # --------------------------------------------------------
    log("\n[Step 2] Build 540x540 adjacency matrix and verify spectrum")
    A = build_adjacency_matrix(face_to_idx, body_centers, bc_face_indices)
    log(f"  Shape: {A.shape}, symmetric: {np.allclose(A, A.T)}")

    # Direct spectrum check
    eigvals_full = np.sort(np.linalg.eigvalsh(A))
    grouped = group_eigenvalues(eigvals_full)
    log(f"  Full spectrum:")
    for rep, deg, cum in grouped:
        log(f"    lambda={rep:10.6f}  deg={deg:4d}  cumul={cum:4d}")

    n_flat_full = sum(1 for e in eigvals_full if abs(e - (-3.0)) < 1e-4)
    log(f"  lambda=-3 count in full spectrum: {n_flat_full}")

    # --------------------------------------------------------
    # Step 3: Build hopping matrices T(R) and H(k)
    # --------------------------------------------------------
    log("\n[Step 3] Build hopping matrices T(R) via BC-block reordering")
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

    # Verify T matrices are integer
    max_frac = max(np.max(np.abs(mat - np.round(mat))) for mat in T.values())
    log(f"  Max fractional part of T(R) entries: {max_frac:.3e} (expect ~0 for integer)")
    T_int = {R: np.round(mat).astype(int) for R, mat in T.items()}

    nonzero_T = {R: mat for R, mat in T.items() if np.linalg.norm(mat, 'fro') > 1e-10}
    log(f"  Nonzero T(R) matrices: {len(nonzero_T)}")

    # --------------------------------------------------------
    # Step 4: Build H(k) for all 27 k-points
    # --------------------------------------------------------
    log("\n[Step 4] Build H(k) = sum_R exp(i k.R) T(R) for all 27 k-points")
    kvecs = list(itertools.product(range(3), repeat=3))

    def build_H_k(n_tuple):
        k = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H_k = np.zeros((20, 20), dtype=complex)
        for R, mat in nonzero_T.items():
            R_vec = np.array(R, dtype=float)
            phase = np.exp(1j * np.dot(k, R_vec))
            H_k += phase * mat
        return H_k

    H_all = {n: build_H_k(n) for n in kvecs}

    # Verify Bloch spectrum matches
    all_bloch = np.sort(np.concatenate([np.linalg.eigvalsh(H_all[n]) for n in kvecs]))
    max_diff = float(np.max(np.abs(all_bloch - eigvals_full)))
    log(f"  Max |lambda_Bloch - lambda_direct| = {max_diff:.3e}")
    assert max_diff < 1e-6, f"Bloch decomposition failed: {max_diff}"

    def bz_label(n_tuple):
        nonzero = sum(1 for n in n_tuple if n != 0)
        return ["Gamma", "BZ-edge(1d)", "BZ-face(2d)", "BZ-corner(3d)"][nonzero]

    # --------------------------------------------------------
    # Step 5: At each k-point, find lambda=-3 eigenvectors
    # --------------------------------------------------------
    log("\n[Step 5] Find lambda=-3 eigenvectors at each k-point")
    log(f"  Tolerance for eigenvalue = -3: 1e-4")

    tol_ev = 1e-4
    lam3_per_k = {}
    for n_tuple in kvecs:
        H_k = H_all[n_tuple]
        evals, evecs = np.linalg.eigh(H_k)
        hits = [(i, ev) for i, ev in enumerate(evals) if abs(ev - (-3.0)) < tol_ev]
        if hits:
            vecs = np.array([evecs[:, i] for i, _ in hits])  # shape (m, 20)
            lam3_per_k[n_tuple] = vecs
        else:
            lam3_per_k[n_tuple] = np.zeros((0, 20), dtype=complex)

    for n_tuple in sorted(kvecs):
        m = lam3_per_k[n_tuple].shape[0]
        if m > 0:
            log(f"  k=2pi/3*{n_tuple}  ({bz_label(n_tuple)}): {m} lambda=-3 state(s)")

    total_lam3 = sum(v.shape[0] for v in lam3_per_k.values())
    log(f"  Total lambda=-3 eigenstates across Bloch: {total_lam3} (expect {n_flat_full})")

    # --------------------------------------------------------
    # Step 6: Project onto face types at each k-point
    # --------------------------------------------------------
    log("\n[Step 6] Projection weights of lambda=-3 states onto face types")
    log(f"  Face indices within each BC (local, 0-19):")
    log(f"    Pure-corner (no BC vertex): {pure_corner_idx}  [8 faces]")
    log(f"    BC-containing: {bc_containing_idx}  [12 faces]")
    log()

    pure_set = set(pure_corner_idx)
    bc_set = set(bc_containing_idx)

    # For each k-point with lambda=-3 states, compute per-face-type projection
    # P_m = |<face_m | psi>|^2, then sum over face type
    k_results = {}
    for n_tuple in sorted(kvecs):
        vecs = lam3_per_k[n_tuple]  # shape (m, 20)
        if vecs.shape[0] == 0:
            continue

        # For each eigenvector, compute |psi_m|^2 for each of 20 face indices
        # Then sum within face type
        # vecs[i, m] = amplitude of face m in eigenvector i
        # Weight on face m (summed over degenerate eigenvectors):
        # w[m] = sum_i |vecs[i, m]|^2

        w = np.sum(np.abs(vecs)**2, axis=0)  # shape (20,)
        total_w = np.sum(w)

        w_pure = np.sum(w[pure_corner_idx])
        w_bc = np.sum(w[bc_containing_idx])

        # Normalized fractions
        f_pure = w_pure / total_w
        f_bc = w_bc / total_w

        k_results[n_tuple] = {
            'n_states': vecs.shape[0],
            'w': w,
            'w_pure': w_pure,
            'w_bc': w_bc,
            'f_pure': f_pure,
            'f_bc': f_bc,
            'total_w': total_w,
        }

        label = bz_label(n_tuple)
        log(f"  k=2pi/3*{n_tuple}  ({label})  m={vecs.shape[0]}:")
        log(f"    w_pure={w_pure:.8f}  w_bc={w_bc:.8f}  total={total_w:.8f}")
        log(f"    f_pure={f_pure:.8f}  f_bc={f_bc:.8f}  (sum={f_pure+f_bc:.8f})")

        # Per-face weights
        w_str = "  ".join(f"[{m}]:{w[m]:.4f}" for m in range(20))
        log(f"    per-face: {w_str[:100]}...")

    # --------------------------------------------------------
    # Step 7: Average projection weights over k-points
    # --------------------------------------------------------
    log("\n[Step 7] Average projection weights over all k-points with lambda=-3")

    all_f_pure = [r['f_pure'] for r in k_results.values()]
    all_f_bc   = [r['f_bc']   for r in k_results.values()]
    all_w_pure = [r['w_pure'] for r in k_results.values()]
    all_w_bc   = [r['w_bc']   for r in k_results.values()]
    all_n      = [r['n_states'] for r in k_results.values()]

    avg_f_pure = float(np.mean(all_f_pure))
    avg_f_bc   = float(np.mean(all_f_bc))

    # Weighted average (by number of states at each k)
    total_n = sum(all_n)
    wavg_f_pure = sum(r['n_states'] * r['f_pure'] for r in k_results.values()) / total_n
    wavg_f_bc   = sum(r['n_states'] * r['f_bc']   for r in k_results.values()) / total_n

    log(f"  Number of k-points with lambda=-3: {len(k_results)}")
    log(f"  Total lambda=-3 states: {total_n}")
    log()
    log(f"  Unweighted averages across k-points:")
    log(f"    <f_pure> = {avg_f_pure:.10f}")
    log(f"    <f_bc>   = {avg_f_bc:.10f}")
    log(f"    sum      = {avg_f_pure + avg_f_bc:.10f}")
    log()
    log(f"  State-count-weighted averages:")
    log(f"    <f_pure>_w = {wavg_f_pure:.10f}")
    log(f"    <f_bc>_w   = {wavg_f_bc:.10f}")
    log(f"    sum        = {wavg_f_pure + wavg_f_bc:.10f}")

    # --------------------------------------------------------
    # Step 8: Rationality check
    # --------------------------------------------------------
    log("\n[Step 8] Rationality and algebraic structure check")

    def check_and_report(label, x):
        r = find_rational(x, max_denom=1000, tol=1e-7)
        alg = check_algebraic_simple(x, tol=1e-7)
        log(f"  {label} = {x:.12f}")
        log(f"    rational approx: {r}")
        log(f"    algebraic check: {alg}")

        # Multiply by small integers to find integer multiples
        for mult in [2, 3, 4, 6, 8, 9, 10, 12, 16, 18, 20, 24, 27, 36, 48, 54, 72]:
            val = x * mult
            r2 = find_rational(val, max_denom=100, tol=1e-6)
            if r2 is not None and r2.denominator == 1:
                log(f"    {mult} * x = {float(r2)} (integer!)")
        log()

    log("\n  --- Per-k-point f_pure values ---")
    for n_tuple in sorted(k_results.keys()):
        r = k_results[n_tuple]
        check_and_report(f"f_pure at k={n_tuple} ({bz_label(n_tuple)})", r['f_pure'])

    log("\n  --- Averages ---")
    check_and_report("avg f_pure (unweighted)", avg_f_pure)
    check_and_report("avg f_bc (unweighted)", avg_f_bc)
    check_and_report("wavg f_pure (weighted)", wavg_f_pure)
    check_and_report("wavg f_bc (weighted)", wavg_f_bc)

    # --------------------------------------------------------
    # Step 9: Per-face weight detail at each k
    # --------------------------------------------------------
    log("\n[Step 9] Per-face weight detail (all 20 faces) at each k with lambda=-3")
    log(f"  Pure-corner face indices: {pure_corner_idx}")
    log(f"  BC-containing face indices: {bc_containing_idx}")
    log()

    for n_tuple in sorted(k_results.keys()):
        r = k_results[n_tuple]
        w = r['w']
        label = bz_label(n_tuple)
        n_states = r['n_states']
        log(f"  k=2pi/3*{n_tuple} ({label}), {n_states} state(s), total_w={r['total_w']:.6f}:")
        log(f"  {'face_m':>7} | {'type':>14} | {'weight':>14} | {'frac':>10} | {'rational?':>20}")
        log(f"  {'-'*7}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}-+-{'-'*20}")
        for m in range(20):
            ftype = "pure-corner" if m in pure_set else "BC-contain"
            wm = float(w[m])
            fm = wm / r['total_w']
            r_approx = find_rational(fm, max_denom=200, tol=1e-6)
            r_str = str(r_approx) if r_approx is not None else "?"
            log(f"  {m:7d} | {ftype:>14} | {wm:14.8f} | {fm:10.6f} | {r_str:>20}")
        log()

    # --------------------------------------------------------
    # Step 10: Gamma-point detailed analysis
    # --------------------------------------------------------
    log("\n[Step 10] Gamma-point (0,0,0) H(0) detailed analysis")
    H_gamma = H_all[(0, 0, 0)]
    H_gamma_int = T_int[(0, 0, 0)].copy()
    for R, mat in T_int.items():
        H_gamma_int = H_gamma_int + mat  # actually should rebuild from T
    # Correct: H(0) = sum_R T(R)
    H_gamma_exact = sum(mat for mat in T_int.values())
    log(f"  H(0) = sum_R T(R) is integer matrix: {np.allclose(H_gamma_exact, np.real(H_gamma), atol=1e-6)}")
    evals_g, evecs_g = np.linalg.eigh(H_gamma_exact.astype(float))
    log(f"  H(0) eigenvalues (sorted):")
    grouped_g = group_eigenvalues(np.sort(evals_g))
    for rep, deg, cum in grouped_g:
        log(f"    lambda={rep:10.4f}  deg={deg}")

    hits_g = [(i, ev) for i, ev in enumerate(evals_g) if abs(ev - (-3.0)) < 1e-4]
    log(f"  lambda=-3 at Gamma: {len(hits_g)} state(s)")
    if hits_g:
        for i, ev in hits_g:
            v = evecs_g[:, i]
            w_pure_g = float(np.sum(np.abs(v[pure_corner_idx])**2))
            w_bc_g   = float(np.sum(np.abs(v[bc_containing_idx])**2))
            log(f"    Eigvec {i}: w_pure={w_pure_g:.8f} w_bc={w_bc_g:.8f}")
            r_p = find_rational(w_pure_g, max_denom=200)
            r_b = find_rational(w_bc_g, max_denom=200)
            log(f"    Rational: w_pure ~ {r_p}, w_bc ~ {r_b}")

    # --------------------------------------------------------
    # Step 11: Deviation from uniform distribution
    # --------------------------------------------------------
    log("\n[Step 11] Deviation from uniform face distribution")
    log(f"  Uniform expectation:")
    log(f"    If lambda=-3 states were uniform over all 20 faces:")
    log(f"    f_pure_uniform = 8/20 = {8/20:.6f}")
    log(f"    f_bc_uniform   = 12/20 = {12/20:.6f}")
    log()

    uniform_pure = 8.0/20.0
    uniform_bc   = 12.0/20.0

    for n_tuple in sorted(k_results.keys()):
        r = k_results[n_tuple]
        dev_pure = r['f_pure'] - uniform_pure
        dev_bc   = r['f_bc']   - uniform_bc
        label = bz_label(n_tuple)
        log(f"  k=2pi/3*{n_tuple} ({label}): dev_pure={dev_pure:+.8f} dev_bc={dev_bc:+.8f}")

    log()
    log(f"  Average deviation from uniform:")
    avg_dev_pure = float(np.mean([k_results[n]['f_pure'] - uniform_pure
                                   for n in k_results]))
    avg_dev_bc   = float(np.mean([k_results[n]['f_bc'] - uniform_bc
                                   for n in k_results]))
    log(f"    <dev_pure> = {avg_dev_pure:.10f}")
    log(f"    <dev_bc>   = {avg_dev_bc:.10f}")
    check_and_report("avg_dev_pure", avg_dev_pure)
    check_and_report("avg_dev_bc", avg_dev_bc)

    # --------------------------------------------------------
    # Step 12: Check if flat-band subspace weights are k-independent
    # --------------------------------------------------------
    log("\n[Step 12] Are per-face weights k-independent (flat band property)?")

    # Collect f_pure at each BZ type
    by_type = defaultdict(list)
    for n_tuple, r in k_results.items():
        t = bz_label(n_tuple)
        by_type[t].append(r['f_pure'])

    for t, vals in sorted(by_type.items()):
        arr = np.array(vals)
        log(f"  {t}: f_pure values: min={arr.min():.8f} max={arr.max():.8f} "
            f"spread={arr.max()-arr.min():.3e} n={len(arr)}")

    all_f_pure_arr = np.array([r['f_pure'] for r in k_results.values()])
    spread = float(all_f_pure_arr.max() - all_f_pure_arr.min())
    log(f"\n  Overall f_pure spread across all k-points: {spread:.3e}")
    if spread < 1e-8:
        log(f"  -> f_pure is CONSTANT across k-points (flat band property confirmed)")
        log(f"  -> EXACT VALUE: f_pure = {all_f_pure_arr[0]:.12f}")
        check_and_report("f_pure (constant across all k)", float(all_f_pure_arr[0]))
    else:
        log(f"  -> f_pure varies across k-points (not k-independent)")

    # --------------------------------------------------------
    # Step 13: Summary
    # --------------------------------------------------------
    log("\n" + "=" * 78)
    log("[SUMMARY: Corner Deviation Analysis for Periodic BCC 3x3x3]")
    log("=" * 78)

    log(f"\n1. Flat band: lambda=-3 has {total_lam3} states total across {len(k_results)} k-points")
    log(f"   Degeneracy per k-point: {set(all_n)}")

    log(f"\n2. Face classification:")
    log(f"   Pure-corner faces (both simplices, no BC vertex): 8 per BC")
    log(f"   BC-containing faces: 12 per BC")
    log(f"   Uniform expectation: f_pure=8/20=0.4, f_bc=12/20=0.6")

    log(f"\n3. Projection weight on pure-corner faces:")
    log(f"   Weighted average: {wavg_f_pure:.10f}")
    log(f"   Uniform would be: {uniform_pure:.10f}")
    log(f"   Deviation: {wavg_f_pure - uniform_pure:+.10f}")

    if spread < 1e-8:
        fval = float(all_f_pure_arr[0])
        r = find_rational(fval, max_denom=1000, tol=1e-7)
        log(f"\n4. EXACT RATIONAL RESULT:")
        log(f"   f_pure = {r} = {fval:.12f}  (constant across ALL k-points)")
        log(f"   f_bc   = {1 - fval:.12f} = {find_rational(1-fval, max_denom=1000)}")
        dev = fval - uniform_pure
        dev_r = find_rational(dev, max_denom=1000, tol=1e-7)
        log(f"   Corner deviation analog = f_pure - 8/20 = {dev_r} = {dev:.12f}")
    else:
        log(f"\n4. f_pure is not constant across k-points (spread={spread:.3e})")
        log(f"   Average: {wavg_f_pure:.10f}")
        alg = check_algebraic_simple(wavg_f_pure)
        log(f"   Algebraic structure: {alg}")

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
            print(f"WARNING: failed to write results: {e}", file=sys.stderr)
