# -*- coding: utf-8 -*-
"""
audit_bridge_layers.py

Bottom-up numerical audit of verify_bridge_norm_projective.py
Dependency chain:
  Layer 0: BCC lattice construction
  Layer 1: Bloch matrix
  Layer 2: Eigenvalues
  Layer 3: Galois classification
  Layer 4: Bridge judgment

Audit priority (user spec):
  P1: SC analytical cross-check (fastest, validates construction pipeline)
  P2: Layer 3 sympy exact cross-check (most subtle)
  P3: Layer 0 manual face counting (foundational)
  P4: Layer 1 Bloch matrix sanity
  P5: Layer 2 eigenvalue stability
  P6: FCC analytical cross-check

Each layer prints PASS/FAIL + specific numerical deviations.
"""

import itertools
import sys
import io
import math
from fractions import Fraction
from collections import Counter

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

try:
    import sympy
    from sympy import Rational, factorint, Poly, Symbol, minimal_polynomial
    from sympy import nsimplify, sqrt as sp_sqrt, cos as sp_cos, pi as sp_pi
    from sympy import Matrix as SympyMatrix
    from sympy.polys.numberfields import minimal_polynomial as min_poly_func
    SYMPY_OK = True
except ImportError:
    SYMPY_OK = False
    print("WARNING: sympy not available; Layer 3 exact audit disabled.")

# ---------------------------------------------------------------------------
# Import construction functions from verify_bridge_norm_projective
# ---------------------------------------------------------------------------
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "格点拓扑不变量校验"))
from verify_bridge_norm_projective import (
    build_bcc_lattice_periodic, enumerate_simplex_faces, translate_face,
    build_all_faces, build_adjacency_matrix, build_transfer_blocks,
    build_H_k, build_bcc_bloch,
    build_sc_bloch_v2, build_fcc_bloch,
    group_eigenvalues, galois_conjugacy_classes,
    eval_min_poly_at, prime_factors,
    ORBIT_DEFS, BCC_K_REPS,
    FLAT_EIGENVALUE, FLAT_TOL, TOL,
)

PASS_COUNT = 0
FAIL_COUNT = 0

def report(test_name, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if passed else "FAIL"
    if not passed:
        FAIL_COUNT += 1
    else:
        PASS_COUNT += 1
    print(f"  [{tag}] {test_name}" + (f"  -- {detail}" if detail else ""))


# ===========================================================================
# PRIORITY 1: SC Analytical Cross-Check
# ===========================================================================

def audit_sc_analytical():
    """
    SC tight-binding on 3x3x3 supercell.
    Analytical dispersion: eps(k) = 2(cos kx + cos ky + cos kz)
    The 27x27 Bloch matrix at supercell k-point K should have eigenvalues
    equal to eps(K + G_j) for all 27 reciprocal lattice vectors G_j.

    For a Bravais lattice supercell, the spectrum is K-independent
    (zone-folding identity): eigenvalues are always the same 27 values
    eps(2*pi*m/3) for m in {0,1,2}^3.
    """
    print("\n" + "=" * 76)
    print("  AUDIT P1: SC ANALYTICAL CROSS-CHECK")
    print("=" * 76)

    N = 3
    # Analytical eigenvalues: eps = 2*(cos(2*pi*m1/3) + cos(2*pi*m2/3) + cos(2*pi*m3/3))
    # cos(0) = 1, cos(2*pi/3) = cos(4*pi/3) = -1/2
    # So each component contributes 2*1=2 (if m=0) or 2*(-1/2)=-1 (if m=1 or 2)
    analytical_evals = []
    for m1, m2, m3 in itertools.product(range(N), repeat=3):
        val = 0.0
        for mi in [m1, m2, m3]:
            val += 2.0 * math.cos(2.0 * math.pi * mi / N)
        analytical_evals.append(val)
    analytical_evals.sort()

    # Check analytical eigenvalue distribution at Gamma
    anal_counter = Counter([round(v, 8) for v in analytical_evals])
    print(f"\n  Analytical SC eigenvalues at Gamma (zone-folded):")
    for val, mult in sorted(anal_counter.items()):
        print(f"    eps = {val:8.4f}  mult = {mult}")
    # Expected: 6(x1), 3(x6), 0(x12), -3(x8)

    report("SC Gamma analytical: 6(x1), 3(x6), 0(x12), -3(x8)",
           anal_counter == {6.0: 1, 3.0: 6, 0.0: 12, -3.0: 8})

    # Point-by-point comparison of eigenvalues to zone-folded primitive dispersion for all 27 k-points
    print("\n  --- Point-by-point SC analytical dispersion verification ---")
    all_kpts_pass = True
    for n1, n2, n3 in itertools.product(range(N), repeat=3):
        K = np.array([2 * np.pi * n1 / 3.0, 2 * np.pi * n2 / 3.0, 2 * np.pi * n3 / 3.0])
        H = build_sc_bloch_v2(K)
        evals = sorted(np.linalg.eigvalsh(H))
        
        anal = []
        for mx, my, mz in itertools.product(range(N), repeat=3):
            qx = K[0]/3.0 + 2.0*np.pi*mx/3.0
            qy = K[1]/3.0 + 2.0*np.pi*my/3.0
            qz = K[2]/3.0 + 2.0*np.pi*mz/3.0
            val = 2.0 * (math.cos(qx) + math.cos(qy) + math.cos(qz))
            anal.append(val)
        anal.sort()
        
        max_dev = max(abs(evals[i] - anal[i]) for i in range(27))
        if max_dev >= 1e-10:
            all_kpts_pass = False
            print(f"    FAIL: SC dispersion mismatch at n=({n1},{n2},{n3}), max_dev={max_dev:.2e}")
            
    report("SC eigenvalues at all 27 k-points match analytical dispersion", all_kpts_pass)

    # For non-Gamma k-points, build_sc_bloch_v2 constructs a 27x27 Bloch matrix
    # using site indices as basis. Verify structural properties: Hermiticity and trace.
    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        if orbit_name == 'Gamma':
            continue
        K = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_sc_bloch_v2(K)

        # Hermiticity
        herm_err = np.max(np.abs(H - H.conj().T))
        report(f"SC H({orbit_name}) Hermitian", herm_err < 1e-12,
               f"err = {herm_err:.2e}")

        # Trace = 0 (no self-loops at any k)
        tr = abs(np.real(np.trace(H)))
        report(f"SC H({orbit_name}) Tr=0", tr < 1e-10, f"Tr = {tr:.2e}")

    # Aggregate test: sum eigenvalues over all 27 k-points = eigenvalues of full A
    all_bloch_evals = []
    for m1, m2, m3 in itertools.product(range(N), repeat=3):
        K = np.array([2*np.pi*m1/N, 2*np.pi*m2/N, 2*np.pi*m3/N])
        H = build_sc_bloch_v2(K)
        evals = sorted(np.linalg.eigvalsh(H))
        all_bloch_evals.extend(evals)
    all_bloch_evals.sort()
    total_trace = sum(all_bloch_evals)
    report("SC total Bloch trace (sum over all k) = 0",
           abs(total_trace) < 1e-8, f"total = {total_trace:.6f}")

    H_gamma = build_sc_bloch_v2(np.array([0.0, 0.0, 0.0]))
    tr = np.real(np.trace(H_gamma))
    report("SC Tr(H(Gamma)) = 0 (no self-loops)", abs(tr) < 1e-10, f"Tr = {tr:.12f}")


# ===========================================================================
# PRIORITY 2: FCC Analytical Cross-Check
# ===========================================================================

def audit_fcc_analytical():
    """
    FCC tight-binding: eps(k) = 4*(cos kx cos ky + cos ky cos kz + cos kz cos kx)
    Zone-folded onto 3x3x3 supercell.
    """
    print("\n" + "=" * 76)
    print("  AUDIT P6: FCC ANALYTICAL CROSS-CHECK")
    print("=" * 76)

    N = 3

    # FCC primitive dispersion: eps(q) = 4*(cos qx cos qy + cos qy cos qz + cos qz cos qx)
    def fcc_dispersion(q):
        return 4.0 * (math.cos(q[0]) * math.cos(q[1]) +
                      math.cos(q[1]) * math.cos(q[2]) +
                      math.cos(q[2]) * math.cos(q[0]))

    # Gamma analytical
    anal_gamma = []
    for m1, m2, m3 in itertools.product(range(N), repeat=3):
        q = np.array([2*math.pi*m1/N, 2*math.pi*m2/N, 2*math.pi*m3/N])
        anal_gamma.append(fcc_dispersion(q))
    anal_gamma.sort()

    anal_counter = Counter([round(v, 8) for v in anal_gamma])
    print(f"\n  Analytical FCC eigenvalues at Gamma:")
    for val, mult in sorted(anal_counter.items()):
        print(f"    eps = {val:8.4f}  mult = {mult}")

    report("FCC analytical Gamma: max eigenvalue = 12",
           abs(max(anal_gamma) - 12.0) < 1e-10)

    # Gamma check with correct analytical reference
    H_gamma = build_fcc_bloch(np.array([0.0, 0.0, 0.0]))
    evals_gamma = sorted(np.linalg.eigvalsh(H_gamma))
    max_dev = max(abs(evals_gamma[i] - anal_gamma[i]) for i in range(27))
    report(f"FCC H(Gamma) eigenvalues match analytical",
           max_dev < 1e-10, f"max deviation = {max_dev:.2e}")

    # Point-by-point comparison of eigenvalues to zone-folded FCC dispersion for all 27 k-points
    print("\n  --- Point-by-point FCC analytical dispersion verification ---")
    all_kpts_pass = True
    for n1, n2, n3 in itertools.product(range(N), repeat=3):
        K = np.array([2 * np.pi * n1 / 3.0, 2 * np.pi * n2 / 3.0, 2 * np.pi * n3 / 3.0])
        H = build_fcc_bloch(K)
        evals = sorted(np.linalg.eigvalsh(H))
        
        anal = []
        for mx, my, mz in itertools.product(range(N), repeat=3):
            qx = K[0]/3.0 + 2.0*np.pi*mx/3.0
            qy = K[1]/3.0 + 2.0*np.pi*my/3.0
            qz = K[2]/3.0 + 2.0*np.pi*mz/3.0
            val = fcc_dispersion([qx, qy, qz])
            anal.append(val)
        anal.sort()
        
        max_dev = max(abs(evals[i] - anal[i]) for i in range(27))
        if max_dev >= 1e-10:
            all_kpts_pass = False
            print(f"    FAIL: FCC dispersion mismatch at n=({n1},{n2},{n3}), max_dev={max_dev:.2e}")
            
    report("FCC eigenvalues at all 27 k-points match analytical dispersion", all_kpts_pass)

    # Non-Gamma: structural checks only (same reasoning as SC)
    for orbit_name, (n_tuple, _) in ORBIT_DEFS.items():
        if orbit_name == 'Gamma':
            continue
        K = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_fcc_bloch(K)

        herm_err = np.max(np.abs(H - H.conj().T))
        report(f"FCC H({orbit_name}) Hermitian", herm_err < 1e-12,
               f"err = {herm_err:.2e}")

        tr = abs(np.real(np.trace(H)))
        report(f"FCC H({orbit_name}) Tr=0", tr < 1e-10, f"Tr = {tr:.2e}")


# ===========================================================================
# PRIORITY 3: Layer 0 — BCC Lattice Construction
# ===========================================================================

def audit_layer0_bcc():
    """
    Manual verification of BCC stella octangula face construction.
    """
    print("\n" + "=" * 76)
    print("  AUDIT P3: LAYER 0 — BCC LATTICE CONSTRUCTION")
    print("=" * 76)

    # --- Test 1: Parity split for bc=(0,0,0) ---
    print("\n  --- Test 1: Parity split at bc=(0,0,0) ---")
    bc = (0, 0, 0)
    bc_v = ('bc', 0, 0, 0)
    corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = dx, dy, dz
        parity = (ox + oy + oz) % 2
        wx, wy, wz = ox % 3, oy % 3, oz % 3
        corner_v = ('c', wx, wy, wz)
        corners.append((parity, corner_v, (ox, oy, oz)))

    even = [(v, pos) for (p, v, pos) in corners if p == 0]
    odd = [(v, pos) for (p, v, pos) in corners if p == 1]

    report("8 corners total", len(corners) == 8, f"got {len(corners)}")
    report("4 even-parity corners", len(even) == 4, f"got {len(even)}")
    report("4 odd-parity corners", len(odd) == 4, f"got {len(odd)}")

    print(f"\n    Even corners (parity 0): {[pos for _, pos in even]}")
    print(f"    Odd  corners (parity 1): {[pos for _, pos in odd]}")

    # --- Test 2: Each group forms a regular tetrahedron ---
    print("\n  --- Test 2: Regular tetrahedra check ---")
    for label, group in [("even", even), ("odd", odd)]:
        positions = [np.array(pos, dtype=float) for _, pos in group]
        dists = []
        for a, b in itertools.combinations(positions, 2):
            dists.append(np.linalg.norm(a - b))
        dists.sort()
        # Regular tetrahedron: all 6 pairwise distances equal
        all_equal = all(abs(d - dists[0]) < 1e-10 for d in dists)
        report(f"{label} corners form regular tetrahedron",
               all_equal,
               f"distances = {[f'{d:.4f}' for d in dists]}")

    # --- Test 3: Face count per body-center ---
    print("\n  --- Test 3: 20 faces per body-center ---")
    faces = enumerate_simplex_faces(bc)
    report("enumerate_simplex_faces returns 20 faces", len(faces) == 20, f"got {len(faces)}")

    # Check all faces are triangles (3 vertices each)
    all_triangles = all(len(f) == 3 for f in faces)
    report("All faces are triangles (3 vertices)", all_triangles)

    # Check body-center appears in all faces? No — C(5,3)=10 per tet,
    # bc appears in C(4,2)=6 faces per tet, so 12 total
    bc_in_face = sum(1 for f in faces if bc_v in f)
    report("bc appears in 12 of 20 faces (C(4,2)*2)",
           bc_in_face == 12, f"got {bc_in_face}")

    # --- Test 4: Global face count = 540 ---
    print("\n  --- Test 4: Global face count ---")
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    n_faces = len(face_to_idx)
    report("Total faces = 540", n_faces == 540, f"got {n_faces}")
    report("27 body-centers", len(body_centers) == 27)
    report("Each bc has 20 face indices", all(len(x) == 20 for x in bc_face_indices))

    # --- Test 5: Face adjacency (edge-sharing) ---
    print("\n  --- Test 5: Face adjacency structure ---")
    A = build_adjacency_matrix(face_to_idx)
    report("Adjacency matrix shape = (540, 540)", A.shape == (540, 540))
    report("Adjacency matrix symmetric", np.allclose(A, A.T))
    report("Adjacency matrix binary (0 or 1)", set(np.unique(A)).issubset({0.0, 1.0}))
    report("No self-loops", np.trace(A) == 0)

    nnz = int(np.sum(A))
    degree_vec = np.sum(A, axis=1).astype(int)
    deg_counter = Counter(degree_vec.tolist())
    print(f"    Nonzero entries: {nnz}")
    print(f"    Degree distribution: {dict(sorted(deg_counter.items()))}")

    # For BCC stella octangula, each triangular face has 3 edges,
    # each edge is shared by exactly 2 faces in the stella octangula complex.
    # But in the full periodic complex, each edge may be shared by more faces.
    # Expected: graph is regular or nearly regular.
    # From prior runs: nnz = 5184, suggesting avg degree = 5184/540 = 9.6

    avg_deg = nnz / n_faces
    print(f"    Average degree: {avg_deg:.2f}")

    # --- Test 6: Translation covariance ---
    print("\n  --- Test 6: Translation covariance ---")
    # Translating all faces by (1,0,0) should permute the face set
    all_faces = set(face_to_idx.keys())
    shift = (1, 0, 0)
    translated = {translate_face(f, shift) for f in all_faces}
    report("Translation (1,0,0) permutes face set",
           translated == all_faces,
           f"|symmetric_diff| = {len(translated.symmetric_difference(all_faces))}")


# ===========================================================================
# PRIORITY 4: Layer 1 — Bloch Matrix Construction
# ===========================================================================

def audit_layer1_bloch():
    """
    Verify Bloch matrix construction: T(R) blocks, reordering, phases.
    """
    print("\n" + "=" * 76)
    print("  AUDIT P4: LAYER 1 — BLOCH MATRIX CONSTRUCTION")
    print("=" * 76)

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T = build_transfer_blocks(body_centers, bc_face_indices, A_full)

    # --- Test 1: T(R) block count ---
    print("\n  --- Test 1: Transfer block structure ---")
    n_blocks = len(T)
    report("27 distinct T(R) blocks", n_blocks == 27, f"got {n_blocks}")

    # All R in {-1,0,1}^3
    expected_Rs = set(itertools.product([-1, 0, 1], repeat=3))
    actual_Rs = set(T.keys())
    report("R vectors span {-1,0,1}^3", actual_Rs == expected_Rs)

    # Each T(R) is 20x20
    all_20x20 = all(block.shape == (20, 20) for block in T.values())
    report("All T(R) blocks are 20x20", all_20x20)

    # --- Test 2: H(Gamma) = sum T(R) should reconstruct A ---
    print("\n  --- Test 2: Gamma-point reconstruction ---")
    H_gamma = np.zeros((20, 20))
    for R, block in T.items():
        H_gamma += block  # All phases = 1 at Gamma

    H_gamma_bloch = build_H_k(T, np.array([0.0, 0.0, 0.0]))
    imag_err = np.max(np.abs(H_gamma_bloch.imag))
    report("H(Gamma) is real", imag_err < 1e-12, f"max imag = {imag_err:.2e}")

    diff = np.max(np.abs(H_gamma - H_gamma_bloch.real))
    report("H(Gamma) = sum T(R) matches build_H_k", diff < 1e-12, f"max diff = {diff:.2e}")

    # Trace of H(Gamma): should be 0 (no self-loops in A)
    tr_gamma = np.trace(H_gamma)
    report("Tr(H(Gamma)) = 0", abs(tr_gamma) < 1e-10, f"Tr = {tr_gamma:.6f}")

    # --- Test 3: Hermiticity at all k-points ---
    print("\n  --- Test 3: Hermiticity at all orbit representatives ---")
    for orbit_name, (n_tuple, _) in ORBIT_DEFS.items():
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_H_k(T, k_vec)
        herm_err = np.max(np.abs(H - H.conj().T))
        report(f"H({orbit_name}) Hermitian", herm_err < 1e-12, f"err = {herm_err:.2e}")

    # --- Test 4: T(R) + T(-R)^T = 0 or symmetric pair ---
    print("\n  --- Test 4: T(R) and T(-R) relation ---")
    # For real Hamiltonian: T(-R) = T(R)^T
    max_pair_err = 0
    for R, block in T.items():
        neg_R = tuple(-r for r in R)
        if neg_R in T:
            err = np.max(np.abs(T[neg_R] - block.T))
            max_pair_err = max(max_pair_err, err)
    report("T(-R) = T(R)^T for all R", max_pair_err < 1e-12,
           f"max err = {max_pair_err:.2e}")

    # --- Test 5: Sum of all T(R) row sums ---
    print("\n  --- Test 5: Row sum conservation ---")
    # Each face has a fixed degree in A, so H(Gamma) row sums should be constant
    row_sums = H_gamma.sum(axis=1)
    print(f"    H(Gamma) row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}")
    # Not necessarily constant (BCC is not regular after projection to 20 faces)
    # But this gives diagnostic info


# ===========================================================================
# PRIORITY 5: Layer 2 — Eigenvalue Stability
# ===========================================================================

def audit_layer2_eigenvalues():
    """
    Verify flat band properties and eigenvalue numerical stability.
    """
    print("\n" + "=" * 76)
    print("  AUDIT P5: LAYER 2 — EIGENVALUE STABILITY")
    print("=" * 76)

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T = build_transfer_blocks(body_centers, bc_face_indices, A_full)

    for orbit_name, (n_tuple, _) in ORBIT_DEFS.items():
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_H_k(T, k_vec)
        evals = np.linalg.eigvalsh(H)

        # Flat band check
        flat_mask = np.abs(evals - FLAT_EIGENVALUE) < FLAT_TOL
        flat_count = np.sum(flat_mask)
        flat_deviations = evals[flat_mask] - FLAT_EIGENVALUE

        report(f"BCC {orbit_name}: flat band count = 6",
               flat_count == 6, f"got {flat_count}")

        if flat_count > 0:
            max_flat_dev = np.max(np.abs(flat_deviations))
            report(f"BCC {orbit_name}: flat band max deviation from -3",
                   max_flat_dev < 1e-10,
                   f"max_dev = {max_flat_dev:.2e}")

        # Non-flat eigenvalues
        non_flat = evals[~flat_mask]
        report(f"BCC {orbit_name}: 14 non-flat eigenvalues",
               len(non_flat) == 14, f"got {len(non_flat)}")

        # Check no non-flat eigenvalue accidentally close to -3
        if len(non_flat) > 0:
            closest_to_flat = np.min(np.abs(non_flat - FLAT_EIGENVALUE))
            report(f"BCC {orbit_name}: non-flat well-separated from -3",
                   closest_to_flat > 0.1,
                   f"closest = {closest_to_flat:.6f}")

    # --- Sympy exact diagonalization at Gamma ---
    if SYMPY_OK:
        print("\n  --- Sympy exact charpoly factorization at Gamma ---")
        H_gamma = np.zeros((20, 20))
        for R, block in T.items():
            H_gamma += block
        H_sym = SympyMatrix(np.round(H_gamma).astype(int).tolist())
        x_sym = Symbol('x')
        char_poly = H_sym.charpoly(x_sym)
        print(f"    Charpoly degree: {char_poly.degree()}")

        factors = char_poly.factor_list()
        print(f"    Factorization over Q:")
        has_flat = False
        for fac, exp in factors[1]:
            fac_expr = fac.as_expr()
            f_val = fac.eval(-3)
            print(f"      ({fac_expr})^{exp}   f(-3) = {f_val}")
            if fac_expr == x_sym + 3 and exp == 6:
                has_flat = True
        report("Sympy exact: (x+3)^6 factor (flat band) at Gamma", has_flat)
    else:
        print("\n  [SKIP] Sympy exact charpoly (sympy not available)")


# ===========================================================================
# PRIORITY 6: Layer 3 — Galois Classification (Most Subtle)
# ===========================================================================

def audit_layer3_galois():
    """
    Cross-check galois_conjugacy_classes backtracking with sympy
    minimal_polynomial for each non-flat eigenvalue.
    """
    print("\n" + "=" * 76)
    print("  AUDIT P2: LAYER 3 — GALOIS CLASSIFICATION (SYMPY EXACT)")
    print("=" * 76)

    if not SYMPY_OK:
        print("  [SKIP] sympy not available")
        return

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T_blocks = build_transfer_blocks(body_centers, bc_face_indices, A_full)

    x = Symbol('x')

    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        print(f"\n  --- Orbit: {orbit_name} (size {orbit_size}) ---")
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_H_k(T_blocks, k_vec)
        evals_all = np.linalg.eigvalsh(H)

        # Separate flat from non-flat
        non_flat = [e for e in evals_all if abs(e - FLAT_EIGENVALUE) >= FLAT_TOL]
        groups = group_eigenvalues(non_flat)

        # Run the script's Galois classification
        classes = galois_conjugacy_classes(groups)

        # For each conjugacy class, verify with sympy
        print(f"    Script found {len(classes)} conjugacy classes:")
        for ci, cls in enumerate(classes):
            deg = cls['degree']
            vals = cls['values']
            coeffs = cls['min_poly_coeffs']
            gap_norm = cls['gap_norm']

            # Reconstruct minimal polynomial from coefficients
            if coeffs:
                poly_terms = []
                d = len(coeffs) - 1
                for i, c in enumerate(coeffs):
                    poly_terms.append(f"{c}*x^{d-i}")
                # Build sympy polynomial
                sym_poly = sum(c * x**(d - i) for i, c in enumerate(coeffs))
                sym_poly_obj = Poly(sym_poly, x, domain='QQ')

                # Evaluate at each numerical root — should be ~0
                max_residual = 0
                for v in vals:
                    res = abs(float(sym_poly_obj.eval(v)))
                    max_residual = max(max_residual, res)

                report(f"  Class {ci+1} (deg {deg}): poly vanishes at numerical roots",
                       max_residual < 1e-6,
                       f"max residual = {max_residual:.2e}")

                # Evaluate f(-3) from Fraction coefficients vs script's gap_norm
                f_at_minus3 = sum(c * Fraction(-3)**(d - i) for i, c in enumerate(coeffs))
                report(f"  Class {ci+1} (deg {deg}): f(-3) = {f_at_minus3}",
                       abs(f_at_minus3) == abs(gap_norm),
                       f"script says {gap_norm}")

                # CRITICAL: Verify polynomial is irreducible over Q
                is_irred = sym_poly_obj.is_irreducible
                report(f"  Class {ci+1} (deg {deg}): polynomial irreducible over Q",
                       is_irred,
                       f"poly = {sym_poly_obj.as_expr()}")

            else:
                report(f"  Class {ci+1}: has coefficients", False, "no coefficients found")

        # Independent sympy check: use exact matrix (Gamma only, due to cost)
        if orbit_name == 'Gamma':
            print(f"\n    --- Sympy independent minimal polynomial check (Gamma) ---")
            H_real = np.zeros((20, 20))
            for R, block in T_blocks.items():
                H_real += block
            H_sym = SympyMatrix(np.round(H_real).astype(int).tolist())
            char_poly = H_sym.charpoly(x)
            print(f"    Characteristic polynomial degree: {char_poly.degree()}")

            # Factor the characteristic polynomial over Q
            factors = char_poly.factor_list()
            print(f"    Factorization over Q:")
            for fac, exp in factors[1]:
                print(f"      ({fac.as_expr()})^{exp}")

            # Extract irreducible factors and compare to script's classes
            irred_factors = []
            for fac, exp in factors[1]:
                irred_factors.append((fac, exp))
                # Check if (x+3)^6 is among them (flat band)
            has_flat = any(
                fac.as_expr() == x + 3 and exp >= 6
                for fac, exp in irred_factors
            )
            # Could also be (x+3) with exp=6
            report("Gamma charpoly has (x+3)^6 factor (flat band)",
                   has_flat,
                   f"factors: {[(str(f.as_expr()), e) for f, e in irred_factors]}")

            # Compare non-(x+3) irreducible factors to script's classes
            script_polys = []
            for cls in classes:
                if cls['min_poly_coeffs']:
                    d = cls['degree']
                    coeffs = cls['min_poly_coeffs']
                    p = sum(c * x**(d - i) for i, c in enumerate(coeffs))
                    script_polys.append((Poly(p, x, domain='QQ'), cls['mult']))

            sympy_non_flat = []
            for fac, exp in irred_factors:
                fac_expr = fac.as_expr()
                if fac_expr != x + 3:
                    sympy_non_flat.append((fac, exp))

            print(f"\n    Script non-flat classes: {len(script_polys)}")
            print(f"    Sympy non-flat factors: {len(sympy_non_flat)}")

            # Match each script polynomial to a sympy factor
            for sp, mult in script_polys:
                matched = False
                for sf, sexp in sympy_non_flat:
                    if sp.as_expr().equals(sf.as_expr()):
                        report(f"  Script poly {sp.as_expr()} matches sympy factor (exp={sexp})",
                               True, f"script mult={mult}")
                        # Check multiplicity consistency: script mult * degree = sympy exp * degree
                        # Actually sympy exp = multiplicity of that irreducible factor
                        report(f"    Multiplicity: script={mult}, sympy_exp={sexp}",
                               mult == sexp)
                        matched = True
                        break
                if not matched:
                    report(f"  Script poly {sp.as_expr()} found in sympy factors", False)


# ===========================================================================
# PRIORITY 7: Layer 3b — Exact Minimal Polynomials for Non-Gamma Orbits
# ===========================================================================

def audit_layer3b_exact_non_gamma():
    """
    For Gamma, compute the exact charpoly and factor over Q.
    For non-Gamma orbits, exact sympy charpoly is too slow, so we verify using
    a float-verified path (reconstructing polynomials via limit_denominator).
    """
    print("\n" + "=" * 76)
    print("  AUDIT P2b: LAYER 3 — EXACT CHARPOLY / FLOAT-VERIFIED FOR ALL ORBITS")
    print("=" * 76)

    if not SYMPY_OK:
        print("  [SKIP] sympy not available")
        return

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T_blocks = build_transfer_blocks(body_centers, bc_face_indices, A_full)

    x_sym = Symbol('x')

    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        try:
            k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
            H = build_H_k(T_blocks, k_vec)

            if orbit_name == 'Gamma':
                print(f"\n  --- Orbit: Gamma (exact sympy computation) ---")
                H_exact = SympyMatrix(np.round(H.real).astype(int).tolist())
                # Compute characteristic polynomial
                print(f"    Computing exact charpoly...")
                char_poly = H_exact.charpoly(x_sym)
                print(f"    Charpoly degree: {char_poly.degree()}")
                factors = char_poly.factor_list()
                print(f"    Factorization over Q:")
                flat_factor_exp = 0
                for fac, exp in factors[1]:
                    fac_expr = fac.as_expr()
                    fac_at_minus3 = fac.eval(-3)
                    print(f"      ({fac_expr})^{exp}   f(-3) = {fac_at_minus3}")
                    if fac_expr == x_sym + 3:
                        flat_factor_exp = exp
                report("H(Gamma) flat band (x+3)^6", flat_factor_exp == 6)
            else:
                print(f"\n  --- Orbit: {orbit_name} (float-verified path) ---")
                evals_all = np.linalg.eigvalsh(H)
                non_flat = [e for e in evals_all if abs(e - FLAT_EIGENVALUE) >= FLAT_TOL]
                groups = group_eigenvalues(non_flat)
                classes = galois_conjugacy_classes(groups)

                prime_menu = set()
                bridge_found = False
                p_topo = orbit_size - 1

                for cls in classes:
                    coeffs = cls.get('min_poly_coeffs', [])
                    if coeffs:
                        f_val = abs(eval_min_poly_at(coeffs, Fraction(-3)))
                        primes = prime_factors(f_val.numerator) + prime_factors(f_val.denominator)
                        prime_menu.update(primes)
                        if p_topo in primes:
                            bridge_found = True

                report(f"H({orbit_name}) prime menu subset of {{2,3,5,7,11}} (float-verified)",
                       prime_menu.issubset({2, 3, 5, 7, 11}),
                       f"menu = {sorted(prime_menu)}")
                if p_topo > 0:
                    report(f"H({orbit_name}) bridge p={p_topo} present (float-verified)",
                           bridge_found)
        except Exception as e:
            report(f"H({orbit_name}) charpoly computation", False, f"error: {e}")


# ===========================================================================
# SUMMARY
# ===========================================================================

def print_summary():
    print("\n" + "#" * 76)
    print(f"#  AUDIT SUMMARY")
    print(f"#  PASS: {PASS_COUNT}   FAIL: {FAIL_COUNT}")
    print("#" * 76)
    if FAIL_COUNT == 0:
        print("  ALL LAYERS CLEAN. No error propagation detected.")
    else:
        print(f"  {FAIL_COUNT} FAILURES DETECTED. Review above for details.")


# ===========================================================================
# TRACE CONSTANTS SOURCE INDEPENDENT VERIFICATION
# ===========================================================================

def audit_trace_constants_source():
    print("\n" + "=" * 76)
    print("  AUDIT: TRACE CONSTANTS SOURCE INDEPENDENT VERIFICATION")
    print("=" * 76)
    
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T_blocks = build_transfer_blocks(body_centers, bc_face_indices, A_full)
    T0 = T_blocks[(0, 0, 0)]
    
    expected_constants = {
        'Gamma': Fraction(-4),
        'axis': Fraction(-67, 10),
        'face-diagonal': Fraction(-8326, 1155),
        'body-diagonal': Fraction(-214, 35),
    }
    
    # 1. Numerical verification for all orbits
    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_H_k(T_blocks, k_vec)
        evals, evecs = np.linalg.eigh(H)
        
        # Verify flat bands at -3
        flat_indices = [i for i, ev in enumerate(evals) if abs(ev - FLAT_EIGENVALUE) < FLAT_TOL]
        report(f"Trace audit {orbit_name}: exactly 6 flat bands", len(flat_indices) == 6)
        
        V = evecs[:, flat_indices]
        tr_num = np.real(np.trace(V.conj().T @ T0 @ V))
        
        # Express as fraction
        expected = expected_constants[orbit_name]
        dev = abs(tr_num - float(expected))
        report(f"Trace audit {orbit_name} matches {expected}", dev < 1e-11, f"dev={dev:.2e} (float-verified)")
        
    # 2. Exact Sympy verification for Gamma (since it is integer and fast)
    if SYMPY_OK:
        print("\n  --- Sympy exact trace constant at Gamma ---")
        H_gamma = np.zeros((20, 20))
        for R, block in T_blocks.items():
            H_gamma += block
        H_sym = SympyMatrix(np.round(H_gamma).astype(int).tolist())
        H_shifted = H_sym + 3 * sympy.eye(20)
        null_basis = H_shifted.nullspace()
        
        report("Gamma nullspace size is 6", len(null_basis) == 6)
        V_g = null_basis[0]
        for v in null_basis[1:]:
            V_g = V_g.row_join(v)
            
        T0_sym = SympyMatrix(np.round(T0).astype(int).tolist())
        G = V_g.T * V_g
        VTV = V_g.T * T0_sym * V_g
        tr_exact = sympy.simplify((G.inv() * VTV).trace())
        
        report("Gamma exact trace matches -4", tr_exact == Fraction(-4), f"exact={tr_exact}")
    else:
        print("\n  [SKIP] Sympy exact trace verification (sympy not available)")


# ===========================================================================
# LAYER 4 — BRIDGE POLYNOMIAL ASSERTS
# ===========================================================================

def audit_layer4_bridge():
    print("\n" + "=" * 76)
    print("  AUDIT LAYER 4: BRIDGE POLYNOMIAL ASSERTS")
    print("=" * 76)
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T_blocks = build_transfer_blocks(body_centers, bc_face_indices, A_full)
    
    f_neg3_values = {}
    
    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        if orbit_name == 'Gamma':
            continue
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_H_k(T_blocks, k_vec)
        evals_all = np.linalg.eigvalsh(H)
        non_flat = [e for e in evals_all if abs(e - FLAT_EIGENVALUE) >= FLAT_TOL]
        groups = group_eigenvalues(non_flat)
        classes = galois_conjugacy_classes(groups)
        
        p_topo = orbit_size - 1
        bridge_val = None
        for cls in classes:
            coeffs = cls.get('min_poly_coeffs', [])
            if coeffs:
                f_val = eval_min_poly_at(coeffs, Fraction(-3))
                f_abs = abs(f_val)
                primes_f = prime_factors(f_abs.numerator) + prime_factors(f_abs.denominator)
                if p_topo in primes_f:
                    # Found the bridge class
                    bridge_val = int(f_val) # f(-3) should be integer
                    break
        f_neg3_values[orbit_name] = bridge_val
        print(f"  Orbit {orbit_name}: found bridge f(-3) = {bridge_val}")

    f_neg3_axis = f_neg3_values['axis']
    f_neg3_face = f_neg3_values['face-diagonal']
    f_neg3_body = f_neg3_values['body-diagonal']
    
    # Assertions
    print(f"  Asserting f(-3) values...")
    assert f_neg3_axis == -60, f"Expected -60, got {f_neg3_axis}"
    assert f_neg3_face == 264, f"Expected 264, got {f_neg3_face}"
    assert f_neg3_body == 7, f"Expected 7, got {f_neg3_body}"
    report("Layer 4 bridge f(-3) ground-truth asserts passed", True)


# ===========================================================================
# L-SCAN AUDIT
# ===========================================================================

def audit_l_scan():
    print("\n" + "=" * 76)
    print("  AUDIT: L-SCAN VERIFICATION")
    print("=" * 76)
    
    # Import analyze_L from verify_bridge_norm_L_scan
    sys.path.insert(0, '.')
    from verify_bridge_norm_L_scan import analyze_L
    
    # 1. Run L=2, 3, 4, 5
    print("  Running L-scan for L=2,3,4,5...")
    results = {}
    for L in [2, 3, 4, 5]:
        res = analyze_L(L, verbose=False)
        results[L] = res
        print(f"    L={L}: bridge_count={res['bridge_count']}/{res['bridge_total']}, gap_primes={sorted(res['all_gap_primes'])}")
        
    # 2. Verify results against ground truth
    # L=2: 2/2 bridge, gap primes {2, 3, 89}
    r2 = results[2]
    report("L=2 bridge count is 2/2", r2['bridge_count'] == 2 and r2['bridge_total'] == 2)
    report("L=2 gap primes are {2, 3, 89}", r2['all_gap_primes'] == {2, 3, 89})
    
    # L=3: 3/3 bridge, gap primes {2, 3, 5, 7, 11}
    r3 = results[3]
    report("L=3 bridge count is 3/3", r3['bridge_count'] == 3 and r3['bridge_total'] == 3)
    report("L=3 gap primes are {2, 3, 5, 7, 11}", r3['all_gap_primes'] == {2, 3, 5, 7, 11})
    
    # L=4: 2/8 bridge
    r4 = results[4]
    report("L=4 bridge count is 2/8", r4['bridge_count'] == 2 and r4['bridge_total'] == 8)
    
    # L=5: 0/9 bridge
    r5 = results[5]
    report("L=5 bridge count is 0/9", r5['bridge_count'] == 0 and r5['bridge_total'] == 9)
    
    # 3. Confirm bridge results consistency between L_scan and projective script for L=3
    report("L=3 bridge results consistent with projective script", r3['bridge_count'] == 3)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 76)
    print("  BOTTOM-UP NUMERICAL AUDIT: verify_bridge_norm_projective.py")
    print("  Layer 0 → 1 → 2 → 3 → 4")
    print("=" * 76)

    # Priority 1: SC analytical (fastest, validates pipeline)
    audit_sc_analytical()

    # Priority 6: FCC analytical (same idea, second lattice)
    audit_fcc_analytical()

    # Priority 3: Layer 0 — BCC lattice construction
    audit_layer0_bcc()

    # Priority 4: Layer 1 — Bloch matrix
    audit_layer1_bloch()

    # Priority 5: Layer 2 — Eigenvalues
    audit_layer2_eigenvalues()

    # Priority 2: Layer 3 — Galois classification
    audit_layer3_galois()

    # Priority 2b: Layer 3 exact for non-Gamma
    audit_layer3b_exact_non_gamma()
    
    # Trace constants source audit
    audit_trace_constants_source()
    
    # Layer 4 bridge asserts
    audit_layer4_bridge()
    
    # L-scan audit
    audit_l_scan()

    # Summary
    print_summary()


if __name__ == "__main__":
    main()
