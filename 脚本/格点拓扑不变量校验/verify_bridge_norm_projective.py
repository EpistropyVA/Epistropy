# -*- coding: utf-8 -*-
"""
verify_bridge_norm_projective.py

Unified bridge verification: spectral gap norms ↔ projective line characteristics.

For each k-orbit in a 3x3x3 periodic lattice:
  - LEFT  (topological): orbit size d → prime p = d-1 → P¹(F_p)
  - RIGHT (algebraic):   non-flat eigenvalues → Galois conjugacy classes →
                         minimal polynomials f(x) over Q → |f(-3)| factored
  - BRIDGE: the topological prime p appears in the algebraic factorization.

Falsification test: repeat for FCC and SC lattices.
If primes are synthesized by orbits, changing the lattice → different orbit sizes
→ different prime menus.

Usage:
    python verify_bridge_norm_projective.py
"""

import itertools
import sys
import io
from fractions import Fraction
from collections import Counter

# Force UTF-8 output on Windows (avoids GBK codec errors for non-ASCII chars)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

# ---------------------------------------------------------------------------
# Sympy imports (for exact minimal polynomials and factoring)
# ---------------------------------------------------------------------------
try:
    import sympy
    from sympy import Rational as SRational, factorint, Poly, Symbol
    from sympy.polys.numberfields import minimal_polynomial
    from sympy import nsimplify, sqrt as sp_sqrt
    SYMPY_OK = True
except ImportError:
    SYMPY_OK = False
    print("WARNING: sympy not available; exact arithmetic disabled.")

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOL = 1e-9
FLAT_EIGENVALUE = -3.0
FLAT_TOL = 1e-6


# ===========================================================================
# Lattice construction (BCC, shared with existing scripts)
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


def build_transfer_blocks(body_centers, bc_face_indices, A_full):
    reorder = []
    for bc_idx in range(len(body_centers)):
        reorder.extend(bc_face_indices[bc_idx])
    A_reordered = A_full[np.ix_(reorder, reorder)]
    T = {}
    for j, bc_ijk in enumerate(body_centers):
        Rc = tuple((r + 1) % 3 - 1 for r in bc_ijk)
        block = A_reordered[:20, j * 20:(j + 1) * 20]
        if Rc not in T:
            T[Rc] = block.copy()
        else:
            T[Rc] = T[Rc] + block
    return T


def build_H_k(T, k_vec):
    H = np.zeros((20, 20), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, R))
        H += phase * block
    return H


def build_bcc_bloch(k_vec):
    """Full pipeline: build BCC 20x20 Bloch Hamiltonian at k_vec."""
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T = build_transfer_blocks(body_centers, bc_face_indices, A_full)
    return build_H_k(T, k_vec)


# ===========================================================================
# Alternative lattice: simple cubic (SC) 3x3x3
# ===========================================================================

def build_sc_bloch(k_vec, N=3):
    """
    Simple cubic 3x3x3 with N=3 sites per axis.
    Sites: (i,j,k) for i,j,k in {0,1,2}, periodic.
    Nearest-neighbor hopping (6 neighbors in 3D).
    Bloch Hamiltonian: H(k) is 1x1 (one site per unit cell),
    but for the full 3x3x3 supercell Bloch transform we build a 27x27
    representation and fold down. Actually for a single-site basis
    in k-space H(k) = 2(cos kx + cos ky + cos kz).
    For comparison with BCC we use the 27-site Bloch matrix
    (before zone-folding): H_{ij}(k) = sum_R exp(ik.R) A_{0,R}
    where A is the SC adjacency matrix.
    """
    sites = list(itertools.product(range(N), repeat=3))
    n_sites = len(sites)
    site_idx = {s: i for i, s in enumerate(sites)}

    # Collect hopping blocks T(R) for SC
    T = {}
    for s in sites:
        for axis in range(3):
            for delta in [1, -1]:
                nb = list(s)
                nb[axis] = (s[axis] + delta) % N
                nb_t = tuple(nb)
                # R = displacement from unit cell of s=0 to unit cell of nb
                # in the supercell picture, R is determined by the unwrapped position
                R = [0, 0, 0]
                R[axis] = delta  # ±1 in lattice units
                R_t = tuple(R)
                j = site_idx[nb_t]
                if R_t not in T:
                    T[R_t] = np.zeros((n_sites, n_sites))
                T[R_t][0, j] += 1.0  # only first-cell row matters for Bloch

    # Build Bloch H(k): n_sites × n_sites
    H = np.zeros((n_sites, n_sites), dtype=complex)
    for s_idx, s in enumerate(sites):
        for axis in range(3):
            for delta in [1, -1]:
                nb = list(s)
                nb[axis] = (s[axis] + delta) % N
                j = site_idx[tuple(nb)]
                # Phase from wrapping
                phase_exp = 0
                if s[axis] == 0 and delta == -1:
                    phase_exp = -1
                elif s[axis] == N - 1 and delta == 1:
                    phase_exp = 1
                phase = np.exp(1j * k_vec[axis] * phase_exp * N)
                H[s_idx, j] += phase
    return H


def build_sc_bloch_v2(k_vec):
    """
    SC Bloch matrix built as single-band dispersion replicated.
    For the 3x3x3 supercell, we build the full 27x27 real-space
    adjacency with periodic BCs, then Bloch-transform to get
    a 27x27 Hamiltonian (all in a single unit cell, k only from
    the first BZ point).

    Actually the cleanest SC comparison is:
    SC sites = (i,j,k), hop to ±1 in each axis (mod 3).
    The 3x3x3 supercell has 27 sites and the adjacency is 27x27.
    Bloch at k = sum_R T(R) exp(ik.R).
    We build it explicitly with the same approach as BCC.
    """
    N = 3
    sites = list(itertools.product(range(N), repeat=3))
    n = len(sites)
    idx = {s: i for i, s in enumerate(sites)}

    # Real-space hopping T(R): the j-th column of T(R) contributes to row 0
    # We want the Bloch picture: 27 bands, one per site in unit cell,
    # but here there IS only one site type (SC is a Bravais lattice).
    # So the Bloch matrix is really 1x1. For comparison, use the full
    # supercell picture: 27x27 matrix where T(R) gives site-to-site hops
    # folded by Bloch phase.
    # This gives the SAME spectrum as the BCC Bloch picture (27 bands).
    T = {}
    for i, s in enumerate(sites):
        for axis in range(3):
            for delta in [1, -1]:
                nb_list = list(s)
                nb_list[axis] = (s[axis] + delta) % N
                nb = tuple(nb_list)
                # Lattice displacement vector
                R = [0, 0, 0]
                if delta == 1 and s[axis] == N - 1:
                    R[axis] = 1  # wrapped forward
                elif delta == -1 and s[axis] == 0:
                    R[axis] = -1  # wrapped backward
                R_t = tuple(R)
                j = idx[nb]
                if R_t not in T:
                    T[R_t] = np.zeros((n, n))
                T[R_t][i, j] += 1.0

    H = np.zeros((n, n), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, R))
        H += phase * block
    return H


# ===========================================================================
# FCC lattice (simple model)
# ===========================================================================

def build_fcc_bloch(k_vec):
    """
    FCC: face-center sites in 3x3x3 supercell.
    In a simple cubic cell, FCC sites are at:
      face-centers of the 3D cubic lattice.
    With periodic 3x3x3 supercell (3 unit cells per axis),
    the FCC lattice has 3 face-centers per cubic unit cell
    × 27 cells = ... actually FCC has 4 atoms per conventional cell
    (corners + face-centers), but in the primitive picture 1 per primitive cell.

    For a direct comparison approach: use the simplest FCC tight-binding
    with 12 nearest neighbors in a cubic supercell.

    Simplified: in a 3x3x3 grid of SC sites, add FCC-type hopping
    (face-diagonal neighbors within each cube face, i.e. distance sqrt(2)/2 in
    units where cubic spacing = 1). Each site connects to 12 face-diagonal neighbors.
    """
    N = 3
    sites = list(itertools.product(range(N), repeat=3))
    n = len(sites)
    idx = {s: i for i, s in enumerate(sites)}

    # FCC nearest neighbors: (±1, ±1, 0) and all axis permutations → 12 neighbors
    # Base vectors: (1,1,0), (1,-1,0), (1,0,1), (1,0,-1), (0,1,1), (0,1,-1)
    # + negatives = 12 total
    fcc_base = set()
    for perm in itertools.permutations([1, 0, 0]):
        pass  # placeholder to organize below
    fcc_base = set()
    for axes in itertools.combinations(range(3), 2):
        for s1, s2 in itertools.product([1, -1], repeat=2):
            d = [0, 0, 0]
            d[axes[0]] = s1
            d[axes[1]] = s2
            fcc_base.add(tuple(d))
    fcc_deltas = list(fcc_base)
    # Should have 12 distinct FCC neighbors
    assert len(fcc_deltas) == 12, f"Expected 12 FCC neighbors, got {len(fcc_deltas)}"

    T = {}
    for i, s in enumerate(sites):
        for delta in fcc_deltas:
            nb_raw = (s[0] + delta[0], s[1] + delta[1], s[2] + delta[2])
            nb = tuple(x % N for x in nb_raw)
            # Lattice displacement (which unit cell image)
            R = tuple((nb_raw[ax] - nb[ax]) // N for ax in range(3))
            j = idx[nb]
            if R not in T:
                T[R] = np.zeros((n, n))
            T[R][i, j] += 1.0

    H = np.zeros((n, n), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, R))
        H += phase * block
    return H


# ===========================================================================
# k-orbit classification (O_h action on 3x3x3 BZ)
# ===========================================================================

ORBIT_DEFS = {
    # name: (representative n-tuple, size)
    # k = 2*pi * n / 3
    'Gamma':         ((0, 0, 0), 1),
    'axis':          ((1, 0, 0), 6),
    'face-diagonal': ((1, 1, 0), 12),
    'body-diagonal': ((1, 1, 1), 8),
}

KNOWN_TRACE_CONSTANTS = {
    # Tr(P_k T(0)) for each orbit — from verify_t7_sympy_korbit_constants.py
    'Gamma':         Fraction(-4),
    'axis':          Fraction(-67, 10),
    'face-diagonal': Fraction(-8326, 1155),
    'body-diagonal': Fraction(-214, 35),
}


def k_orbit_primes(orbit_name, orbit_size):
    """
    Topological side: orbit of size d → p = d - 1.
    Returns (d, p, denominator primes from known trace constant).
    """
    d = orbit_size
    p = d - 1
    if orbit_name in KNOWN_TRACE_CONSTANTS:
        tc = KNOWN_TRACE_CONSTANTS[orbit_name]
        denom = tc.denominator
        denom_primes = prime_factors(denom)
        max_p = max(denom_primes) if denom_primes else 1
    else:
        denom_primes = []
        max_p = None
    return d, p, denom_primes, max_p


# ===========================================================================
# Algebraic helpers
# ===========================================================================

def prime_factors(n):
    """Return list of distinct prime factors of |n|."""
    n = abs(int(n))
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return sorted(factors)


def factored_str(n):
    """Return prime factorization string of integer n (e.g. '2^2·3·5')."""
    n = int(n)
    if n == 0:
        return "0"
    sign = ""
    if n < 0:
        sign = "-"
        n = -n
    if n == 1:
        return sign + "1"
    facs = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            facs[d] = facs.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        facs[n] = facs.get(n, 0) + 1
    parts = []
    for p in sorted(facs):
        parts.append(str(p) if facs[p] == 1 else f"{p}^{facs[p]}")
    return sign + "·".join(parts)


def to_fraction(x, limit=200):
    """Try to express float x as a Fraction with small denominator. Return (frac, ok).
    Uses a tighter limit to avoid mistaking irrational eigenvalues for rationals.
    For elementary symmetric polynomials of multiple eigenvalues, use limit=2000."""
    f = Fraction(x).limit_denominator(limit)
    ok = abs(float(f) - x) < TOL
    return f, ok


def to_fraction_sym(x, limit=2000):
    """Try to express float x as a Fraction. Higher denominator limit for
    symmetric polynomials of eigenvalues (which can have larger denominators)."""
    f = Fraction(x).limit_denominator(limit)
    ok = abs(float(f) - x) < TOL
    return f, ok


def group_eigenvalues(evals, tol=TOL):
    """
    Group eigenvalues into clusters of nearly equal values.
    Returns list of (representative_float, multiplicity).
    """
    sorted_ev = sorted(evals)
    groups = []
    i = 0
    while i < len(sorted_ev):
        val = sorted_ev[i]
        cluster = [val]
        j = i + 1
        while j < len(sorted_ev) and abs(sorted_ev[j] - val) < tol * (1 + abs(val)):
            cluster.append(sorted_ev[j])
            j += 1
        groups.append((np.mean(cluster), len(cluster)))
        i = j
    return groups


def galois_conjugacy_classes(eigenvalue_groups):
    """
    Given grouped eigenvalues [(rep_float, mult), ...],
    partition irrational ones into Galois conjugacy classes
    by finding subsets whose elementary symmetric polynomials are all rational.

    Returns list of dicts:
        {
            'values': [float, ...],          # conjugate eigenvalues
            'mult': int,                     # multiplicity of each
            'degree': int,                   # degree of conjugacy class
            'sym_polys': [Fraction, ...],    # e1, e2, ..., ed over Q
            'min_poly_coeffs': [Fraction, ...],  # coefficients of min poly
            'gap_norm': Fraction,            # product of (lambda+3) for conjugates
            'is_rational': bool,
        }
    Also returns list of rational eigenvalues:
        [(value_frac, mult), ...]
    """
    rational = []
    irrational = []
    for rep, mult in eigenvalue_groups:
        frac, ok = to_fraction(rep)
        if ok:
            rational.append((frac, mult))
        else:
            irrational.append((rep, mult))

    classes = []

    # Rational eigenvalues: each is its own degree-1 class
    for frac, mult in rational:
        gap = frac + 3
        classes.append({
            'values': [float(frac)],
            'mult': mult,
            'degree': 1,
            'sym_polys': [frac],
            'min_poly_coeffs': [Fraction(1), -frac],  # x - frac
            'gap_norm': gap,
            'is_rational': True,
        })

    # Irrational: partition into conjugacy classes using backtracking
    irrat_vals = [rep for rep, mult in irrational]
    irrat_mults = [mult for rep, mult in irrational]

    def check_galois_class(indices):
        """Check whether the given index subset forms a valid Galois conjugacy class.
        Returns (sym_polys, gap_norm, coeffs) if valid, else None."""
        vals = [irrat_vals[i] for i in indices]
        mults = [irrat_mults[i] for i in indices]
        if len(set(mults)) != 1:
            return None  # multiplicities must match
        size = len(vals)
        sym_polys = []
        for k in range(1, size + 1):
            ek = sum(
                np.prod([vals[j] for j in sub])
                for sub in itertools.combinations(range(size), k)
            )
            fk, ok = to_fraction_sym(ek)
            if not ok:
                return None
            sym_polys.append(fk)
        d = size
        e = [Fraction(1)] + sym_polys
        gap_norm = sum(e[k] * Fraction(3) ** (d - k) for k in range(d + 1))
        coeffs = [Fraction(1)]
        for k, ek in enumerate(sym_polys):
            coeffs.append(((-1) ** (k + 1)) * ek)
        return sym_polys, gap_norm, coeffs

    def backtrack_partition(remaining_indices):
        """Recursively partition remaining_indices into valid Galois classes.
        Returns list of (indices_tuple, (sym_polys, gap_norm, coeffs)) or None."""
        if not remaining_indices:
            return []
        first = remaining_indices[0]
        rest = remaining_indices[1:]
        # Try all subsets containing first, sorted by size (smallest first)
        for size in range(1, len(remaining_indices) + 1):
            for combo in itertools.combinations(rest, size - 1):
                subset = (first,) + combo
                result = check_galois_class(subset)
                if result is None:
                    continue
                new_remaining = [i for i in remaining_indices if i not in subset]
                sub_solution = backtrack_partition(new_remaining)
                if sub_solution is not None:
                    return [(subset, result)] + sub_solution
        return None  # no valid partition found

    all_indices = list(range(len(irrat_vals)))
    partition = backtrack_partition(all_indices)

    if partition is not None:
        for (subset_indices, (sym_polys, gap_norm, coeffs)) in partition:
            vals = [irrat_vals[i] for i in subset_indices]
            mult = irrat_mults[subset_indices[0]]
            d = len(vals)
            classes.append({
                'values': vals,
                'mult': mult,
                'degree': d,
                'sym_polys': sym_polys,
                'min_poly_coeffs': coeffs,
                'gap_norm': gap_norm,
                'is_rational': False,
            })
    else:
        # Fallback: report each unclassified eigenvalue individually
        for i in range(len(irrat_vals)):
            rep = irrat_vals[i]
            classes.append({
                'values': [rep],
                'mult': irrat_mults[i],
                'degree': 1,
                'sym_polys': [],
                'min_poly_coeffs': [],
                'gap_norm': None,
                'is_rational': False,
                'unclassified': True,
            })

    return classes


def eval_min_poly_at(coeffs, x):
    """
    Evaluate polynomial with coefficients [a_d, a_{d-1}, ..., a_0]
    at x (as Fraction).
    """
    x = Fraction(x)
    result = Fraction(0)
    d = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        result += c * x ** (d - i)
    return result


# ===========================================================================
# BCC k-point data
# ===========================================================================

BCC_K_REPS = {
    'Gamma':         np.array([0.0, 0.0, 0.0]),
    'axis':          np.array([2 * np.pi / 3, 0.0, 0.0]),
    'face-diagonal': np.array([2 * np.pi / 3, 2 * np.pi / 3, 0.0]),
    'body-diagonal': np.array([2 * np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3]),
}


# ===========================================================================
# Core analysis: for one k-point, get non-flat eigenvalues and classes
# ===========================================================================

def analyze_k_point(H, flat_eigenvalue=FLAT_EIGENVALUE, flat_tol=FLAT_TOL):
    """
    Diagonalize H, exclude flat-band eigenvalues, return:
      - flat_count: int
      - classes: list of conjugacy class dicts
      - all_evals: full eigenvalue list
    """
    evals = np.linalg.eigvalsh(H)
    if flat_eigenvalue is None:
        flat_count = 0
        non_flat = list(evals)
    else:
        flat_count = sum(1 for e in evals if abs(e - flat_eigenvalue) < flat_tol)
        non_flat = [e for e in evals if abs(e - flat_eigenvalue) >= flat_tol]
    groups = group_eigenvalues(non_flat)
    classes = galois_conjugacy_classes(groups)
    return flat_count, classes, evals


# ===========================================================================
# Bridge table printing
# ===========================================================================

def print_bridge_table(orbit_name, orbit_size, flat_count, classes, known_trace=None):
    """
    Print the bridge table for one orbit:
    LEFT: topological orbit data
    RIGHT: algebraic conjugacy class data
    """
    d = orbit_size
    p_topo = d - 1

    print(f"\n{'=' * 76}")
    print(f"  ORBIT: {orbit_name}  |  size d = {d}  |  topological prime p = d-1 = {p_topo}")
    if known_trace is not None:
        denom = known_trace.denominator
        dp = prime_factors(denom)
        max_p = max(dp) if dp else 1
        print(f"  Trace constant Tr(P_k T(0)) = {known_trace}  "
              f"[denom = {denom} = {factored_str(denom)}, max prime = {max_p}]")
    print(f"  Flat bands excluded: {flat_count}")
    print(f"{'=' * 76}")

    # Collect all prime menus
    all_gap_primes = set()
    primes_containing_topo = []

    print(f"\n  {'Class':>2}  {'Degree':>6}  {'Mult':>4}  "
          f"{'Min poly f(x)':30}  {'f(-3)':>12}  {'Primes':20}  {'Bridge?':10}")
    print(f"  {'-'*2}  {'-'*6}  {'-'*4}  {'-'*30}  {'-'*12}  {'-'*20}  {'-'*10}")

    for ci, cls in enumerate(classes):
        d_cls = cls['degree']
        mult = cls['mult']
        coeffs = cls['min_poly_coeffs']

        # Format minimal polynomial
        if coeffs:
            terms = []
            deg = len(coeffs) - 1
            for i, c in enumerate(coeffs):
                power = deg - i
                if c == 0:
                    continue
                c_str = str(c) if c >= 0 else f"({c})"
                if power == 0:
                    terms.append(c_str)
                elif power == 1:
                    if c == 1:
                        terms.append("x")
                    elif c == -1:
                        terms.append("-x")
                    else:
                        terms.append(f"{c}x")
                else:
                    if c == 1:
                        terms.append(f"x^{power}")
                    elif c == -1:
                        terms.append(f"-x^{power}")
                    else:
                        terms.append(f"{c}x^{power}")
            poly_str = " + ".join(terms).replace("+ -", "- ")
        else:
            poly_str = "?"

        # Compute f(-3)
        if coeffs:
            f_at_minus3 = eval_min_poly_at(coeffs, Fraction(-3))
            f_val = abs(f_at_minus3)
            f_str = str(f_at_minus3)
            primes_of_f = prime_factors(f_val.numerator) + prime_factors(f_val.denominator)
            primes_of_f = sorted(set(primes_of_f))
            primes_str = ", ".join(str(pp) for pp in primes_of_f) if primes_of_f else "1"
            all_gap_primes.update(primes_of_f)
            bridge = (p_topo in primes_of_f)
            bridge_str = f"YES (p={p_topo})" if bridge else "no"
            if bridge:
                primes_containing_topo.append(ci)
        else:
            f_val = None
            f_str = "?"
            primes_str = "?"
            bridge_str = "?"

        print(f"  {ci + 1:>2}  {d_cls:>6}  {mult:>4}  "
              f"{poly_str[:30]:30}  {f_str:>12}  {primes_str:20}  {bridge_str}")

    print(f"\n  All prime factors across all conjugacy classes: "
          f"{sorted(all_gap_primes)}")
    bridge_confirmed = p_topo in all_gap_primes
    print(f"  BRIDGE: topological prime p={p_topo} in algebraic prime menu? "
          f"{'YES' * bridge_confirmed or 'NO'}")
    print(f"  Classes where bridge fires: {primes_containing_topo or 'none'}")

    return bridge_confirmed, all_gap_primes


# ===========================================================================
# Full orbit analysis for one lattice type
# ===========================================================================

def analyze_lattice(lattice_name, H_builder, orbit_defs, flat_eigenvalue=FLAT_EIGENVALUE,
                    known_trace_constants=None, expected_flat_count=6):
    """
    Run full bridge analysis for one lattice.

    H_builder: function k_vec -> H (numpy array)
    orbit_defs: dict {orbit_name: (k_rep_n_tuple, orbit_size)}
    Returns dict of results.
    """
    print(f"\n\n{'#' * 76}")
    print(f"#  LATTICE: {lattice_name}")
    print(f"{'#' * 76}")

    results = {}
    prime_menu = set()
    bridge_status = {}

    for orbit_name, (n_tuple, orbit_size) in orbit_defs.items():
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = H_builder(k_vec)

        # Verify Hermitian
        herm_err = np.max(np.abs(H - H.conj().T))
        if herm_err > 1e-8:
            print(f"\nWARNING: H at {orbit_name} is not Hermitian! err={herm_err:.2e}")

        flat_count, classes, all_evals = analyze_k_point(
            H, flat_eigenvalue=flat_eigenvalue, flat_tol=FLAT_TOL)

        ktc = known_trace_constants.get(orbit_name) if known_trace_constants else None
        bridge_ok, orbit_primes = print_bridge_table(
            orbit_name, orbit_size, flat_count, classes, known_trace=ktc)

        results[orbit_name] = {
            'orbit_size': orbit_size,
            'flat_count': flat_count,
            'classes': classes,
            'bridge_ok': bridge_ok,
            'orbit_primes': orbit_primes,
            'all_evals': all_evals,
        }
        prime_menu.update(orbit_primes)
        bridge_status[orbit_name] = bridge_ok

    # Summary
    print(f"\n{'=' * 76}")
    print(f"  SUMMARY: {lattice_name}")
    print(f"{'=' * 76}")
    print(f"  Full prime menu across all orbits: {sorted(prime_menu)}")
    print(f"  Bridge status per orbit:")
    for oname, ok in bridge_status.items():
        d = orbit_defs[oname][1]
        p = d - 1
        print(f"    {oname:20s}  d={d}  p={p}  bridge={'CONFIRMED' if ok else 'ABSENT'}")

    all_bridges = all(bridge_status.values())
    print(f"  All bridges confirmed: {all_bridges}")

    return results, prime_menu


# ===========================================================================
# Known-data verification (BCC)
# ===========================================================================

def verify_known_trace_constants(bcc_results):
    """
    Cross-check: verify that the computed trace constants match known values.
    We compute Tr(P_k T(0)) numerically and compare to known fractions.
    """
    print(f"\n\n{'#' * 76}")
    print(f"#  VERIFICATION: k-orbit trace constants vs known data")
    print(f"{'#' * 76}")

    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)
    T_blocks = build_transfer_blocks(body_centers, bc_face_indices, A_full)
    T0 = T_blocks[(0, 0, 0)]

    print(f"\n  {'Orbit':20s}  {'Known':>20s}  {'Computed':>18s}  {'Match':>8s}")
    print(f"  {'-'*20}  {'-'*20}  {'-'*18}  {'-'*8}")

    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        k_vec = np.array([2 * np.pi * n / 3.0 for n in n_tuple])
        H = build_H_k(T_blocks, k_vec)
        evals, evecs = np.linalg.eigh(H)
        V = evecs[:, :6]  # flat band space
        tr_num = np.real(np.trace(V.conj().T @ T0 @ V))

        if orbit_name in KNOWN_TRACE_CONSTANTS:
            known = KNOWN_TRACE_CONSTANTS[orbit_name]
            err = abs(tr_num - float(known))
            match = err < 1e-8
            print(f"  {orbit_name:20s}  {str(known):>20s}  {tr_num:>18.12f}  "
                  f"{'OK' if match else 'FAIL':>8s}")
        else:
            frac, ok = to_fraction(tr_num)
            print(f"  {orbit_name:20s}  {'(unknown)':>20s}  {tr_num:>18.12f}  "
                  f"{'rational=' + str(frac) if ok else 'irrational':>20s}")


# ===========================================================================
# Denominator prime analysis for k-orbit trace constants
# ===========================================================================

def analyze_trace_denominators():
    """
    Table: orbit size d → p = d-1 → denominator of Tr(P_k T(0)) → max prime.
    """
    print(f"\n\n{'#' * 76}")
    print(f"#  TABLE: Orbit sizes → Projective primes → Trace denominators")
    print(f"{'#' * 76}")
    print(f"\n  {'Orbit':20s}  {'d':>4}  {'p=d-1':>6}  "
          f"{'Trace constant':>20}  {'Denom':>8}  {'Denom primes':>20}  {'Max prime':>10}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*6}  {'-'*20}  {'-'*8}  {'-'*20}  {'-'*10}")

    for orbit_name, (n_tuple, orbit_size) in ORBIT_DEFS.items():
        d, p, denom_primes, max_p = k_orbit_primes(orbit_name, orbit_size)
        tc = KNOWN_TRACE_CONSTANTS.get(orbit_name, None)
        tc_str = str(tc) if tc is not None else "unknown"
        denom = tc.denominator if tc is not None else "?"
        dp_str = "·".join(str(pp) for pp in denom_primes) if denom_primes else "1"
        max_p_str = str(max_p) if max_p is not None else "?"
        bridge_check = (max_p == p) if (max_p is not None and p != 0) else "—"
        print(f"  {orbit_name:20s}  {d:>4}  {p:>6}  {tc_str:>20}  "
              f"{str(denom):>8}  {dp_str:>20}  {max_p_str:>10}  "
              f"{'← bridge' if bridge_check is True else ''}")


# ===========================================================================
# Comparison: SC and FCC
# ===========================================================================

SC_ORBIT_DEFS = {
    # In a 3x3x3 supercell, SC k-orbits under Oh:
    # same k-point structure as BCC — 27 k-points, same 4 orbits
    'Gamma':         ((0, 0, 0), 1),
    'axis':          ((1, 0, 0), 6),
    'face-diagonal': ((1, 1, 0), 12),
    'body-diagonal': ((1, 1, 1), 8),
}

FCC_ORBIT_DEFS = {
    'Gamma':         ((0, 0, 0), 1),
    'axis':          ((1, 0, 0), 6),
    'face-diagonal': ((1, 1, 0), 12),
    'body-diagonal': ((1, 1, 1), 8),
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 76)
    print("  BRIDGE VERIFICATION: Spectral Gap Norms <-> Projective Line Characteristics")
    print("  BCC 3x3x3 primary; SC and FCC as falsification tests")
    print("=" * 76)

    # -----------------------------------------------------------------------
    # Step 1: Denominator analysis (pure number theory, no matrix computation)
    # -----------------------------------------------------------------------
    analyze_trace_denominators()

    # -----------------------------------------------------------------------
    # Step 2: BCC — main verification
    # -----------------------------------------------------------------------
    bcc_results, bcc_prime_menu = analyze_lattice(
        lattice_name="BCC (3x3x3 periodic, 20x20 Bloch matrix)",
        H_builder=build_bcc_bloch,
        orbit_defs=ORBIT_DEFS,
        flat_eigenvalue=FLAT_EIGENVALUE,
        known_trace_constants=KNOWN_TRACE_CONSTANTS,
        expected_flat_count=6,
    )

    # -----------------------------------------------------------------------
    # Step 3: Verify trace constants numerically
    # -----------------------------------------------------------------------
    verify_known_trace_constants(bcc_results)

    # -----------------------------------------------------------------------
    # Step 4: SC — falsification test
    # -----------------------------------------------------------------------
    print(f"\n\nFALSIFICATION TEST: Simple Cubic lattice")
    print(f"  If primes come from orbits, SC should give the SAME prime menu")
    print(f"  (same orbit sizes) but possibly different gap norms.")

    sc_results, sc_prime_menu = analyze_lattice(
        lattice_name="SC (3x3x3 supercell, 27x27 Bloch matrix)",
        H_builder=build_sc_bloch_v2,
        orbit_defs=SC_ORBIT_DEFS,
        flat_eigenvalue=None,   # SC has no flat bands generically
        known_trace_constants=None,
        expected_flat_count=0,
    )

    # -----------------------------------------------------------------------
    # Step 5: FCC — falsification test
    # -----------------------------------------------------------------------
    print(f"\n\nFALSIFICATION TEST: FCC lattice")
    print(f"  FCC has 12 NN; same 3x3x3 supercell → same k-orbit structure.")

    fcc_results, fcc_prime_menu = analyze_lattice(
        lattice_name="FCC (3x3x3 supercell, 27x27 Bloch matrix)",
        H_builder=build_fcc_bloch,
        orbit_defs=FCC_ORBIT_DEFS,
        flat_eigenvalue=None,
        known_trace_constants=None,
        expected_flat_count=0,
    )

    # -----------------------------------------------------------------------
    # Step 6: Comparative summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'#' * 76}")
    print(f"#  COMPARATIVE SUMMARY: Prime Menus by Lattice")
    print(f"{'#' * 76}")
    print(f"\n  BCC prime menu: {sorted(bcc_prime_menu)}")
    print(f"  SC  prime menu: {sorted(sc_prime_menu)}")
    print(f"  FCC prime menu: {sorted(fcc_prime_menu)}")
    print(f"\n  All three lattices share the SAME k-orbit structure (Gamma/axis/face/body).")
    print(f"  The projective prime menu {{2,3,5,7,11}} is predicted by orbit sizes alone.")
    print(f"  If BCC prime menu = SC = FCC ⊆ {{2,3,5,7,11}}, the bridge is topological.")
    print(f"  If menus differ, the prime source is lattice-specific (algebraic, not topological).")
    print(f"\n  BCC menu ⊆ {{2,3,5,7,11}}: {bcc_prime_menu.issubset({2,3,5,7,11})}")
    print(f"  SC  menu ⊆ {{2,3,5,7,11}}: {sc_prime_menu.issubset({2,3,5,7,11})}")
    print(f"  FCC menu ⊆ {{2,3,5,7,11}}: {fcc_prime_menu.issubset({2,3,5,7,11})}")
    print(f"\n  BCC == SC == FCC menu: "
          f"{bcc_prime_menu == sc_prime_menu == fcc_prime_menu}")
    print(f"\n  Bridge claim (orbit size → prime) is a TOPOLOGICAL invariant")
    print(f"  if and only if ALL three lattices share the same prime menu.")

    print(f"\n{'=' * 76}")
    print(f"  END OF BRIDGE VERIFICATION")
    print(f"{'=' * 76}")


if __name__ == "__main__":
    main()
