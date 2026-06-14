# -*- coding: utf-8 -*-
"""
prime_generation_L_scan.py

BCC lattice prime generation scan: as L grows from L_min to L_max,
which NEW primes appear in the gap product factorizations?

Correct algorithm (verified against existing bridge scan for L=3):
  For each k-point (O_h orbit representative):
    1. Diagonalize H(k), exclude flat eigenvalues (lambda ~ -3)
    2. Partition remaining eigenvalues into Galois conjugacy classes over Q
       (each class has a minimal polynomial f(x) with rational coefficients)
    3. Evaluate f(-3) for each class, factor numerator and denominator
    4. Collect all prime factors as "gap primes" emitted at this k-point

Key optimization: T(R) transfer blocks are INDEPENDENT of L for L >= 3.
Build them once from L=3, reuse for all larger L (different k-grid, same T).

Usage:
    python prime_generation_L_scan.py [--L-min 3] [--L-max 50] [--verbose]
"""

import sys
import io
import itertools
import time
import argparse
from fractions import Fraction
from collections import defaultdict
from math import gcd

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

try:
    import mpmath
    MPMATH_OK = True
except ImportError:
    MPMATH_OK = False

SYMPY_OK = False

# ===========================================================================
# Constants
# ===========================================================================

FLAT_EIGENVALUE = -3.0
FLAT_TOL = 1e-6
TOL = 1e-9


# ===========================================================================
# O_h group: 48 signed permutation matrices
# ===========================================================================

def build_oh_ops():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            mat = np.zeros((3, 3), dtype=int)
            for col, (row, s) in enumerate(zip(perm, signs)):
                mat[row, col] = s
            ops.append(mat)
    assert len(ops) == 48
    return ops


OH_OPS = build_oh_ops()


# ===========================================================================
# BCC lattice construction (general L)
# ===========================================================================

def build_bcc_lattice(L):
    return list(itertools.product(range(L), repeat=3))


def enumerate_simplex_faces(bc_ijk, L):
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)
    orig_corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = i + dx, j + dy, k + dz
        wx, wy, wz = ox % L, oy % L, oz % L
        orig_corners.append(((ox + oy + oz) % 2, ('c', wx, wy, wz)))
    even_corners = [v for (par, v) in orig_corners if par == 0]
    odd_corners  = [v for (par, v) in orig_corners if par == 1]
    faces = []
    for tet_corners in (even_corners, odd_corners):
        simplex_verts = [bc_v] + tet_corners
        for combo in itertools.combinations(simplex_verts, 3):
            faces.append(frozenset(combo))
    return faces


def translate_vertex(v, shift, L):
    si, sj, sk = shift
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i + si) % L, (j + sj) % L, (k + sk) % L)
    else:
        _, cx, cy, cz = v
        return ('c', (cx + si) % L, (cy + sj) % L, (cz + sk) % L)


def translate_face(face, shift, L):
    return frozenset(translate_vertex(v, shift, L) for v in face)


def build_all_faces(body_centers, L):
    ref_bc = (0, 0, 0)
    ref_faces = enumerate_simplex_faces(ref_bc, L)
    assert len(ref_faces) == 20

    face_to_idx = {}
    bc_face_indices = []
    global_idx = 0

    for bc_ijk in body_centers:
        local_indices = []
        for ref_face in ref_faces:
            shifted_face = translate_face(ref_face, bc_ijk, L)
            if shifted_face not in face_to_idx:
                face_to_idx[shifted_face] = global_idx
                global_idx += 1
            local_indices.append(face_to_idx[shifted_face])
        bc_face_indices.append(local_indices)

    return face_to_idx, bc_face_indices, ref_faces


def build_adjacency_matrix(face_to_idx):
    N = len(face_to_idx)
    A = np.zeros((N, N), dtype=float)
    all_faces_list = [None] * N
    for face, idx in face_to_idx.items():
        all_faces_list[idx] = face

    vertex_to_faces = defaultdict(set)
    for face, idx in face_to_idx.items():
        for v in face:
            vertex_to_faces[v].add(idx)

    candidate_pairs = set()
    for fset in vertex_to_faces.values():
        flist = sorted(fset)
        for a in range(len(flist)):
            for b in range(a + 1, len(flist)):
                candidate_pairs.add((flist[a], flist[b]))

    for i, j in candidate_pairs:
        if len(all_faces_list[i] & all_faces_list[j]) == 2:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


def build_transfer_blocks(body_centers, bc_face_indices, A_full, L):
    """
    Extract 20x20 transfer blocks T(R) for R in {-1,0,1}^3 (for L=3).
    For L=3: R = (r+1)%3-1 maps {0,1,2} -> {1,2,0} -> {0,1,-1}.
    Wait: (0+1)%3-1=0, (1+1)%3-1=1, (2+1)%3-1=-1. Correct: {-1,0,1}.
    """
    n_bc = len(body_centers)
    reorder = []
    for bc_idx in range(n_bc):
        reorder.extend(bc_face_indices[bc_idx])
    A_reordered = A_full[np.ix_(reorder, reorder)]

    T = {}
    for j, bc_ijk in enumerate(body_centers):
        # Map coordinates to displacement relative to (0,0,0)
        Rc = tuple((r + 1) % L - 1 for r in bc_ijk)
        block = A_reordered[:20, j * 20:(j + 1) * 20]
        if Rc not in T:
            T[Rc] = block.copy()
        else:
            T[Rc] = T[Rc] + block
    return T


def build_transfer_blocks_L3():
    """
    Build the canonical T(R) blocks from L=3 BCC lattice.
    R in {-1,0,1}^3 (27 blocks, some may be zero).
    These blocks encode hopping in real space and are L-independent.
    Reused for all L >= 3 via H(k) = sum_R T[R] * exp(i k.R).
    """
    L = 3
    body_centers = build_bcc_lattice(L)
    face_to_idx, bc_face_indices, _ = build_all_faces(body_centers, L)
    A_full = build_adjacency_matrix(face_to_idx)
    T = build_transfer_blocks(body_centers, bc_face_indices, A_full, L)
    return T


# ===========================================================================
# H(k) from T(R) blocks
# ===========================================================================

def build_H_k(T, k_vec):
    """H(k) = sum_R T[R] * exp(i k.R), 20x20 complex Hermitian."""
    H = np.zeros((20, 20), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, np.array(R, dtype=float)))
        H += phase * block
    return H


# ===========================================================================
# O_h orbit enumeration for a given L
# ===========================================================================

def compute_oh_orbits(L):
    """
    Partition (Z/LZ)^3 into O_h orbits.
    Returns list of (rep, frozenset_of_orbit_members), sorted by rep.
    """
    remaining = set(itertools.product(range(L), repeat=3))
    orbits = []
    while remaining:
        rep = min(remaining)
        orbit = set()
        n = np.array(rep, dtype=int)
        for M in OH_OPS:
            rotated = tuple(int(x) % L for x in M @ n)
            orbit.add(rotated)
        orbits.append((rep, frozenset(orbit)))
        remaining -= orbit
    orbits.sort(key=lambda x: x[0])
    return orbits


# ===========================================================================
# Prime factorization
# ===========================================================================

def prime_factors_dict(n):
    """Return dict {prime: exponent} for |n|. Returns {} for |n| <= 1."""
    n = abs(int(n))
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def prime_factors_set(n, bound=10**6):
    """Return set of prime factors of |n| up to bound. Ignores large cofactors."""
    facs, _ = prime_factors_bounded(abs(int(n)), bound)
    return set(facs.keys())


def prime_factors_bounded(n, bound=10**6):
    """
    Trial division up to bound. Returns (factors_dict, cofactor).
    If loop exits because d*d > n, remaining n is proven prime → included in factors.
    If loop exits because d > bound, remaining n is unfactored cofactor.
    """
    n = abs(int(n))
    if n <= 1:
        return {}, 1
    factors = {}
    d = 2
    while d * d <= n and d <= bound:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        if d * d > n:
            factors[n] = factors.get(n, 0) + 1
            n = 1
    cofactor = n if n > 1 else 1
    return factors, cofactor


def factorint_robust(n):
    """Factor |n| with bounded trial division. Never hangs."""
    facs, cofactor = prime_factors_bounded(abs(int(n)))
    if cofactor > 1:
        facs[cofactor] = 1
    return facs


def factored_str(n):
    n = abs(int(n))
    if n <= 1:
        return str(n)
    facs = factorint_robust(n)
    return " * ".join(
        f"{p}^{e}" if e > 1 else str(p)
        for p, e in sorted(facs.items())
    )


# ===========================================================================
# Raw gap product per orbit (no Galois eigenvalue classification needed)
# ===========================================================================

def collect_raw_gap_product(orbit_rep, T, L):
    """
    Compute raw gap product = product(lambda_i + 3) for non-flat eigenvalues
    at k = 2*pi*orbit_rep/L.

    Returns (gap_product_float, n_flat, n_nonflat).
    """
    k_vec = np.array([2.0 * np.pi * float(n) / L for n in orbit_rep])
    H = build_H_k(T, k_vec)
    evals = np.linalg.eigvalsh(H)

    non_flat = [e for e in evals if abs(e - FLAT_EIGENVALUE) >= FLAT_TOL]
    n_flat = len(evals) - len(non_flat)

    if not non_flat:
        return 0.0, n_flat, 0

    product = 1.0
    for e in non_flat:
        product *= (e + 3.0)

    return product, n_flat, len(non_flat)


def collect_raw_gap_product_mp(orbit_rep, T, L, dps=50):
    """
    Same as collect_raw_gap_product but using mpmath for higher precision.
    Returns (mpf gap_product, n_flat, n_nonflat).
    """
    k_vec = np.array([2.0 * np.pi * float(n) / L for n in orbit_rep])
    H = build_H_k(T, k_vec)
    evals = np.linalg.eigvalsh(H)

    non_flat = [e for e in evals if abs(e - FLAT_EIGENVALUE) >= FLAT_TOL]
    n_flat = len(evals) - len(non_flat)

    if not non_flat:
        return mpmath.mpf(0), n_flat, 0

    with mpmath.workdps(dps):
        product = mpmath.mpf(1)
        for e in non_flat:
            product *= mpmath.mpf(e) + 3

    return product, n_flat, len(non_flat)


def try_as_rational(x, tol=1e-6):
    """
    Try to identify float x as a rational number.
    Strategy: integer first, then small-denominator fractions.
    NEVER use limit_denominator with huge limits on near-integer floats
    (parasitic prime factors from binary representation).
    Returns (Fraction, ok).
    """
    if abs(x) < tol:
        return Fraction(0), True
    # Try as integer
    rounded = round(x)
    if rounded != 0 and abs(x - rounded) < tol * max(1.0, abs(rounded)):
        return Fraction(int(rounded)), True
    # Try as fraction with small denominator (powers of 2,3 up to ~10000)
    for denom_limit in [100, 1000, 10000, 100000]:
        f = Fraction(x).limit_denominator(denom_limit)
        if abs(float(f) - x) < tol * max(1.0, abs(x)):
            return f, True
    return None, False


# ===========================================================================
# Per-L analysis: raw gap product + Galois orbit norm
# ===========================================================================

def analyze_L(L, T, verbose=False):
    """
    Strategy:
    1. Compute raw gap product for each O_h orbit (fast, no backtracking)
    2. Try each product as rational — if yes, factor directly
    3. For irrational products: group O_h orbits into Galois classes,
       multiply products across class (= norm to Q), factor the norm
    4. mpmath used for precision when Galois class size > 1
    """
    t0 = time.time()

    orbits = compute_oh_orbits(L)

    # Step 1: raw gap products
    orbit_products = []  # float64
    orbit_flats = []
    for rep, orb in orbits:
        prod, n_flat, n_nf = collect_raw_gap_product(rep, T, L)
        orbit_products.append(prod)
        orbit_flats.append(n_flat)

    # Step 2: build Galois classes of O_h orbits
    units = [a for a in range(1, L) if gcd(a, L) == 1]
    orbit_map = {}
    for idx, (rep, orb) in enumerate(orbits):
        for pt in orb:
            orbit_map[pt] = idx

    visited = [False] * len(orbits)
    galois_classes = []  # list of lists of orbit indices
    for i in range(len(orbits)):
        if visited[i]:
            continue
        gclass = {i}
        rep_i = orbits[i][0]
        for a in units:
            scaled = tuple((a * x) % L for x in rep_i)
            j = orbit_map.get(scaled)
            if j is not None:
                gclass.add(j)
        gclass = sorted(gclass)
        for idx in gclass:
            visited[idx] = True
        galois_classes.append(gclass)

    # Step 3: extract primes from each Galois class
    all_primes = set()
    n_rational_orbits = 0
    n_galois_norm_ok = 0
    n_precision_fail = 0

    for gclass in galois_classes:
        if len(gclass) == 1:
            # Single orbit = Galois-trivial => gap product should be rational
            idx = gclass[0]
            prod = orbit_products[idx]
            frac, ok = try_as_rational(prod)
            if ok and frac != 0:
                n_rational_orbits += 1
                all_primes.update(prime_factors_set(abs(frac.numerator)))
                if frac.denominator > 1:
                    all_primes.update(prime_factors_set(frac.denominator))
            elif abs(prod) > 1e-10:
                # Not rational despite singleton class — try harder
                frac2 = Fraction(prod).limit_denominator(10**15)
                if abs(float(frac2) - prod) < 1e-4 * max(1, abs(prod)):
                    n_rational_orbits += 1
                    all_primes.update(prime_factors_set(abs(frac2.numerator)))
                    if frac2.denominator > 1:
                        all_primes.update(prime_factors_set(frac2.denominator))
                else:
                    n_precision_fail += 1
        else:
            # Multi-orbit Galois class: multiply products => norm => rational
            class_size = len(gclass)

            if MPMATH_OK:
                # Recompute with mpmath for precision
                dps = max(50, class_size * 20)
                with mpmath.workdps(dps):
                    norm = mpmath.mpf(1)
                    for idx in gclass:
                        rep = orbits[idx][0]
                        mp_prod, _, _ = collect_raw_gap_product_mp(rep, T, L, dps)
                        norm *= mp_prod

                    # Try integer first, then small-denominator fraction
                    norm_rounded = int(mpmath.nint(norm))
                    err = abs(norm - norm_rounded)
                    if err < mpmath.mpf(0.5):
                        n_galois_norm_ok += 1
                        if abs(norm_rounded) > 1:
                            primes = set(factorint_robust(abs(norm_rounded)).keys())
                            all_primes.update(primes)
                        if verbose:
                            print(f"    Galois class size={class_size}: "
                                  f"norm={norm_rounded} = {factored_str(abs(norm_rounded))}")
                    else:
                        n_precision_fail += 1
                        if verbose:
                            print(f"    Galois class size={class_size}: "
                                  f"norm precision fail, err={float(err):.2e}")
            else:
                # No mpmath: try float64 multiplication (lossy for large classes)
                norm_f = 1.0
                for idx in gclass:
                    norm_f *= orbit_products[idx]
                frac, ok = try_as_rational(norm_f, tol=1e-3)
                if ok and frac != 0:
                    n_galois_norm_ok += 1
                    all_primes.update(prime_factors_set(abs(frac.numerator)))
                    if frac.denominator > 1:
                        all_primes.update(prime_factors_set(frac.denominator))
                else:
                    n_precision_fail += 1

    t1 = time.time()

    if verbose:
        print(f"  L={L}: {len(orbits)} orbits, {len(galois_classes)} Galois classes, "
              f"{n_rational_orbits} rational, {n_galois_norm_ok} norm-ok, "
              f"{n_precision_fail} precision-fail")

    return {
        'L': L,
        'n_orbits': len(orbits),
        'n_galois_classes': len(galois_classes),
        'primes': all_primes,
        'time': t1 - t0,
        'n_rational_orbits': n_rational_orbits,
        'n_galois_norm_ok': n_galois_norm_ok,
        'n_precision_fail': n_precision_fail,
    }


# ===========================================================================
# Sieve: primes up to bound
# ===========================================================================

def primes_up_to(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BCC prime generation L-scan: track which primes appear as L grows."
    )
    parser.add_argument('--L-min', type=int, default=3,
                        help='Minimum L value (default: 3)')
    parser.add_argument('--L-max', type=int, default=50,
                        help='Maximum L value (default: 50)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-orbit details')
    args = parser.parse_args()

    L_min = args.L_min
    L_max = args.L_max

    output_lines = []

    def emit(s=""):
        print(s)
        output_lines.append(s)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    emit("=" * 70)
    emit("=== PRIME GENERATION L-SCAN ===")
    emit(f"  BCC 20x20 Bloch Hamiltonian, L from {L_min} to {L_max}")
    emit(f"  T(R) blocks built once from L=3, reused for all L")
    emit(f"  Gap primes: prime factors of f(-3) for each Galois eigenvalue class")
    emit("=" * 70)

    # -----------------------------------------------------------------------
    # Build T(R) once from L=3
    # -----------------------------------------------------------------------
    emit("")
    emit("Building T(R) from L=3...")
    t_build = time.time()
    T = build_transfer_blocks_L3()
    t_build_done = time.time()

    n_nonzero = sum(1 for b in T.values() if np.any(b != 0))
    max_entries = max(int(np.sum(b != 0)) for b in T.values()) if T else 0
    emit(f"T(R) built in {t_build_done - t_build:.2f}s: "
         f"{len(T)} blocks, {n_nonzero} nonzero, "
         f"max nonzero entries per block: {max_entries}")

    # -----------------------------------------------------------------------
    # Verify L=3 baseline
    # -----------------------------------------------------------------------
    emit("")
    emit("Verifying L=3 baseline (expect gap primes = {2,3,5,7,11})...")
    test_res = analyze_L(3, T, verbose=args.verbose)
    expected_L3 = {2, 3, 5, 7, 11}
    match = expected_L3 <= test_res['primes']
    extra = test_res['primes'] - expected_L3
    emit(f"  Found: {sorted(test_res['primes'])}")
    if match and not extra:
        emit(f"  PASS: exactly matches expected {sorted(expected_L3)}")
    elif match:
        emit(f"  PASS (superset): expected subset present, extra primes: {sorted(extra)}")
    else:
        missing = expected_L3 - test_res['primes']
        emit(f"  WARNING: missing expected primes {sorted(missing)}, "
             f"found: {sorted(test_res['primes'])}")
    emit("")

    # -----------------------------------------------------------------------
    # Main scan
    # -----------------------------------------------------------------------
    emit("-" * 70)
    emit(f"{'L':<6}  {'orbits':<8}  {'Gcls':<6}  "
         f"{'rat':<5}  {'norm':<5}  {'fail':<5}  "
         f"{'primes at this L':<40}  {'new primes':<30}  {'time':<8}")
    emit("-" * 70)

    all_results = {}
    cumulative_primes = set()
    first_appearance = {}  # prime -> first L where it appeared

    for L in range(L_min, L_max + 1):
        if args.verbose:
            print(f"\n--- L={L} ---")

        result = analyze_L(L, T, verbose=args.verbose)
        all_results[L] = result

        new_primes = result['primes'] - cumulative_primes
        for p in new_primes:
            if p not in first_appearance:
                first_appearance[p] = L
        cumulative_primes.update(result['primes'])

        primes_str = "{" + ",".join(str(p) for p in sorted(result['primes'])) + "}"
        new_str = "{" + ",".join(str(p) for p in sorted(new_primes)) + "}"

        line = (f"L={L:<4}  orbits={result['n_orbits']:<5}  "
                f"Gcls={result['n_galois_classes']:<4}  "
                f"rat={result['n_rational_orbits']:<4}  "
                f"norm={result['n_galois_norm_ok']:<4}  "
                f"fail={result['n_precision_fail']:<4}  "
                f"primes={primes_str:<40}  "
                f"new={new_str:<30}  "
                f"time={result['time']:.2f}s")
        emit(line)

    # -----------------------------------------------------------------------
    # Prime coverage table
    # -----------------------------------------------------------------------
    P_bound = max(L_max + 30, 100)
    all_small_primes = primes_up_to(P_bound)

    emit("")
    emit("=" * 70)
    emit("=== PRIME COVERAGE TABLE ===")
    emit("")

    for p in all_small_primes:
        if p <= L_max + 20:  # only show reasonably small primes
            if p in first_appearance:
                emit(f"  p={p:<5}  first at L={first_appearance[p]}")
            else:
                emit(f"  p={p:<5}  NOT FOUND (L up to {L_max})")

    # Also report any very large primes found
    large_primes = sorted(p for p in first_appearance if p > P_bound)
    if large_primes:
        emit("")
        emit(f"  Large primes also found (p > {P_bound}):")
        for p in large_primes[:20]:
            emit(f"    p={p}  first at L={first_appearance[p]}")
        if len(large_primes) > 20:
            emit(f"    ... and {len(large_primes) - 20} more")

    # -----------------------------------------------------------------------
    # Missing primes
    # -----------------------------------------------------------------------
    missing_bound = max(L_max, 100)
    missing_primes = [p for p in primes_up_to(missing_bound) if p not in first_appearance]

    emit("")
    emit("=" * 70)
    emit(f"=== MISSING PRIMES <= {missing_bound} (not found in L={L_min}..{L_max}) ===")
    if missing_primes:
        emit(f"  {missing_primes}")
    else:
        emit(f"  (none -- all primes <= {missing_bound} appeared)")

    # -----------------------------------------------------------------------
    # Pattern analysis
    # -----------------------------------------------------------------------
    emit("")
    emit("=" * 70)
    emit("=== PRIME GENERATION PATTERN ===")
    emit("")

    pattern_at_p = []
    pattern_at_p1 = []
    pattern_at_pm1 = []
    pattern_other = []

    for p in sorted(p for p in first_appearance if p >= 2):
        L_first = first_appearance[p]
        if L_first == p:
            pattern_at_p.append(p)
        elif L_first == p + 1:
            pattern_at_p1.append(p)
        elif L_first == p - 1:
            pattern_at_pm1.append(p)
        else:
            pattern_other.append((p, L_first))

    if pattern_at_p:
        emit(f"  Primes p with first appearance at L=p:   {pattern_at_p}")
    if pattern_at_p1:
        emit(f"  Primes p with first appearance at L=p+1: {pattern_at_p1}")
    if pattern_at_pm1:
        emit(f"  Primes p with first appearance at L=p-1: {pattern_at_pm1}")
    if pattern_other:
        emit(f"  Other (p, first_L):")
        for p, L_f in sorted(pattern_other):
            if p < 200:  # only small primes
                ratio = L_f / p
                emit(f"    p={p:<6}  first_L={L_f:<5}  L/p={ratio:.3f}")

    # Cumulative count progression
    emit("")
    emit("  Cumulative prime count by L:")
    prev_count = 0
    for L in range(L_min, L_max + 1):
        n_by_L = sum(1 for fl in first_appearance.values() if fl <= L)
        if n_by_L != prev_count:
            new_at_L = sorted(p for p, fl in first_appearance.items() if fl == L)
            emit(f"    L={L:<4}: {n_by_L:3d} distinct primes total  (+{new_at_L})")
            prev_count = n_by_L

    # Summary stats
    emit("")
    small_primes_in_range = [p for p in primes_up_to(L_max) if p >= 2]
    found_in_range = [p for p in small_primes_in_range if p in first_appearance]
    emit(f"  Primes <= L_max={L_max}: {len(small_primes_in_range)} total, "
         f"{len(found_in_range)} found "
         f"({100*len(found_in_range)/max(1,len(small_primes_in_range)):.1f}%)")
    emit(f"  Total distinct primes found (any size): {len(first_appearance)}")

    # Check if all primes <= L_max appear
    all_covered = all(p in first_appearance for p in small_primes_in_range)
    if all_covered:
        emit(f"  ALL primes <= {L_max} found within this scan range.")
    else:
        missing_in_range = [p for p in small_primes_in_range if p not in first_appearance]
        emit(f"  Missing primes <= {L_max}: {missing_in_range}")

    emit("")
    emit("=" * 70)
    emit("=== END ===")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "prime_generation_L_scan_results.txt")
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\nResults saved to: {results_path}")
    except Exception as e:
        print(f"\nWARNING: Could not save results: {e}")


if __name__ == "__main__":
    main()
