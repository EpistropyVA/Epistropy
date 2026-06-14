# -*- coding: utf-8 -*-
"""
verify_bridge_norm_L_scan.py

Extends the bridge verification (spectral gap norms <-> projective line
characteristics) to arbitrary BCC lattice sizes L.

For each L in {1, 2, 3, 4, 5}:
  - Build BCC L×L×L with L³ body-centers, 20L³ faces (before sharing collapse)
  - Compute the 20×20 Bloch matrix H(k) at each k = 2pi*(n1,n2,n3)/L
  - Classify k-points into O_h orbits (enumerate all 48 operations)
  - For each orbit representative:
      * Extract non-flat eigenvalues (flat = -3)
      * Galois-conjugacy classify, compute minimal poly f(x)
      * Evaluate f(-3); factor; check p = orbit_size - 1 divides |f(-3)|
  - Print per-orbit bridge table and cross-L summary

Usage:
    python verify_bridge_norm_L_scan.py [--L 1,2,3,4,5] [--no-detail]
"""

import itertools
import sys
import io
import time
import argparse
from fractions import Fraction
from collections import defaultdict

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

# ---------------------------------------------------------------------------
# Sympy (optional, for exact minimal polynomials)
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    SYMPY_OK = _ilu.find_spec("sympy") is not None
except Exception:
    SYMPY_OK = False

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOL = 1e-9
FLAT_EIGENVALUE = -3.0
FLAT_TOL = 1e-6


# ===========================================================================
# O_h symmetry group (48 operations acting on integer triples mod L)
# ===========================================================================

def _oh_generators():
    """Return the 48 matrices of the Oh point group (as integer 3x3 arrays)."""
    ops = set()
    # Generate all signed permutations: 3! * 2^3 = 48
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            mat = np.zeros((3, 3), dtype=int)
            for col, (row, s) in enumerate(zip(perm, signs)):
                mat[row, col] = s
            ops.add(tuple(mat.flatten()))
    mats = [np.array(op, dtype=int).reshape(3, 3) for op in ops]
    assert len(mats) == 48, f"Oh should have 48 ops, got {len(mats)}"
    return mats

OH_OPS = _oh_generators()


def k_orbit(n_tuple, L):
    """
    Compute the O_h orbit of k-point n_tuple = (n1,n2,n3) in {0,...,L-1}^3.
    Returns frozenset of tuples (canonical mod L in [0,L)).
    """
    orbit = set()
    n_arr = np.array(n_tuple, dtype=int)
    for M in OH_OPS:
        rotated = M @ n_arr
        # Map to canonical representative in [0, L)^3
        canonical = tuple(int(x % L) for x in rotated)
        orbit.add(canonical)
    return frozenset(orbit)


def compute_all_orbits(L):
    """
    Enumerate all k-points in {0,...,L-1}^3 and partition into O_h orbits.
    Returns list of (representative_tuple, orbit_frozenset).
    """
    all_kpts = set(itertools.product(range(L), repeat=3))
    remaining = set(all_kpts)
    orbits = []
    while remaining:
        # Pick lexicographically smallest as representative
        rep = min(remaining)
        orb = k_orbit(rep, L)
        orbits.append((rep, orb))
        remaining -= orb
    # Sort by representative
    orbits.sort(key=lambda x: x[0])
    return orbits


# ===========================================================================
# BCC lattice construction (general L)
# ===========================================================================

def build_bcc_lattice_L(L):
    """Return list of body-center indices (i,j,k) for i,j,k in {0,...,L-1}."""
    return list(itertools.product(range(L), repeat=3))


def enumerate_simplex_faces_L(bc_ijk, L):
    """
    Enumerate the 20 triangular faces of the stella octangula at body-center bc_ijk.
    Corner parity is (x+y+z) % 2 of the *original* (unwrapped) coordinates,
    since parity is a global invariant.
    Corners wrap mod L.
    Returns list of 20 frozensets, each containing 3 vertex tuples.
    """
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


def translate_vertex_L(v, shift_ijk, L):
    si, sj, sk = shift_ijk
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i + si) % L, (j + sj) % L, (k + sk) % L)
    else:
        _, cx, cy, cz = v
        return ('c', (cx + si) % L, (cy + sj) % L, (cz + sk) % L)


def translate_face_L(face, shift_ijk, L):
    return frozenset(translate_vertex_L(v, shift_ijk, L) for v in face)


def build_all_faces_L(body_centers, L):
    """
    Build all faces for the L×L×L BCC lattice.
    Returns:
      face_to_idx: dict { frozenset_face -> global_index }
      bc_face_indices: list of lists; bc_face_indices[bc_idx] = list of 20 face indices
      ref_faces: 20 faces of the (0,0,0) body-center (before wrapping/sharing)
    """
    ref_bc = (0, 0, 0)
    ref_faces = enumerate_simplex_faces_L(ref_bc, L)
    assert len(ref_faces) == 20, f"Expected 20 faces, got {len(ref_faces)}"

    face_to_idx = {}
    bc_face_indices = []
    global_idx = 0

    for bc_ijk in body_centers:
        local_indices = []
        for ref_face in ref_faces:
            shifted_face = translate_face_L(ref_face, bc_ijk, L)
            if shifted_face not in face_to_idx:
                face_to_idx[shifted_face] = global_idx
                global_idx += 1
            local_indices.append(face_to_idx[shifted_face])
        bc_face_indices.append(local_indices)

    return face_to_idx, bc_face_indices, ref_faces


def build_adjacency_matrix_L(face_to_idx):
    """Build the full face-adjacency matrix (edge-sharing: exactly 2 shared vertices)."""
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


def build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L):
    """
    Extract 20×20 transfer blocks T(R) from the full adjacency matrix,
    where R is the lattice displacement between body-centers.

    For general L, R = (bc_ijk[0] - 0, ...) but we need the *minimal image*
    displacement, i.e. the shift that maps bc_0=(0,0,0) to bc_ijk.
    Since body-centers are at integer coordinates and wrap mod L, the displacement
    vector R lies in {-(L//2),...,L//2}^3 (using minimum image convention is not
    strictly needed for the Bloch transform, but the representation R -> T(R) must
    be consistent: T(R) accumulates hops FROM cell 0 TO cell at displacement R).
    """
    n_bc = len(body_centers)
    reorder = []
    for bc_idx in range(n_bc):
        reorder.extend(bc_face_indices[bc_idx])
    A_reordered = A_full[np.ix_(reorder, reorder)]

    T = {}
    for j, bc_ijk in enumerate(body_centers):
        # Displacement from (0,0,0) to bc_ijk, interpreted as lattice vector
        # For Bloch transform: R = bc_ijk as a displacement vector
        # (coordinates are already in {0,...,L-1} but we keep them as-is;
        # the phase is exp(i k . R) where k = 2pi*n/L)
        Rc = tuple(bc_ijk)
        block = A_reordered[:20, j * 20:(j + 1) * 20]
        if Rc not in T:
            T[Rc] = block.copy()
        else:
            T[Rc] = T[Rc] + block
    return T


def build_H_k_L(T, k_vec):
    """Build 20×20 Bloch Hamiltonian H(k) from transfer blocks."""
    H = np.zeros((20, 20), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, np.array(R, dtype=float)))
        H += phase * block
    return H


def build_bcc_bloch_L(k_vec, L, T_cache=None):
    """
    Full pipeline: build BCC 20×20 Bloch Hamiltonian at k_vec for L×L×L lattice.
    T_cache: if provided, dict {L: T_blocks} to avoid rebuilding.
    """
    if T_cache is not None and L in T_cache:
        T = T_cache[L]
    else:
        body_centers = build_bcc_lattice_L(L)
        face_to_idx, bc_face_indices, _ = build_all_faces_L(body_centers, L)
        A_full = build_adjacency_matrix_L(face_to_idx)
        T = build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L)
        if T_cache is not None:
            T_cache[L] = T
    return build_H_k_L(T, k_vec)


# ===========================================================================
# Algebraic helpers (shared with original script)
# ===========================================================================

def prime_factors(n):
    """Return sorted list of distinct prime factors of |n|."""
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
    """Return prime factorization string of integer n."""
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
    f = Fraction(x).limit_denominator(limit)
    ok = abs(float(f) - x) < TOL
    return f, ok


def to_fraction_sym(x, limit=2000):
    f = Fraction(x).limit_denominator(limit)
    ok = abs(float(f) - x) < TOL
    return f, ok


def group_eigenvalues(evals, tol=TOL):
    """Cluster nearly equal eigenvalues."""
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
    Partition eigenvalues into Galois conjugacy classes over Q.
    Returns list of class dicts with 'gap_norm', 'min_poly_coeffs', etc.
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

    for frac, mult in rational:
        gap = frac + 3
        classes.append({
            'values': [float(frac)],
            'mult': mult,
            'degree': 1,
            'sym_polys': [frac],
            'min_poly_coeffs': [Fraction(1), -frac],
            'gap_norm': gap,
            'is_rational': True,
        })

    irrat_vals = [rep for rep, mult in irrational]
    irrat_mults = [mult for rep, mult in irrational]

    def check_galois_class(indices):
        vals = [irrat_vals[i] for i in indices]
        mults = [irrat_mults[i] for i in indices]
        if len(set(mults)) != 1:
            return None
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
        if not remaining_indices:
            return []
        first = remaining_indices[0]
        rest = remaining_indices[1:]
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
        return None

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
        for i in range(len(irrat_vals)):
            classes.append({
                'values': [irrat_vals[i]],
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
    x = Fraction(x)
    result = Fraction(0)
    d = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        result += c * x ** (d - i)
    return result


def analyze_k_point(H, flat_eigenvalue=FLAT_EIGENVALUE, flat_tol=FLAT_TOL):
    """Diagonalize H, exclude flat-band eigenvalues, return classes."""
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
# Per-orbit bridge check
# ===========================================================================

def bridge_check_orbit(orbit_rep, orbit_size, T, L, verbose=True):
    """
    For one orbit representative (tuple of n-values), build H(k),
    analyze eigenvalues, check bridge.

    Returns dict:
      orbit_rep, orbit_size, topo_prime, flat_count,
      classes, all_gap_primes, bridge_fires, notes
    """
    n_tuple = orbit_rep
    k_vec = np.array([2 * np.pi * float(n) / L for n in n_tuple])
    H = build_H_k_L(T, k_vec)

    # Hermitian check
    herm_err = np.max(np.abs(H - H.conj().T))
    if herm_err > 1e-8 and verbose:
        print(f"    WARNING: H not Hermitian at n={n_tuple}, err={herm_err:.2e}")

    flat_count, classes, all_evals = analyze_k_point(H)

    p_topo = orbit_size - 1
    all_gap_primes = set()
    bridge_fires = False
    notes = []

    for cls in classes:
        coeffs = cls.get('min_poly_coeffs', [])
        if coeffs:
            f_val = eval_min_poly_at(coeffs, Fraction(-3))
            f_abs = abs(f_val)
            if f_abs.numerator != 0:
                primes_f = prime_factors(f_abs.numerator) + prime_factors(f_abs.denominator)
                all_gap_primes.update(set(primes_f))
                if p_topo in set(primes_f):
                    bridge_fires = True
            else:
                # f(-3) = 0 means -3 is an eigenvalue (already excluded as flat?
                # or a coincidence). Flag it.
                notes.append("f(-3)=0 for one class")

    if p_topo == 0:
        notes.append("p=0 (orbit size=1, Gamma point); bridge undefined")
        bridge_fires = None  # undefined

    if not classes:
        notes.append("no non-flat eigenvalues")
        bridge_fires = None

    return {
        'orbit_rep': orbit_rep,
        'orbit_size': orbit_size,
        'topo_prime': p_topo,
        'flat_count': flat_count,
        'classes': classes,
        'all_gap_primes': all_gap_primes,
        'bridge_fires': bridge_fires,
        'notes': notes,
        'all_evals': all_evals,
    }


def print_orbit_detail(result):
    """Print detailed bridge table for one orbit."""
    rep = result['orbit_rep']
    d = result['orbit_size']
    p = result['topo_prime']
    flat_count = result['flat_count']
    classes = result['classes']

    print(f"\n  --- Orbit rep n={rep}  |  size d={d}  |  topo prime p={p}  "
          f"|  flat bands={flat_count} ---")

    if not classes:
        print(f"    (no non-flat eigenvalues)")
        return

    for ci, cls in enumerate(classes):
        coeffs = cls.get('min_poly_coeffs', [])
        d_cls = cls['degree']
        mult = cls['mult']

        # Format polynomial
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

        if coeffs:
            f_val = eval_min_poly_at(coeffs, Fraction(-3))
            f_abs = abs(f_val)
            primes_f = sorted(set(prime_factors(f_abs.numerator) + prime_factors(f_abs.denominator)))
            bridge = p in primes_f and p > 0
            print(f"    [{ci+1}] deg={d_cls} mult={mult}  f(x)={poly_str[:40]}  "
                  f"f(-3)={f_val}={factored_str(int(f_val)) if f_val.denominator==1 else f_val}  "
                  f"primes={primes_f}  bridge={'YES' if bridge else 'no'}")
        else:
            print(f"    [{ci+1}] deg={d_cls} mult={mult}  UNCLASSIFIED")

    gap_primes = result['all_gap_primes']
    bf = result['bridge_fires']
    print(f"    All gap primes: {sorted(gap_primes)}  |  Bridge fires: {bf}")
    if result['notes']:
        print(f"    Notes: {'; '.join(result['notes'])}")


# ===========================================================================
# Per-L analysis
# ===========================================================================

def analyze_L(L, verbose=True):
    """
    Full bridge analysis for BCC L×L×L.
    Returns summary dict.
    """
    t0 = time.time()

    if verbose:
        print(f"\n{'#' * 70}")
        print(f"#  L = {L}  (BCC {L}x{L}x{L},  {L**3} body-centers,  "
              f"k-points = {L**3})")
        print(f"{'#' * 70}")

    # Build transfer blocks once for this L
    body_centers = build_bcc_lattice_L(L)
    face_to_idx, bc_face_indices, _ = build_all_faces_L(body_centers, L)

    t_adj = time.time()
    A_full = build_adjacency_matrix_L(face_to_idx)
    t_adj_done = time.time()

    T = build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L)

    t_built = time.time()
    if verbose:
        print(f"  Construction: faces={len(face_to_idx)}  "
              f"adj_time={t_adj_done-t_adj:.2f}s  total_build={t_built-t0:.2f}s")

    # Compute orbits
    orbits = compute_all_orbits(L)
    if verbose:
        print(f"  k-points: {L**3}  |  O_h orbits: {len(orbits)}")
        orbit_sizes = sorted([len(orb) for _, orb in orbits])
        print(f"  Orbit sizes: {orbit_sizes}")

    # Analyze each orbit
    orbit_results = []
    for rep, orb in orbits:
        result = bridge_check_orbit(rep, len(orb), T, L, verbose=verbose)
        orbit_results.append(result)
        if verbose:
            print_orbit_detail(result)

    # Aggregate
    all_gap_primes = set()
    all_topo_primes = set()
    bridge_count = 0
    bridge_total = 0  # orbits where bridge is defined (p > 0)

    for r in orbit_results:
        all_gap_primes.update(r['all_gap_primes'])
        if r['topo_prime'] > 0:
            all_topo_primes.add(r['topo_prime'])
            if r['bridge_fires'] is not None:
                bridge_total += 1
                if r['bridge_fires']:
                    bridge_count += 1

    t1 = time.time()

    if verbose:
        print(f"\n  SUMMARY for L={L}:")
        print(f"  Topo primes (orbit_size - 1, p>0): {sorted(all_topo_primes)}")
        print(f"  Gap primes (from f(-3) factorizations): {sorted(all_gap_primes)}")
        print(f"  Bridge fires: {bridge_count}/{bridge_total} "
              f"orbits where p>0 and eigenvalues exist")
        print(f"  Total time: {t1-t0:.2f}s")

    return {
        'L': L,
        'n_kpts': L ** 3,
        'orbits': orbits,
        'orbit_results': orbit_results,
        'all_gap_primes': all_gap_primes,
        'all_topo_primes': all_topo_primes,
        'bridge_count': bridge_count,
        'bridge_total': bridge_total,
        'time': t1 - t0,
    }


# ===========================================================================
# Cross-L summary table
# ===========================================================================

def print_summary_table(all_results):
    """Print comparative summary across all L values."""
    print(f"\n\n{'=' * 90}")
    print(f"  CROSS-L COMPARATIVE SUMMARY")
    print(f"{'=' * 90}")

    header = (f"{'L':>3}  {'k-pts':>6}  {'#orbits':>7}  "
              f"{'orbit sizes':25}  {'topo primes':18}  "
              f"{'gap primes':20}  {'bridge':10}  {'time':>8}")
    print(f"\n  {header}")
    print(f"  {'-' * (len(header) + 2)}")

    for res in all_results:
        L = res['L']
        n_kpts = res['n_kpts']
        orbits = res['orbits']
        n_orb = len(orbits)
        sizes = sorted([len(orb) for _, orb in orbits])
        sizes_str = str(sizes)[:24]
        topo_str = str(sorted(res['all_topo_primes']))[:17]
        gap_str = str(sorted(res['all_gap_primes']))[:19]
        bc = res['bridge_count']
        bt = res['bridge_total']
        bridge_str = f"{bc}/{bt}" if bt > 0 else "N/A"
        t = res['time']
        print(f"  {L:>3}  {n_kpts:>6}  {n_orb:>7}  "
              f"{sizes_str:25}  {topo_str:18}  "
              f"{gap_str:20}  {bridge_str:10}  {t:>7.2f}s")

    # Cross-L comparisons
    print(f"\n  Cross-L observations:")

    # Check if L=3 reproduces known result
    l3 = next((r for r in all_results if r['L'] == 3), None)
    if l3:
        known_primes = {2, 3, 5, 7, 11}
        gap = l3['all_gap_primes']
        print(f"  L=3 gap primes={sorted(gap)}  "
              f"matches known {{2,3,5,7,11}}? {gap == known_primes}")

    # Check if prime menus grow / are consistent
    all_gap_union = set()
    for r in all_results:
        all_gap_union.update(r['all_gap_primes'])
    print(f"  Union of all gap primes across all L: {sorted(all_gap_union)}")

    topo_union = set()
    for r in all_results:
        topo_union.update(r['all_topo_primes'])
    print(f"  Union of all topo primes across all L: {sorted(topo_union)}")

    # Bridge universality
    all_fire = all(r['bridge_count'] == r['bridge_total'] and r['bridge_total'] > 0
                   for r in all_results if r['bridge_total'] > 0)
    print(f"  Bridge fires for ALL eligible orbits across ALL L? {all_fire}")

    # Prime stability: do the primes from L=3 persist?
    if l3:
        l3_primes = l3['all_gap_primes']
        for r in all_results:
            if r['L'] != 3:
                shared = l3_primes & r['all_gap_primes']
                new_primes = r['all_gap_primes'] - l3_primes
                print(f"  L={r['L']} vs L=3: shared primes={sorted(shared)}  "
                      f"new primes={sorted(new_primes)}")

    print(f"\n  Column notes:")
    print(f"    'orbit sizes': sorted list of O_h orbit sizes for this L")
    print(f"    'topo primes': set of (orbit_size - 1) for all non-trivial orbits")
    print(f"    'gap primes':  union of prime factors of |f(-3)| over all orbit classes")
    print(f"    'bridge':      #orbits where topo prime p appears in gap primes / "
          f"#orbits with p>0")

    print(f"\n{'=' * 90}")
    print(f"  END OF CROSS-L BRIDGE SCAN")
    print(f"{'=' * 90}\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BCC bridge norm L-scan: verify bridge across lattice sizes L.")
    parser.add_argument('--L', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of L values (default: 1,2,3,4,5)')
    parser.add_argument('--no-detail', action='store_true',
                        help='Suppress per-orbit detail output')
    args = parser.parse_args()

    L_values = [int(x.strip()) for x in args.L.split(',')]
    verbose = not args.no_detail

    print("=" * 70)
    print("  BRIDGE NORM L-SCAN")
    print("  BCC lattice: varies L from 1 to max")
    print(f"  L values: {L_values}")
    print("=" * 70)
    print(f"  For each L: build LxLxL BCC, 20x20 Bloch matrix,")
    print(f"  classify k-points into O_h orbits, check bridge:")
    print(f"  orbit_size - 1 = topo prime p  |  p | f(-3)  => bridge fires")
    print()

    all_results = []
    t_total = time.time()

    for L in L_values:
        res = analyze_L(L, verbose=verbose)
        all_results.append(res)

    print(f"\n  Total scan time: {time.time()-t_total:.2f}s")

    print_summary_table(all_results)


if __name__ == "__main__":
    main()
