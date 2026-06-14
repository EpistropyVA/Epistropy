"""
verify_prime_suite_spectroscopy.py
Conjugate-suite spectroscopy of H(k) for 3x3x3 periodic BCC Bloch transfer matrix.
"""

import itertools
import numpy as np
from fractions import Fraction

# ---------------------------------------------------------------------------
# Replicate lattice construction from verify_pending_4b_rank19_shell.py
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
# Build Bloch transfer blocks T(R)
# ---------------------------------------------------------------------------

def build_transfer_blocks(body_centers, bc_face_indices, A_full):
    """
    Reorder 540x540 adjacency by parent cell (20 faces per cell).
    For cell j at offset bc_ijk relative to cell 0, Rc = tuple((r+1)%3-1 for r in bc_ijk).
    T[Rc] = A_reordered[:20, j*20:(j+1)*20]
    """
    # Build reordered index list: cell 0 first 20, cell 1 next 20, etc.
    reorder = []
    for bc_idx in range(len(body_centers)):
        reorder.extend(bc_face_indices[bc_idx])

    A_reordered = A_full[np.ix_(reorder, reorder)]

    T = {}
    for j, bc_ijk in enumerate(body_centers):
        Rc = tuple((r + 1) % 3 - 1 for r in bc_ijk)
        block = A_reordered[:20, j*20:(j+1)*20]
        if Rc not in T:
            T[Rc] = block.copy()
        else:
            T[Rc] = T[Rc] + block  # accumulate (same R from different translations? shouldn't happen)
    return T


def build_H_k(T, k_vec):
    """H(k) = sum_R exp(i k.R) T(R), 20x20 Hermitian."""
    H = np.zeros((20, 20), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, R))
        H += phase * block
    return H

# ---------------------------------------------------------------------------
# Algebraic identification helpers
# ---------------------------------------------------------------------------

TOL = 1e-9

def to_rational(x):
    """Try to express x as Fraction with limit_denominator 2000. Return (frac, ok)."""
    f = Fraction(x).limit_denominator(2000)
    ok = abs(float(f) - x) < TOL
    return f, ok


def group_degenerate(evals, tol=TOL):
    """Group eigenvalues by degeneracy. Returns list of (representative_float, multiplicity, list_of_values)."""
    sorted_ev = sorted(evals)
    groups = []
    i = 0
    while i < len(sorted_ev):
        val = sorted_ev[i]
        cluster = [val]
        j = i + 1
        while j < len(sorted_ev) and abs(sorted_ev[j] - val) < tol * max(1, abs(val)) + tol:
            cluster.append(sorted_ev[j])
            j += 1
        groups.append((np.mean(cluster), len(cluster), cluster))
        i = j
    return groups


def find_quadratic_suites(distinct_vals):
    """
    Among distinct irrational values, find pairs (a, b) s.t. a+b and a*b are both rational.
    Returns list of (a, b, s_frac, p_frac, D_squarefree, min_poly_str).
    """
    suites = []
    used = set()
    n = len(distinct_vals)
    for i in range(n):
        if i in used:
            continue
        for j in range(i+1, n):
            if j in used:
                continue
            a, b = distinct_vals[i], distinct_vals[j]
            s = a + b
            p = a * b
            sf, s_ok = to_rational(s)
            pf, p_ok = to_rational(p)
            if s_ok and p_ok:
                # discriminant
                disc_f = sf*sf - 4*pf
                disc_float = float(disc_f)
                # squarefree kernel
                D = squarefree_kernel(disc_f)
                suites.append((a, b, sf, pf, D, f"x^2 - ({sf})x + ({pf})"))
                used.add(i)
                used.add(j)
                break
    unpaired = [distinct_vals[i] for i in range(n) if i not in used]
    return suites, unpaired


def squarefree_kernel(frac_disc):
    """Given a Fraction discriminant, return the squarefree integer kernel D."""
    # disc = n/d, squarefree kernel of n*d (up to sign convention)
    # We want D such that disc = D * (something)^2 as rational
    # disc_f = p/q -> D = squarefree part of p*q (with sign)
    n = frac_disc.numerator
    d = frac_disc.denominator
    val = n * d  # disc * d^2 = n*d, so sqrt(disc) = sqrt(n*d)/d
    if val == 0:
        return 0
    sign = 1 if val > 0 else -1
    val = abs(val)
    # remove perfect square factors
    D = 1
    v = val
    p = 2
    while p * p <= v:
        cnt = 0
        while v % p == 0:
            v //= p
            cnt += 1
        if cnt % 2 == 1:
            D *= p
        p += 1
    if v > 1:
        D *= v
    return sign * D


def factored_str(n):
    """Return prime factorization string of integer n."""
    if n == 0:
        return "0"
    sign = ""
    if n < 0:
        sign = "-"
        n = -n
    if n == 1:
        return sign + "1"
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    parts = []
    for p in sorted(factors):
        if factors[p] == 1:
            parts.append(str(p))
        else:
            parts.append(f"{p}^{factors[p]}")
    return sign + "·".join(parts)


def gap_norm_suite(suite_vals, multiplicities=None):
    """Product of (lambda+3) for suite members WITHOUT multiplicity."""
    prod = Fraction(1)
    for v in suite_vals:
        g = v + 3.0
        gf, ok = to_rational(g)
        if ok:
            prod *= gf
        else:
            # irrational gap - handle as float product for now
            return None, float(prod) * (v + 3.0)
    return prod, None


def gap_product_all(eigenvalues_with_mult):
    """Product of (lambda+3) over all eigenvalues WITH multiplicity, as Fraction or float."""
    prod = Fraction(1)
    for val, mult, _ in eigenvalues_with_mult:
        g = val + 3.0
        gf, ok = to_rational(g)
        if ok:
            prod *= gf ** mult
        else:
            return None
    return prod


# ---------------------------------------------------------------------------
# Main spectroscopy per orbit
# ---------------------------------------------------------------------------

EXPECTED_FULL_PRODUCTS = {
    'Gamma':         Fraction(2**18 * 3**3),
    'axis':          Fraction(2**18 * 3**2 * 5),
    'face-diagonal': Fraction(2**8 * 3**3 * 5**2 * 7 * 11),
    'body-diagonal': Fraction(2**6 * 3**3 * 5**3 * 7**2),
}

K_POINTS = {
    'Gamma':         np.array([0.0, 0.0, 0.0]),
    'axis':          np.array([2*np.pi/3, 0.0, 0.0]),
    'face-diagonal': np.array([2*np.pi/3, 2*np.pi/3, 0.0]),
    'body-diagonal': np.array([2*np.pi/3, 2*np.pi/3, 2*np.pi/3]),
}

def analyze_orbit(orbit_name, evals_all, outfile):
    lines = []
    def w(s=""):
        lines.append(s)
        print(s)

    w(f"\n{'='*70}")
    w(f"ORBIT: {orbit_name}")
    w(f"All 20 eigenvalues: {sorted(evals_all)}")

    # Identify and exclude flat band at -3
    flat_count = sum(1 for e in evals_all if abs(e + 3) < 1e-6)
    non_flat = [e for e in evals_all if abs(e + 3) > 1e-6]
    w(f"Flat band (-3) count: {flat_count}  (expected 6)")
    w(f"Remaining 14 eigenvalues: {sorted(non_flat)}")
    assert len(non_flat) == 14, f"Expected 14 non-flat, got {len(non_flat)}"

    # Group degenerate
    groups = group_degenerate(non_flat)
    w(f"\nDegenerate groups ({len(groups)} distinct values):")
    for rep, mult, cluster in groups:
        rf, r_ok = to_rational(rep)
        rat_str = f"= {rf}" if r_ok else f"≈ {rep:.15f}"
        w(f"  λ {rat_str}  (mult {mult})")

    # Separate rational vs irrational distinct values
    rational_groups = []
    irrational_groups = []
    for rep, mult, cluster in groups:
        rf, r_ok = to_rational(rep)
        if r_ok:
            rational_groups.append((rep, mult, rf))
        else:
            irrational_groups.append((rep, mult))

    # Find quadratic suites among irrational distinct values
    irrat_vals = [v for v, m in irrational_groups]
    quad_suites, still_unpaired = find_quadratic_suites(irrat_vals)

    w(f"\nRational eigenvalues ({len(rational_groups)}):")
    for val, mult, frac in rational_groups:
        gap_val = frac + 3
        w(f"  λ={frac}  mult={mult}  gap=(λ+3)={gap_val}  gap_factored={factored_str(gap_val.numerator)}{'/' + factored_str(gap_val.denominator) if gap_val.denominator != 1 else ''}")

    w(f"\nQuadratic suites ({len(quad_suites)}):")
    for (a, b, sf, pf, D, minpoly) in quad_suites:
        ga = Fraction(a + 3).limit_denominator(2000)
        gb = Fraction(b + 3).limit_denominator(2000)
        # gap norm = (a+3)(b+3) without multiplicity
        gap_norm_frac = (Fraction(a+3).limit_denominator(2000)) * (Fraction(b+3).limit_denominator(2000))
        gap_norm_frac_check = ga * gb
        # Use exact arithmetic via minpoly: (a+3)(b+3) = ab + 3(a+b) + 9 = p + 3s + 9
        exact_gap_norm = pf + 3*sf + 9
        w(f"  {{{a:.10f}, {b:.10f}}}")
        w(f"    min poly: {minpoly}  →  Q(√{D})")
        w(f"    sum={sf}  product={pf}  disc={sf*sf - 4*pf}  D(squarefree)={D}")
        w(f"    gap norm (no mult) = (a+3)(b+3) = {exact_gap_norm} = {factored_str(exact_gap_norm.numerator)}")
        # Find multiplicities of a and b
        mult_a = next(m for v,m in irrational_groups if abs(v-a) < TOL)
        mult_b = next(m for v,m in irrational_groups if abs(v-b) < TOL)
        w(f"    mult_a={mult_a}  mult_b={mult_b}")
        gap_with_mult = exact_gap_norm ** min(mult_a, mult_b)
        # full product with multiplicities
        full = (Fraction(a+3).limit_denominator(2000)**mult_a) * (Fraction(b+3).limit_denominator(2000)**mult_b)
        w(f"    gap product WITH mult = {full} = {factored_str(full.numerator)}" +
          (f"/{factored_str(full.denominator)}" if full.denominator != 1 else ""))

    if still_unpaired:
        w(f"\nUnpaired irrational values (may be higher-degree suites):")
        for v in still_unpaired:
            w(f"  {v:.15f}")
        # Try triples
        if len(still_unpaired) >= 3:
            for tri in itertools.combinations(range(len(still_unpaired)), 3):
                a,b,c = [still_unpaired[i] for i in tri]
                s = a+b+c
                p2 = a*b+a*c+b*c
                p3 = a*b*c
                sf, s_ok = to_rational(s)
                p2f, p2_ok = to_rational(p2)
                p3f, p3_ok = to_rational(p3)
                if s_ok and p2_ok and p3_ok:
                    w(f"  Cubic suite: {a:.10f}, {b:.10f}, {c:.10f}")
                    w(f"    sum={sf} sum-pairs={p2f} product={p3f}")

    # Full gap product with multiplicities — algebraic via suite structure
    w(f"\nSanity check — full gap product (WITH mult) over 14 non-flat:")
    # Strategy: rational eigenvalues contribute gf^mult directly.
    # Quadratic suite {a,b} with mults ma,mb: since the suite is always symmetric with ma==mb,
    # contribution = ((a+3)(b+3))^ma = (pf + 3sf + 9)^ma, which is rational.
    # Unpaired irrationals: try to group as cubic suites and use their norm.
    full_gap = Fraction(1)
    accounted_vals = set()

    # Rational eigenvalues
    for val, mult, frac in rational_groups:
        gf = frac + 3
        full_gap *= gf ** mult
        accounted_vals.add(round(val, 9))

    # Quadratic suites
    for (a, b, sf, pf, D, minpoly) in quad_suites:
        exact_gap_norm = pf + 3*sf + 9  # (a+3)(b+3) as Fraction
        mult_a = next(m for v,m in irrational_groups if abs(v-a) < TOL)
        mult_b = next(m for v,m in irrational_groups if abs(v-b) < TOL)
        # For well-formed suites mult_a == mult_b; use min to be safe
        if mult_a == mult_b:
            full_gap *= exact_gap_norm ** mult_a
        else:
            # asymmetric — use algebraic product with individual mults
            # (a+3)^ma * (b+3)^mb — hard to keep rational if ma != mb
            # fall back: report mismatch
            w(f"  WARNING: asymmetric mult for Q(√{D}) suite: mult_a={mult_a} mult_b={mult_b}")
            full_gap = None
            break
        accounted_vals.add(round(a, 9))
        accounted_vals.add(round(b, 9))

    if full_gap is not None:
        from math import comb as math_comb

        remaining = list(still_unpaired)
        n_rem = len(remaining)

        def check_suite(vals_list):
            """Check if a set of values forms an algebraic suite (all sym polys rational).
            Returns (syms_list, gap_norm) or None."""
            d = len(vals_list)
            syms = []
            for k in range(1, d+1):
                s = sum(np.prod([vals_list[i] for i in idx])
                        for idx in itertools.combinations(range(d), k))
                sf, ok = to_rational(s)
                if not ok:
                    return None
                syms.append(sf)
            e = [Fraction(1)] + list(syms)
            # product(v+3) = sum_{k=0}^{d} e_k * 3^(d-k), e_0=1
            gap_norm = sum(e[k] * Fraction(3)**(d-k) for k in range(d+1))
            return syms, gap_norm

        def partition_into_suites(indices, current_partition):
            """Backtracking: try all ways to partition indices into algebraic suites."""
            if not indices:
                return current_partition  # success
            # try all subsets starting from index indices[0]
            first = indices[0]
            rest = indices[1:]
            for size in range(1, len(indices)+1):
                for combo in itertools.combinations(rest, size-1):
                    subset = [first] + list(combo)
                    subset_vals = [remaining[i] for i in subset]
                    result = check_suite(subset_vals)
                    if result is not None:
                        remaining_indices = [i for i in indices if i not in subset]
                        sol = partition_into_suites(remaining_indices, current_partition + [(subset, result)])
                        if sol is not None:
                            return sol
            return None  # no valid partition found

        all_indices = list(range(n_rem))
        partition = partition_into_suites(all_indices, [])

        higher_suites = []
        if partition is not None:
            for (subset, (syms, gap_norm)) in partition:
                vals_combo = [remaining[i] for i in subset]
                mult_vals = [next(m2 for v2,m2 in irrational_groups if abs(v2-v) < TOL)
                             for v in vals_combo]
                higher_suites.append((len(subset), vals_combo, syms, gap_norm, subset, mult_vals))

            for (degree, vals_combo, syms, gap_norm, combo, mult_vals) in higher_suites:
                s_str = ", ".join(f"{v:.8f}" for v in vals_combo)
                w(f"  Degree-{degree} suite: {{{s_str}}}")
                w(f"    sym polys (e1..e{min(3,degree)}): {syms[:min(3,degree)]}...")
                w(f"    gap norm (no mult) = {gap_norm} = {factored_str(gap_norm.numerator)}")
                if len(set(mult_vals)) == 1:
                    full_gap *= gap_norm ** mult_vals[0]
                    w(f"    mult={mult_vals[0]} -> contribution = {gap_norm**mult_vals[0]}")
                else:
                    w(f"  WARNING: asymmetric suite mults {mult_vals}")
                    full_gap = None
                    break
        else:
            w(f"  WARNING: could not partition {n_rem} unpaired irrationals into algebraic suites")
            for v in remaining:
                w(f"    {v:.15f}")
            full_gap = None

    expected = EXPECTED_FULL_PRODUCTS.get(orbit_name)
    if full_gap is not None and expected is not None:
        match = (full_gap == expected)
        w(f"  Computed: {full_gap} = {factored_str(full_gap.numerator)}" +
          (f"/{factored_str(full_gap.denominator)}" if full_gap.denominator != 1 else ""))
        w(f"  Expected: {expected} = {factored_str(expected.numerator)}")
        w(f"  MATCH: {match}")
        if not match:
            w(f"  MISMATCH: computed={full_gap}, expected={expected}")
    elif full_gap is None:
        w(f"  Could not compute (asymmetric mult or unaccounted irrationals)")
    else:
        w(f"  No expected value to compare")

    w(f"{'='*70}")
    return lines, groups, quad_suites, rational_groups, irrational_groups, still_unpaired, full_gap


def main():
    output_lines = []
    def out(s=""):
        output_lines.append(s)
        print(s)

    out("=== Prime Suite Spectroscopy of H(k), 3x3x3 periodic BCC ===")
    out()

    # Build lattice
    body_centers = build_bcc_lattice_periodic()
    face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
    A_full = build_adjacency_matrix(face_to_idx)

    out(f"Lattice: 27 cells, {len(face_to_idx)} faces total (expected 540)")
    out(f"Faces per cell (ref): {len(ref_faces)} (expected 20)")

    # Build transfer blocks
    T = build_transfer_blocks(body_centers, bc_face_indices, A_full)
    out(f"Transfer blocks T(R): {len(T)} distinct R vectors")
    out(f"R vectors: {sorted(T.keys())}")
    out()

    # Spectroscopy at each k-point
    orbit_results = {}
    for orbit_name, k_vec in K_POINTS.items():
        H = build_H_k(T, k_vec)
        # Verify Hermitian
        herm_err = np.max(np.abs(H - H.conj().T))
        evals = np.linalg.eigvalsh(H)
        out(f"k={orbit_name}: H Hermitian error={herm_err:.2e}")

        lines, groups, quad_suites, rat_groups, irrat_groups, unpaired, full_gap = \
            analyze_orbit(orbit_name, evals, output_lines)
        output_lines.extend(lines)
        orbit_results[orbit_name] = {
            'evals': evals,
            'groups': groups,
            'quad_suites': quad_suites,
            'rat_groups': rat_groups,
            'irrat_groups': irrat_groups,
            'unpaired': unpaired,
            'full_gap': full_gap,
        }

    # ---------------------------------------------------------------------------
    # Wanted-list answers
    # ---------------------------------------------------------------------------
    out("\n" + "="*70)
    out("WANTED-LIST ANSWERS")
    out("="*70)

    # Q1: Does prime 5 arise from rational λ=2 (gap 5) in axis orbit?
    out("\nQ1: Prime 5 from λ=2 in axis orbit?")
    axis_rat = orbit_results['axis']['rat_groups']
    axis_found_2 = [(v,m,f) for v,m,f in axis_rat if abs(v-2.0)<TOL]
    if axis_found_2:
        out(f"  YES: λ=2 (gap=5) found in axis orbit with mult={axis_found_2[0][1]}")
    else:
        out(f"  NO: λ=2 not found in axis orbit. Rational eigenvalues: {[(str(f),m) for v,m,f in axis_rat]}")
        # Which suite emits 5?
        out("  Searching for suite with 5 in gap factorization across all orbits...")
        for orb, res in orbit_results.items():
            for rg in res['rat_groups']:
                v, m, f = rg
                gf = f + 3
                if gf.numerator % 5 == 0:
                    out(f"  → orbit={orb}: rational λ={f}, gap={gf} (contains factor 5)")
            for qs in res['quad_suites']:
                a, b, sf, pf, D, minpoly = qs
                exact_gap_norm = pf + 3*sf + 9
                n = exact_gap_norm.numerator
                d = exact_gap_norm.denominator
                if n % 5 == 0:
                    out(f"  → orbit={orb}: quad suite Q(√{D}), gap norm={exact_gap_norm} (contains factor 5)")

    # Q2: Does prime 7 arise from ±√2-type suite in some orbits?
    out("\nQ2: Prime 7 from Q(√2) suite (gaps 3±√2)?")
    found_7_from_sqrt2 = False
    for orb, res in orbit_results.items():
        for qs in res['quad_suites']:
            a, b, sf, pf, D, minpoly = qs
            if D == 2:
                exact_gap_norm = pf + 3*sf + 9
                out(f"  orbit={orb}: Q(√2) suite, eigenvalues≈{a:.6f},{b:.6f}, gap norm={exact_gap_norm}={factored_str(exact_gap_norm.numerator)}")
                if exact_gap_norm.numerator % 7 == 0:
                    out(f"    → contains factor 7 ✓")
                    found_7_from_sqrt2 = True
    if not found_7_from_sqrt2:
        out("  Q(sqrt2) gap norm = 8 = 2^3 -- prime 7 does NOT arise from Q(sqrt2) suites.")
        orbits_with_7 = []
        for orb, res in orbit_results.items():
            for qs in res['quad_suites']:
                a, b, sf, pf, D, minpoly = qs
                gn = pf + 3*sf + 9
                if D == 21 and gn.numerator % 7 == 0:
                    orbits_with_7.append(orb)
        out(f"  Q(√21) suite (gap norm=7) appears in: {orbits_with_7}")

    # Q3: Prime 11 in face-diagonal orbit
    out("\nQ3: Prime 11 in face-diagonal orbit?")
    fd_res = orbit_results['face-diagonal']
    out("  Rational eigenvalues in face-diagonal:")
    for v, m, f in fd_res['rat_groups']:
        gf = f + 3
        out(f"    λ={f} mult={m} gap={gf} {'← contains 11' if gf.numerator % 11 == 0 else ''}")
    out("  Quad suites in face-diagonal:")
    for a, b, sf, pf, D, minpoly in fd_res['quad_suites']:
        exact_gap_norm = pf + 3*sf + 9
        out(f"    Q(√{D}) suite: λ≈{a:.8f},{b:.8f}  gap norm={exact_gap_norm}={factored_str(exact_gap_norm.numerator)} {'← 11 here' if exact_gap_norm.numerator % 11 == 0 else ''}")
    out("  Searching for 11 in cubic suite gap norms (face-diagonal):")
    unpaired_fd = fd_res['unpaired']
    for tri in itertools.combinations(range(len(unpaired_fd)), 3):
        a,b,c = [unpaired_fd[i] for i in tri]
        s = a+b+c; p2 = a*b+a*c+b*c; p3 = a*b*c
        sf3, s_ok = to_rational(s); p2f, p2_ok = to_rational(p2); p3f, p3_ok = to_rational(p3)
        if s_ok and p2_ok and p3_ok:
            gn = p3f + 3*p2f + 9*sf3 + 27
            out(f"    Cubic triple ≈{a:.6f},{b:.6f},{c:.6f}: gap norm={gn}={factored_str(gn.numerator)} {'← 11 here' if gn.numerator % 11 == 0 else ''}")
    # Also check the full orbit gap product for 11
    fg = fd_res['full_gap']
    if fg is not None:
        out(f"  Full gap product contains 11: {fg.numerator % 11 == 0}")

    # Q4: Q(√6) suites anywhere?
    out("\nQ4: Q(√6) suites anywhere?")
    found_sqrt6 = False
    for orb, res in orbit_results.items():
        for qs in res['quad_suites']:
            a, b, sf, pf, D, minpoly = qs
            if D == 6:
                found_sqrt6 = True
                out(f"  YES: orbit={orb}, eigenvalues≈{a:.8f},{b:.8f}, min poly={minpoly}")
    if not found_sqrt6:
        out("  No Q(√6) suites found in any orbit.")

    # Q5: Primes emitted not in {2,3,5,7,11}
    out("\nQ5: Primes emitted NOT in {2,3,5,7,11}:")
    known_primes = {2, 3, 5, 7, 11}
    exotic = set()
    for orb, res in orbit_results.items():
        if res['full_gap'] is not None:
            n = abs(res['full_gap'].numerator)
            d = res['full_gap'].denominator
            for val in [n, d]:
                v = val
                p = 2
                while p * p <= v:
                    if v % p == 0:
                        if p not in known_primes:
                            exotic.add(p)
                        while v % p == 0:
                            v //= p
                    p += 1
                if v > 1 and v not in known_primes:
                    exotic.add(v)
    if exotic:
        out(f"  Exotic primes: {sorted(exotic)}")
    else:
        out("  None found — all primes in {2,3,5,7,11}")

    out("\n=== END ===")

    # Write results file
    results_path = r"d:/AI thoery/.agent/scripts/verify_prime_suite_spectroscopy_results.txt"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    print(f"\nResults written to: {results_path}")


if __name__ == "__main__":
    main()
