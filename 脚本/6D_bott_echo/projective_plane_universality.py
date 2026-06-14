"""
projective_plane_universality.py
Verify β₁(P²(F_q); F₂) = q³ for q = 5, 7, 8, 9, 11

Simplicial complex:
  Vertices : q²+q+1 points of P²(F_q)
  Edges    : ALL pairs  (complete graph K_V)
  2-faces  : ALL collinear triples
"""

import itertools
import time
import numpy as np
from math import comb

# ───────────────────────────────────────────────────────────────
#  Finite field helpers
# ───────────────────────────────────────────────────────────────

def field_prime(p):
    """Return (add, mul, elems) tables for F_p (p prime)."""
    elems = list(range(p))
    def add(a, b): return (a + b) % p
    def mul(a, b): return (a * b) % p
    return add, mul, elems

def build_F8():
    """
    F_8 = F_2[x]/(x^3+x+1).
    Represent element a0 + a1*x + a2*x^2 as integer a0 + 2*a1 + 4*a2.
    Reduction table: x^3=x+1, x^4=x^2+x, x^5=x^2+x+1.
    """
    elems = list(range(8))

    def add(a, b):
        return a ^ b  # XOR = addition over F_2

    # Reduction constants for degrees 3,4,5
    _reduce = {3: 0b011, 4: 0b110, 5: 0b111}

    def mul(a, b):
        # Expand product (may reach degree 4)
        r = 0
        for i in range(3):
            if (b >> i) & 1:
                r ^= (a << i)
        # Reduce from highest degree down
        for deg in range(5, 2, -1):
            if (r >> deg) & 1:
                r ^= (1 << deg)         # clear bit
                r ^= _reduce[deg]       # replace with reduced form
        return r & 7

    return add, mul, elems

def build_F9():
    """
    F_9 = F_3[x]/(x²+1).  x² = -1 = 2 mod 3.
    Elements: pairs (a, b) representing a + b*x, a,b in {0,1,2}.
    Map to integers 0..8 via idx = a + 3*b.
    """
    elems = list(range(9))

    def to_pair(n):
        return n % 3, n // 3

    def from_pair(a, b):
        return (a % 3) + 3 * (b % 3)

    def add(n1, n2):
        a1, b1 = to_pair(n1)
        a2, b2 = to_pair(n2)
        return from_pair((a1 + a2) % 3, (b1 + b2) % 3)

    def mul(n1, n2):
        a, b = to_pair(n1)
        c, d = to_pair(n2)
        # (a+bx)(c+dx) = ac + adx + bcx + bd*x² = ac + bd*2 + (ad+bc)x
        real = (a * c + b * d * 2) % 3
        imag = (a * d + b * c) % 3
        return from_pair(real, imag)

    return add, mul, elems

# ───────────────────────────────────────────────────────────────
#  Projective plane P²(F_q)
# ───────────────────────────────────────────────────────────────

def build_projective_plane(q, add, mul, elems):
    """
    Return (points, lines) where:
      points = list of canonical representatives [x:y:z] in P²(F_q)
      lines  = list of sets of point indices

    Canonical representative: first nonzero coordinate is 1.
    """
    n = len(elems)
    assert n == q

    # Non-zero check: element != 0
    def is_nonzero(e):
        return e != 0

    # Scalar multiplication: k * (a,b,c)
    def scale(k, triple):
        return tuple(mul(k, x) for x in triple)

    # Precompute inverses
    inv_table = {}
    for e in elems:
        if is_nonzero(e):
            for f in elems:
                if mul(e, f) == 1:
                    inv_table[e] = f
                    break

    # Find canonical form of [a:b:c]: scale so first nonzero coord = 1
    def canonical(a, b, c):
        for v in (a, b, c):
            if is_nonzero(v):
                inv_v = inv_table[v]
                return tuple(mul(inv_v, x) for x in (a, b, c))
        raise ValueError("Zero point")

    # Enumerate all points
    seen = set()
    points = []
    for a in elems:
        for b in elems:
            for c in elems:
                if not (a == 0 and b == 0 and c == 0):
                    p = canonical(a, b, c)
                    if p not in seen:
                        seen.add(p)
                        points.append(p)

    pt_to_idx = {p: i for i, p in enumerate(points)}
    V = len(points)
    assert V == q*q + q + 1, f"Expected {q*q+q+1} points, got {V}"

    # Enumerate lines: a line [a:b:c] is the set {P : a*px + b*py + c*pz = 0}
    # Lines are dual to points, so same count
    seen_lines = set()
    lines = []
    for a in elems:
        for b in elems:
            for c in elems:
                if not (a == 0 and b == 0 and c == 0):
                    lc = canonical(a, b, c)
                    if lc not in seen_lines:
                        seen_lines.add(lc)
                        la, lb, lc_coeff = lc
                        # Find all points on this line
                        line_pts = []
                        for i, (px, py, pz) in enumerate(points):
                            # dot product = 0
                            dot = add(add(mul(la, px), mul(lb, py)), mul(lc_coeff, pz))
                            if dot == 0:
                                line_pts.append(i)
                        assert len(line_pts) == q + 1, f"Line has {len(line_pts)} points, expected {q+1}"
                        lines.append(line_pts)

    assert len(lines) == q*q + q + 1

    return points, lines

# ───────────────────────────────────────────────────────────────
#  F₂ homology via Gaussian elimination
# ───────────────────────────────────────────────────────────────

def rank_mod2(mat):
    """Compute rank of matrix over F₂ using Gaussian elimination (XOR)."""
    if mat.size == 0:
        return 0
    m, n = mat.shape
    A = mat.copy().astype(np.uint8)
    pivot_row = 0
    for col in range(n):
        # Find pivot
        rows_with_one = np.where(A[pivot_row:, col] == 1)[0]
        if len(rows_with_one) == 0:
            continue
        r = rows_with_one[0] + pivot_row
        if pivot_row != r:
            temp = A[pivot_row].copy()
            A[pivot_row] = A[r]
            A[r] = temp
        # Eliminate
        rows_to_elim = np.where(A[:, col] == 1)[0]
        rows_to_elim = rows_to_elim[rows_to_elim != pivot_row]
        A[rows_to_elim] ^= A[pivot_row]
        pivot_row += 1
        if pivot_row == m:
            break
    return pivot_row

def compute_homology(q, add, mul, elems, timeout=180):
    """Compute β₀, β₁, β₂ of the simplicial complex for P²(F_q)."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  q = {q}")
    print(f"{'='*60}")

    print(f"  Building P²(F_{q})...", end=" ", flush=True)
    points, lines = build_projective_plane(q, add, mul, elems)
    V = len(points)
    print(f"done. V={V}")

    # Edge index mapping
    edges = list(itertools.combinations(range(V), 2))
    E = len(edges)
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    print(f"  E = {E}")

    # Collinear triples
    faces = []
    for line in lines:
        for triple in itertools.combinations(sorted(line), 3):
            faces.append(triple)
    F = len(faces)
    print(f"  F = {F} (collinear triples)")

    expected_F = len(lines) * comb(q + 1, 3)
    assert F == expected_F, f"F mismatch: {F} vs {expected_F}"

    # Build ∂₁: E × V  (columns = edges, rows = vertices? No: ∂₁: C₁→C₀)
    # Standard: ∂₁ is V×E, ∂₁[v, e] = 1 if v is endpoint of e
    print(f"  Building ∂₁ ({V}×{E})...", end=" ", flush=True)
    d1 = np.zeros((V, E), dtype=np.uint8)
    for j, (u, v) in enumerate(edges):
        d1[u, j] = 1
        d1[v, j] = 1
    print("done")

    # Build ∂₂: E×F, ∂₂[e, f] = 1 if e is face of triangle f
    print(f"  Building ∂₂ ({E}×{F})...", end=" ", flush=True)
    d2 = np.zeros((E, F), dtype=np.uint8)
    for j, (a, b, c) in enumerate(faces):
        d2[edge_to_idx[(a, b)], j] = 1
        d2[edge_to_idx[(a, c)], j] = 1
        d2[edge_to_idx[(b, c)], j] = 1
    print("done")

    elapsed = time.time() - t0
    if elapsed > timeout * 0.5:
        print(f"  [Warning] Already {elapsed:.1f}s elapsed before rank computation")

    print(f"  Computing rank(∂₁)...", end=" ", flush=True)
    r1 = rank_mod2(d1)
    print(f"rank = {r1}")

    print(f"  Computing rank(∂₂)...", end=" ", flush=True)
    r2 = rank_mod2(d2)
    print(f"rank = {r2}")

    # β₀ = V - rank(∂₁)  [connected components]  ... wait, standard:
    # β₀ = dim ker(∂₀) - rank(∂₁) but ∂₀=0, so β₀ = V - rank(∂₁)
    # β₁ = dim ker(∂₁) - rank(∂₂) = (E - rank(∂₁)) - rank(∂₂)
    # β₂ = dim ker(∂₂) - rank(∂₃) = F - rank(∂₂)   [no 3-faces]
    beta0 = V - r1
    beta1 = (E - r1) - r2
    beta2 = F - r2

    chi = beta0 - beta1 + beta2
    chi_alt = V - E + F

    total_time = time.time() - t0
    print(f"  b0={beta0}, b1={beta1}, b2={beta2}")
    print(f"  q^3={q**3}  match={'YES' if beta1 == q**3 else 'NO MISMATCH'}")
    print(f"  chi={chi} (V-E+F={chi_alt})")
    print(f"  Time: {total_time:.1f}s")

    return {
        'q': q, 'V': V, 'E': E, 'F': F,
        'beta0': beta0, 'beta1': beta1, 'beta2': beta2,
        'q3': q**3, 'match': beta1 == q**3,
        'chi': chi, 'time': total_time
    }

# ───────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Verifying b1(P^2(F_q); F_2) = q^3")
    print("Known: q=2 (b1=8), q=3 (b1=27), q=4 (b1=64)")
    print()

    results = []

    # q=5 (prime)
    add5, mul5, elems5 = field_prime(5)
    results.append(compute_homology(5, add5, mul5, elems5))

    # q=7 (prime)
    add7, mul7, elems7 = field_prime(7)
    results.append(compute_homology(7, add7, mul7, elems7))

    # q=8 = 2³ (F_8)
    add8, mul8, elems8 = build_F8()
    results.append(compute_homology(8, add8, mul8, elems8))

    # q=9 = 3² (F_9)
    add9, mul9, elems9 = build_F9()
    results.append(compute_homology(9, add9, mul9, elems9))

    # q=11 (prime) — may be slow
    add11, mul11, elems11 = field_prime(11)
    results.append(compute_homology(11, add11, mul11, elems11))

    # Summary table
    print("\n")
    print("=" * 95)
    print("SUMMARY TABLE")
    print("=" * 95)
    header = f"{'q':>4} | {'V':>6} | {'E':>6} | {'F':>6} | {'b0':>4} | {'b1':>6} | {'q^3':>6} | {'match?':>7} | {'b2':>6} | {'chi':>5} | {'time(s)':>7}"
    print(header)
    print("-" * 95)
    for r in results:
        q = r['q']
        print(f"{q:>4} | {r['V']:>6} | {r['E']:>6} | {r['F']:>6} | "
              f"{r['beta0']:>4} | {r['beta1']:>6} | {r['q3']:>6} | "
              f"{'YES' if r['match'] else 'NO':>7} | "
              f"{r['beta2']:>6} | {r['chi']:>5} | {r['time']:>7.1f}")
    print("=" * 95)

    mismatches = [r for r in results if not r['match']]
    if mismatches:
        print(f"\n!!! MISMATCH DETECTED for q = {[r['q'] for r in mismatches]} !!!")
    else:
        print(f"\nAll verified: b1 = q^3 holds for q = {[r['q'] for r in results]}")
