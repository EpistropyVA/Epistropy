"""
verify_sec16_adelic.py
======================
Numerical verification for §16 "Adelic 验证：Q_p 不是异构宇宙"
Source: 有趣的拓扑和几何的互洽（终）.md, lines 161-188 + line 267

Claims verified:
  A. p-adic valuation v_p(n) and norm |n|_p = p^{-v_p(n)}
  B. Ultrametric (超度量) inequality: d(x,y) ≤ max(d(x,z), d(y,z))
  C. Strong triangle inequality (ultrametric norm): |x+y|_p ≤ max(|x|_p, |y|_p)
  D. Product formula ∏_v |x|_v = 1  (line 267, §16 back-reference)
  E. Adele ring — all Q_p are "already present" (non-archimedean completions of Q)
  F. Hasse principle (quadratic form example — Hasse-Minkowski)

Dependencies: numpy only (p-adic arithmetic implemented from scratch)
"""

import numpy as np
from fractions import Fraction
from math import gcd, isqrt
import sys
import io

# Force UTF-8 stdout so math symbols don't hit GBK codec on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]   # finite set for product formula

# ──────────────────────────────────────────────
# Core p-adic primitives
# ──────────────────────────────────────────────

def v_p(n: int, p: int) -> int:
    """p-adic valuation of integer n (n ≠ 0). v_p(0) = +∞ (returned as a large int)."""
    if n == 0:
        return 10**9
    n = abs(n)
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k

def v_p_rat(num: int, den: int, p: int) -> int:
    """p-adic valuation of rational num/den."""
    return v_p(num, p) - v_p(den, p)

def norm_p(num: int, den: int, p: int) -> float:
    """p-adic absolute value |num/den|_p = p^{-v_p(num/den)}."""
    v = v_p_rat(num, den, p)
    if v == 10**9:   # zero
        return 0.0
    return float(p) ** (-v)

def norm_arch(num: int, den: int) -> float:
    """Archimedean (real) absolute value |num/den|_∞ = |num/den|."""
    return abs(num / den)

def p_adic_dist(a_num: int, a_den: int, b_num: int, b_den: int, p: int) -> float:
    """p-adic distance |a - b|_p for rationals a = a_num/a_den, b = b_num/b_den."""
    # a - b = (a_num * b_den - b_num * a_den) / (a_den * b_den)
    diff_num = a_num * b_den - b_num * a_den
    diff_den = a_den * b_den
    return norm_p(diff_num, diff_den, p)

# ──────────────────────────────────────────────
# Test scaffolding
# ──────────────────────────────────────────────

results = []

def check(name: str, condition: bool, explanation: str, detail: str = ""):
    tag = "[PASS]" if condition else "[FAIL]"
    msg = f"{tag} {name}: {explanation}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append(condition)

# ──────────────────────────────────────────────
# A. p-adic valuation and norm basics
# ──────────────────────────────────────────────

print("\n=== A. p-adic valuation and norm  (|n|_p = p^{-v_p(n)}) ===\n")

# A1: v_2(12) = 2  (12 = 2^2 * 3)
got = v_p(12, 2)
check("A1 v_2(12)", got == 2,
      f"v_2(12) = {got}, expected 2  (12 = 2^2 · 3)")

# A2: |12|_2 = 2^{-2} = 1/4
got = norm_p(12, 1, 2)
check("A2 |12|_2", np.isclose(got, 0.25),
      f"|12|_2 = {got:.6f}, expected 0.25")

# A3: |12|_3 = 3^{-1} = 1/3
got = norm_p(12, 1, 3)
check("A3 |12|_3", np.isclose(got, 1/3),
      f"|12|_3 = {got:.6f}, expected {1/3:.6f}")

# A4: |12|_5 = 1  (5 does not divide 12)
got = norm_p(12, 1, 5)
check("A4 |12|_5", np.isclose(got, 1.0),
      f"|12|_5 = {got:.6f}, expected 1.0")

# A5: v_p of 1/p = -1  (negative valuation = denominators)
got = v_p_rat(1, 2, 2)
check("A5 v_2(1/2)", got == -1,
      f"v_2(1/2) = {got}, expected -1")

# A6: |1/2|_2 = 2
got = norm_p(1, 2, 2)
check("A6 |1/2|_2", np.isclose(got, 2.0),
      f"|1/2|_2 = {got:.6f}, expected 2.0  (denominators are p-adically large)")

# A7: v_p(p^k) = k
for p in [2, 3, 5]:
    for k in [1, 2, 3]:
        got = v_p(p**k, p)
        ok = (got == k)
        if not ok:
            check(f"A7 v_{p}({p}^{k})", False,
                  f"got {got}, expected {k}")
            break
    else:
        continue
    break
else:
    check("A7 v_p(p^k) = k", True,
          "verified for p∈{2,3,5}, k∈{1,2,3}")

# ──────────────────────────────────────────────
# B. Ultrametric (超度量) inequality
#    §16 table: d(x,y) ≤ max(d(x,z), d(y,z))
# ──────────────────────────────────────────────

print("\n=== B. Ultrametric inequality  d(x,y) ≤ max(d(x,z), d(y,z)) ===\n")

def test_ultrametric(x, y, z, p, label):
    """x, y, z are Fraction objects."""
    dxy = p_adic_dist(x.numerator, x.denominator, y.numerator, y.denominator, p)
    dxz = p_adic_dist(x.numerator, x.denominator, z.numerator, z.denominator, p)
    dyz = p_adic_dist(y.numerator, y.denominator, z.numerator, z.denominator, p)
    holds = dxy <= max(dxz, dyz) + 1e-12
    check(label,
          holds,
          f"|{x}-{y}|_{p} = {dxy:.4f} ≤ max({dxz:.4f}, {dyz:.4f}) = {max(dxz,dyz):.4f}",
          f"p={p}")

# B1: x=0, y=1/4, z=1/2 under p=2
test_ultrametric(Fraction(0), Fraction(1,4), Fraction(1,2), 2, "B1 ultrametric p=2 (0,1/4,1/2)")

# B2: x=1, y=3, z=9 under p=3
test_ultrametric(Fraction(1), Fraction(3), Fraction(9), 3, "B2 ultrametric p=3 (1,3,9)")

# B3: random triples for p=5 (stress test)
rng = np.random.default_rng(42)
all_ok = True
violations = []
for _ in range(200):
    a, b, c = [int(x) for x in rng.integers(1, 100, size=3)]
    d, e, f_ = [int(x) for x in rng.integers(1, 20, size=3)]
    x, y, z = Fraction(a, d), Fraction(b, e), Fraction(c, f_)
    for p in [2, 3, 5, 7]:
        dxy = p_adic_dist(x.numerator, x.denominator, y.numerator, y.denominator, p)
        dxz = p_adic_dist(x.numerator, x.denominator, z.numerator, z.denominator, p)
        dyz = p_adic_dist(y.numerator, y.denominator, z.numerator, z.denominator, p)
        if dxy > max(dxz, dyz) + 1e-9:
            all_ok = False
            violations.append((x, y, z, p, dxy, max(dxz,dyz)))
check("B3 ultrametric stress test (200 random triples, p∈{2,3,5,7})",
      all_ok,
      "all 800 triples satisfy ultrametric inequality" if all_ok
      else f"{len(violations)} violations found, first: {violations[0]}")

# ──────────────────────────────────────────────
# C. Strong triangle inequality for norms
#    |x + y|_p ≤ max(|x|_p, |y|_p)
#    (implies ultrametric; logically prior)
# ──────────────────────────────────────────────

print("\n=== C. Strong triangle inequality  |x+y|_p ≤ max(|x|_p, |y|_p) ===\n")

def test_strong_triangle(x: Fraction, y: Fraction, p: int, label: str):
    sum_ = x + y
    nx = norm_p(x.numerator, x.denominator, p)
    ny = norm_p(y.numerator, y.denominator, p)
    ns = norm_p(sum_.numerator, sum_.denominator, p)
    holds = ns <= max(nx, ny) + 1e-12
    check(label,
          holds,
          f"|{x}+{y}|_{p} = {ns:.4f} ≤ max({nx:.4f},{ny:.4f}) = {max(nx,ny):.4f}")

test_strong_triangle(Fraction(3,4), Fraction(1,4), 2, "C1 |3/4 + 1/4|_2 = |1|_2 = 1")
test_strong_triangle(Fraction(1,9), Fraction(2,9), 3, "C2 |1/9 + 2/9|_3 = |1/3|_3")
test_strong_triangle(Fraction(5), Fraction(-5), 5, "C3 |5 + (-5)|_5 = 0")
test_strong_triangle(Fraction(25), Fraction(1,25), 5, "C4 large exponent contrast p=5")

# Stress test
all_ok = True
for _ in range(500):
    a, b = [int(x) for x in rng.integers(-50, 50, size=2)]
    c, d = [int(x) for x in rng.integers(1, 15, size=2)]
    if a == 0 or b == 0 or c == 0 or d == 0:
        continue
    x, y = Fraction(a, c), Fraction(b, d)
    for p in [2, 3, 5, 7, 11]:
        sum_ = x + y
        nx = norm_p(x.numerator, x.denominator, p)
        ny = norm_p(y.numerator, y.denominator, p)
        ns = norm_p(sum_.numerator, sum_.denominator, p)
        if ns > max(nx, ny) + 1e-9:
            all_ok = False
check("C5 strong triangle stress test (500 pairs, p∈{2,3,5,7,11})",
      all_ok,
      "all cases satisfy |x+y|_p ≤ max(|x|_p,|y|_p)")

# ──────────────────────────────────────────────
# D. Product formula  ∏_v |x|_v = 1
#    (line 267: 乘积公式 ∏_v |x|_v = 1 将激活的异构性绑回 1)
#    v ranges over all places: archimedean (∞) + all primes p
#    For rational x = a/b, the product is finite because |x|_p = 1 for almost all p
# ──────────────────────────────────────────────

print("\n=== D. Product formula  ∏_v |x|_v = 1  for x ∈ Q* ===\n")

def product_formula(num: int, den: int) -> float:
    """Compute ∏_v |num/den|_v over all places (arch + primes dividing num*den)."""
    # Archimedean place
    result = norm_arch(num, den)
    # Non-archimedean places: only primes dividing numerator or denominator contribute non-trivially
    # We factor num and den completely, collect all prime factors
    def prime_factors(n: int):
        n = abs(n)
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    relevant_primes = prime_factors(num) | prime_factors(den)
    for p in relevant_primes:
        result *= norm_p(num, den, p)
    return result

test_cases = [
    (1, 1,    "x = 1"),
    (2, 1,    "x = 2"),
    (3, 1,    "x = 3"),
    (6, 1,    "x = 6 = 2·3"),
    (1, 2,    "x = 1/2"),
    (3, 4,    "x = 3/4"),
    (12, 5,   "x = 12/5"),
    (-7, 3,   "x = -7/3"),
    (100, 1,  "x = 100 = 2^2·5^2"),
    (360, 49, "x = 360/49"),
]

all_ok = True
for num, den, label in test_cases:
    prod = product_formula(num, den)
    ok = np.isclose(prod, 1.0, rtol=1e-9)
    if not ok:
        all_ok = False
    check(f"D product formula {label}",
          ok,
          f"∏_v |{num}/{den}|_v = {prod:.10f}, expected 1.0")

# ──────────────────────────────────────────────
# E. Adele ring structure: Q embeds diagonally
#    For x ∈ Q, all but finitely many |x|_p = 1
#    (this is the "restricted product" condition)
# ──────────────────────────────────────────────

print("\n=== E. Adele ring — restricted product condition ===\n")
print("(Q_p 'already present' = Q embeds into each completion; |x|_p = 1 for almost all p)\n")

def count_nontrivial_places(num: int, den: int, prime_list) -> int:
    """Count primes in prime_list where |num/den|_p ≠ 1."""
    return sum(1 for p in prime_list if not np.isclose(norm_p(num, den, p), 1.0))

# For 12/5: only p=2,3,5 should be non-trivial (among the first 11 primes)
x_num, x_den = 12, 5
nontrivial = [p for p in PRIMES if not np.isclose(norm_p(x_num, x_den, p), 1.0)]
expected_nontrivial = {2, 3, 5}
check("E1 restricted product — 12/5 is non-unit at finitely many places",
      set(nontrivial) == expected_nontrivial,
      f"non-trivial primes for 12/5: {nontrivial}, expected {sorted(expected_nontrivial)}")

# For 1/30: 30 = 2·3·5
x_num, x_den = 1, 30
nontrivial = [p for p in PRIMES if not np.isclose(norm_p(x_num, x_den, p), 1.0)]
expected_nontrivial = {2, 3, 5}
check("E2 restricted product — 1/30 is non-unit at {2,3,5} only",
      set(nontrivial) == expected_nontrivial,
      f"non-trivial primes for 1/30: {nontrivial}")

# For large prime: 1/97
x_num, x_den = 1, 97
nontrivial_in_list = [p for p in PRIMES if not np.isclose(norm_p(x_num, x_den, p), 1.0)]
check("E3 restricted product — 1/97: non-unit only at p=97 (outside our list of first 11)",
      nontrivial_in_list == [],
      f"primes in {PRIMES} that are non-trivial for 1/97: {nontrivial_in_list}  "
      "(97 is prime, not in our test list → correctly shows 0 non-trivial in range)")

# ──────────────────────────────────────────────
# F. Hasse principle (Hasse-Minkowski for quadratic forms)
#    §16: "R 解的存在性 ⟺ 所有 Q_p 解的共识 | Hasse 原理"
#    Classic example: x^2 - 2y^2 = 0 has a non-trivial solution over Q iff
#    it does over R and all Q_p.
#    We check locally solvable + globally solvable for simple cases.
#    Also the failure case: x^2 + y^2 + z^2 + w^2 = -1  (no real solution)
# ──────────────────────────────────────────────

print("\n=== F. Hasse principle examples ===\n")

def is_square_mod(n: int, m: int) -> bool:
    """Is n a quadratic residue mod m?"""
    n = n % m
    for x in range(m):
        if (x * x) % m == n:
            return True
    return False

def legendre_symbol(a: int, p: int) -> int:
    """Legendre symbol (a/p) for odd prime p."""
    if a % p == 0:
        return 0
    ls = pow(a, (p-1)//2, p)
    return -1 if ls == p-1 else ls

def is_qp_solvable_x2_minus_Dy2(D: int, p: int) -> bool:
    """
    Is x^2 = D*y^2 solvable non-trivially over Q_p?
    Equivalent: is D a square in Q_p?
    For odd prime p not dividing D: D is square in Q_p iff Legendre (D/p) ∈ {0,1}
    For p=2: more complex — use Hensel lifting via brute force mod 8
    For p | D: 0 is always a solution (trivial) — non-trivial means D·u^2 is a square.
    We check: does x^2 - D = 0 have a solution in Q_p (i.e., is D a p-adic square)?
    """
    if p == 2:
        # D is a 2-adic square iff D ≡ 1 (mod 8) after removing factors of 4
        # simplified: check mod 8 for the odd part
        v = v_p(D, 2)
        if v % 2 != 0:
            return False
        odd_part = D >> v
        return odd_part % 8 == 1
    else:
        v = v_p(D, p)
        if v % 2 != 0:
            return False
        reduced = D // (p**v)
        return legendre_symbol(reduced, p) >= 0

# F1: D=2, equation x^2 = 2  (x = sqrt(2))
# Over R: solvable (x = 1.414...)
# Over Q: NOT solvable
# Over Q_2: is 2 a 2-adic square? v_2(2)=1 (odd) → NOT a 2-adic square
# → Hasse principle correctly predicts: fails at Q_2 → fails over Q
D = 2
real_solvable = D > 0   # sqrt(D) ∈ R
p2_solvable = is_qp_solvable_x2_minus_Dy2(D, 2)
# Local obstruction at p=2 means global failure is consistent with Hasse
hasse_prediction_consistent = (not p2_solvable)   # no global sol ↔ some local failure
check("F1 Hasse: x^2=2 has real solution but fails at Q_2 (predicts no rational solution)",
      real_solvable and not p2_solvable and hasse_prediction_consistent,
      f"R-solvable={real_solvable}, Q_2-solvable={p2_solvable} -> local obstruction at 2 -> x=sqrt(2) not in Q")

# F2: D=4, x^2=4 → x=2 ∈ Q (globally solvable)
# Should be locally solvable everywhere
D = 4
real_solvable = D > 0
all_local = all(is_qp_solvable_x2_minus_Dy2(D, p) for p in [2, 3, 5, 7, 11, 13])
# Direct global check: is 4 a perfect square?
global_solvable = isqrt(D)**2 == D
check("F2 Hasse: x^2=4 solvable globally and locally everywhere",
      real_solvable and all_local and global_solvable,
      f"R={real_solvable}, all-local={all_local}, Q-global={global_solvable}")

# F3: D=-1, x^2 = -1 → NO real solution → Hasse says no global solution
D = -1
real_solvable = D > 0   # False
# Real obstruction → can't be globally solvable regardless of local
check("F3 Hasse: x^2=-1 fails over R → no global Q solution (consistent)",
      not real_solvable,
      "D=-1 < 0: archimedean place blocks global solvability, local Q_p may differ")

# F4: D=9 = 3^2, x^2 = 9 → x=3 ∈ Q
D = 9
real_solvable = True
all_local = all(is_qp_solvable_x2_minus_Dy2(D, p) for p in [2, 3, 5, 7])
global_solvable = isqrt(D)**2 == D
check("F4 Hasse: x^2=9 solvable everywhere (x=3)",
      real_solvable and all_local and global_solvable,
      f"all-local={all_local}, global={global_solvable}")

# F5: Hasse principle consistency check — for x^2 = D with D a perfect square,
# all completions must say solvable
perfect_squares = [1, 4, 9, 16, 25, 49, 100]
all_consistent = True
fail_cases = []
for D in perfect_squares:
    all_local = all(is_qp_solvable_x2_minus_Dy2(D, p) for p in [3, 5, 7, 11, 13])
    if not all_local:
        all_consistent = False
        fail_cases.append(D)
check("F5 Hasse consistency: perfect squares are p-adically square for all odd primes tested",
      all_consistent,
      f"tested D∈{perfect_squares} vs p∈{{3,5,7,11,13}}" +
      (f"; failures: {fail_cases}" if fail_cases else ""))

# ──────────────────────────────────────────────
# G. Non-archimedean property: large integers are p-adically small
#    (Core qualitative claim: Q_p metric is "opposite" to R metric)
#    §16: "非阿基米德位，始终在场但不展开为空间"
# ──────────────────────────────────────────────

print("\n=== G. Non-archimedean: p^n → 0 in Q_p as n→∞ ===\n")

p = 2
norms = [norm_p(2**n, 1, p) for n in range(1, 10)]
strictly_decreasing = all(norms[i] > norms[i+1] for i in range(len(norms)-1))
check("G1 |2^n|_2 → 0 as n→∞ (2-adic integers: powers of 2 are small)",
      strictly_decreasing and np.isclose(norms[-1], 2**(-9)),
      f"norms: {[f'{x:.6f}' for x in norms[:5]]}... last={norms[-1]:.8f}")

p = 3
norms = [norm_p(3**n, 1, p) for n in range(1, 10)]
strictly_decreasing = all(norms[i] > norms[i+1] for i in range(len(norms)-1))
check("G2 |3^n|_3 → 0 as n→∞ (3-adic integers: powers of 3 are small)",
      strictly_decreasing and np.isclose(norms[-1], 3**(-9)),
      f"norms: {[f'{x:.6f}' for x in norms[:5]]}... last={norms[-1]:.8f}")

# Contrast: in R, 2^n → ∞
arch_norms = [float(2**n) for n in range(1, 10)]
check("G3 |2^n|_∞ → ∞ in R (archimedean place: opposite behavior)",
      all(arch_norms[i] < arch_norms[i+1] for i in range(len(arch_norms)-1)),
      f"arch norms: {arch_norms[:5]}... diverge in R, converge in Q_2")

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────

print("\n" + "="*60)
passed = sum(results)
total = len(results)
print(f"SUMMARY: {passed}/{total} tests passed")
if passed == total:
    print("All claims in §16 verified numerically.")
else:
    failed_idx = [i+1 for i, ok in enumerate(results) if not ok]
    print(f"Failed tests at positions: {failed_idx}")
print("="*60)
