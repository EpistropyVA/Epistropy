"""
BCC Corner Gauss-Bonnet: Exact rational computation of corner deviation δ
BCC 3×3×3 lattice, 27 body-centers, 540 faces, 6 flat bands at λ=-3
"""

from fractions import Fraction
import math

import sys

# Force UTF-8 stdout so Unicode characters display correctly on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

results = []

def log(s=""):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode('ascii', 'replace').decode('ascii'))
    results.append(str(s))

log("=" * 70)
log("BCC CORNER GAUSS-BONNET — EXACT RATIONAL COMPUTATION")
log("=" * 70)

# ─────────────────────────────────────────────────────────────────
# §1  Known exact f_pure values
# ─────────────────────────────────────────────────────────────────
log("\n§1  Known f_pure values per k-type")
log("-" * 40)

f_gamma  = Fraction(2, 3)
f_edge   = Fraction(23, 40)
f_corner = Fraction(61, 105)

log(f"  Gamma  (1 pt):  f_pure = {f_gamma}  = {float(f_gamma):.15f}")
log(f"  Edge   (6 pts): f_pure = {f_edge}  = {float(f_edge):.15f}")
log(f"  Corner (8 pts): f_pure = {f_corner} = {float(f_corner):.15f}")

# ─────────────────────────────────────────────────────────────────
# §2  Identify f_face = 0.544011544012...
# ─────────────────────────────────────────────────────────────────
log("\n§2  Identifying f_face ≈ 0.544011544012...")
log("-" * 40)

target = 0.544011544011544012  # the repeating pattern suggests a rational

# 2a  Check 377/693
cand_377_693 = Fraction(377, 693)
log(f"  377/693 = {float(cand_377_693):.15f}  (target {target:.15f})")
log(f"    693 = 7×9×11 = {7*9*11},  377 = 13×29 = {13*29}")
log(f"    Match: {abs(float(cand_377_693) - target) < 1e-12}")

# 2b  Scan N up to 100000 for N * target ≈ integer
log("\n  Scanning N × target for near-integers (N ≤ 100000) …")
best_n, best_err, best_frac = 1, 1.0, Fraction(1)
for N in range(1, 100001):
    prod = target * N
    nearest = round(prod)
    err = abs(prod - nearest)
    if err < best_err:
        best_err = err
        best_n   = N
        best_frac = Fraction(nearest, N)
    if err < 1e-9 and N > 1:
        log(f"    N={N:6d}: {N} × target = {prod:.10f}  → {nearest}/{N} = {Fraction(nearest,N)}  err={err:.2e}")

log(f"\n  Best rational found: {best_frac} = {float(best_frac):.15f}  (N={best_n}, err={best_err:.2e})")

# 2c  Use Fraction.limit_denominator
for denom_limit in [100, 1000, 10000, 100000, 1000000]:
    approx = Fraction(target).limit_denominator(denom_limit)
    err = abs(float(approx) - target)
    log(f"  limit_denominator({denom_limit:>7d}): {approx}  err={err:.2e}")

# Accept best rational for f_face
f_face_float = target
# The repeating decimal 0.544011544011... looks like 544011/999999 or similar
# 0.544011544011544... let's check period:
# 0.544 011 544 011 ... period-6 group "544011"?
# 544011 / 999999 = 544011/999999, simplify
g = math.gcd(544011, 999999)
num_period = 544011 // g
den_period = 999999 // g
log(f"\n  Period-6 hypothesis: 544011/999999 = {num_period}/{den_period} = {Fraction(num_period,den_period)} = {float(Fraction(num_period,den_period)):.15f}")

# Another hypothesis: 0.544011544011... with shorter period
# Check period 3: "544" repeating? 544/999 = 0.544544... nope
# The sequence "544011" repeats → period 6
# 0.(544011) = 544011/999999
# Simplify: gcd(544011, 999999)
log(f"  gcd(544011, 999999) = {g}")
log(f"  → {num_period}/{den_period}")

# Check also 0.544011544... as 144/264.7... not integer
# Try: is it p/(p+q) for some structure?
# Best candidate from limit_denominator:
f_face_best = Fraction(f_face_float).limit_denominator(1000000)
log(f"\n  Adopted f_face (limit_denominator 1e6): {f_face_best} = {float(f_face_best):.15f}")

# Use the period-6 fraction if it matches better
f_face_period6 = Fraction(num_period, den_period)
err_period6 = abs(float(f_face_period6) - target)
err_best    = abs(float(f_face_best)    - target)
log(f"  period-6 err = {err_period6:.2e},  limit_denom err = {err_best:.2e}")

if err_period6 < err_best:
    f_face = f_face_period6
    log(f"  → Using period-6: {f_face}")
else:
    f_face = f_face_best
    log(f"  → Using limit_denominator: {f_face}")

# ─────────────────────────────────────────────────────────────────
# §3  BZ average f_avg
# ─────────────────────────────────────────────────────────────────
log("\n§3  BZ-average f_avg = (1·f_Γ + 6·f_edge + 12·f_face + 8·f_corner) / 27")
log("-" * 40)

f_avg = (1*f_gamma + 6*f_edge + 12*f_face + 8*f_corner) / 27
log(f"  f_avg = {f_avg} ≈ {float(f_avg):.15f}")
log(f"  Expected uniform (8/20 = 2/5): {Fraction(2,5)} = {float(Fraction(2,5)):.15f}")
log(f"  Difference f_avg - 2/5 = {f_avg - Fraction(2,5)}")

# Also compute with float f_face for comparison
f_avg_float = (1*float(f_gamma) + 6*float(f_edge) + 12*f_face_float + 8*float(f_corner)) / 27
log(f"  f_avg (float target)   ≈ {f_avg_float:.15f}")

# ─────────────────────────────────────────────────────────────────
# §4  Known δ values — rational identification
# ─────────────────────────────────────────────────────────────────
log("\n§4  Identifying δ values as exact rationals")
log("-" * 40)

delta_floats = {
    'delta_0 (corner)': +0.013782448754,
    'delta_1 (edge)':   -0.002433232828,
    'delta_2 (face)':   -0.010889544909,
    'delta_3 (center)': -0.015723526723,
}

denominators_to_try = [72, 360, 693, 720, 840, 1260, 2520, 5040, 27720, 55440,
                        180180, 360360, 720720,
                        # LCM of small sets
                        math.lcm(8,9,5,7),        # = 2520
                        math.lcm(3,5,7,9,11),      # = 3465
                        math.lcm(3,5,7,11,13),     # = 15015
                        math.lcm(5,7,9,11,13),     # = 45045
                        math.lcm(3,5,7,9,11,13),   # = 45045
                        math.lcm(2,3,5,7,9,11,13), # = 90090
                        ]

delta_rationals = {}
for name, val in delta_floats.items():
    log(f"\n  {name} ≈ {val}")
    # Try limit_denominator at several scales
    best = None
    best_err = 1.0
    for lim in [100, 1000, 10000, 100000, 1000000, 10000000]:
        f = Fraction(val).limit_denominator(lim)
        e = abs(float(f) - val)
        if e < best_err:
            best_err = e
            best = f
        if e < 1e-9:
            log(f"    limit_denominator({lim:>8d}): {str(f):>30s}  err={e:.2e}  EXACT")
            break
        else:
            log(f"    limit_denominator({lim:>8d}): {str(f):>30s}  err={e:.2e}")

    # Try specific denominators
    for d in denominators_to_try:
        prod = val * d
        n = round(prod)
        e = abs(prod - n)
        if e < 1e-6:
            f = Fraction(n, d)
            log(f"    × {d:>7d} → {n}/{d} = {f}  err={e:.2e}")

    delta_rationals[name] = best
    log(f"  → Best: {best} ≈ {float(best):.12f}  (err={best_err:.2e})")

# ─────────────────────────────────────────────────────────────────
# §5  Gauss-Bonnet constraint check
# ─────────────────────────────────────────────────────────────────
log("\n§5  Gauss-Bonnet constraint: 8δ₀ + 12δ₁ + 6δ₂ + δ₃ = 0")
log("-" * 40)

d0_f = float(list(delta_floats.values())[0])
d1_f = float(list(delta_floats.values())[1])
d2_f = float(list(delta_floats.values())[2])
d3_f = float(list(delta_floats.values())[3])

gb_check = 8*d0_f + 12*d1_f + 6*d2_f + d3_f
log(f"  8δ₀ + 12δ₁ + 6δ₂ + δ₃ = {gb_check:.2e}  (should be ~0)")

# With rationals
d0_r = Fraction(d0_f).limit_denominator(10000000)
d1_r = Fraction(d1_f).limit_denominator(10000000)
d2_r = Fraction(d2_f).limit_denominator(10000000)
d3_r = Fraction(d3_f).limit_denominator(10000000)

gb_rational = 8*d0_r + 12*d1_r + 6*d2_r + d3_r
log(f"  Rational check: {gb_rational} ≈ {float(gb_rational):.2e}")

# ─────────────────────────────────────────────────────────────────
# §6  Algebraic relations between δ values
# ─────────────────────────────────────────────────────────────────
log("\n§6  Algebraic relations between δ values")
log("-" * 40)

log(f"  δ₀ + δ₁ + δ₂ + δ₃ = {d0_f+d1_f+d2_f+d3_f:.10f}")
log(f"  δ₀/δ₃ = {d0_f/d3_f:.10f}")
log(f"  3δ₀ + 3δ₁ = {3*d0_f + 3*d1_f:.10f}  (3 missing bonds for corner, 2 for edge)")
log(f"  δ₀ - δ₁ = {d0_f - d1_f:.10f}")
log(f"  δ₁ - δ₂ = {d1_f - d2_f:.10f}")
log(f"  δ₂ - δ₃ = {d2_f - d3_f:.10f}")
log(f"  (δ₀-δ₁)/(δ₁-δ₂) = {(d0_f-d1_f)/(d1_f-d2_f):.10f}")
log(f"  (δ₁-δ₂)/(δ₂-δ₃) = {(d1_f-d2_f)/(d2_f-d3_f):.10f}")

# Missing bond counts: corner=3, edge=2, face=1, center=0
missing = [3, 2, 1, 0]
deltas  = [d0_f, d1_f, d2_f, d3_f]

# Linear fit: δ = a*n + b
# (3a+b, 2a+b, a+b, b) = deltas
# a = (δ₀-δ₃)/3 - (δ₁-δ₃)/2... use least squares
import numpy as np
A = np.array([[n, 1] for n in missing])
b_vec = np.array(deltas)
coeffs, res, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
log(f"\n  Linear fit δ = a·n_missing + b:")
log(f"    a = {coeffs[0]:.10f},  b = {coeffs[1]:.10f}")
log(f"    Residuals: {[float(f'{r:.4e}') for r in b_vec - A@coeffs]}")

# Quadratic fit
A2 = np.array([[n**2, n, 1] for n in missing])
coeffs2, _, _, _ = np.linalg.lstsq(A2, b_vec, rcond=None)
log(f"\n  Quadratic fit δ = a·n² + b·n + c:")
log(f"    a={coeffs2[0]:.10f}, b={coeffs2[1]:.10f}, c={coeffs2[2]:.10f}")
log(f"    Residuals: {[float(f'{r:.4e}') for r in b_vec - A2@coeffs2]}")

# ─────────────────────────────────────────────────────────────────
# §7  Euler / Gauss-Bonnet: alternating-sign combination
# ─────────────────────────────────────────────────────────────────
log("\n§7  Discrete Gauss-Bonnet: alternating-sign check")
log("-" * 40)

log(f"  8δ₀ - 12δ₁ + 6δ₂ - δ₃ = {8*d0_f - 12*d1_f + 6*d2_f - d3_f:.10f}")
log(f"  8δ₀ - 12δ₁ + 6δ₂        = {8*d0_f - 12*d1_f + 6*d2_f:.10f}")
log(f"  Missing-bond-weighted sum: Σ n_missing × δ = {sum(m*d for m,d in zip(missing,deltas)):.10f}")
log(f"  Σ n_missing² × δ        = {sum(m**2*d for m,d in zip(missing,deltas)):.10f}")

# ─────────────────────────────────────────────────────────────────
# §8  δ₀ ∝ f_Γ - f_avg  hypothesis
# ─────────────────────────────────────────────────────────────────
log("\n§8  δ₀ proportional to (f_Γ - f_avg)?")
log("-" * 40)

diff_gamma_avg = float(f_gamma) - f_avg_float
log(f"  f_Γ - f_avg = {diff_gamma_avg:.10f}")
log(f"  δ₀          = {d0_f:.10f}")
log(f"  ratio δ₀ / (f_Γ - f_avg) = {d0_f / diff_gamma_avg:.10f}")

# Check with each k-type deviation
for ktype, f_kt, n_k in [('Gamma',float(f_gamma),1),('Edge',float(f_edge),6),
                           ('Face',f_face_float,12),('Corner',float(f_corner),8)]:
    dev = f_kt - f_avg_float
    log(f"  f_{ktype:<6s} - f_avg = {dev:+.10f}  (× {n_k}  = {n_k*dev:+.10f})")

# ─────────────────────────────────────────────────────────────────
# §9  Shell-local f_pure averages
# ─────────────────────────────────────────────────────────────────
log("\n§9  Shell-local analysis: which k-points 'see' each shell?")
log("-" * 40)
log("  (In 3×3×3 BCC: each BC contributes to which k-types?)")
log("  All k-points are translation-equivalent in periodic BC,")
log("  so each k-point sees the SAME local environment.")
log(f"  f_avg (uniform over BZ) = {f_avg_float:.10f}")

# The pure-corner face count: 8 per BC in 3×3×3
# Corner BCs in open BC: 8 corners have 3 neighbors instead of 6
# The deviation δ₀ accounts for weight redistribution

# ─────────────────────────────────────────────────────────────────
# §10  Direct rational construction of δ₀
# ─────────────────────────────────────────────────────────────────
log("\n§10  Direct construction: δ₀ from BZ-weights")
log("-" * 40)
log("  Hypothesis: δ₀ = Σ_k weight(k,corner_BC) × (f_pure(k) - f_avg)")
log("  In open BC, the corner BC loses 3 of its 6 periodic bonds.")
log("  The missing bonds are the ones wrapping from x=0→x=2 etc.")
log("  k-points that contributed to those bonds lose their periodic phase.")

# For each f_pure type, compute contribution weight in corner-local frame
# Corners of BCC Brillouin zone for 3×3×3 periodic:
# kx,ky,kz ∈ {0, 2π/3, 4π/3}  (or equivalently mod 2π)
# Gamma: (0,0,0)   — 1 point
# Edge:  one zero, two non-zero — wait, for 3³ BZ:
#   types by (kx=0?, ky=0?, kz=0?):
#   (0,0,0): Gamma — 1 point
#   (0,0,1) permutations: 3 coords, 1 zero, 2 nonzero → "face" of BZ cube → 12 points
#   (0,1,1) permutations: 2 zeros → "edge" → 6 points?
# Wait — need to recount for 3³
# In 3×3×3 periodic, k-grid has 3³=27 points
# k = (2πn₁/3, 2πn₂/3, 2πn₃/3) with n_i ∈ {0,1,2}
# mod Z: n=2 is equiv to n=-1 (same shell as n=1 by inversion)
# So # nonzero components:
#   0 nonzero: (0,0,0) Gamma — 1 point
#   1 nonzero: (±,0,0) × 3 dirs × 1 value → 3×2 but n=1 & n=2=−1 are distinct k-points → 6 points  (these are BZ-face centers)
#   2 nonzero: (±,±,0) × 3 pairs × 1×1 = 3×4=12 points  (BZ-edge midpoints? or face?)
#   3 nonzero: (±,±,±) × 8 = 8 points  (BZ corners)
# Total: 1+6+12+8 = 27 ✓

log("\n  k-grid for 3×3×3: n_i ∈ {0,1,2}")
log("  # nonzero components determines k-type:")
log("  0: Gamma (1), 1: BZ-face (6), 2: BZ-edge (12), 3: BZ-corner (8)")
log("  — Note: the labeling in the problem statement uses 'edge' for n_nonzero=1")
log("    and 'face' for n_nonzero=2; let's re-examine...")

# Re-examine: the problem says
#   Gamma (1), BZ-edge (6), BZ-face (12), BZ-corner (8)
# 6 BZ-edge → n_nonzero=1 (face-centers of BZ cube)
# 12 BZ-face → n_nonzero=2 (edge-midpoints of BZ cube)
# 8 BZ-corner → n_nonzero=3 (corners of BZ cube)
log("  Problem naming: edge(6)=1 nonzero, face(12)=2 nonzero, corner(8)=3 nonzero")

# For the OPEN boundary corner BC at (0,0,0):
# It loses the 3 bonds in the -x, -y, -z directions (which wrapped around)
# The flat-band projector P_{-3}(k) = |ψ_k><ψ_k| for each k
# The corner BC's contribution to the projected weight in OPEN BC:

# BZ weighting: in open BC, the corner BC at site r=(0,0,0) sees
# k-dependent weight: w(k,r) ~ |φ_k(r)|² for flat band φ
# For BCC with open BC, this is approximately the same as periodic
# PLUS corrections from the missing bonds.

# The correction to the projector from removing bond (r, r+δ):
# ΔP ≈ (δP/δt) × Δt  where Δt is the hopping strength change
# For λ=-3 flat band in BCC, the correction is exactly computable.

# Key formula (from Aharonov-Bohm / lattice theory):
# δ₀ = (1/27) Σ_k [f_pure(k) - f_avg] × correction_factor(k)
# where correction_factor(k) depends on which bonds are removed

# For corner at (0,0,0) losing bonds in +x,+y,+z directions:
# The Fourier weight at k of the missing bond contribution is:
# ~ cos(kx) + cos(ky) + cos(kz)  — but sign depends on convention

# Compute: for each k-type, what is cos(kx)+cos(ky)+cos(kz)?
# k_i ∈ {0, 2π/3, 4π/3} for n_i ∈ {0, 1, 2}
import cmath

k_vals = [0, 2*math.pi/3, 4*math.pi/3]
k_types = {}
for n1 in range(3):
    for n2 in range(3):
        for n3 in range(3):
            kx, ky, kz = k_vals[n1], k_vals[n2], k_vals[n3]
            n_nonzero = (n1 != 0) + (n2 != 0) + (n3 != 0)
            cos_sum = math.cos(kx) + math.cos(ky) + math.cos(kz)
            phase_sum = cmath.exp(1j*kx) + cmath.exp(1j*ky) + cmath.exp(1j*kz)
            key = n_nonzero
            if key not in k_types:
                k_types[key] = []
            k_types[key].append((kx,ky,kz,cos_sum,phase_sum))

log("\n  cos(kx)+cos(ky)+cos(kz) by k-type:")
type_names = {0:'Gamma',1:'Edge(6)',2:'Face(12)',3:'Corner(8)'}
for nz in sorted(k_types):
    pts = k_types[nz]
    cos_vals = [p[3] for p in pts]
    log(f"    {type_names[nz]}: {set(round(c,6) for c in cos_vals)}")

# ─────────────────────────────────────────────────────────────────
# §11  Rational structure of δ₀ via BZ decomposition
# ─────────────────────────────────────────────────────────────────
log("\n§11  Rational structure: δ₀ as BZ-weighted combination")
log("-" * 40)

# f_pure per k-type and k-counts
k_data = [
    ('Gamma',   1,  f_gamma),
    ('Edge',    6,  f_edge),
    ('Face',   12,  f_face),
    ('Corner',  8,  f_corner),
]

# cos(kx)+cos(ky)+cos(kz) by n_nonzero type:
# n=0 (Gamma): 3×cos(0) = 3
# n=1 (Edge):  1×cos(0) + 2×cos(2π/3) = 1 + 2×(-1/2) = 0
# n=2 (Face):  2×cos(0) + 1×cos(2π/3) = 2 + (-1/2) = 3/2... wait
# Wait for n_nonzero=1: exactly 1 component is nonzero
#   cos(k_nonzero) = cos(2π/3) = -1/2, the other two cos(0)=1
#   sum = 1 + 1 + (-1/2) = 3/2? No wait:
#   If n=1: two components are 0, one is 2π/3
#   cos(0)+cos(0)+cos(2π/3) = 1+1-1/2 = 3/2

# Let me recompute:
log("  Recomputing cos-sums precisely:")
for n1 in range(3):
    for n2 in range(3):
        for n3 in range(3):
            kx,ky,kz = k_vals[n1],k_vals[n2],k_vals[n3]
            n_nonzero = (n1!=0)+(n2!=0)+(n3!=0)
            if (n1,n2,n3) in [(0,0,0),(1,0,0),(1,1,0),(1,1,1)]:
                c = math.cos(kx)+math.cos(ky)+math.cos(kz)
                log(f"    n=({n1},{n2},{n3}) → cos_sum={c:.6f}")

# Exact rational cos values: cos(2π/3) = -1/2
cos_0   = Fraction(1)
cos_2pi3 = Fraction(-1,2)  # exact

cos_sum_by_type = {
    0: cos_0 + cos_0 + cos_0,          # Gamma: 3
    1: cos_0 + cos_0 + cos_2pi3,       # Edge:  3/2
    2: cos_0 + cos_2pi3 + cos_2pi3,    # Face:  0
    3: cos_2pi3 + cos_2pi3 + cos_2pi3, # Corner: -3/2
}
log("\n  Exact rational cos(kx)+cos(ky)+cos(kz) by type:")
for nz,cs in sorted(cos_sum_by_type.items()):
    log(f"    n_nonzero={nz} ({type_names[nz]}): cos_sum = {cs}")

# The Brillouin zone average of cos_sum = (1×3 + 6×3/2 + 12×0 + 8×(-3/2))/27
cos_avg = (1*cos_sum_by_type[0] + 6*cos_sum_by_type[1] + 12*cos_sum_by_type[2] + 8*cos_sum_by_type[3]) / 27
log(f"\n  BZ-average cos_sum = {cos_avg}  (should be 0 by symmetry: {float(cos_avg):.6f})")

# Hypothesis: δ₀ = C × Σ_k f_pure(k) × cos_sum(k) / 27
# where C is some rational constant
weighted_cos = (1*f_gamma*cos_sum_by_type[0] +
                6*f_edge *cos_sum_by_type[1] +
                12*f_face*cos_sum_by_type[2] +
                8*f_corner*cos_sum_by_type[3]) / 27
log(f"\n  Σ_k f_pure(k)·cos_sum(k) / 27 = {weighted_cos} ≈ {float(weighted_cos):.10f}")
log(f"  δ₀ / this = {d0_f / float(weighted_cos):.10f}")

# Also try cos²:
cos2_by_type = {nz: cs**2 for nz,cs in cos_sum_by_type.items()}
weighted_cos2 = (1*f_gamma*cos2_by_type[0] +
                 6*f_edge *cos2_by_type[1] +
                 12*f_face*cos2_by_type[2] +
                 8*f_corner*cos2_by_type[3]) / 27
log(f"  Σ_k f_pure(k)·cos²_sum(k) / 27 = {weighted_cos2} ≈ {float(weighted_cos2):.10f}")

# ─────────────────────────────────────────────────────────────────
# §12  Direct rational candidate for δ₀
# ─────────────────────────────────────────────────────────────────
log("\n§12  Direct rational candidates for all δ values")
log("-" * 40)

# Given f_pure values and the Gauss-Bonnet constraint, try to find
# rational expressions for each δ.

# The deviation δ at each shell s = (projection onto faces of shell s) - (graph degree of shell s)
# In the open-BC system, the flat-band projector P_{-3} has total trace = 6×27 = 162
# Wait: 6 flat bands × 27 BC sites = 162? But open BC has 27 BC sites still.
# Actually in open BC: flat bands may not be exactly flat anymore.
# The problem states δ values ARE computed (numerically), so they exist.

# The formula for δ₀ in terms of BZ sums:
# δ₀ = P_{-3,corner BC}(ff) - 3   (degree of corner in open BC)
# The corner BC has 3 bonds in open BC (vs 6 in periodic)
# So: P_{-3,corner BC}(ff) = 3 + δ₀ = 3 + 0.013782... ≈ 3.01378...

# The flat-band projector trace over corner BC faces = 3.013782448754

# From the BZ analysis:
# In periodic BC: P_{-3}(ff) for any BC = 6 flat bands × (weight on that face's endpoint)
#   = (6/27) × total_weight × f_pure correction
# Actually in periodic BC, by translation symmetry:
#   P_{-3,any BC}(ff) = (total flat-band weight) / 27 = (6×27)/27 = 6
#   Wait: trace(P_{-3}) = 6 (eigenvalue multiplicity), and there are 27×8 = 216 face-vertices?
# Let me reconsider: the system has 27 BC sites, each with up to 6 bonds.
# The flat-band projector P_{-3} is 27×27 (in site space) with trace 6 (6 flat bands).
# In periodic BC: P_{-3}(ii) = 6/27 for each site i (by symmetry).
# The diagonal element = weight on site i = 6/27.

# The "projection onto faces of BC i" = sum over neighbors j of |<i|P_{-3}|j>|²
# Hmm, but the problem defines δ as:
# "Σ_faces_of_corner_BC [P_{-3}]_{ff} - deg(corner_BC)"

# Let's interpret [P_{-3}]_{ff} for face f = bond (i,j):
# This is the projection of the flat-band states onto the bond state |f> = (|i>-|j>)/√2
# So [P_{-3}]_{ff} = <f|P_{-3}|f> = (1/2)(P_{ii} + P_{jj} - P_{ij} - P_{ji})

# In periodic BC: P_{ii} = 6/27 for all i
# P_{ij} for i,j neighbors: by symmetry all equal P_nn
# sum over all 6 neighbors of i: Σ_j P_{ij} = 0 (since P_{-3} projects onto eigenvalue -3,
#   and H|i> = -3|i> + corrections, so the sum is related to eigenvalue equation)

# Actually: P_{-3} satisfies H·P_{-3} = -3·P_{-3}
# (H·P)_{ij} = Σ_k H_{ik}P_{kj} = -3 P_{ij}
# H_{ik} = -1 if (i,k) bond, 0 otherwise (for BCC graph)
# So: -Σ_{k~i} P_{kj} = -3 P_{ij}
# → Σ_{k~i} P_{kj} = 3 P_{ij}
# Setting j=i: Σ_{k~i} P_{ki} = 3 P_{ii}
# But P is Hermitian: P_{ki} = P*_{ik} = P_{ik} (real)
# So: Σ_{k~i} P_{ik} = 3 × (6/27) = 18/27 = 2/3 (in periodic BC)

# Now: Σ_f [P]_{ff} for all faces of site i:
# = Σ_{j~i} (1/2)(P_{ii} + P_{jj} - 2P_{ij})
# = (deg_i/2)(P_{ii} + P_{jj}) - Σ_{j~i} P_{ij}
# In periodic BC, P_{jj} = P_{ii} = 6/27:
# = (deg_i/2)(2×6/27) - Σ_{j~i} P_{ij}
# = deg_i × (6/27) - (2/3)
# For deg_i=6: = 6×(6/27) - 2/3 = 36/27 - 18/27 = 18/27 = 2/3
# Hmm wait, that's the projection, and deg=6, so δ = 2/3 - 6? That's wrong.

# I think the formula means: sum of diagonal P-elements for the 20 faces (tetrahedra)
# that contain the BC site, not the 6 bonds.

# Let me just focus on the rational identification of the given δ values.

# Given target δ₀ ≈ 0.013782448754, find best exact rational.
# Let's try systematically: what denominators d in 1..200 give d×δ₀ near integer?
log("  Scanning for d such that d × δ₀ is near-integer (d ≤ 10000):")
delta0_val = 0.013782448754
near_int_hits = []
for d in range(1, 10001):
    prod = delta0_val * d
    n_near = round(prod)
    err = abs(prod - n_near)
    if err < 0.0001:
        f_r = Fraction(n_near, d)
        near_int_hits.append((d, n_near, err, f_r))

# Show best 10 by error
near_int_hits.sort(key=lambda x: x[2])
log(f"  Top 10 by error:")
for d,n,err,f_r in near_int_hits[:10]:
    log(f"    {n}/{d} = {f_r} ≈ {float(f_r):.12f}  err={err:.2e}")

# ─────────────────────────────────────────────────────────────────
# §13  Summary: candidate exact rationals
# ─────────────────────────────────────────────────────────────────
log("\n§13  Summary: best rational candidates for all δ")
log("=" * 70)

all_deltas = [
    ('δ₀ (corner)',  +0.013782448754),
    ('δ₁ (edge)',    -0.002433232828),
    ('δ₂ (face)',    -0.010889544909),
    ('δ₃ (center)', -0.015723526723),
]

best_rationals = []
for name, val in all_deltas:
    # Scan
    hits = []
    for d in range(1, 1000001):
        prod = val * d
        n_near = round(prod)
        if n_near == 0:
            continue
        err = abs(prod - n_near)
        if err < 1e-7:
            hits.append((d, n_near, err, Fraction(n_near, d)))
    hits.sort(key=lambda x: x[2])
    if hits:
        best = hits[0]
        log(f"  {name} ≈ {val}")
        log(f"    Best: {best[1]}/{best[0]} = {best[3]} ≈ {float(best[3]):.12f}  err={best[2]:.2e}")
        best_rationals.append(best[3])
        # Check Gauss-Bonnet with these
        for h in hits[:5]:
            if h[2] < 1e-8:
                log(f"    Also: {h[1]}/{h[0]} = {h[3]}  err={h[2]:.2e}")
    else:
        log(f"  {name}: no rational found with d≤1e6, err<1e-7")
        best_rationals.append(Fraction(val).limit_denominator(1000000))

log()
if len(best_rationals) == 4:
    d0_r, d1_r, d2_r, d3_r = best_rationals
    gb = 8*d0_r + 12*d1_r + 6*d2_r + d3_r
    log(f"  Gauss-Bonnet check with best rationals: 8δ₀+12δ₁+6δ₂+δ₃ = {gb} ≈ {float(gb):.4e}")
    log(f"  δ₀ = {d0_r}")
    log(f"  δ₁ = {d1_r}")
    log(f"  δ₂ = {d2_r}")
    log(f"  δ₃ = {d3_r}")

# ─────────────────────────────────────────────────────────────────
# §14  f_face exact rational via BZ consistency
# ─────────────────────────────────────────────────────────────────
log("\n§14  Deriving f_face from BZ consistency constraints")
log("-" * 40)

# If we assume δ₀ = exact rational r₀, and we know
# δ₀ = function(f_Γ, f_edge, f_face, f_corner)
# we might back-solve for f_face.

# However, without the explicit formula relating δ to f_pure values,
# we use the period-6 fraction as our best rational for f_face:
log(f"  f_face (period-6 rational): {f_face_period6} = {num_period}/{den_period}")
log(f"  Verification: {float(f_face_period6):.15f}")
log(f"  Target:       {target:.15f}")
log(f"  Error:        {abs(float(f_face_period6)-target):.2e}")

# Check: does period-6 give f_avg with nice denominator?
f_avg_check = (1*f_gamma + 6*f_edge + 12*f_face_period6 + 8*f_corner) / 27
log(f"\n  f_avg with period-6 f_face: {f_avg_check} ≈ {float(f_avg_check):.15f}")
log(f"  Denominator: {f_avg_check.denominator}")

log("\n" + "=" * 70)
log("END OF COMPUTATION")
log("=" * 70)

# Save results
out_path = r"d:\AI thoery\.agent\scripts\bcc_corner_gauss_bonnet_results.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results))
print(f"\nResults saved to: {out_path}")
