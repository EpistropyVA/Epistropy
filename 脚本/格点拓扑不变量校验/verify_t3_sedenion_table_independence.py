# verify_t3_sedenion_table_independence.py
# Compare two sedenion multiplication table implementations and verify algebraic identities.
#
# Table A: sedenion_check.py — numpy recursive _build_table → (_SIGN16, _IDX16)
#   Access: sign_T[i,j] and idx_T[i,j]   →  e_i * e_j = sign_T[i,j] * e_{idx_T[i,j]}
#
# Table B: sedenion_klein_84.py — iterative cd_mult_table(4) → mul_table
#   Access: mul_table[i][j] = (index, sign)   →  e_i * e_j = sign * e_{index}
#
# Compare all 256 entries (16x16). Then verify 5 algebraic identities on the shared table.

import sys
import os
import io
import importlib
import numpy as np

# ---------------------------------------------------------------------------
# Import Table A from sedenion_check.py
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(parent_dir, "十六元数零因子几何"))

# sedenion_check has no top-level prints — safe to import directly
import sedenion_check as sc

sign_A = sc._SIGN16   # shape (16,16) int8
idx_A  = sc._IDX16    # shape (16,16) int16

print("Table A (sedenion_check._build_table) loaded.")
print(f"  sign_A shape: {sign_A.shape}, dtype: {sign_A.dtype}")
print(f"  idx_A  shape: {idx_A.shape},  dtype: {idx_A.dtype}")
print()

# ---------------------------------------------------------------------------
# Import Table B from sedenion_klein_84.py — suppress its print statements
# ---------------------------------------------------------------------------
import contextlib

# Redirect stdout during import to suppress "Building..." / "Done." prints
_stdout_buf = io.StringIO()
with contextlib.redirect_stdout(_stdout_buf):
    # Force fresh import (in case already cached)
    if 'sedenion_klein_84' in sys.modules:
        del sys.modules['sedenion_klein_84']
    import sedenion_klein_84 as sk

mul_table_B = sk.mul_table  # mul_table[i][j] = (index, sign)
print("Table B (sedenion_klein_84.cd_mult_table) loaded.")
print(f"  mul_table_B size: {len(mul_table_B)}x{len(mul_table_B[0])}")
print()

# ---------------------------------------------------------------------------
# Part 1: Compare all 256 entries
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 1: Compare all 256 entries (index AND sign)")
print("=" * 60)

mismatches = []
for i in range(16):
    for j in range(16):
        # Table A: e_i * e_j
        sA = int(sign_A[i, j])
        kA = int(idx_A[i, j])
        # Table B: e_i * e_j
        kB, sB = mul_table_B[i][j]
        kB = int(kB)
        sB = int(sB)
        if kA != kB or sA != sB:
            mismatches.append((i, j, sA, kA, sB, kB))

if not mismatches:
    print("PASS: All 256 entries match (index and sign) between Table A and Table B.")
else:
    print(f"FAIL: {len(mismatches)} mismatches found:")
    for (i, j, sA, kA, sB, kB) in mismatches[:20]:
        print(f"  e_{i}*e_{j}: A=({sA}*e_{kA}), B=({sB}*e_{kB})")
    if len(mismatches) > 20:
        print(f"  ... and {len(mismatches)-20} more")
print()

# ---------------------------------------------------------------------------
# Helper: multiply two sedenion vectors using Table A (numpy-based)
# ---------------------------------------------------------------------------

def mul_sed(x, y):
    """Sedenion product using Table A structure constants."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xy_outer = np.outer(x, y)
    contributions = sign_A * xy_outer
    result = np.zeros(16)
    np.add.at(result, idx_A.ravel(), contributions.ravel())
    return result


def basis(i):
    v = np.zeros(16)
    v[i] = 1.0
    return v


# ---------------------------------------------------------------------------
# Part 2a: Flexibility x(yx) = (xy)x for 200 random integer sedenion pairs
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 2a: Flexibility x(yx) = (xy)x (200 random integer pairs)")
print("=" * 60)

rng = np.random.default_rng(42)
max_flex_residual = 0.0
flex_fail = 0

for _ in range(200):
    x = rng.integers(-5, 6, size=16).astype(float)
    y = rng.integers(-5, 6, size=16).astype(float)
    lhs = mul_sed(x, mul_sed(y, x))  # x(yx)
    rhs = mul_sed(mul_sed(x, y), x)  # (xy)x
    res = np.max(np.abs(lhs - rhs))
    max_flex_residual = max(max_flex_residual, res)
    if res > 1e-10:
        flex_fail += 1

print(f"Max residual |x(yx) - (xy)x|: {max_flex_residual:.3e}")
if flex_fail == 0 and max_flex_residual < 1e-10:
    print("PASS: Flexibility holds for all 200 random pairs.")
else:
    print(f"FAIL: {flex_fail} violations (max residual {max_flex_residual:.3e}).")
print()

# ---------------------------------------------------------------------------
# Part 2b: e_i^2 = -1 for i = 1..15
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 2b: e_i^2 = -1 for i = 1..15")
print("=" * 60)

sq_pass = True
for i in range(1, 16):
    ei = basis(i)
    sq = mul_sed(ei, ei)
    expected = -basis(0)
    res = np.max(np.abs(sq - expected))
    if res > 1e-14:
        print(f"  FAIL: e_{i}^2 = {sq}, expected {expected}, residual {res:.3e}")
        sq_pass = False

if sq_pass:
    print("PASS: e_i^2 = -1 for all i = 1..15.")
else:
    print("FAIL: Some e_i^2 != -1.")
print()

# ---------------------------------------------------------------------------
# Part 2c: Anticommutativity e_i*e_j = -e_j*e_i for i != j >= 1
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 2c: Anticommutativity e_i*e_j = -e_j*e_i for i != j >= 1")
print("=" * 60)

anti_fail = 0
anti_max = 0.0
for i in range(1, 16):
    for j in range(1, 16):
        if i == j:
            continue
        ei, ej = basis(i), basis(j)
        eij = mul_sed(ei, ej)
        eji = mul_sed(ej, ei)
        res = np.max(np.abs(eij + eji))
        anti_max = max(anti_max, res)
        if res > 1e-14:
            anti_fail += 1

print(f"Max residual |e_i*e_j + e_j*e_i|: {anti_max:.3e}")
if anti_fail == 0:
    print("PASS: Anticommutativity holds for all pairs i != j >= 1.")
else:
    print(f"FAIL: {anti_fail} violations.")
print()

# ---------------------------------------------------------------------------
# Part 2d: Quaternion subalgebra {e0,e1,e2,e3}: e1*e2=e3, e2*e3=e1, e3*e1=e2
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 2d: Quaternion subalgebra e1*e2=e3, e2*e3=e1, e3*e1=e2")
print("=" * 60)

e1, e2, e3 = basis(1), basis(2), basis(3)
cyclic = [
    ("e1*e2", mul_sed(e1, e2), e3),
    ("e2*e3", mul_sed(e2, e3), e1),
    ("e3*e1", mul_sed(e3, e1), e2),
]
quat_pass = True
for name, result, expected in cyclic:
    res = np.max(np.abs(result - expected))
    ok = res < 1e-14
    print(f"  {name} = {name.split('*')[0][-1]}_{name.split('*')[1][-1]}: "
          f"residual {res:.3e} — {'PASS' if ok else 'FAIL'}")
    if not ok:
        quat_pass = False

if quat_pass:
    print("PASS: Quaternion subalgebra cyclic relations hold.")
else:
    print("FAIL: Quaternion subalgebra relations violated.")
print()

# ---------------------------------------------------------------------------
# Part 2e: Norm multiplicativity on octonion subalgebra {e0..e7}
# ---------------------------------------------------------------------------
print("=" * 60)
print("PART 2e: Norm multiplicativity |x*y|^2 = |x|^2 * |y|^2 on octonions (100 random)")
print("=" * 60)

rng2 = np.random.default_rng(137)
norm_fail = 0
max_norm_res = 0.0

for _ in range(100):
    # Random integer octonions (components 0..7 only)
    x_full = np.zeros(16)
    y_full = np.zeros(16)
    x_full[:8] = rng2.integers(-10, 11, size=8).astype(float)
    y_full[:8] = rng2.integers(-10, 11, size=8).astype(float)

    xy = mul_sed(x_full, y_full)
    # Check that xy also lives in octonion subalgebra (components 8..15 should be 0)
    oct_check = np.max(np.abs(xy[8:]))
    if oct_check > 1e-10:
        # Product falls outside octonion subalgebra — expected since sedenions are not alt
        # This is actually allowed: octonions (e0..e7) ARE closed under mult
        # If it fails, that's a construction bug
        print(f"  WARNING: product has non-zero high components: {oct_check:.3e}")
        norm_fail += 1
        continue

    norm_x = np.dot(x_full, x_full)
    norm_y = np.dot(y_full, y_full)
    norm_xy = np.dot(xy, xy)
    expected_norm = norm_x * norm_y
    res = abs(norm_xy - expected_norm)
    max_norm_res = max(max_norm_res, res)
    if res > 1e-8:
        norm_fail += 1

print(f"Max residual ||xy|^2 - |x|^2|y|^2|: {max_norm_res:.3e}")
if norm_fail == 0:
    print("PASS: Norm multiplicativity holds on octonion subalgebra for all 100 pairs.")
else:
    print(f"FAIL: {norm_fail} violations (including possible subalgebra closure failures).")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY")
print("=" * 60)
results = {
    "Table comparison (256 entries)": len(mismatches) == 0,
    "2a Flexibility (200 pairs)":     flex_fail == 0 and max_flex_residual < 1e-10,
    "2b e_i^2 = -1 (i=1..15)":        sq_pass,
    "2c Anticommutativity":            anti_fail == 0,
    "2d Quaternion subalgebra":        quat_pass,
    "2e Norm multiplicativity (oct)":  norm_fail == 0,
}
all_pass = True
for name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed:
        all_pass = False
print()
print("OVERALL:", "PASS" if all_pass else "FAIL")
