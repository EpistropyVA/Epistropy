# e16_experiment.py
# Cayley-Dickson doubling: sedenions (16) -> trigintaduonions (32)
# Compare zero-divisor counts before and after adding e_16
#
# Hypothesis: if e_16 could absorb cancellations like e_0/e_8, the original
# 84 zero-divisor pairs would disappear. If they persist, termination is real.

import sys
import io
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================================================
# Build multiplication table via Cayley-Dickson
# Convention: (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
# ============================================================

def cd_mult_table(n):
    """Build multiplication table for 2^n-ion algebra via Cayley-Dickson.
    Returns mul[a][b] = (index, sign) where e_a * e_b = sign * e_index."""
    sz = 1
    current_mul = [[(0, 1)]]

    for level in range(n):
        new_sz = sz * 2
        new_mul = [[None] * new_sz for _ in range(new_sz)]

        for a in range(sz):
            for c in range(sz):
                idx, sgn = current_mul[a][c]
                new_mul[a][c] = (idx, sgn)

                idx2, sgn2 = current_mul[c][a]
                new_mul[a][c + sz] = (idx2 + sz, sgn2)

                conj_sgn_c = 1 if c == 0 else -1
                idx3, sgn3 = current_mul[a][c]
                new_mul[a + sz][c] = (idx3 + sz, sgn3 * conj_sgn_c)

                conj_sgn_c2 = 1 if c == 0 else -1
                idx4, sgn4 = current_mul[c][a]
                new_mul[a + sz][c + sz] = (idx4, -conj_sgn_c2 * sgn4)

        sz = new_sz
        current_mul = new_mul

    return current_mul


def sedenion_mul(x, y, mul_table, dim):
    """Multiply two elements (dicts {index: coeff}) using mul_table."""
    result = {}
    for ai, ac in x.items():
        for bi, bc in y.items():
            idx, sgn = mul_table[ai][bi]
            coeff = ac * bc * sgn
            result[idx] = result.get(idx, 0) + coeff
    return {k: v for k, v in result.items() if v != 0}


def count_zero_divisors(mul_table, dim):
    """Count zero-divisor pairs of the form (e_a + s*e_b)(e_c + t*e_d) = 0
    where 0 < a < b < dim, 0 < c < d < dim, s,t in {+1,-1}.
    Returns (ordered_count, unordered_list_of_configs)."""
    pairs = list(combinations(range(1, dim), 2))
    signs = [1, -1]

    zero_div_ordered = []
    for (a, b) in pairs:
        for s in signs:
            x = {a: 1, b: s}
            for (c, d) in pairs:
                for t in signs:
                    y = {c: 1, d: t}
                    prod = sedenion_mul(x, y, mul_table, dim)
                    if not prod:
                        zero_div_ordered.append(((a, b, s), (c, d, t)))

    # Unordered
    seen = set()
    unordered = []
    for (fac1, fac2) in zero_div_ordered:
        key = tuple(sorted([fac1, fac2]))
        if key not in seen:
            seen.add(key)
            unordered.append((fac1, fac2))

    # Index-pair configs (ignoring signs)
    configs = set()
    for (fac1, fac2) in unordered:
        a, b, s = fac1
        c, d, t = fac2
        configs.add(frozenset([frozenset([a, b]), frozenset([c, d])]))

    return len(zero_div_ordered), unordered, configs


# ============================================================
# Main experiment
# ============================================================

print("=" * 60)
print("SEDENION -> TRIGINTADUONION ZERO-DIVISOR EXPERIMENT")
print("=" * 60)

# Step 1: Build sedenion (2^4 = 16) multiplication table
print("\n[1] Building sedenion multiplication table (dim=16)...")
mul16 = cd_mult_table(4)
print("    Done.")

# Step 2: Count sedenion zero-divisors
print("\n[2] Counting sedenion zero-divisor pairs...")
ord16, unord16, configs16 = count_zero_divisors(mul16, 16)
print(f"    Ordered pairs:   {ord16} (expect 168)")
print(f"    Unordered pairs: {len(unord16)} (expect 84)")
print(f"    Index configs:   {len(configs16)} (expect 84)")

# Step 3: Build trigintaduonion (2^5 = 32) multiplication table
print("\n[3] Building trigintaduonion multiplication table (dim=32)...")
mul32 = cd_mult_table(5)
print("    Done.")

# Verify XOR property still holds
xor_ok = all(mul32[a][b][0] == a ^ b for a in range(32) for b in range(32))
print(f"    XOR property in dim-32: {xor_ok}")

# Step 4: Check if original 84 sedenion zero-divisor pairs survive in dim-32
print("\n[4] Checking survival of original 84 pairs in dim-32...")
survived = 0
destroyed = 0
for (fac1, fac2) in unord16:
    a, b, s = fac1
    c, d, t = fac2
    x = {a: 1, b: s}
    y = {c: 1, d: t}
    prod = sedenion_mul(x, y, mul32, 32)
    if not prod:
        survived += 1
    else:
        destroyed += 1

print(f"    Survived:  {survived} / {len(unord16)}")
print(f"    Destroyed: {destroyed} / {len(unord16)}")

# Step 5: Count ALL zero-divisor pairs in dim-32 (only among e_1..e_31)
# This is expensive: C(30,2)*2 * C(30,2)*2 = 870*870*4 ~ 3M checks
print("\n[5] Counting ALL zero-divisor pairs in dim-32 (e_1..e_31)...")
print("    (This may take a while...)")
ord32, unord32, configs32 = count_zero_divisors(mul32, 32)
print(f"    Ordered pairs:   {ord32}")
print(f"    Unordered pairs: {len(unord32)}")
print(f"    Index configs:   {len(configs32)}")

# Step 6: How many involve ONLY e_1..e_15 (original sedenion indices)?
original_only = []
mixed = []
new_only = []
for (fac1, fac2) in unord32:
    a, b, s = fac1
    c, d, t = fac2
    indices = {a, b, c, d}
    all_low = all(i <= 15 for i in indices)
    all_high = all(i >= 16 for i in indices)
    if all_low:
        original_only.append((fac1, fac2))
    elif all_high:
        new_only.append((fac1, fac2))
    else:
        mixed.append((fac1, fac2))

print(f"\n[6] Breakdown of dim-32 zero-divisor pairs:")
print(f"    Original indices only (e_1..e_15): {len(original_only)}")
print(f"    New indices only (e_16..e_31):     {len(new_only)}")
print(f"    Mixed (cross old/new):             {len(mixed)}")
print(f"    Total:                             {len(original_only) + len(new_only) + len(mixed)}")

# Step 7: Verdict
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)
if survived == len(unord16):
    print("  ALL 84 original sedenion zero-divisor pairs SURVIVE in dim-32.")
    print("  e_16 does NOT absorb the cancellations.")
    print("  -> Termination at dim-16 is REAL: the obstruction is structural,")
    print("     not removable by further doubling.")
else:
    print(f"  {destroyed} of {len(unord16)} original pairs were DESTROYED by doubling.")
    print("  e_16 partially absorbs cancellations.")

if len(configs32) > len(configs16):
    new_count = len(configs32) - len(configs16)
    print(f"  Additionally, {new_count} NEW zero-divisor configs appeared.")
    print(f"  Zero-divisors PROLIFERATE: {len(configs16)} -> {len(configs32)}")
    ratio = len(configs32) / len(configs16)
    print(f"  Growth factor: {ratio:.2f}x")

print("\nDone.")
