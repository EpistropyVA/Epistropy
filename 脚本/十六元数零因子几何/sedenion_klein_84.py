# sedenion_klein_84.py
# Exact sedenion algebra + Klein quartic 84-edge comparison
# All algebra: exact integer arithmetic
# Group theory: sympy.combinatorics

import sys
from itertools import combinations, product as iproduct
from sympy.combinatorics import PermutationGroup, Permutation

# ============================================================
# Build sedenion multiplication table via Cayley-Dickson
# Convention: (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
# conj(a,b) = (conj(a), -b)
# For reals: conj(x) = x
# Basis vectors: e_0..e_15, represented as (index, sign)
# ============================================================

def cd_mult_table(n):
    """
    Build multiplication table for 2^n -ion algebra via Cayley-Dickson.
    Returns mul[a][b] = (index, sign) where e_a * e_b = sign * e_index
    """
    dim = 1 << n
    # Start with reals: mul[0][0] = (0, 1)
    mul = {(0, 0): (0, 1)}

    def conj_sign(a, prev_mul):
        """conjugate of e_a in the current algebra: returns sign multiplier"""
        # conj(e_0) = e_0, conj(e_i) = -e_i for i>0
        # This is encoded in how we build the table
        return 1 if a == 0 else -1

    # Iteratively double
    # At each step, current algebra has size `sz`
    # New algebra has size `2*sz`
    # Basis: e_0..e_{sz-1} are (e_a, 0), e_{sz}..e_{2sz-1} are (0, e_{a-sz})
    # New product: (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
    # In index terms:
    #   For indices in old algebra: a < sz, b < sz
    #   e_{a+sz} represents (0, e_a)
    # Product rules:
    #   e_a * e_b (both < sz): same as old
    #   e_a * e_{b+sz} = (0, e_b * e_a) -> index (b+sz), sign from e_b*e_a
    #     Wait, let me be more careful.
    # (a,0)*(c,0) = (a*c - conj(0)*0, 0*a + 0*conj(c)) = (a*c, 0) -> same as old
    # (a,0)*(0,d) = (a*0 - conj(d)*0, d*a + 0*conj(0)) = (0, d*a) -> index (d*a in new)
    # (0,b)*(c,0) = (0*c - conj(0)*b, 0*0 + b*conj(c)) = (-0*b, b*conj(c))
    #             = (0, b*conj(c))
    # (0,b)*(0,d) = (0*0 - conj(d)*b, d*0 + b*conj(0)) = (-conj(d)*b, b*conj(0))
    #             = (-conj(d)*b, b*1) if 0-component...
    # Let me redo carefully.
    # In Cayley-Dickson, elements are pairs (a, b).
    # Product: (a,b)*(c,d) = (ac - d*conj(b), conj(a)*d + c*b)
    #   [This is one common convention; let's use the one stated in the problem]
    # Problem convention: x*y=(ac - conj(d)b, da + b conj(c))
    # So (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
    # conj(a,b) = (conj(a), -b)

    # Build iteratively
    # current_mul[i][j] = (idx, sign) for the current-level algebra
    sz = 1
    current_mul = [[(0, 1)]]  # 1x1: e_0 * e_0 = e_0

    for level in range(n):
        new_sz = sz * 2
        new_mul = [[None] * new_sz for _ in range(new_sz)]

        # Fill the four quadrants
        for a in range(sz):
            for c in range(sz):
                # (a,0)*(c,0) = (a*c - conj(0)*0, 0*a + 0*conj(c)) = (a*c, 0)
                idx, sgn = current_mul[a][c]
                new_mul[a][c] = (idx, sgn)

                # (a,0)*(0+sz, c+sz) -> (a,0)*(0,e_c):
                # = (a*0 - conj(e_c)*0, e_c*a + 0*conj(0))
                # = (0, e_c * a)
                # conj(e_c) for c>0 is -e_c, for c=0 is e_0
                # But here second component index is c (the e_c in the pair (0, e_c))
                # Product (a,0)*(0,e_c) = (a*0 - conj(e_c)*0, e_c*a + 0*conj(a))
                # = (-0, e_c * a) = (0, e_c * a)
                # e_c * a in current algebra:
                idx2, sgn2 = current_mul[c][a]
                new_mul[a][c + sz] = (idx2 + sz, sgn2)

                # (0+sz, a)*(c,0) -> (0,e_a)*(c,0):
                # = (0*c - conj(0)*e_a, 0*0 + e_a*conj(c))
                # conj(0) = 0 for the real component
                # = (0, e_a * conj(c))
                # conj(e_c) = sign_c * e_c where sign_c = 1 if c==0 else -1
                conj_sgn_c = 1 if c == 0 else -1
                idx3, sgn3 = current_mul[a][c]
                new_mul[a + sz][c] = (idx3 + sz, sgn3 * conj_sgn_c)

                # (0,e_a)*(0,e_c):
                # = (0*0 - conj(e_c)*e_a, e_c*0 + e_a*conj(0))
                # = (-conj(e_c)*e_a, e_a)
                # conj(e_c) = sign_c * e_c
                # -conj(e_c)*e_a = -sign_c * (e_c * e_a)
                conj_sgn_c2 = 1 if c == 0 else -1
                idx4, sgn4 = current_mul[c][a]
                new_mul[a + sz][c + sz] = (idx4, -conj_sgn_c2 * sgn4)

        sz = new_sz
        current_mul = new_mul

    return current_mul

print("Building sedenion multiplication table...")
mul_table = cd_mult_table(4)  # 2^4 = 16 dimensional
print("Done.")
print()

# ============================================================
# Part A1: Verify twisted group algebra (XOR property)
# ============================================================
print("=" * 60)
print("PART A1: Twisted group algebra (XOR property)")
print("=" * 60)

all_pass_A1 = True
fail_cases_A1 = []
for a in range(16):
    for b in range(16):
        idx, sgn = mul_table[a][b]
        expected_idx = a ^ b
        if idx != expected_idx:
            all_pass_A1 = False
            fail_cases_A1.append((a, b, idx, expected_idx, sgn))
        if sgn not in (1, -1):
            all_pass_A1 = False
            fail_cases_A1.append((a, b, idx, expected_idx, sgn))

if all_pass_A1:
    print("A1: PASS - index(e_a * e_b) == a XOR b for all a,b in 0..15, coefficient in {+1,-1}")
else:
    print(f"A1: FAIL - {len(fail_cases_A1)} violations")
    for case in fail_cases_A1[:5]:
        print(f"  e_{case[0]} * e_{case[1]}: got idx={case[2]}, expected {case[3]}, sign={case[4]}")
print()

# ============================================================
# Part A2: Exhaustive exact polarization check
# ============================================================
print("=" * 60)
print("PART A2: Polarization check (exact)")
print("=" * 60)

def sedenion_add(x, y):
    """Add two sedenions represented as dicts {index: coeff}"""
    result = dict(x)
    for idx, coeff in y.items():
        result[idx] = result.get(idx, 0) + coeff
    return {k: v for k, v in result.items() if v != 0}

def sedenion_scale(x, s):
    return {k: v * s for k, v in x.items() if v * s != 0}

def basis_vec(i):
    return {i: 1}

def sedenion_mul_basis(a, b):
    """Returns sedenion dict for e_a * e_b"""
    idx, sgn = mul_table[a][b]
    return {idx: sgn}

all_pass_A2 = True
fail_cases_A2 = []

for i in range(16):
    for j in range(16):
        # Compute e_i * e_j + e_j * e_i
        eij = sedenion_mul_basis(i, j)
        eji = sedenion_mul_basis(j, i)
        anticomm = sedenion_add(eij, eji)

        # Expected:
        if j == 0:
            expected = {i: 2}
        elif i == 0:
            expected = {j: 2}
        elif i == j:
            expected = {0: -2}  # -2*e_0
        else:
            expected = {}  # zero

        if anticomm != expected:
            all_pass_A2 = False
            fail_cases_A2.append((i, j, anticomm, expected))

if all_pass_A2:
    print("A2: PASS - all 256 polarization cases exact")
    print()
    print("Conclusion chain:")
    print("  polarization exact")
    print("  ==> {e_i,e_j} = 2d_{i0}e_j + 2d_{j0}e_i - 2d_{ij}e_0 (i,j>0)")
    print("  ==> quadratic identity x^2 - 2x_0*x + ||x||^2*1 = 0 holds EXACTLY for all")
    print("     real-coefficient sedenions (follows by bilinearity from polarization)")
    print("  ==> u^2 = 1 has only u = +/-1 (from quadratic: u_0 = +/-1, all other components 0)")
    print("  ==> no nontrivial idempotents e^2 = e (would need e(1-e)=0 with e!=0,1)")
    print("  Residual is literally 0, not 1e-14")
else:
    print(f"A2: FAIL - {len(fail_cases_A2)} violations")
    for case in fail_cases_A2[:3]:
        print(f"  i={case[0]}, j={case[1]}: got {case[2]}, expected {case[3]}")
print()

# ============================================================
# Part B1: Zero-divisor enumeration
# ============================================================
print("=" * 60)
print("PART B1: Zero-divisor enumeration")
print("=" * 60)

def sedenion_mul(x, y):
    """Multiply two sedenions (dicts) using mul_table"""
    result = {}
    for ai, ac in x.items():
        for bi, bc in y.items():
            idx, sgn = mul_table[ai][bi]
            coeff = ac * bc * sgn
            result[idx] = result.get(idx, 0) + coeff
    return {k: v for k, v in result.items() if v != 0}

# x = e_a + s*e_b, y = e_c + t*e_d
# 0 < a < b <= 15, s in {+1,-1}
# 0 < c < d <= 15, t in {+1,-1}

pairs = list(combinations(range(1, 16), 2))  # (a,b) with 0<a<b<=15
signs = [1, -1]

zero_div_ordered = []  # list of ((a,b,s),(c,d,t))

print(f"Checking {len(pairs)*2} x {len(pairs)*2} = {len(pairs)*2 * len(pairs)*2} combinations...")

for (a, b) in pairs:
    for s in signs:
        x = {a: 1, b: s}
        for (c, d) in pairs:
            for t in signs:
                y = {c: 1, d: t}
                prod = sedenion_mul(x, y)
                if not prod:  # exactly zero
                    zero_div_ordered.append(((a, b, s), (c, d, t)))

print(f"Ordered zero-divisor pairs: {len(zero_div_ordered)} (expect 168)")

# Unordered pairs: {x, y}  -  but x and y are labeled (a,b,s),(c,d,t)
# Two ordered pairs ((a,b,s),(c,d,t)) and ((c,d,t),(a,b,s)) represent same unordered pair
seen = set()
zero_div_unordered = []
for (fac1, fac2) in zero_div_ordered:
    key = tuple(sorted([fac1, fac2]))
    if key not in seen:
        seen.add(key)
        zero_div_unordered.append((fac1, fac2))

print(f"Unordered zero-divisor pairs: {len(zero_div_unordered)} (expect 84)")
print()

# ============================================================
# Part B2: Structure extraction
# ============================================================
print("=" * 60)
print("PART B2: Structure extraction  -  XOR grouping")
print("=" * 60)

# For each unordered pair, record XOR values
xor_same = True
xor_fail_count = 0
xor_groups = {}  # u -> list of pairs

for (fac1, fac2) in zero_div_unordered:
    a, b, s = fac1
    c, d, t = fac2
    u1 = a ^ b
    u2 = c ^ d
    if u1 != u2:
        xor_same = False
        xor_fail_count += 1
    u = u1
    if u not in xor_groups:
        xor_groups[u] = []
    xor_groups[u].append((fac1, fac2))

if xor_same:
    print("XOR match: a XOR b == c XOR d holds within EVERY unordered pair: YES")
else:
    print(f"XOR match: FAILS in {xor_fail_count} pairs")

print()
print(f"Distinct XOR values (u = a XOR b): {sorted(xor_groups.keys())}")
print(f"Number of distinct u values: {len(xor_groups)}")
print()
print("Counts per u value:")
for u in sorted(xor_groups.keys()):
    print(f"  u={u:2d} (binary {u:04b}): {len(xor_groups[u])} pairs")

total = sum(len(v) for v in xor_groups.values())
print(f"Total: {total}")

# Check 7 x 12 hypothesis
if len(xor_groups) == 7 and all(len(v) == 12 for v in xor_groups.values()):
    print("Box-kite structure: 7 values x 12 pairs each - CONFIRMED")
else:
    print(f"Box-kite structure: {len(xor_groups)} values, counts {sorted(len(v) for v in xor_groups.values())}")

print()
print("Index distribution (low 1-7 vs high 8-15) within each u-group:")
for u in sorted(xor_groups.keys()):
    low_counts = []
    high_counts = []
    for (fac1, fac2) in xor_groups[u]:
        for fac in [fac1, fac2]:
            a, b, s = fac
            low_in_fac = sum(1 for x in [a, b] if 1 <= x <= 7)
            high_in_fac = sum(1 for x in [a, b] if 8 <= x <= 15)
            low_counts.append(low_in_fac)
            high_counts.append(high_in_fac)
    # Summarize
    from collections import Counter
    low_dist = Counter(low_counts)
    print(f"  u={u}: low-idx-count distribution per factor: {dict(sorted(low_dist.items()))}")
print()

# ============================================================
# Part B3: GL(4,2) symmetry
# ============================================================
print("=" * 60)
print("PART B3: GL(4,2) symmetry of the 84-set")
print("=" * 60)

# GL(4,2): invertible 4x4 matrices over F_2
# Index i in 1..15 represented as 4-bit vector (bits 0-3 = bit positions)
# Matrix acts: new_idx = apply matrix to 4-bit vector of i

def idx_to_vec(i):
    """Convert index 1..15 to F_2^4 vector as tuple of 4 bits"""
    return tuple((i >> k) & 1 for k in range(4))

def vec_to_idx(v):
    """Convert F_2^4 vector to index"""
    return sum(v[k] << k for k in range(4))

def mat_vec_f2(M, v):
    """Multiply 4x4 F_2 matrix M (list of 4 rows, each a tuple of 4 bits) by vector v"""
    result = []
    for row in M:
        val = sum(row[k] * v[k] for k in range(4)) % 2
        result.append(val)
    return tuple(result)

def mat_det_f2(M):
    """Compute determinant of 4x4 F_2 matrix mod 2"""
    # Gaussian elimination over F_2
    m = [list(row) for row in M]
    det = 1
    for col in range(4):
        # Find pivot
        pivot = None
        for row in range(col, 4):
            if m[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            return 0
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            # det *= -1, but in F_2 -1 = 1
        for row in range(4):
            if row != col and m[row][col] == 1:
                for k in range(4):
                    m[row][k] = (m[row][k] + m[col][k]) % 2
    return 1

print("Enumerating GL(4,2) (order 20160)...")
gl42 = []
# All 4x4 F_2 matrices: 2^16 candidates
for bits in range(1 << 16):
    M = []
    for row in range(4):
        M.append(tuple((bits >> (row * 4 + col)) & 1 for col in range(4)))
    if mat_det_f2(M) == 1:
        gl42.append(M)

print(f"GL(4,2) order: {len(gl42)} (expect 20160)")

# The 84 unordered pairs as frozensets of index-pair configs (ignoring signs)
# Represent each pair as frozenset of frozenset({a,b}, {c,d})
# We need to represent the 84 pairs by their INDEX structure only (signs forgotten)
# Each unordered zero-div pair: ((a,b,s),(c,d,t)) -> {{a,b},{c,d}} as a frozenset

zero_div_84_set = set()
zero_div_84_list = []
for (fac1, fac2) in zero_div_unordered:
    a, b, s = fac1
    c, d, t = fac2
    key = frozenset([frozenset([a, b]), frozenset([c, d])])
    if key not in zero_div_84_set:
        zero_div_84_set.add(key)
        zero_div_84_list.append(key)

print(f"Distinct index-pair configurations: {len(zero_div_84_set)} (expect 84)")

# Apply matrix M to a frozenset config
def apply_mat_to_config(M, config):
    """Apply GL(4,2) matrix to a zero-div config (frozenset of two frozensets of indices)"""
    new_pairs = []
    for pair in config:
        new_pair = frozenset(vec_to_idx(mat_vec_f2(M, idx_to_vec(i))) for i in pair)
        new_pairs.append(new_pair)
    return frozenset(new_pairs)

# Check: does apply produce valid indices (1..15)?
def config_valid(config):
    for pair in config:
        for i in pair:
            if i == 0 or i > 15:
                return False
        if len(pair) != 2:  # shouldn't collapse to same index
            return False
    return True

print("Finding symmetry subgroup G_ZD...")
G_ZD = []
for M in gl42:
    preserves = True
    for config in zero_div_84_list:
        new_config = apply_mat_to_config(M, config)
        if not config_valid(new_config) or new_config not in zero_div_84_set:
            preserves = False
            break
    if preserves:
        G_ZD.append(M)

print(f"Order of G_ZD (symmetry subgroup): {len(G_ZD)}")
print()

# ============================================================
# Part B4: PSL(2,7) = GL(3,2) inside GL(4,2)
# ============================================================
print("=" * 60)
print("PART B4: PSL(2,7) = GL(3,2) block embedding")
print("=" * 60)

# Block embedding: A in GL(3,2) acts on bits 1-3 (indices 1-7)
# High index 8+i -> 8 + A(i) for i in 1..7, 8->8
# Bits: index i, bits 0-3
# Low indices 1-7: bit3=0, bits 0-2 give 1..7
# High indices 8-15: bit3=1, bits 0-2 give 0..7 -> but 8+0=8, 8+A(i) for i in 1-7

# Actually: index = bit3*8 + bit2*4 + bit1*2 + bit0*1
# Low indices (1-7): bit3=0
# High indices (8-15): bit3=1
# The block embedding A in GL(3,2) (3x3 invertible F_2 matrix):
#   For low index i (1<=i<=7, bit3=0): new_i = A applied to bits 0-2 of i (as F_2^3)
#   For high index i=8+j: new_i = 8 + A(j) where j = i-8 (but j can be 0..7)
#     Wait: "8+i -> 8 + A(i), 8->8" means 8+0->8, so j=0->0, j=i for i in 1..7->A(i)
# But A is 3x3, acts on F_2^3 (indices 1-7 as nonzero vectors).
# For j=0 (index 8): stays at 8
# For j in 1-7 (index 8+j): maps to 8+A(j) -- but A(j) might hit 0!
# We need A that maps 1-7 to 1-7 (nonzero vectors), which is exactly GL(3,2).
# Also A(0)=0 by linearity, so 8->8 is automatic.

# Represent the GL(3,2) -> GL(4,2) embedding
def gl32_to_gl42(A3):
    """
    A3: 3x3 F_2 matrix (list of 3 rows, each a list/tuple of 3 bits)
    Returns 4x4 F_2 matrix acting as: bits 0-2 by A3, bit 3 unchanged
    """
    M4 = []
    for row in range(4):
        if row < 3:
            # The new row-th component
            # new_bit_row = A3[row] . old_bits_0-2
            new_row = list(A3[row]) + [0]  # bit3 doesn't affect low bits
        else:
            # bit3 component: unchanged
            new_row = [0, 0, 0, 1]
        M4.append(tuple(new_row))
    return M4

# Enumerate GL(3,2) (order 168)
print("Enumerating GL(3,2) (order 168)...")
gl32 = []
for bits in range(1 << 9):
    A = []
    for row in range(3):
        A.append(tuple((bits >> (row * 3 + col)) & 1 for col in range(3)))
    # Check det != 0 over F_2 (matrix is invertible)
    # Use 3x3 det mod 2
    m = [list(row) for row in A]
    det = 1
    invertible = True
    for col in range(3):
        pivot = None
        for row in range(col, 3):
            if m[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            invertible = False
            break
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        for row in range(3):
            if row != col and m[row][col] == 1:
                for k in range(3):
                    m[row][k] = (m[row][k] + m[col][k]) % 2
    if invertible:
        gl32.append(A)

print(f"GL(3,2) order: {len(gl32)} (expect 168)")

# Check if each GL(3,2) element preserves the 84-set
block_preserves_all = True
block_fail_count = 0
block_elements_gl42 = []

for A3 in gl32:
    M4 = gl32_to_gl42(A3)
    block_elements_gl42.append(M4)
    preserves = True
    for config in zero_div_84_list:
        new_config = apply_mat_to_config(M4, config)
        if not config_valid(new_config) or new_config not in zero_div_84_set:
            preserves = False
            block_preserves_all = False
            block_fail_count += 1
            break

if block_preserves_all:
    print("Block GL(3,2) embedding PRESERVES the 84-set: YES")
else:
    print(f"Block GL(3,2) embedding: {block_fail_count} elements do NOT preserve the 84-set")

# Build induced permutation action on the 84 pairs
# Map each config to its index in zero_div_84_list
config_to_idx = {config: i for i, config in enumerate(zero_div_84_list)}

def mat_to_perm_on_84(M):
    """Build permutation of 0..83 induced by matrix M"""
    perm = []
    for config in zero_div_84_list:
        new_config = apply_mat_to_config(M, config)
        perm.append(config_to_idx[new_config])
    return perm

print()
print("Building induced permutation action of GL(3,2) on 84 pairs...")
perms_84 = [mat_to_perm_on_84(M4) for M4 in block_elements_gl42]

# Build PermutationGroup
sympy_perms = [Permutation(p) for p in perms_84]
G_induced = PermutationGroup(*sympy_perms)

print(f"(i) Order of induced action: {G_induced.order()} (expect 168)")

# (ii) Transitivity
is_transitive = G_induced.is_transitive()
print(f"(ii) Is transitive on 84? {is_transitive} (expect True)")

# (iii) Point stabilizer
stab = G_induced.stabilizer(0)
stab_order = stab.order()
print(f"(iii) Point stabilizer order: {stab_order} (expect 2)")

if stab_order >= 1:
    # Get generator(s) of stabilizer
    stab_gens = stab.generators
    if stab_gens:
        gen = stab_gens[0]
        # Find order of generator
        gen_order = gen.order()
        print(f"     Stabilizer generator order: {gen_order} (expect 2)")
        is_order_2 = (gen_order == 2)
        print(f"     Stabilizer generator has order 2: {is_order_2}")
    else:
        print("     Stabilizer is trivial (identity only)")
        is_order_2 = False
else:
    is_order_2 = False

print()

# Also report orbit structure of block GL(3,2) on 84 (for completeness)
orbits = G_induced.orbits()
orbit_sizes = sorted([len(o) for o in orbits])
print(f"Orbit structure of block GL(3,2) on 84: {orbit_sizes}")
print()

# ============================================================
# Part B5: Klein criterion and verdict
# ============================================================
print("=" * 60)
print("PART B5: Klein criterion and verdict")
print("=" * 60)

# PSL(2,7) = GL(3,2) -- confirm involution count
# An involution is an element of order 2
print("Counting involutions in GL(3,2) (order-2 elements)...")

involution_count = 0
involutions = []
for A3 in gl32:
    # Compute A3^2 mod 2
    A3_sq = []
    for i in range(3):
        row = []
        for j in range(3):
            val = sum(A3[i][k] * A3[k][j] for k in range(3)) % 2
            row.append(val)
        A3_sq.append(tuple(row))
    # Is A3^2 = identity?
    identity = [(1,0,0),(0,1,0),(0,0,1)]
    is_inv = (A3_sq == [list(r) for r in identity] or
              [tuple(r) for r in A3_sq] == [(1,0,0),(0,1,0),(0,0,1)])
    # Also check A3 != identity
    is_id = (A3 == [(1,0,0),(0,1,0),(0,0,1)] or
             [tuple(r) for r in A3] == [(1,0,0),(0,1,0),(0,0,1)])
    if is_inv and not is_id:
        involution_count += 1
        involutions.append(A3)

print(f"Number of involutions in GL(3,2): {involution_count} (expect 21)")

# Check single conjugacy class of involutions
# Two elements are conjugate if there exists g in GL(3,2) with g*A*g^{-1} = B
# For our purposes, count conjugacy classes of involutions
# Quick check: if |class| = 21 and there's only one class

def mat3_mul(A, B):
    C = []
    for i in range(3):
        row = []
        for j in range(3):
            val = sum(A[i][k] * B[k][j] for k in range(3)) % 2
            row.append(val)
        C.append(tuple(row))
    return C

def mat3_inv(A):
    """Compute inverse of A in GL(3,2)"""
    # Augmented matrix [A | I]
    aug = [list(A[i]) + [1 if i==j else 0 for j in range(3)] for i in range(3)]
    for col in range(3):
        pivot = None
        for row in range(col, 3):
            if aug[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(3):
            if row != col and aug[row][col] == 1:
                for k in range(6):
                    aug[row][k] = (aug[row][k] + aug[col][k]) % 2
    return [tuple(aug[i][3:]) for i in range(3)]

# Find conjugacy classes of involutions
inv_classes = []
inv_remaining = list(range(len(involutions)))
inv_list = [tuple(tuple(r) for r in A) for A in involutions]

while inv_remaining:
    rep_idx = inv_remaining[0]
    rep = involutions[rep_idx]
    # Conjugacy class: {g * rep * g^{-1} : g in GL(3,2)}
    conj_class = set()
    for g in gl32:
        g_inv = mat3_inv(g)
        if g_inv is None:
            continue
        conj = mat3_mul(mat3_mul([list(r) for r in g], rep), [list(r) for r in g_inv])
        conj_key = tuple(tuple(r) for r in conj)
        conj_class.add(conj_key)
    inv_classes.append(len(conj_class))
    # Remove all class members from remaining
    inv_remaining = [i for i in inv_remaining if inv_list[i] not in conj_class]

print(f"Conjugacy classes of involutions: {len(inv_classes)} class(es), sizes: {inv_classes}")
single_conj_class = (len(inv_classes) == 1 and inv_classes[0] == 21)
print(f"PSL(2,7) has exactly 21 involutions in a single conjugacy class: {single_conj_class}")

print()
print("Classification fact:")
print("  A transitive G-set of size 84 with |G|=168 and point stabilizer of order 2")
print("  is unique up to G-set isomorphism when G = PSL(2,7).")
print("  The Klein quartic has exactly 84 edges, and PSL(2,7) acts on them with")
print("  edge stabilizer Z_2 (flip), giving |orbit| = 168/2 = 84.")
print()

# Final verdict
induced_order = G_induced.order()
faithful = (induced_order == 168)
transitive = is_transitive
stab2 = (stab_order == 2)

print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)
print(f"  Induced action order: {induced_order} (faithful: {faithful})")
print(f"  Transitive on 84: {transitive}")
print(f"  Point stabilizer order: {stab_order} (is 2: {stab2})")
print()

if faithful and transitive and stab2:
    print("SAME-84")
    print("  The sedenion zero-divisor 84 and the Klein quartic 84 edges are")
    print("  G-set isomorphic as PSL(2,7)-sets:")
    print("  both are transitive PSL(2,7)-sets of size 84 with point stabilizer Z_2,")
    print("  and such G-sets are unique up to isomorphism.")
else:
    conditions = []
    if not faithful:
        conditions.append(f"NOT faithful (order={induced_order}, expected 168)")
    if not transitive:
        conditions.append("NOT transitive")
    if not stab2:
        conditions.append(f"stabilizer order={stab_order}, expected 2")
    print(f"NOT-SAME or PARTIAL: " + "; ".join(conditions))
    # Report orbit structure
    print(f"  Orbit structure of block GL(3,2) on 84: {orbit_sizes}")
    print(f"  G_ZD order: {len(G_ZD)}")

print()
print("Script complete.")
