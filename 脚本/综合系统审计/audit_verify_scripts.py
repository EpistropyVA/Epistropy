# audit_verify_scripts.py
# Adversarial audit of verify_pending_{4-10}.py
# Purpose: catch false-PASS logic by injecting wrong data or checking edge cases.
# Each section probes one script. Output: per-item verdict.

import numpy as np
import sys
from itertools import combinations

print("=" * 70)
print("ADVERSARIAL AUDIT OF VERIFICATION SCRIPTS 4-10")
print("=" * 70)
issues = []

# ====================================================================
# AUDIT 4: verify_pending_4_three54.py
# ====================================================================
print("\n--- AUDIT 4: THREE 54 MARKERS ---")

# ISSUE 4a: The PASS condition is (rank==54) OR (rank_pi==54 AND rank_simplex==54).
# The overlap matrix rank was only 19, not 54.
# So the script is NOT passing on the overlap criterion.
# It passes because both row spaces have rank 54 independently.
# But "both have rank 54" only means they each span 54-dim subspaces of R^540.
# It does NOT prove they span the SAME 54-dim subspace (1-to-1 correspondence).
# Two random 54-dim subspaces of R^540 will generically have rank 54 each,
# with overlap rank ~54*54/540 ~ 5.4, not 54.
# The actual overlap rank 19 is somewhere in between.
# 
# VERDICT: The fallback criterion is too weak. rank_pi=54 AND rank_simplex=54
# is trivially satisfied for any two 54-dim subsets. The 1-to-1 correspondence
# claim requires overlap_rank = 54, which FAILS (it's 19).
#
# However, the claim "54 pi-modes correspond 1-to-1 to 54 simplices" might be
# about a dimensional/counting match, not a subspace identity.
# Still, the script's own comment says "expected 54 for 1-to-1 correspondence"
# for the overlap rank, and it gets 19 -- then sidesteps this by the fallback.

print("ISSUE 4a: PASS condition is too loose.")
print("  The overlap matrix between pi-modes and simplices has rank 19, not 54.")
print("  The script passes via fallback: rank(pi_modes)=54 AND rank(simplices)=54.")
print("  This fallback is trivially true for any 54 vectors in R^540 in general position.")
print("  It does NOT prove 1-to-1 correspondence between the two specific sets.")
print("  SEVERITY: HIGH -- the core claim is not validated by this criterion.")

# ISSUE 4b: Wilson loop band selection uses deg_bands = range(6), i.e. the
# lowest 6 eigenvalues. This is the flat band. But the code does not verify
# that these 6 bands are actually degenerate (flat). If they happen to be
# non-degenerate at some k-point, the Wilson loop construction is wrong.
print("ISSUE 4b: Band selection deg_bands=range(6) is hard-coded, not validated.")
print("  No check that these 6 bands are actually degenerate across all k-points.")
print("  SEVERITY: MEDIUM")

# ISSUE 4c: pi-phase threshold is 0.1 radians around |phase|=pi.
# That's abs(abs(phase) - pi) < 0.1, which is generous.
# If phases cluster near pi but aren't exactly pi, this could over-count.
print("ISSUE 4c: pi-phase detection threshold is 0.1 radian -- fairly loose.")
print("  SEVERITY: LOW (output count 54 matches expected, so likely fine)")

issues.append(("4", "HIGH", "Overlap rank=19 not 54; fallback criterion trivially true"))

# ====================================================================
# AUDIT 5: verify_pending_5_ghz_fano.py
# ====================================================================
print("\n--- AUDIT 5: GHZ <-> FANO COCYCLE ---")

# ISSUE 5a: The script hard-codes h_QM = [0, 1, 1, 1] as given constants.
# It does NOT actually compute the quantum expectation values from the GHZ state.
# It simply verifies that 0+1+1+1 = 3 = 1 mod 2.
# This is tautological: the "computation" is just checking 1+1+1 is odd.
print("ISSUE 5a: Expectation values h_QM = [0,1,1,1] are HARD-CODED, not computed.")
print("  The script does not construct the GHZ state vector or compute <GHZ|M_i|GHZ>.")
print("  The 'verification' reduces to checking that 0+1+1+1 = 1 mod 2.")
print("  SEVERITY: HIGH -- no actual quantum mechanical calculation is performed.")

# ISSUE 5b: The Fano plane 2-cycle check (every point appears even times)
# is correct in logic. The 4 lines are hard-coded consistently with the Fano plane.
# But the mapping from Fano points to Mermin operators is asserted, not derived.
print("ISSUE 5b: The Fano-to-Mermin mapping is asserted, not derived from structure.")
print("  Still, the specific mapping is standard (Abramsky-Brandenburger).")
print("  SEVERITY: LOW")

# Let me verify the hard-coded values are at least correct:
# GHZ = (|000> + |111>)/sqrt(2)
# sigma_x = [[0,1],[1,0]], sigma_y = [[0,-1j],[1j,0]]
sx = np.array([[0,1],[1,0]], dtype=complex)
sy = np.array([[0,-1j],[1j,0]], dtype=complex)
I2 = np.eye(2, dtype=complex)

GHZ = np.zeros(8, dtype=complex)
GHZ[0] = 1/np.sqrt(2)  # |000>
GHZ[7] = 1/np.sqrt(2)  # |111>

def kron3(A, B, C):
    return np.kron(A, np.kron(B, C))

M1 = kron3(sx, sx, sx)   # X1 X2 X3
M2 = kron3(sx, sy, sy)   # X1 Y2 Y3
M3 = kron3(sy, sx, sy)   # Y1 X2 Y3
M4 = kron3(sy, sy, sx)   # Y1 Y2 X3

exp_vals = [np.real(GHZ.conj() @ M @ GHZ) for M in [M1, M2, M3, M4]]
h_computed = [0 if v > 0 else 1 for v in exp_vals]

print(f"  Actual computed <GHZ|M_i|GHZ>: {[round(v,6) for v in exp_vals]}")
print(f"  Converted to F2: {h_computed}")
print(f"  Hard-coded in script: [0, 1, 1, 1]")
print(f"  Match: {h_computed == [0, 1, 1, 1]}")

if h_computed != [0, 1, 1, 1]:
    print("  *** CRITICAL: Hard-coded values are WRONG! ***")
    issues.append(("5", "CRITICAL", "Hard-coded h_QM values are incorrect"))
else:
    print("  Hard-coded values are correct, but should have been computed.")
    issues.append(("5", "HIGH", "Expectation values hard-coded, not computed from GHZ state"))

# ====================================================================
# AUDIT 6: verify_pending_6_bvn_ijk.py
# ====================================================================
print("\n--- AUDIT 6: BvN NON-DISTRIBUTIVE LATTICE ---")

# The script constructs P_i = (I + sigma_x)/2, P_j = (I + sigma_y)/2, P_k = (I + sigma_z)/2
# These are rank-1 projections onto spin-up states along x, y, z axes.
# The meet/join are implemented via eigenvalue decomposition.

# ISSUE 6a: Let's verify the meet implementation is correct.
# Meet of P_A and P_B = projection onto intersection of their ranges.
# Implementation: eigenspace of (P_A + P_B)/2 with eigenvalue 1.
# If v is in range(P_A) AND range(P_B), then P_A v = v and P_B v = v,
# so (P_A + P_B)/2 v = v. Conversely, if (P_A+P_B)/2 v = v, then
# P_A v + P_B v = 2v. Since ||P_A v|| <= ||v|| and ||P_B v|| <= ||v||,
# equality requires P_A v = v and P_B v = v. This is correct.
print("ISSUE 6a: Meet via eigenvalue=1 of (P_A+P_B)/2 is mathematically correct.")
print("  SEVERITY: NONE")

# ISSUE 6b: Join implementation: range projection of P_A + P_B.
# The range of P_A + P_B equals span(range(P_A), range(P_B)) = join.
# This is correct: if v in range(P_A), then (P_A + P_B)v = v + P_B v,
# which is nonzero (so v is in range of the sum), and conversely if
# (P_A + P_B)w != 0 then w has a component in either range.
# Actually more carefully: range(P_A + P_B) = range(P_A) + range(P_B)
# since P_A + P_B is positive semidefinite and range(P_A + P_B) = range(P_A) + range(P_B).
print("ISSUE 6b: Join via range projection of P_A+P_B is mathematically correct.")
print("  SEVERITY: NONE")

# ISSUE 6c: The specific predictions are verified against exact values.
# LHS should equal P_i (rank 1), RHS should equal 0 (rank 0).
# The script checks both np.allclose(LHS, P_i) and np.allclose(RHS, 0).
# This is a strong check.

# Let me verify the expected result independently:
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2c = np.eye(2, dtype=complex)

P_i = 0.5 * (I2c + sigma_x)
P_j = 0.5 * (I2c + sigma_y)
P_k = 0.5 * (I2c + sigma_z)

# P_j and P_k are rank-1 projections onto different directions.
# Their join is the full C^2 (since two non-parallel rank-1 projections span C^2).
# P_i meet (full space) = P_i. So LHS = P_i. Rank 1. Correct.
# P_i meet P_j: intersection of two non-parallel 1D subspaces = {0}. Rank 0. Correct.
# P_i meet P_k: same reasoning. Rank 0. Correct.
# Join of 0 and 0 = 0. Rank 0. Correct.

# Let me check that P_j, P_k spans are indeed non-parallel
eigvals_j = np.linalg.eigh(P_j)[1][:, 1]  # eigenvector for eigenvalue 1
eigvals_k = np.linalg.eigh(P_k)[1][:, 1]
overlap = abs(np.vdot(eigvals_j, eigvals_k))
print(f"  |<psi_j|psi_k>| = {overlap:.6f} (should be < 1 for non-parallel)")
if overlap > 1 - 1e-10:
    print("  *** WARNING: P_j and P_k project onto parallel subspaces! ***")
else:
    print("  P_j and P_k are non-parallel. BvN result is correct.")

print("VERDICT 6: Script logic is sound. PASS is genuine.")
# No issue

# ====================================================================
# AUDIT 7: verify_pending_7_orientation_forgetting.py
# ====================================================================
print("\n--- AUDIT 7: 336/168/84 ORIENTATION FORGETTING ---")

# ISSUE 7a: The CD multiplication table in the script uses a different
# implementation than sedenion_check.py. Need to verify they agree.
# The scripts use a recursive formula. Let me check the CD formula used.
# In scripts 7, 8, 9, the CD table uses:
#   new_mul[a][c] = current_mul[a][c]                           -- (a,0)*(c,0)
#   new_mul[a][c+sz] = (idx2+sz, sgn2) where current_mul[c][a]  -- (a,0)*(0,c) = (0, c*a)
#   new_mul[a+sz][c] = (idx3+sz, sgn3*conj_sgn_c)              -- (0,a)*(c,0) = (0, a*conj(c))
#   new_mul[a+sz][c+sz] = (idx4, -conj_sgn_c2*sgn4) where current_mul[c][a]  -- (0,a)*(0,c) = (-conj(c)*a, 0)
#
# Compare with sedenion_check.py which uses:
#   (A,0)*(C,0) = (A*C, 0)
#   (A,0)*(0,D) = (0, D*A)
#   (0,B)*(C,0) = (0, B*conj(C))
#   (0,B)*(0,D) = (-conj(D)*B, 0)
#
# Let me verify the verify scripts' CD formula matches this.
# In the verify scripts:
#   new_mul[a][c+sz]: represents e_a * e_{c+sz}. Here a < sz (first half), c+sz >= sz (second half).
#     This should be (A,0)*(0,D) = (0, D*A).
#     The code: current_mul[c][a] => this is D*A in the half-algebra. idx2+sz puts it in the second half. Good.
#   new_mul[a+sz][c]: represents e_{a+sz} * e_c. Here a+sz >= sz (second half), c < sz (first half).
#     This should be (0,B)*(C,0) = (0, B*conj(C)).
#     The code: current_mul[a][c] (which is B*C in half), then multiplied by conj_sgn_c.
#     conj_sgn_c = 1 if c==0 else -1. So result = B*C * conj_sgn_c.
#     But conj(e_c) = e_c if c==0 else -e_c.
#     So B*conj(C) = B*(conj_sgn_c * C) ... no, that's not right.
#     conj(C) = conj_sgn_c * C only for basis elements.
#     B * conj(e_c) = B * (conj_sgn_c * e_c) = conj_sgn_c * (B * e_c)
#     = conj_sgn_c * current_mul[a][c] in terms of (idx, sgn).
#     Code gives: (idx3+sz, sgn3 * conj_sgn_c) where idx3, sgn3 = current_mul[a][c].
#     So result index = idx3+sz (in second half), sign = sgn3 * conj_sgn_c.
#     This equals conj_sgn_c * (sgn3 * e_{idx3}) in second half = conj_sgn_c * B*C in second half.
#     Which is B * conj(C) projected to second half. Correct.
#
#   new_mul[a+sz][c+sz]: represents e_{a+sz} * e_{c+sz} = (0,B)*(0,D) = (-conj(D)*B, 0).
#     conj(D) = conj(e_c) = conj_sgn_c * e_c.
#     conj(D)*B = conj_sgn_c * e_c * e_a = conj_sgn_c * current_mul[c][a].
#     -conj(D)*B = -conj_sgn_c * sgn4 * e_{idx4}
#     Code gives: (idx4, -conj_sgn_c2 * sgn4) where idx4, sgn4 = current_mul[c][a].
#     This is (-conj_sgn_c * sgn4) * e_{idx4} in first half. Correct.

print("ISSUE 7a: CD multiplication table verified against sedenion_check.py formula.")
print("  The recursive construction matches the standard Cayley-Dickson product rules.")
print("  SEVERITY: NONE")

# ISSUE 7b: The quotient 336 -> 168 forgets the sign s,t in (a,b,s) and (c,d,t).
# It maps ((a,b,s),(c,d,t)) -> ((a,b),(c,d)).
# But wait -- is this correct? The sign-forgetting should map e_a + s*e_b to {e_a, e_b}
# (the pair of indices, regardless of sign). Two signed pairs (a,b,+1) and (a,b,-1)
# should map to the same unsigned pair (a,b). Let me check:
# The signed ordered set has entries ((a,b,s), (c,d,t)).
# When we forget signs, we get ((a,b), (c,d)).
# But different signs s,t might give different ordered pairs or the same one.
# The assertion is that |unsigned_ordered| = 168 = 336/2.
# This means each unsigned ordered pair corresponds to exactly 2 signed versions.
# Let me verify this is meaningful: for each pair (a,b), there are 2 sign choices for x,
# and for each (c,d), there are 2 sign choices for y. But xy=0 might only hold for
# specific sign combinations. If for every (a,b),(c,d) that works, exactly 2 of the
# 4 sign combinations give xy=0, then 336/2 = 168.

# Actually, looking more carefully: the code checks ALL (a,b,s) vs ALL (c,d,t).
# For each unsigned ordered pair ((a,b),(c,d)), how many (s,t) combos give xy=0?
# Let's verify this is exactly 2.
print("ISSUE 7b: Checking that each unsigned ordered pair has exactly 2 sign realizations...")

# Import CD table from script 7 logic
def cd_mult_table(n):
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

mul_table = cd_mult_table(4)

def sedenion_mul_dict(x, y):
    result = {}
    for ai, ac in x.items():
        for bi, bc in y.items():
            idx, sgn = mul_table[ai][bi]
            coeff = ac * bc * sgn
            result[idx] = result.get(idx, 0) + coeff
    return {k: v for k, v in result.items() if v != 0}

pairs = list(combinations(range(1, 16), 2))
signs = [1, -1]

# Count sign realizations per unsigned ordered pair
from collections import Counter
sign_count = Counter()
for (a, b) in pairs:
    for s in signs:
        x = {a: 1, b: s}
        for (c, d) in pairs:
            for t in signs:
                y = {c: 1, d: t}
                if not sedenion_mul_dict(x, y):
                    sign_count[((a, b), (c, d))] += 1

sign_vals = set(sign_count.values())
print(f"  Sign multiplicities per unsigned ordered pair: {sign_vals}")
if sign_vals == {2}:
    print("  Every unsigned ordered pair has exactly 2 sign realizations. Correct.")
else:
    print(f"  *** UNEXPECTED multiplicities: {sign_vals} ***")
    issues.append(("7", "HIGH", f"Sign multiplicity is not uniformly 2: {sign_vals}"))

# ISSUE 7c: The GL(3,2) block embedding fixes the 4th coordinate.
# This means only matrices of the form [[A3, 0], [0, 1]] act on F2^4.
# But sedenion indices 1-15 are mapped to F2^4 via bit decomposition of index.
# The 4th bit corresponds to e_8 (the doubling basis). Fixing it means the
# GL(3,2) action doesn't mix octonion and "doubled" parts.
# This is the standard embedding and is correct for the automorphism of the
# Fano plane embedded in F2^3 (the first 3 bits of the index).
print("ISSUE 7c: GL(3,2) block embedding is standard. Verified.")
print("  SEVERITY: NONE")

# ISSUE 7d: apply_mat_to_pair applies to a pair of INDICES, but the pair
# is supposed to represent {e_a, e_b}. The function sorts the result.
# Let me check: configs are frozensets of two frozensets like frozenset([a,b]).
# apply_mat_to_pair takes a pair (which is a frozenset like frozenset([a,b]))
# and applies M to each index a and b separately, returning frozenset of results.
# Wait, let me re-read the code...
# In script 7 line 152: apply_mat_to_pair(M, pair) where pair is a tuple (a,b)
# from unsigned_ordered elements like ((a,b),(c,d)).
# But in script 9 line 170-171: apply_mat_to_pair takes pair as a frozenset
# and iterates over its elements with idx_to_vec.
# These are different representations! Let me check script 7 more carefully.

# Script 7 line 151: def apply_mat_to_pair(M, pair):
#     return tuple(sorted(vec_to_idx(mat_vec_f2(M, idx_to_vec(i))) for i in pair))
# And unsigned_ordered contains tuples like ((a,b),(c,d)) where (a,b) and (c,d) are tuples.
# So pair = (a,b), and "for i in pair" iterates over a and b.
# Then it maps each index to F2^4, applies M, maps back, and sorts.
# This is correct: it transforms the index pair under the GL(3,2) action.
print("ISSUE 7d: apply_mat_to_pair correctly transforms index pairs. Verified.")
print("  SEVERITY: NONE")

print("VERDICT 7: Script logic is sound. PASS is genuine.")

# ====================================================================
# AUDIT 8: verify_pending_8_rank_drop4.py
# ====================================================================
print("\n--- AUDIT 8: RANK DROP 4 ---")

# ISSUE 8a: The script uses numpy floating-point for rank computation.
# np.linalg.matrix_rank uses SVD with a default tolerance.
# Since the multiplication table gives integer entries and x has integer entries,
# L_x is an integer matrix. We could use exact rank via sympy.
# But for 16x16 integer matrices, numpy's SVD should be reliable.
print("ISSUE 8a: Rank computed via numpy SVD on integer matrices. Should be reliable.")
print("  SEVERITY: NONE")

# ISSUE 8b: The zero-divisor enumeration only considers x = e_a + s*e_b
# (sum/difference of two basis elements). This is the standard Moreno form.
# But are there zero-divisors of other forms?
# In sedenions, ALL zero-divisors of norm sqrt(2) have this form (up to scaling).
# The script correctly restricts to this class.
print("ISSUE 8b: Zero-divisor enumeration is standard Moreno form. Correct.")
print("  SEVERITY: NONE")

# ISSUE 8c: The kernel spanning check.
# For each x, the script finds ALL y such that xy=0 (the "partners").
# It checks len(partners)==4, that they share the same "box" (a^b == c^d),
# that they're linearly independent (rank 4), and that Lx @ y = 0 for each.
# BUT: it does NOT verify that these 4 partners SPAN ker(Lx).
# It only checks that they lie IN ker(Lx) and are independent.
# Since ker(Lx) has dimension 4 (because rank=12 in a 16-dim space),
# 4 independent vectors in a 4-dim space automatically span it.
# So this is actually correct!
print("ISSUE 8c: 4 independent vectors in 4-dim kernel must span it. Logic correct.")
print("  SEVERITY: NONE")

# ISSUE 8d: The zero-divisor deduplication uses np.allclose, which could
# miss near-duplicates or wrongly merge distinct vectors. But since all
# entries are exactly 0.0 or +-1.0, np.allclose with default tol is fine.
print("ISSUE 8d: Deduplication via np.allclose on {0,+-1} vectors is fine.")
print("  SEVERITY: NONE")

# Let me verify the CD table agrees with sedenion_check.py by spot-checking
# a few products:
def cd_mul_vec(x, y, table):
    res = np.zeros(16)
    for i in range(16):
        for j in range(16):
            idx, sgn = table[i][j]
            res[idx] += x[i] * y[j] * sgn
    return res

# e1 * e2 should give some specific e_k
e1 = np.zeros(16); e1[1] = 1
e2 = np.zeros(16); e2[2] = 1
prod = cd_mul_vec(e1, e2, mul_table)
nz = np.nonzero(prod)[0]
print(f"  Spot check: e1*e2 = {'+' if prod[nz[0]]>0 else '-'}e{nz[0]}")

# e3 * e5 
e3 = np.zeros(16); e3[3] = 1
e5 = np.zeros(16); e5[5] = 1
prod2 = cd_mul_vec(e3, e5, mul_table)
nz2 = np.nonzero(prod2)[0]
print(f"  Spot check: e3*e5 = {'+' if prod2[nz2[0]]>0 else '-'}e{nz2[0]}")

# Cross-check with sedenion_check.py's table
# sedenion_check uses _build_table(16) which gives sign_T and idx_T.
# The verify scripts use cd_mult_table(4) which gives list-of-lists of (idx, sign).
# Let me verify they agree on a sample:
print("  (Cross-check with sedenion_check.py would require importing it -- skipped)")

print("VERDICT 8: Script logic is sound. PASS is genuine.")

# ====================================================================
# AUDIT 9: verify_pending_9_moreno_g2.py
# ====================================================================
print("\n--- AUDIT 9: MORENO G2 ---")

# ISSUE 9a: The adjacency relation is "share at least one index."
# A config is a frozenset of two frozensets like {{a,b},{c,d}}.
# The indices are {a,b,c,d} (a set of 3 or 4 indices depending on overlaps).
# Two configs are adjacent if they share any index.
# With 84 configs, each involving at most 4 indices out of 15,
# the expected adjacency by random chance would be:
# P(share at least 1) = 1 - C(11,4)/C(15,4) for 4-element subsets
# = 1 - 330/1365 = 1 - 0.242 = 0.758
# So ~76% adjacency expected, which for 83 neighbors = ~63.
# The actual degree is 61. This is close to random expectation.
#
# CONCERN: Is "share at least one index" the right adjacency for the 
# zero-divisor configuration graph? The task says "共享元素定义邻接".
# This is what's implemented. The regularity (degree 61) and PSL(2,7)
# invariance are non-trivial facts that wouldn't hold for arbitrary
# 84 4-element subsets of a 15-element set.
print("ISSUE 9a: Adjacency = shared index. Degree 61 (regular) with PSL(2,7) symmetry.")
print("  This is non-trivial -- random 4-subsets wouldn't give constant degree.")
print("  SEVERITY: NONE (the graph properties are genuine)")

# ISSUE 9b: The PSL(2,7) check only verifies adjacency preservation,
# not that PSL(2,7) is the FULL automorphism group. The full automorphism
# group could be larger.
print("ISSUE 9b: Script verifies PSL(2,7) IS an automorphism subgroup,")
print("  but does not verify it's the FULL automorphism group.")
print("  SEVERITY: LOW (the claim is about PSL(2,7) acting, not being maximal)")

# ISSUE 9c: The Moreno theorem summary is hard-coded text, not verified.
# The script prints literature claims without computational verification.
# The claim that "PSL(2,7) is a maximal discrete subgroup of G2" is a 
# mathematical fact, but it's asserted, not proved here.
print("ISSUE 9c: Literature claims about G2 are asserted text, not computed.")
print("  SEVERITY: MEDIUM (acceptable for a literature check item)")

print("VERDICT 9: Script logic is sound for what it claims to verify.")
print("  The graph-theoretic computations are genuine.")

# ====================================================================
# AUDIT 10: verify_pending_10_orientation_transport.py
# ====================================================================
print("\n--- AUDIT 10: ORIENTATION TRANSPORT ---")

# ISSUE 10a: The holonomy check uses random walks that happen to return
# to the starting face. With 500 attempts and random walks of length up to 12,
# only 72 loops were found. This is a small sample.
# More importantly: ANY product of +-1 signs is +-1.
# So holonomy = product of +-1 transition signs is ALWAYS +-1 by construction.
# The check "all holonomies quantized to Z2" is TAUTOLOGICALLY TRUE
# because each transition sign t_sign = - d2[e,f1] * d2[e,f2] where d2 entries
# are +1 or -1, so t_sign is always +1 or -1, and any product of +-1 is +-1.
print("ISSUE 10a: *** CRITICAL: Holonomy is ALWAYS +-1 by construction! ***")
print("  Each transition sign t_sign = -d2[e,f1]*d2[e,f2] is +-1 (product of +-1).")
print("  Product of +-1 values is always +-1.")
print("  The check 'all holonomies in {+1,-1}' is TAUTOLOGICALLY TRUE.")
print("  This does NOT verify any non-trivial Z2 quantization!")
print("  The real claim should be about the DISTRIBUTION of holonomies")
print("  (e.g., certain loops give -1 while others give +1),")
print("  or about non-trivial homology classes having specific holonomy values.")
print("  SEVERITY: CRITICAL -- the verification is vacuous.")

issues.append(("10", "CRITICAL", "Holonomy Z2 check is tautological: product of +-1 is always +-1"))

# ISSUE 10b: The diagonal body loop search failed ("No diagonal loop found").
# This was supposed to be a key test: wrapping along the body diagonal.
# The BFS depth limit (20000 queue entries) might be too small, or the
# adjacency structure might not connect faces along body diagonals efficiently.
print("ISSUE 10b: Diagonal loop search failed. Key geometric test not executed.")
print("  SEVERITY: HIGH")

# ISSUE 10c: The face complex construction differs from script 4.
# Script 4 uses enumerate_simplex_faces with vertex tuples like ('bc',i,j,k).
# Script 10 uses integer vertex IDs (0-53) with explicit tet offsets.
# These should produce the same face complex, but the implementations are different.
# Let me check: script 10 produces 540 faces with 378 edges.
# Script 4 produces 540 faces. 
# The edge count wasn't checked in script 4, but 378 is consistent with a 
# face complex where each face has 3 edges and faces share edges.
print("ISSUE 10c: Different face construction from script 4, but both get 540 faces.")
print("  SEVERITY: LOW")

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

for item, severity, desc in issues:
    print(f"  Item {item} [{severity}]: {desc}")

critical_count = sum(1 for _, s, _ in issues if s == "CRITICAL")
high_count = sum(1 for _, s, _ in issues if s == "HIGH")

print(f"\nTotal CRITICAL issues: {critical_count}")
print(f"Total HIGH issues: {high_count}")
if critical_count > 0:
    print("\n*** SCRIPTS WITH CRITICAL ISSUES NEED REWRITE ***")
