# verify_pending_7_orientation_forgetting.py
# Verification of Item 7: 336/168/84 orientation forgetting ladder
# We verify the quotient structure: signed ordered (336) -> unsigned ordered (168) -> unsigned unordered (84)
# and compare it with the Klein quartic automorphism group chains.

import sys
from itertools import combinations
from sympy.combinatorics import PermutationGroup, Permutation

def cd_mult_table(n):
    """Build Cayley-Dickson multiplication table (same as sedenion_klein_84.py)."""
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

def main():
    print("=" * 70)
    print("VERIFYING PENDING 7: 336/168/84 ORIENTATION FORGETTING LADDER")
    print("=" * 70)

    # 1. Build sedenion multiplication table
    mul_table = cd_mult_table(4) # 16 dimensional

    def sedenion_mul(x, y):
        result = {}
        for ai, ac in x.items():
            for bi, bc in y.items():
                idx, sgn = mul_table[ai][bi]
                coeff = ac * bc * sgn
                result[idx] = result.get(idx, 0) + coeff
        return {k: v for k, v in result.items() if v != 0}

    # 2. Enumerate signed ordered zero-divisor pairs
    # x = e_a + s*e_b, y = e_c + t*e_d
    pairs = list(combinations(range(1, 16), 2))
    signs = [1, -1]
    
    zero_div_ordered_signed = [] # size 336
    for (a, b) in pairs:
        for s in signs:
            x = {a: 1, b: s}
            for (c, d) in pairs:
                for t in signs:
                    y = {c: 1, d: t}
                    if not sedenion_mul(x, y):
                        zero_div_ordered_signed.append(((a, b, s), (c, d, t)))

    print(f"1. Signed ordered zero-divisor pairs: {len(zero_div_ordered_signed)} (expected 336)")
    assert len(zero_div_ordered_signed) == 336, f"Expected 336, got {len(zero_div_ordered_signed)}"

    # 3. quotient 336 -> 168 (forgetting signs)
    # Identify pairs by their indices only: ((a,b,s),(c,d,t)) -> ((a,b),(c,d))
    unsigned_ordered = set()
    for (fac1, fac2) in zero_div_ordered_signed:
        a, b, s = fac1
        c, d, t = fac2
        unsigned_ordered.add(((a, b), (c, d)))
        
    print(f"2. Unsigned ordered zero-divisor pairs (forgetting signs): {len(unsigned_ordered)} (expected 168)")
    assert len(unsigned_ordered) == 168, f"Expected 168, got {len(unsigned_ordered)}"

    # 4. quotient 168 -> 84 (forgetting order)
    # Identify ((a,b),(c,d)) with ((c,d),(a,b))
    unsigned_unordered = set()
    for (p1, p2) in unsigned_ordered:
        key = frozenset([p1, p2])
        unsigned_unordered.add(key)
        
    print(f"3. Unsigned unordered zero-divisor pairs (forgetting order): {len(unsigned_unordered)} (expected 84)")
    assert len(unsigned_unordered) == 84, f"Expected 84, got {len(unsigned_unordered)}"

    # 5. Group theory verification: GL(3,2) action on the 168 unsigned ordered pairs
    # and 84 unsigned unordered pairs.
    # GL(3,2) represents the automorphism group PSL(2,7) of order 168.
    # We build the 168 elements of GL(3,2) and check their action.
    # Representation of GL(3,2) block embedding in GL(4,2) (same as sedenion_klein_84.py):
    
    # We will build GL(3,2) (order 168) and embed in GL(4,2)
    def mat3_det_f2(M):
        m = [list(row) for row in M]
        for col in range(3):
            pivot = None
            for row in range(col, 3):
                if m[row][col] == 1:
                    pivot = row
                    break
            if pivot is None:
                return 0
            if pivot != col:
                m[col], m[pivot] = m[pivot], m[col]
            for row in range(3):
                if row != col and m[row][col] == 1:
                    for k in range(3):
                        m[row][k] = (m[row][k] + m[col][k]) % 2
        return 1

    gl32 = []
    for bits in range(1 << 9):
        A = []
        for row in range(3):
            A.append(tuple((bits >> (row * 3 + col)) & 1 for col in range(3)))
        if mat3_det_f2(A) == 1:
            gl32.append(A)

    print(f"\nGL(3,2) group order: {len(gl32)} (expected 168)")
    
    # Block embedding: GL(3,2) -> GL(4,2)
    def gl32_to_gl42(A3):
        M4 = []
        for row in range(4):
            if row < 3:
                new_row = list(A3[row]) + [0]
            else:
                new_row = [0, 0, 0, 1]
            M4.append(tuple(new_row))
        return M4

    block_gl42 = [gl32_to_gl42(A) for A in gl32]

    # Map index to vector in F2^4
    def idx_to_vec(i):
        return tuple((i >> k) & 1 for k in range(4))

    def vec_to_idx(v):
        return sum(v[k] << k for k in range(4))

    def mat_vec_f2(M, v):
        result = []
        for row in M:
            val = sum(row[k] * v[k] for k in range(4)) % 2
            result.append(val)
        return tuple(result)

    def apply_mat_to_pair(M, pair):
        return tuple(sorted(vec_to_idx(mat_vec_f2(M, idx_to_vec(i))) for i in pair))

    # The 84 configurations as frozensets of frozensets
    zero_div_84_list = list(unsigned_unordered)
    
    # Verify G preserves the 84-set
    preserves_all = True
    for M in block_gl42:
        for config in zero_div_84_list:
            # config is frozenset of two pairs
            new_config = frozenset(apply_mat_to_pair(M, pair) for pair in config)
            if new_config not in unsigned_unordered:
                preserves_all = False
                break
        if not preserves_all:
            break
            
    print(f"GL(3,2) preserves the 84-set: {preserves_all}")
    assert preserves_all, "GL(3,2) does not preserve the 84-set!"

    # Build permutations on the 168 unsigned ordered pairs
    unsigned_ordered_list = list(unsigned_ordered)
    ordered_to_idx = {p: i for i, p in enumerate(unsigned_ordered_list)}

    perms_168 = []
    for M in block_gl42:
        perm = []
        for p in unsigned_ordered_list:
            new_p = (apply_mat_to_pair(M, p[0]), apply_mat_to_pair(M, p[1]))
            perm.append(ordered_to_idx[new_p])
        perms_168.append(Permutation(perm))
        
    G_induced = PermutationGroup(*perms_168)
    print(f"Induced group order on 168: {G_induced.order()} (expected 168)")
    print(f"  Is transitive on 168? {G_induced.is_transitive()} (expected True)")
    
    # Point stabilizer of G_induced on 168
    stab_168 = G_induced.stabilizer(0)
    print(f"  Point stabilizer order on 168: {stab_168.order()} (expected 1)")
    
    # Point stabilizer of G_induced on 84 (unsigned unordered)
    perms_84 = []
    unordered_to_idx = {c: i for i, c in enumerate(zero_div_84_list)}
    for M in block_gl42:
        perm = []
        for config in zero_div_84_list:
            new_config = frozenset(apply_mat_to_pair(M, pair) for pair in config)
            perm.append(unordered_to_idx[new_config])
        perms_84.append(Permutation(perm))
        
    G_induced_84 = PermutationGroup(*perms_84)
    print(f"Induced group order on 84: {G_induced_84.order()} (expected 168)")
    print(f"  Is transitive on 84? {G_induced_84.is_transitive()} (expected True)")
    stab_84 = G_induced_84.stabilizer(0)
    print(f"  Point stabilizer order on 84: {stab_84.order()} (expected 2)")

    # Comparison with Klein quartic flag/edge chains:
    # Klein quartic flag count = 336 (full automorphism group Aut(X) order 336)
    # Aut^+(X) = PSL(2,7) (order 168) acts simply transitively on directed edges (order 168)
    # Aut^+(X) acts on 84 undirected edges with Z2 stabilizer
    # These match our quotients exactly:
    # 336 signed ordered -> 168 unsigned ordered -> 84 unsigned unordered.
    # Therefore, the sign-forgetting and order-forgetting are isomorphic to flag-forgetting and orientation-forgetting.
    
    ok = (G_induced.order() == 168 and G_induced.is_transitive() and stab_168.order() == 1 
          and G_induced_84.order() == 168 and G_induced_84.is_transitive() and stab_84.order() == 2)
          
    if ok:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
