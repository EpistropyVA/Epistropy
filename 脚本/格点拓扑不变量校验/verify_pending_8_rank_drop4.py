# verify_pending_8_rank_drop4.py
# Verification of Item 8: rank drop 4 mechanism in sedenions
# We verify:
#   1. rank(L_x) = 12 for all zero-divisors x
#   2. rank(L_x L_y) = 8 for all zero-divisor pairs x, y with xy = 0
#   3. ker(L_x) (4-dimensional) is spanned by its box partners in the box-kite structure.

import numpy as np
from itertools import combinations

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

def main():
    print("=" * 70)
    print("VERIFYING PENDING 8: RANK DROP 4 MECHANISM")
    print("=" * 70)

    # 1. Build sedenion multiplication table
    mul_table = cd_mult_table(4)

    def cd_mul(x, y):
        res = np.zeros(16)
        for i in range(16):
            for j in range(16):
                idx, sgn = mul_table[i][j]
                res[idx] += x[i] * y[j] * sgn
        return res

    def get_L_matrix(x):
        L = np.zeros((16, 16))
        for j in range(16):
            ej = np.zeros(16)
            ej[j] = 1.0
            L[:, j] = cd_mul(x, ej)
        return L

    # 2. Generate all zero divisors (x = e_a + s*e_b)
    pairs = list(combinations(range(1, 16), 2))
    signs = [1, -1]
    
    zero_divisors = []
    zero_div_pairs = []
    
    for (a, b) in pairs:
        for s in signs:
            x_vec = np.zeros(16)
            x_vec[a] = 1.0
            x_vec[b] = s
            # We want to identify all zero-divisors by checking if there is any partner
            for (c, d) in pairs:
                for t in signs:
                    y_vec = np.zeros(16)
                    y_vec[c] = 1.0
                    y_vec[d] = t
                    if np.linalg.norm(cd_mul(x_vec, y_vec)) < 1e-10:
                        zero_div_pairs.append((x_vec, y_vec))
                        if not any(np.allclose(x_vec, z) for z in zero_divisors):
                            zero_divisors.append(x_vec)

    print(f"Total unique zero-divisors found (norm sqrt(2)): {len(zero_divisors)}")
    print(f"Total zero-divisor pairs (xy = 0): {len(zero_div_pairs)}")

    # 3. Verify rank(L_x) = 12 for all zero-divisors x
    rank_12_ok = True
    for x in zero_divisors:
        Lx = get_L_matrix(x)
        rank = np.linalg.matrix_rank(Lx)
        if rank != 12:
            rank_12_ok = False
            print(f"  Failed for x: rank(L_x) = {rank}")
            break
    print(f"Verification: rank(L_x) = 12 for all zero-divisors: {rank_12_ok}")
    assert rank_12_ok, "rank(L_x) is not 12 for all zero-divisors!"

    # 4. Verify rank(L_x L_y) = 8 for all zero-divisor pairs with xy = 0
    rank_8_ok = True
    for x, y in zero_div_pairs:
        Lx = get_L_matrix(x)
        Ly = get_L_matrix(y)
        rank = np.linalg.matrix_rank(Lx @ Ly)
        if rank != 8:
            rank_8_ok = False
            print(f"  Failed for pair: rank(L_x L_y) = {rank}")
            break
    print(f"Verification: rank(L_x L_y) = 8 for all pairs with xy = 0: {rank_8_ok}")
    assert rank_8_ok, "rank(L_x L_y) is not 8 for all zero-divisor pairs!"

    # 5. Verify ker(L_x) is spanned by its box partners
    # For a given zero-divisor x = e_a + s*e_b, its box is determined by u = a XOR b.
    # Its partners are the other zero-divisors y in the same box such that xy = 0.
    # Let's verify that for each x, the 4 partners y in the same box span the 4D ker(L_x).
    ker_spanned_ok = True
    for x in zero_divisors:
        a, b = np.nonzero(x)[0]
        u = a ^ b
        # Find all y in the zero_div_pairs that pair with x
        partners = []
        for x_p, y_p in zero_div_pairs:
            if np.allclose(x_p, x):
                partners.append(y_p)
        
        # Check there are exactly 4 partners
        if len(partners) != 4:
            ker_spanned_ok = False
            print(f"  x has {len(partners)} partners instead of 4!")
            break
            
        # Check that the 4 partners are in the same box (c ^ d == u)
        for y in partners:
            c, d = np.nonzero(y)[0]
            if c ^ d != u:
                ker_spanned_ok = False
                print(f"  Partner y indices {c}, {d} XOR to {c^d}, expected {u}!")
                break
                
        # Check that the 4 partners are linearly independent (so they span a 4D subspace)
        partner_matrix = np.column_stack(partners)
        rank_partners = np.linalg.matrix_rank(partner_matrix)
        if rank_partners != 4:
            ker_spanned_ok = False
            print(f"  Partners are not linearly independent! Rank = {rank_partners}")
            break
            
        # Check that the partners lie in ker(L_x)
        Lx = get_L_matrix(x)
        for y in partners:
            if np.linalg.norm(Lx @ y) > 1e-10:
                ker_spanned_ok = False
                print(f"  Lx @ y is not zero: norm = {np.linalg.norm(Lx @ y)}")
                break

    print(f"Verification: ker(L_x) is spanned by the 4 box partners: {ker_spanned_ok}")
    assert ker_spanned_ok, "ker(L_x) is not spanned by its box partners!"

    # 6. Algebraic explanation
    print("\nAlgebraic commentary:")
    print("  - Sedenions are built from Octonions via Cayley-Dickson doubling (S = O + e_8 O).")
    print("  - Sedenion zero-divisors have one component in O and one component in e_8 O.")
    print("  - For x = e_a + s*e_b, the singular subspace of L_x is exactly 4-dimensional.")
    print("  - This 4-dimensional kernel represents a quaternionic subalgebra H (4D) associated")
    print("    with the Fano point of the Fano plane, showing how non-associativity is restricted")
    print("    by the 4D quaternion division algebra structure within the sedenions.")

    if rank_12_ok and rank_8_ok and ker_spanned_ok:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
