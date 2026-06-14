# -*- coding: utf-8 -*-
"""
verify_klein_84_edge.py

Verification of Item 23 / taskboard:
84 = Klein edge traversal verification (SAME-84 foundation).
We verify:
1. Sedenion zero-divisor index configurations yield exactly 84 unordered pairs.
2. The automorphism group GL(4,2) (order 20160) acting on F2^4 preserves this 84-set.
3. The stabilizer subgroup H in GL(4,2) that preserves the 84-set has order 168.
4. H is isomorphic to GL(3,2) (which is PSL(2,7), the automorphism group of the Klein quartic).
5. H acts transitively on the 84 pairs, and the stabilizer of any single pair has order 2 (168 / 84 = 2).
"""

import sys
import io
import os
import itertools
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Reuse the CD multiplication table construction
from verify_pending_8_rank_drop4 import cd_mult_table

def get_84_zero_divisor_configs(mul_table):
    """Find the 84 index configurations {{a,b}, {c,d}} representing base zero divisors."""
    pairs = list(itertools.combinations(range(1, 16), 2))
    signs = [1, -1]
    
    def cd_mul(x_vec, y_vec):
        res = np.zeros(16)
        for i in range(16):
            for j in range(16):
                idx, sgn = mul_table[i][j]
                res[idx] += x_vec[i] * y_vec[j] * sgn
        return res

    zd_configs = set()
    for (a, b) in pairs:
        for s in signs:
            x_vec = np.zeros(16)
            x_vec[a] = 1.0
            x_vec[b] = s
            for (c, d) in pairs:
                for t in signs:
                    y_vec = np.zeros(16)
                    y_vec[c] = 1.0
                    y_vec[d] = t
                    if np.linalg.norm(cd_mul(x_vec, y_vec)) < 1e-9:
                        # Construct configuration {{a,b}, {c,d}}
                        config = frozenset([frozenset([a, b]), frozenset([c, d])])
                        zd_configs.add(config)
                        
    return zd_configs

def get_gl42_matrices():
    """Generate all 20160 invertible 4x4 matrices over F2."""
    # Enumerate all 4x4 matrices over F2 and check invertibility
    # A faster way: build columns step by step to ensure linear independence
    # Col 1: any non-zero vector (15 options)
    # Col 2: any vector not in span(Col 1) (14 options)
    # Col 3: any vector not in span(Col 1, Col 2) (12 options)
    # Col 4: any vector not in span(Col 1, Col 2, Col 3) (8 options)
    # Total = 15 * 14 * 12 * 8 = 20160
    
    all_vecs = [np.array(list(seq), dtype=int) for seq in itertools.product((0, 1), repeat=4)]
    non_zero_vecs = [v for v in all_vecs if np.sum(v) > 0]
    
    matrices = []
    
    # We will build matrices
    for c1 in non_zero_vecs:
        span1 = {tuple(c1), (0,0,0,0)}
        for c2 in non_zero_vecs:
            if tuple(c2) in span1:
                continue
            span2 = {tuple(c2), tuple((c1 + c2) % 2)} | span1
            for c3 in non_zero_vecs:
                if tuple(c3) in span2:
                    continue
                # compute span of c1, c2, c3
                span3 = set()
                for c1_coeff in (0, 1):
                    for c2_coeff in (0, 1):
                        for c3_coeff in (0, 1):
                            v = (c1_coeff*c1 + c2_coeff*c2 + c3_coeff*c3) % 2
                            span3.add(tuple(v))
                for c4 in non_zero_vecs:
                    if tuple(c4) in span3:
                        continue
                    M = np.column_stack([c1, c2, c3, c4])
                    matrices.append(M)
                    
    assert len(matrices) == 20160
    return matrices

def apply_matrix_to_index(M, a):
    # Convert index a in {1..15} to binary 4D vector
    v = np.array([a & 1, (a >> 1) & 1, (a >> 2) & 1, (a >> 3) & 1], dtype=int)
    # Apply M
    v_new = (M @ v) % 2
    # Convert back to index
    a_new = int(v_new[0] + (v_new[1] << 1) + (v_new[2] << 2) + (v_new[3] << 3))
    return a_new

def main():
    out_file_path = os.path.join(os.path.dirname(__file__), "verify_klein_84_edge_results.txt")
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    log_print("=" * 80)
    log_print("  VERIFYING 84 = KLEIN EDGE TRAVERSAL (SAME-84 FOUNDATION)")
    log_print("=" * 80)

    # 1. Build table and extract 84 configurations
    mul_table = cd_mult_table(4)
    zd_configs = get_84_zero_divisor_configs(mul_table)
    log_print(f"  Total zero-divisor configurations found: {len(zd_configs)} (expect 84)")
    assert len(zd_configs) == 84
    log_print("  PASS: Exactly 84 zero-divisor configurations found.")
    
    # Helper to apply matrix to a configuration
    def apply_matrix_to_config(M, config):
        pairs_list = list(config)
        p1 = list(pairs_list[0])
        p2 = list(pairs_list[1])
        np1 = frozenset([apply_matrix_to_index(M, p1[0]), apply_matrix_to_index(M, p1[1])])
        np2 = frozenset([apply_matrix_to_index(M, p2[0]), apply_matrix_to_index(M, p2[1])])
        return frozenset([np1, np2])

    # Print the 84 configurations grouped by their box labels
    # Box label u = a ^ b
    boxes = {}
    for config in zd_configs:
        p1 = list(list(config)[0])
        u = p1[0] ^ p1[1]
        # format configuration for printing
        pairs_list = sorted([sorted(list(p)) for p in config])
        boxes.setdefault(u, []).append(pairs_list)
        
    log_print("\n  Grouped by Box labels (u = a XOR b):")
    for u in sorted(boxes.keys()):
        log_print(f"    Box {u} (size {len(boxes[u])}): {sorted(boxes[u])}")
        assert len(boxes[u]) == 12
    log_print("  PASS: 7 boxes of size 12 verified (7 * 12 = 84).")

    # 2. Generate GL(4,2) matrices
    log_print("\n  Generating GL(4,2) matrices...")
    gl42 = get_gl42_matrices()
    log_print(f"  GL(4,2) size: {len(gl42)} (expect 20160)")
    
    # 3. Find subgroup H that preserves the 84-set
    log_print("\n  Finding the stabilizer subgroup H in GL(4,2)...")
    H = []
    for M in gl42:
        # Check if M preserves zd_configs set
        preserves = True
        for config in zd_configs:
            if apply_matrix_to_config(M, config) not in zd_configs:
                preserves = False
                break
        if preserves:
            H.append(M)
            
    log_print(f"  Stabilizer subgroup H order: {len(H)} (expect 168)")
    assert len(H) == 168
    log_print("  PASS: Subgroup H order is exactly 168.")
    
    # 4. Verify Transitivity of H on the 84 configurations
    log_print("\n  Checking transitivity of H on the 84 configurations...")
    ref_config = list(zd_configs)[0]
    reached = set()
    
    # Find all orbits under H for ref_config
    for M in H:
        reached.add(apply_matrix_to_config(M, ref_config))
        
    log_print(f"  Size of reached set under H from a single configuration: {len(reached)} (expect 84)")
    is_transitive = (len(reached) == 84)
    assert is_transitive
    log_print("  PASS: H acts transitively on the 84 configurations.")
    
    # 5. Check stabilizer size of a single configuration
    log_print("\n  Checking stabilizer size of a single configuration...")
    stab_count = 0
    for M in H:
        if apply_matrix_to_config(M, ref_config) == ref_config:
            stab_count += 1
            
    log_print(f"  Stabilizer of a single configuration size: {stab_count} (expect 2)")
    assert stab_count == 2
    log_print("  PASS: Stabilizer size is exactly 2.")
    
    # 6. Verify H isomorphic to GL(3,2) (PSL(2,7))
    # GL(3,2) is a simple group of order 168.
    # To check that H is isomorphic to GL(3,2), we check that H is a simple group of order 168.
    # The only simple group of order 168 is PSL(2,7) ~ GL(3,2).
    # Thus, if H is a subgroup of GL(4,2) of order 168, and it has no normal subgroups (except id and H),
    # it must be simple, hence isomorphic to PSL(2,7).
    #
    # Let's check simplicity: check all conjugacy classes of H and see if their sums can form normal subgroups.
    # We represent H elements as permutation representation on {0..167} of H (Cayley table) to analyze group properties.
    # This is a robust isomorphism verification.
    
    # Build Cayley table for H
    H_keys = [tuple(M.flatten()) for M in H]
    H_dict = {key: idx for idx, key in enumerate(H_keys)}
    
    # Check closure
    cayley = np.zeros((168, 168), dtype=int)
    for i, M_i in enumerate(H):
        for j, M_j in enumerate(H):
            prod = (M_i @ M_j) % 2
            prod_key = tuple(prod.flatten())
            assert prod_key in H_dict
            cayley[i, j] = H_dict[prod_key]
            
    # Find conjugacy classes
    id_idx = H_dict[tuple(np.eye(4, dtype=int).flatten())]
    conj_classes = []
    visited = [False] * 168
    for i in range(168):
        if visited[i]:
            continue
        c_class = set()
        # g^-1 * h * g for all g
        for g in range(168):
            # find g_inv
            g_inv = next(gi for gi in range(168) if cayley[gi, g] == id_idx)
            # compute conjugate
            conj = cayley[cayley[g_inv, i], g]
            c_class.add(conj)
        for c in c_class:
            visited[c] = True
        conj_classes.append(c_class)
        
    class_sizes = sorted([len(c) for c in conj_classes])
    log_print(f"\n  H conjugacy class sizes: {class_sizes} (expect simple group of order 168 profile: [1, 21, 24, 24, 42, 56])")
    
    # Simple group check
    is_simple = (class_sizes == [1, 21, 24, 24, 42, 56])
    log_print(f"  Is H a simple group isomorphic to PSL(2,7)? {is_simple}")
    assert is_simple
    log_print("  PASS: H is a simple group of order 168, which is uniquely isomorphic to PSL(2,7) / GL(3,2).")

    log_print("\n" + "=" * 80)
    log_print("  ALL VERIFICATIONS PASSED: H ~ PSL(2,7) acts transitively on 84 configurations")
    log_print("=" * 80)

    log_file.close()

if __name__ == "__main__":
    main()
