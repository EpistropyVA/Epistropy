# -*- coding: utf-8 -*-
"""
verify_eu_tensor_product.py

Verifies the character tensor product identity Eu = A2u ⊗ Eg under the full octahedral group Oh.
Performs class-by-class verification and projection computation to confirm the isomorphism.
"""

import sys
import io
import os
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def main():
    # Oh conjugacy classes:
    # E, 8C3, 6C2, 6C4, 3C2'(3C4^2), i, 8S6, 6sigma_d, 6S4, 3sigma_h
    classes = ['E', '8C3', '6C2', '6C4', '3C2_prime', 'i', '8S6', '6sigma_d', '6S4', '3sigma_h']
    class_sizes = [1, 8, 6, 6, 3, 1, 8, 6, 6, 3]
    order = sum(class_sizes)
    assert order == 48

    # Character rows (from standard Oh character table)
    # A2u: 1D pseudo-scalar representation (ungerade)
    chi_A2u = [1, 1, -1, -1, 1, -1, -1, 1, 1, -1]
    
    # Eg: 2D representation (gerade)
    chi_Eg  = [2, -1, 0, 0, 2, 2, -1, 0, 0, 2]
    
    # Eu: 2D representation (ungerade)
    chi_Eu  = [2, -1, 0, 0, 2, -2, 1, 0, 0, -2]

    # Compute tensor product character: chi_{A2u ⊗ Eg}(g) = chi_A2u(g) * chi_Eg(g)
    chi_product = [chi_A2u[i] * chi_Eg[i] for i in range(10)]

    out_file_path = os.path.join(os.path.dirname(__file__), "verify_eu_tensor_product_results.txt")
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    log_print("=" * 80)
    log_print("  Oh IRREP TENSOR PRODUCT VERIFICATION: Eu = A2u ⊗ Eg")
    log_print("=" * 80)
    log_print(f"  Oh Group Order: {order}")
    log_print("\n   conjugacy class-by-class character values:")
    log_print(f"  {'Class':12s} {'Size':4s} | {'A2u':>4s}   {'Eg':>4s} | {'A2u ⊗ Eg':>8s} | {'Eu (Target)':>11s} | {'Match?':>6s}")
    log_print("-" * 80)
    
    all_match = True
    for i in range(10):
        prod = chi_product[i]
        target = chi_Eu[i]
        match = (prod == target)
        if not match:
            all_match = False
        log_print(f"  {classes[i]:12s} {class_sizes[i]:4d} | {chi_A2u[i]:+4d}   {chi_Eg[i]:+4d} | {prod:+8d} | {target:+11d} | {'PASS' if match else 'FAIL'}")
    
    # Project product onto Eu: <chi_product, chi_Eu> = (1/|G|) * sum_g size(g) * chi_product(g) * chi_Eu(g)*
    proj = sum(class_sizes[i] * chi_product[i] * chi_Eu[i] for i in range(10)) / float(order)
    
    log_print("-" * 80)
    log_print(f"  Projection <A2u ⊗ Eg, Eu>: {proj:.4f} (expected 1.0000 for isomorphism)")
    
    # Project product onto all other irreps to verify no residue
    # All 10 Oh irreps: A1g, A2g, Eg, T1g, T2g, A1u, A2u, Eu, T1u, T2u
    irreps = {
        'A1g': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'A2g': [1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
        'Eg' : [2, -1, 0, 0, 2, 2, -1, -1, 0, 0], # wait, Eg and Eu have 0 in some columns
        # let's write them down properly
        # actually, just projecting on Eu and checking it equals 1 is mathematically sufficient
        # since dim(A2u ⊗ Eg) = 2, and dim(Eu) = 2, projection=1 means they are isomorphic.
    }
    
    isomorphic = (all_match and np.isclose(proj, 1.0))
    log_print(f"\n  Isomorphism Eu = A2u ⊗ Eg confirmed? {isomorphic}")
    log_print(f"  Overall Verdict: {'PASS' if isomorphic else 'FAIL'}")
    
    log_file.close()

if __name__ == "__main__":
    main()
