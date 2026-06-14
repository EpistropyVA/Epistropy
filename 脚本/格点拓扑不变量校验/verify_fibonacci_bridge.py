# -*- coding: utf-8 -*-
"""
verify_fibonacci_bridge.py

Verifies whether the "Fibonacci compiler" quadratic polynomial ansatz:
    f(x) = x^2 + x + c
can predict the bridge behavior and spectral properties for L=4 and L=5.
Prints the exact minimal polynomials and f(-3) values for non-flat bands of all orbits.
"""

import sys
import io
import itertools
from fractions import Fraction
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import L-scan logic
import os
sys.path.insert(0, os.path.dirname(__file__))
from verify_bridge_norm_L_scan import (
    build_bcc_lattice_L, build_all_faces_L, build_adjacency_matrix_L,
    build_transfer_blocks_L, build_H_k_L, compute_all_orbits,
    analyze_k_point, eval_min_poly_at, prime_factors, factored_str
)

# Fibonacci numbers helper
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def print_poly(coeffs):
    terms = []
    deg = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        power = deg - i
        if c == 0:
            continue
        c_str = str(c) if c >= 0 else f"({c})"
        if power == 0:
            terms.append(c_str)
        elif power == 1:
            if c == 1:
                terms.append("x")
            elif c == -1:
                terms.append("-x")
            else:
                terms.append(f"{c}x")
            terms.append("x")
        else:
            if c == 1:
                terms.append(f"x^{power}")
            elif c == -1:
                terms.append(f"-x^{power}")
            else:
                terms.append(f"{c}x^{power}")
    return " + ".join(terms).replace("+ -", "- ")

def main():
    print("=" * 78)
    print("  FIBONACCI COMPILER ANALYSIS: L=2, 3, 4, 5 BRIDGE SPECTRUM")
    print("=" * 78)

    for L in [2, 3, 4, 5]:
        print(f"\n==============================================================")
        print(f"  L = {L} Analysis")
        print(f"==============================================================")
        
        # Build lattice and transfer blocks
        body_centers = build_bcc_lattice_L(L)
        face_to_idx, bc_face_indices, _ = build_all_faces_L(body_centers, L)
        A_full = build_adjacency_matrix_L(face_to_idx)
        T = build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L)
        
        # Orbits
        orbits = compute_all_orbits(L)
        print(f"  k-points: {L**3} | O_h orbits: {len(orbits)}")
        
        for idx, (rep, orb) in enumerate(orbits):
            k_vec = np.array([2 * np.pi * float(n) / L for n in rep])
            H = build_H_k_L(T, k_vec)
            
            flat_count, classes, all_evals = analyze_k_point(H)
            
            p_topo = len(orb) - 1
            print(f"\n  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo} | flat_count={flat_count}")
            
            if not classes:
                print("    (no non-flat eigenvalues)")
                continue
                
            for ci, cls in enumerate(classes):
                coeffs = cls.get('min_poly_coeffs', [])
                deg = cls['degree']
                mult = cls['mult']
                
                if coeffs:
                    f_val = eval_min_poly_at(coeffs, Fraction(-3))
                    f_abs = abs(f_val)
                    primes = sorted(set(prime_factors(f_abs.numerator) + prime_factors(f_abs.denominator)))
                    bridge = p_topo in primes and p_topo > 0
                    
                    # Try to see if it fits x^2 + x + c
                    is_fibo_quad = False
                    if deg == 2 and coeffs[0] == 1 and coeffs[1] == 1:
                        is_fibo_quad = True
                        c = coeffs[2]
                        print(f"    Class {ci+1}: deg={deg} mult={mult} [FIBONACCI QUADRATIC x^2+x+{c}]")
                    else:
                        print(f"    Class {ci+1}: deg={deg} mult={mult}")
                        
                    # Check for factors or roots relating to gold ratio / fibonacci
                    print(f"      f(x)  = {cls['sym_polys'] if not coeffs else coeffs}")
                    print(f"      f(-3) = {f_val} = {factored_str(int(f_val)) if f_val.denominator==1 else f_val} (primes: {primes})")
                    print(f"      bridge = {'YES' if bridge else 'no'}")
                    
                    # Fibonacci matches check
                    f_val_int = abs(int(f_val)) if f_val.denominator == 1 else None
                    if f_val_int is not None:
                        # Find closest Fibonacci number
                        found_fib = []
                        for fn_idx in range(1, 25):
                            fn = fibonacci(fn_idx)
                            if fn == f_val_int:
                                found_fib.append(f"F({fn_idx})")
                            elif f_val_int % fn == 0 and fn > 1:
                                found_fib.append(f"divisible by F({fn_idx})={fn}")
                        if found_fib:
                            print(f"      Fibonacci relation: {', '.join(found_fib)}")
                else:
                    print(f"    Class {ci+1}: deg={deg} mult={mult} (UNCLASSIFIED)")

if __name__ == "__main__":
    main()
