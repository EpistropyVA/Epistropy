# -*- coding: utf-8 -*-
"""
verify_l4_gaussian_bridge.py

Computes the exact characteristic polynomials for L=4 BCC orbits over the Gaussian rational field Q(i).
Performs exact factorization using extension=sp.I, evaluates f(-3) = A + B*i,
computes the Gaussian norm A^2 + B^2, and checks if it is divisible by the topological prime p.
"""

import sys
import io
import os
import time
from fractions import Fraction
import numpy as np
import sympy as sp

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Import L-scan logic from parent directory if needed, or import from current
sys.path.insert(0, os.path.dirname(__file__))
from verify_bridge_norm_L_scan import (
    build_bcc_lattice_L, build_all_faces_L, build_adjacency_matrix_L,
    build_transfer_blocks_L, compute_all_orbits
)

def prime_factors(n):
    n = abs(int(n))
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return sorted(factors)

def factored_str(n):
    n = int(n)
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n == 1:
        return sign + "1"
    facs = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            facs[d] = facs.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        facs[n] = facs.get(n, 0) + 1
    parts = []
    for p in sorted(facs):
        parts.append(str(p) if facs[p] == 1 else f"{p}^{facs[p]}")
    return sign + "·".join(parts)

def get_gaussian_norm_and_primes(val):
    val = sp.expand(val)
    # Check if there is I
    a = sp.re(val)
    b = sp.im(val)
    norm = a**2 + b**2
    try:
        norm_int = abs(int(norm))
    except TypeError as e:
        print(f"ERROR: val={val}, norm={norm}, type(norm)={type(norm)}, free_symbols={norm.free_symbols}")
        raise e
    return norm_int, prime_factors(norm_int)

def clean_poly_print(poly, x):
    return str(sp.expand(poly))

def main():
    L = 4
    out_file_path = os.path.join(os.path.dirname(__file__), "verify_l4_gaussian_bridge_results.txt")
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    log_print("=" * 80)
    log_print("  L=4 GAUSSIAN RATIONAL FIELD Q(i) EXACT ALGEBRAIC BRIDGE ANALYSIS")
    log_print("=" * 80)
    
    x = sp.Symbol('x', real=True)
    
    # Build lattice and transfer blocks
    body_centers = build_bcc_lattice_L(L)
    face_to_idx, bc_face_indices, _ = build_all_faces_L(body_centers, L)
    A_full = build_adjacency_matrix_L(face_to_idx)
    T = build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L)
    
    # Orbits
    orbits = compute_all_orbits(L)
    log_print(f"  k-points: {L**3} | O_h orbits: {len(orbits)}")
    
    for idx, (rep, orb) in enumerate(orbits):
        is_gamma = all(n == 0 for n in rep)
        p_topo = len(orb) - 1
        
        # Construct H(k) using exact arithmetic in Q(i)
        H_sympy = sp.zeros(20, 20)
        for R, block in T.items():
            dot_product = rep[0]*R[0] + rep[1]*R[1] + rep[2]*R[2]
            phase = sp.I ** (dot_product % 4)
            
            for r in range(20):
                for c in range(20):
                    val = block[r, c]
                    if val != 0:
                        val_int = int(round(val))
                        H_sympy[r, c] += val_int * phase
        
        # Compute characteristic polynomial without re() projection
        t0 = time.time()
        P = H_sympy.charpoly(x).as_expr()
        
        # Substitute internal complex x with our real x symbol
        x_syms = [s for s in P.free_symbols if s.name == 'x']
        if x_syms:
            P = P.subs(x_syms[0], x)
        
        P_field = sp.expand(P)
        charpoly_time = time.time() - t0
        
        # Factor polynomial over Q(i)
        t0 = time.time()
        factored = sp.factor(P_field, extension=sp.I)
        factor_time = time.time() - t0
        
        log_print(f"\n  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo}")
        log_print(f"    Charpoly in {charpoly_time:.2f}s, factored in {factor_time:.2f}s")
        
        # Parse factored terms
        args = sp.Mul.make_args(factored)
        class_idx = 1
        
        # Find the actual x symbol in factored
        x_syms = [s for s in factored.free_symbols if s.name == 'x']
        x_var = x_syms[0] if x_syms else x
        
        for arg in args:
            if isinstance(arg, sp.Pow):
                base, mult = arg.as_base_exp()
            else:
                base = arg
                mult = 1
            
            if base.is_number:
                continue
                
            is_flat = (base == x_var + 3)
            base_str = clean_poly_print(base, x_var)
            deg = sp.degree(base, x_var)
            
            # Evaluate at x = -3
            f_val = base.subs(x_var, -3)
            norm_val, primes = get_gaussian_norm_and_primes(f_val)
            
            # Check bridge
            bridge_status = "no"
            if p_topo > 0 and not is_flat:
                if p_topo in primes:
                    bridge_status = "YES"
            elif p_topo == 0:
                bridge_status = "N/A (Gamma)"
            
            if is_flat:
                log_print(f"    Class {class_idx} [FLAT BAND]: deg={deg} mult={mult} f(x)={base_str}")
            else:
                log_print(f"    Class {class_idx}: deg={deg} mult={mult}")
                log_print(f"      f(x)   = {base_str}")
                log_print(f"      f(-3)  = {f_val}")
                log_print(f"      Norm   = {norm_val} = {factored_str(norm_val)} (primes: {primes})")
                log_print(f"      bridge = {bridge_status}")
                
            class_idx += 1
            
    log_file.close()
    print(f"\nGaussian exact results written to {out_file_path}")

if __name__ == "__main__":
    main()
