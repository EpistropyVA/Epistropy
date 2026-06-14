# -*- coding: utf-8 -*-
"""
verify_exact_fibonacci_bridge.py

Performs exact algebraic charpoly factorization of the BCC Bloch Hamiltonian
at all orbits for L=2, 3, 4, 5. Computes the exact minimal polynomials,
evaluates f(-3), takes the Galois field norm over Q, and verifies the 
bridge prime division and Fibonacci compiler quadratic ansatz behavior.
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

# Import L-scan logic
sys.path.insert(0, os.path.dirname(__file__))
from verify_bridge_norm_L_scan import (
    build_bcc_lattice_L, build_all_faces_L, build_adjacency_matrix_L,
    build_transfer_blocks_L, compute_all_orbits
)

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

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

def clean_poly_print(poly, x):
    # Print poly in standard form
    terms = sp.expand(poly)
    return str(terms)

def get_norm_and_primes(val, u):
    val = sp.expand(val)
    if val.is_rational:
        val_int = abs(int(val))
        return val_int, prime_factors(val_int)
    
    # It contains sqrt(5)
    val_u = val.subs(sp.sqrt(5), u)
    a = val_u.subs(u, 0)
    b = val_u.coeff(u)
    
    norm = a**2 - 5 * b**2
    try:
        norm_int = abs(int(norm))
    except TypeError as e:
        print(f"ERROR: norm={norm}, type(norm)={type(norm)}, free_symbols={norm.free_symbols}")
        raise e
    return norm_int, prime_factors(norm_int)

def main():
    out_file_path = os.path.join(os.path.dirname(__file__), "verify_exact_fibonacci_bridge_results.txt")
    
    # We will write to a file and also print to stdout
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    log_print("=" * 80)
    log_print("  EXACT ALGEBRAIC BRIDGE AND FIBONACCI COMPILER ANALYSIS: L=2, 3, 4, 5")
    log_print("=" * 80)
    
    x = sp.Symbol('x', real=True)
    w = sp.Symbol('w')
    u = sp.Symbol('u') # used for extracting sqrt(5) coefficients
    
    for L in [2, 3, 4, 5]:
        log_print(f"\n==============================================================")
        log_print(f"  L = {L} Exact Analysis")
        log_print(f"==============================================================")
        
        # Build lattice and transfer blocks
        body_centers = build_bcc_lattice_L(L)
        face_to_idx, bc_face_indices, _ = build_all_faces_L(body_centers, L)
        A_full = build_adjacency_matrix_L(face_to_idx)
        T = build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L)
        
        # Orbits
        orbits = compute_all_orbits(L)
        log_print(f"  k-points: {L**3} | O_h orbits: {len(orbits)}")
        
        # Set up field extensions
        if L == 3:
            field_poly = w**2 + w + 1
        elif L == 5:
            field_poly = w**4 + w**3 + w**2 + w + 1
            
        for idx, (rep, orb) in enumerate(orbits):
            # Check if Gamma point
            is_gamma = all(n == 0 for n in rep)
            
            # Construct H(k) using exact arithmetic in SymPy
            H_sympy = sp.zeros(20, 20)
            for R, block in T.items():
                dot_product = rep[0]*R[0] + rep[1]*R[1] + rep[2]*R[2]
                
                if L == 2:
                    phase = (-1) ** (dot_product % 2)
                elif L == 3:
                    phase = w ** (dot_product % 3)
                    phase = sp.rem(sp.expand(phase), field_poly, w)
                elif L == 4:
                    phase = sp.I ** (dot_product % 4)
                elif L == 5:
                    phase = w ** (dot_product % 5)
                    phase = sp.rem(sp.expand(phase), field_poly, w)
                
                for r in range(20):
                    for c in range(20):
                        val = block[r, c]
                        if val != 0:
                            val_int = int(round(val))
                            H_sympy[r, c] += val_int * phase
            
            # Simplify H entries
            if L in [3, 5]:
                for r in range(20):
                    for c in range(20):
                        H_sympy[r, c] = sp.rem(H_sympy[r, c], field_poly, w)
            
            # Compute characteristic polynomial
            t0 = time.time()
            P = H_sympy.charpoly(x).as_expr()
            
            # Substitute internal complex x with our real x symbol
            x_syms = [s for s in P.free_symbols if s.name == 'x']
            if x_syms:
                P = P.subs(x_syms[0], x)
            
            if L == 3:
                P = sp.rem(P, field_poly, w)
                P_rational = sp.expand(P)
                # P_rational should be rational (no w)
                P_field = P_rational
            elif L == 4:
                P_field = sp.expand(sp.re(P))
            elif L == 5:
                P = sp.rem(P, field_poly, w)
                # Convert P to Q(sqrt(5))
                P_collected = sp.collect(sp.expand(P), x)
                P_field = 0
                for deg_x in range(21):
                    coeff_x = P_collected.coeff(x, deg_x)
                    if coeff_x != 0:
                        c0 = coeff_x.subs(w, 0)
                        c2 = coeff_x.coeff(w, 2)
                        coeff_u = (c0 - c2/2) - (c2/2)*u
                        P_field += coeff_u * x**deg_x
                P_field = P_field.subs(u, sp.sqrt(5))
            else:
                P_field = P
                
            charpoly_time = time.time() - t0
            
            # Factor polynomial over the appropriate extension
            t0 = time.time()
            if L == 5:
                factored = sp.factor(P_field, extension=sp.sqrt(5))
            else:
                factored = sp.factor(P_field)
            factor_time = time.time() - t0
            
            p_topo = len(orb) - 1
            log_print(f"\n  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo}")
            log_print(f"    Charpoly in {charpoly_time:.2f}s, factored in {factor_time:.2f}s")
            
            # Parse factored terms
            # factored can be a single Mul or a single Add or Pow
            args = sp.Mul.make_args(factored)
            class_idx = 1
            
            # Find the actual x symbol in factored
            x_syms = [s for s in factored.free_symbols if s.name == 'x']
            x_var = x_syms[0] if x_syms else x
            
            for arg in args:
                # Check if it is a power term: base**exponent
                if isinstance(arg, sp.Pow):
                    base, mult = arg.as_base_exp()
                else:
                    base = arg
                    mult = 1
                
                # Check if it is a constant factor
                if base.is_number:
                    continue
                    
                # base is a polynomial factor
                # check if it is the flat band (x + 3)
                is_flat = False
                if base == x_var + 3:
                    is_flat = True
                
                base_str = clean_poly_print(base, x_var)
                deg = sp.degree(base, x_var)
                
                # Evaluate at x = -3
                f_val = base.subs(x_var, -3)
                norm_val, primes = get_norm_and_primes(f_val, u)
                
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
                    # Check if it is a Fibonacci Quadratic: x^2 + x + c
                    is_fibo_quad = False
                    coeffs = sp.Poly(base, x_var).all_coeffs()
                    if deg == 2 and coeffs[0] == 1 and coeffs[1] == 1:
                        is_fibo_quad = True
                        c = coeffs[2]
                        log_print(f"    Class {class_idx}: deg={deg} mult={mult} [FIBONACCI QUADRATIC x^2+x+{c}]")
                    else:
                        log_print(f"    Class {class_idx}: deg={deg} mult={mult}")
                        
                    log_print(f"      f(x)   = {base_str}")
                    log_print(f"      f(-3)  = {f_val}")
                    if L == 5 and not f_val.is_rational:
                        log_print(f"      Norm   = {norm_val} = {factored_str(norm_val)} (primes: {primes})")
                    else:
                        log_print(f"      f(-3)  = {f_val} = {factored_str(f_val)} (primes: {primes})")
                    log_print(f"      bridge = {bridge_status}")
                    
                    # Fibonacci relation check
                    if norm_val > 0:
                        found_fib = []
                        for fn_idx in range(1, 25):
                            fn = fibonacci(fn_idx)
                            if fn == norm_val:
                                found_fib.append(f"F({fn_idx})")
                            elif norm_val % fn == 0 and fn > 1:
                                found_fib.append(f"divisible by F({fn_idx})={fn}")
                        if found_fib:
                            log_print(f"      Fibonacci relation: {', '.join(found_fib)}")
                
                class_idx += 1
                
    log_file.close()
    print(f"\nExact algebraic results written to {out_file_path}")

if __name__ == "__main__":
    main()
