# -*- coding: utf-8 -*-
"""
verify_l7_cubic_bridge.py

Computes the exact characteristic polynomials for L=7 BCC orbits over the cubic field Q(cos(2pi/7)).
Performs algebraic factorization using extension=sp.AlgebraicNumber(theta),
evaluates f(-3), takes the degree-3 Galois norm over Q, and verifies if the topological prime p
divides the Galois norm (bridge fires).
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

def compute_galois_norm_l7(val, theta, theta_poly):
    """
    Computes Galois Norm over Q(theta) for a value val in Q(theta).
    theta satisfies theta^3 + theta^2 - 2*theta - 1 = 0.
    The three conjugates of theta are:
      t1 = theta
      t2 = theta^2 - 2
      t3 = -theta^2 - theta + 1
    """
    val = sp.expand(val)
    if val.is_rational:
        val_int = abs(int(val))
        # Norm in degree 3 of a rational q is q^3
        norm_val = val_int ** 3
        return norm_val, prime_factors(norm_val)
        
    # Substitute conjugates into val
    # val is a polynomial in theta
    val_1 = val
    val_2 = val.subs(theta, theta**2 - 2)
    val_3 = val.subs(theta, -theta**2 - theta + 1)
    
    norm = sp.expand(val_1 * val_2 * val_3)
    norm_simplified = sp.rem(norm, theta_poly, theta)
    
    # norm_simplified must be a rational number
    norm_simplified = sp.expand(norm_simplified)
    assert norm_simplified.is_rational, f"Norm is not rational: {norm_simplified}"
    
    norm_int = abs(int(norm_simplified))
    return norm_int, prime_factors(norm_int)

def clean_poly_print(poly, x):
    return str(sp.expand(poly))

def main():
    L = 7
    out_file_path = os.path.join(os.path.dirname(__file__), "verify_l7_cubic_bridge_results.txt")
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        sys.stdout.flush()
        log_file.write(msg + "\n")
        log_file.flush()

    log_print("=" * 80)
    log_print("  L=7 CUBIC RATIONAL FIELD Q(cos(2pi/7)) EXACT ALGEBRAIC BRIDGE ANALYSIS")
    log_print("=" * 80)
    
    x = sp.Symbol('x', real=True)
    w = sp.Symbol('w')
    t = sp.Symbol('t')
    
    # theta satisfies t^3 + t^2 - 2*t - 1 = 0
    theta_poly = t**3 + t**2 - 2*t - 1
    theta_val = sp.rootof(theta_poly, 1) # get exact AlgebraicNumber in SymPy
    
    # Cyclotomic field poly for w
    field_poly = w**6 + w**5 + w**4 + w**3 + w**2 + w + 1
    
    # Build lattice and transfer blocks
    body_centers = build_bcc_lattice_L(L)
    face_to_idx, bc_face_indices, _ = build_all_faces_L(body_centers, L)
    A_full = build_adjacency_matrix_L(face_to_idx)
    T = build_transfer_blocks_L(body_centers, bc_face_indices, A_full, L)
    
    # Orbits
    orbits = compute_all_orbits(L)
    log_print(f"  k-points: {L**3} | O_h orbits: {len(orbits)}")
    
    # Define target representative points to calculate (Gamma + typical axis/face/body)
    target_reps = {(0,0,0), (0,0,1), (0,0,2), (0,0,3), (0,1,1), (1,1,1)}
    
    for idx, (rep, orb) in enumerate(orbits):
        is_gamma = all(n == 0 for n in rep)
        p_topo = len(orb) - 1
        
        log_print(f"\n  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo}")
        if rep not in target_reps:
            log_print("    [Skipped calculation to save time]")
            continue
            
        t_start = time.time()
        
        # Construct H(k) using exact arithmetic in Q(w)
        H_sympy = sp.zeros(20, 20)
        for R, block in T.items():
            dot_product = rep[0]*R[0] + rep[1]*R[1] + rep[2]*R[2]
            phase = w ** (dot_product % 7)
            phase = sp.rem(sp.expand(phase), field_poly, w)
            
            for r in range(20):
                for c in range(20):
                    val = block[r, c]
                    if val != 0:
                        val_int = int(round(val))
                        H_sympy[r, c] += val_int * phase
        
        # Simplify H entries
        for r in range(20):
            for c in range(20):
                H_sympy[r, c] = sp.rem(H_sympy[r, c], field_poly, w)
        
        # Compute characteristic polynomial P(x, w)
        t0 = time.time()
        P = H_sympy.charpoly(x).as_expr()
        
        # Substitute internal complex x with our real x symbol
        x_syms = [s for s in P.free_symbols if s.name == 'x']
        if x_syms:
            P = P.subs(x_syms[0], x)
            
        P = sp.rem(P, field_poly, w)
        
        # Convert P(x, w) to P(x, theta) in Q(theta)
        # Using the relation w^2 - t*w + 1 = 0
        P_collected = sp.collect(sp.expand(P), x)
        P_field = 0
        for deg_x in range(21):
            coeff_x = P_collected.coeff(x, deg_x)
            if coeff_x != 0:
                # Modulo w^2 - t*w + 1 to eliminate w
                # Since P is real, the remainder must be of degree 0 in w after simplifying t
                coeff_t = sp.rem(coeff_x, w**2 - t*w + 1, w)
                coeff_t = sp.rem(coeff_t, theta_poly, t)
                coeff_t = sp.expand(coeff_t)
                # Ensure no w remains
                assert w not in coeff_t.free_symbols, f"Failed to eliminate w: {coeff_t}"
                P_field += coeff_t * x**deg_x
                
        charpoly_time = time.time() - t0
        
        # Factor polynomial over Q(theta)
        t0 = time.time()
        # In SymPy, we can factor over AlgebraicNumber theta_val
        factored = sp.factor(P_field, extension=theta_val)
        factor_time = time.time() - t0
        
        log_print(f"    Charpoly in {charpoly_time:.2f}s, factored in {factor_time:.2f}s (total: {time.time()-t_start:.2f}s)")
        
        # Parse factored terms
        args = sp.Mul.make_args(factored)
        class_idx = 1
        
        # Find the actual x symbol in factored
        x_syms = [s for s in factored.free_symbols if s.name == 'x']
        x_var = x_syms[0] if x_syms else x
        
        # Extract t symbol in factored (might be named 't' or match theta_val)
        t_syms = [s for s in factored.free_symbols if s.name == 't']
        t_var = t_syms[0] if t_syms else t
        
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
            norm_val, primes = compute_galois_norm_l7(f_val, t_var, theta_poly)
            
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
    print(f"\nCubic Galois results written to {out_file_path}")

if __name__ == "__main__":
    main()
