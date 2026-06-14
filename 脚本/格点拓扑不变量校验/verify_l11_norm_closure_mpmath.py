# -*- coding: utf-8 -*-
"""
verify_l11_norm_closure_mpmath.py

Performs a high-precision arbitrary-precision (100 decimal digits) numerical 
Galois Norm reconstruction of the BCC Bloch Hamiltonian at L=3, L=5, and L=11
to verify if any native bridges exist, thereby testing the "outer boundary" 
Riemann-Fano limit of 11.

Uses mpmath to bypass the double-precision float limitation of 15 decimal digits.
"""

import sys
import io
import os
import time
import mpmath
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
from verify_bridge_norm_L_scan import (
    build_bcc_lattice_L, build_all_faces_L, build_adjacency_matrix_L,
    build_transfer_blocks_L, compute_all_orbits
)

# Set high precision for mpmath
mpmath.mp.dps = 100

def build_H_k_mpmath(T, k_pt, L):
    """Build 20×20 Bloch matrix exactly using mpmath."""
    H = mpmath.matrix(20, 20)
    for R, block in T.items():
        dot_product = int(k_pt[0]*R[0] + k_pt[1]*R[1] + k_pt[2]*R[2])
        # phase = exp(2 * pi * j * dot_product / L)
        angle = mpmath.mpf(2) * mpmath.pi() * dot_product / L
        phase = mpmath.exp(mpmath.j * angle)
        
        # Add to H using rounded integer block elements
        for r in range(20):
            for c in range(20):
                val = block[r, c]
                if val != 0:
                    H[r, c] += phase * int(round(val))
    return H

def compute_orbit_total_norm_mpmath(rep, L, T, p_topo):
    """
    Computes the total Galois Norm of P_nonflat(-3) using mpmath.
    For prime L, the Galois conjugates of a k-point representative
    are given by a*rep mod L for a in 1, ..., (L-1)/2.
    """
    conjugates = list(range(1, (L - 1) // 2 + 1))
    
    total_norm = mpmath.mpf(1)
    
    for a in conjugates:
        k_pt = tuple((a * x) % L for x in rep)
        
        # Build H(k) using mpmath
        H = build_H_k_mpmath(T, k_pt, L)
        
        # Get eigenvalues (using eighe for Hermitian matrix)
        evals = mpmath.eighe(H, eigvals_only=True)
        evals_list = list(evals)
        
        # Separate flat band eigenvalues (near -3)
        flat_tol = mpmath.mpf('1e-15')
        non_flat_evals = [ev for ev in evals_list if mpmath.absmax(ev - mpmath.mpf(-3)) > flat_tol]
        
        assert len(non_flat_evals) == 14, f"Expected 14 non-flat evals, got {len(non_flat_evals)} at rep={rep}, a={a}"
        
        # Product of (-3 - lambda)
        p_val = mpmath.mpf(1)
        for ev in non_flat_evals:
            p_val *= (mpmath.mpf(-3) - ev)
            
        total_norm *= p_val
        
    # Round to nearest integer
    exact_norm = int(mpmath.nint(total_norm))
    
    # Check divisibility
    is_bridge = False
    if p_topo > 0:
        is_bridge = (exact_norm % p_topo == 0)
        
    return exact_norm, is_bridge

def factored_str(n, limit=1000000):
    n = int(n)
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n == 1:
        return sign + "1"
    facs = {}
    d = 2
    while d * d <= n and d < limit:
        while n % d == 0:
            facs[d] = facs.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        if d >= limit:
            facs[f"[unfactored>{limit}]"] = n
        else:
            facs[n] = facs.get(n, 0) + 1
    parts = []
    keys_sorted = sorted(facs.keys(), key=lambda x: (1, x) if isinstance(x, str) else (0, x))
    for p in keys_sorted:
        if isinstance(p, int):
            parts.append(str(p) if facs[p] == 1 else f"{p}^{facs[p]}")
        else:
            parts.append(f"{p}")
    return sign + "·".join(parts)


def main():
    out_file_path = os.path.join(os.path.dirname(__file__), "verify_l11_norm_closure_mpmath_results.txt")
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        sys.stdout.flush()
        log_file.write(msg + "\n")
        log_file.flush()

    log_print("=" * 80)
    log_print("  MPMATH HIGH-PRECISION GALOIS NORM RECONSTRUCTION: L=3, 5, 11 ANALYSIS")
    log_print("=" * 80)
    
    # ── Benchmark with L=3 ──
    log_print("\n[BENCHMARK] Verifying L=3 (Fano Orbit Bridge)...")
    L3 = 3
    bcs3 = build_bcc_lattice_L(L3)
    f2i3, bcf3, _ = build_all_faces_L(bcs3, L3)
    A3 = build_adjacency_matrix_L(f2i3)
    T3 = build_transfer_blocks_L(bcs3, bcf3, A3, L3)
    orbits3 = compute_all_orbits(L3)
    
    for idx, (rep, orb) in enumerate(orbits3):
        is_gamma = all(n == 0 for n in rep)
        p_topo = len(orb) - 1
        if is_gamma:
            continue
        norm, bridge = compute_orbit_total_norm_mpmath(rep, L3, T3, p_topo)
        log_print(f"  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo} | Norm={factored_str(norm)} | bridge={'YES' if bridge else 'no'}")
        
    # ── Benchmark with L=5 ──
    log_print("\n[BENCHMARK] Verifying L=5 (phi-residual Bridge)...")
    L5 = 5
    bcs5 = build_bcc_lattice_L(L5)
    f2i5, bcf5, _ = build_all_faces_L(bcs5, L5)
    A5 = build_adjacency_matrix_L(f2i5)
    T5 = build_transfer_blocks_L(bcs5, bcf5, A5, L5)
    orbits5 = compute_all_orbits(L5)
    
    for idx, (rep, orb) in enumerate(orbits5):
        is_gamma = all(n == 0 for n in rep)
        p_topo = len(orb) - 1
        if is_gamma:
            continue
        norm, bridge = compute_orbit_total_norm_mpmath(rep, L5, T5, p_topo)
        log_print(f"  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo} | Norm={factored_str(norm)} | bridge={'YES' if bridge else 'no'}")
        
    # ── Test L=11 ──
    log_print("\n[TEST] Verifying L=11 (Riemann-Fano Boundary test)...")
    L11 = 11
    t0 = time.time()
    bcs11 = build_bcc_lattice_L(L11)
    f2i11, bcf11, _ = build_all_faces_L(bcs11, L11)
    A11 = build_adjacency_matrix_L(f2i11)
    T11 = build_transfer_blocks_L(bcs11, bcf11, A11, L11)
    orbits11 = compute_all_orbits(L11)
    log_print(f"  L=11 Lattice built in {time.time()-t0:.2f}s. Total orbits: {len(orbits11)}")
    
    bridges_found = 0
    
    for idx, (rep, orb) in enumerate(orbits11):
        is_gamma = all(n == 0 for n in rep)
        p_topo = len(orb) - 1
        if is_gamma:
            continue
        
        t_start = time.time()
        norm, bridge = compute_orbit_total_norm_mpmath(rep, L11, T11, p_topo)
        
        # Check if the norm is divisible by any of the base primes or other primes
        # We also want to know if there is a real bridge
        log_print(f"  Orbit {idx+1:2d}: rep={rep} | size={len(orb):2d} | p={p_topo:2d} | bridge={'YES' if bridge else 'no'} (time: {time.time()-t_start:.4f}s)")
        
        # Print norm factored if small, otherwise print first few characters of factored_str
        f_str = factored_str(norm)
        if len(f_str) < 100:
            log_print(f"    Norm = {f_str}")
        else:
            log_print(f"    Norm = {f_str[:90]}... (len: {len(f_str)})")
            
        if bridge:
            bridges_found += 1
            
    log_print("\n" + "=" * 80)
    log_print(f"  L=11 Closure Test complete. Total native bridges found: {bridges_found}")
    log_print(f"  Is L=11 a closed outer boundary? {bridges_found == 0}")
    log_print("=" * 80)
    
    log_file.close()

if __name__ == "__main__":
    main()
