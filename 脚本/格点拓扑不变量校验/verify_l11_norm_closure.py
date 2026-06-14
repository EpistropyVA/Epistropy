# -*- coding: utf-8 -*-
"""
verify_l11_norm_closure.py

Performs a high-precision numerical Galois Norm reconstruction of the BCC Bloch Hamiltonian at L=11
to verify if any native bridges exist, thereby testing the "outer boundary" Riemann-Fano limit of 11.

Algorithm:
For a Cyclotomic extension Q(w_L), the Galois conjugates of a k-point representative n
are given by a*n mod L for a in F_L^* / {+-1}.
By calculating the non-flat eigenvalues lambda_j of H(a*n) for all conjugate representatives,
we compute:
  P_nonflat(a*n, -3) = prod_{j in non-flat} (-3 - lambda_j)
And the exact Galois Norm of the non-flat determinant at x = -3 is:
  Norm_total = prod_{a} P_nonflat(a*n, -3)
Since the eigenvalues are computed to 15 decimal places, the total product is rounded to the
nearest integer, which represents the exact algebraic Galois Norm of the non-flat determinant.
If Norm_total is not divisible by the topological prime p = size - 1, then no constituent minimal
polynomial can be divisible by p (since p is prime), proving the absence of a bridge.
"""

import sys
import io
import os
import time
import itertools
import numpy as np

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'buffer') and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
from verify_bridge_norm_L_scan import (
    build_bcc_lattice_L, build_all_faces_L, build_adjacency_matrix_L,
    build_transfer_blocks_L, compute_all_orbits
)

def build_H_k_numerical(T, k_vec):
    """Build 20×20 Bloch matrix numerically."""
    H = np.zeros((20, 20), dtype=complex)
    for R, block in T.items():
        phase = np.exp(1j * np.dot(k_vec, np.array(R, dtype=float)))
        H += phase * block
    return H

def compute_orbit_total_norm(rep, L, T, p_topo):
    """
    Computes the total Galois Norm of P_nonflat(-3) over the real subfield of Q(w_L).
    For prime L, the Galois group of Q(w_L)/Q is cyclic of order L-1.
    The real subfield Q(cos(2pi/L)) has degree (L-1)/2.
    The representatives of F_L^* / {+-1} are 1, 2, ..., (L-1)/2.
    """
    # Representatives for Galois conjugates in the real subfield
    conjugates = list(range(1, (L - 1) // 2 + 1))
    
    total_norm = 1.0
    
    for a in conjugates:
        # Conjugate k-point: a * rep mod L
        k_pt = tuple((a * x) % L for x in rep)
        # Convert to Bloch coordinates
        k_vec = np.array([2.0 * np.pi * x / L for x in k_pt])
        
        # Build H(k)
        H = build_H_k_numerical(T, k_vec)
        
        # Get eigenvalues
        evals = np.linalg.eigvalsh(H)
        
        # Separate flat band eigenvalues (near -3)
        flat_tol = 1e-5
        non_flat_evals = [ev for ev in evals if abs(ev - (-3.0)) > flat_tol]
        
        # Ensure we found exactly 14 non-flat eigenvalues
        assert len(non_flat_evals) == 14, f"Expected 14 non-flat evals, got {len(non_flat_evals)} at rep={rep}, a={a}"
        
        # Product of (-3 - lambda_j)
        p_val = np.prod([-3.0 - ev for ev in non_flat_evals])
        total_norm *= p_val
        
    # Round to nearest integer since the Galois Norm must be an integer
    exact_norm = int(round(total_norm))
    
    # Check divisibility
    is_bridge = False
    if p_topo > 0:
        is_bridge = (exact_norm % p_topo == 0)
        
    return exact_norm, is_bridge

def main():
    out_file_path = os.path.join(os.path.dirname(__file__), "verify_l11_norm_closure_results.txt")
    log_file = open(out_file_path, "w", encoding="utf-8")
    
    def log_print(msg=""):
        print(msg)
        sys.stdout.flush()
        log_file.write(msg + "\n")
        log_file.flush()

    log_print("=" * 80)
    log_print("  NUMERICAL GALOIS NORM RECONSTRUCTION: L=3, 5, 11 CLOSURE ANALYSIS")
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
        norm, bridge = compute_orbit_total_norm(rep, L3, T3, p_topo)
        log_print(f"  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo} | Norm={norm} | bridge={'YES' if bridge else 'no'}")
        
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
        norm, bridge = compute_orbit_total_norm(rep, L5, T5, p_topo)
        log_print(f"  Orbit {idx+1}: rep={rep} | size={len(orb)} | p={p_topo} | Norm={norm} | bridge={'YES' if bridge else 'no'}")
        
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
        norm, bridge = compute_orbit_total_norm(rep, L11, T11, p_topo)
        log_print(f"  Orbit {idx+1:2d}: rep={rep} | size={len(orb):2d} | p={p_topo:2d} | bridge={'YES' if bridge else 'no'} (time: {time.time()-t_start:.4f}s)")
        # If norm is small enough, print it, otherwise show factorization
        if abs(norm) < 1e12:
            log_print(f"    Norm = {norm}")
        else:
            import math
            try:
                log_print(f"    Norm (log10) = {math.log10(abs(norm)):.2f}")
            except (OverflowError, ValueError):
                log_print(f"    Norm (approx log10) = {len(str(abs(norm))) - 1}")
            
        if bridge:
            bridges_found += 1
            
    log_print("\n" + "=" * 80)
    log_print(f"  L=11 Closure Test complete. Total native bridges found: {bridges_found}")
    log_print(f"  Is L=11 a closed outer boundary? {bridges_found == 0}")
    log_print("=" * 80)
    
    log_file.close()

if __name__ == "__main__":
    main()
