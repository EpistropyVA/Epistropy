# -*- coding: utf-8 -*-
"""
verify_s0_logic.py
==================
Verification of §5: S0 self-product (S0 x S0) generates all 16 binary logical connectives.

Mathematical basis:
  - S0 = {0, 1} (represented in GF(2))
  - S0 x S0 has 4 elements: (0,0), (0,1), (1,0), (1,1)
  - There are 2^(2^2) = 16 unique mappings f: S0 x S0 -> S0
  - Every boolean function has a unique representation as a polynomial over GF(2) 
    (known as Algebraic Normal Form / Zhegalkin Polynomial).

This script:
  1. Enumerates all 16 possible truth tables.
  2. Map each truth table to its standard Logic Gate name, logic notation,
     and unique GF(2) polynomial representation.
  3. Validates that the GF(2) representation matches the truth table exactly.
"""

import sys
import io

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# The 4 input pairs in S0 x S0
INPUTS = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# Standard logical metadata for each of the 16 binary functions
# Keyed by the 4-bit output vector (f(0,0), f(0,1), f(1,0), f(1,1))
LOGIC_METADATA = {
    (0, 0, 0, 0): {
        "name": "Contradiction (Constant False)",
        "symbol": "0",
        "poly": lambda x, y: 0,
        "poly_str": "0"
    },
    (0, 0, 0, 1): {
        "name": "AND (Conjunction)",
        "symbol": "x ∧ y",
        "poly": lambda x, y: x * y,
        "poly_str": "x*y"
    },
    (0, 0, 1, 0): {
        "name": "Inhibition (Material Non-implication)",
        "symbol": "x ↛ y",
        "poly": lambda x, y: (x + x * y) % 2,
        "poly_str": "x + x*y"
    },
    (0, 0, 1, 1): {
        "name": "Projection x (Identity A)",
        "symbol": "x",
        "poly": lambda x, y: x,
        "poly_str": "x"
    },
    (0, 1, 0, 0): {
        "name": "Converse Non-implication",
        "symbol": "x ↜ y",
        "poly": lambda x, y: (y + x * y) % 2,
        "poly_str": "y + x*y"
    },
    (0, 1, 0, 1): {
        "name": "Projection y (Identity B)",
        "symbol": "y",
        "poly": lambda x, y: y,
        "poly_str": "y"
    },
    (0, 1, 1, 0): {
        "name": "XOR (Exclusive OR / Addition)",
        "symbol": "x ⊕ y",
        "poly": lambda x, y: (x + y) % 2,
        "poly_str": "x + y"
    },
    (0, 1, 1, 1): {
        "name": "OR (Disjunction)",
        "symbol": "x ∨ y",
        "poly": lambda x, y: (x + y + x * y) % 2,
        "poly_str": "x + y + x*y"
    },
    (1, 0, 0, 0): {
        "name": "NOR (Joint Denial)",
        "symbol": "x ↓ y",
        "poly": lambda x, y: (1 + x + y + x * y) % 2,
        "poly_str": "1 + x + y + x*y"
    },
    (1, 0, 0, 1): {
        "name": "XNOR (Equivalence / Biconditional)",
        "symbol": "x ↔ y",
        "poly": lambda x, y: (1 + x + y) % 2,
        "poly_str": "1 + x + y"
    },
    (1, 0, 1, 0): {
        "name": "Negation of y",
        "symbol": "¬y",
        "poly": lambda x, y: (1 + y) % 2,
        "poly_str": "1 + y"
    },
    (1, 0, 1, 1): {
        "name": "Converse Implication",
        "symbol": "x ↫ y",
        "poly": lambda x, y: (1 + y + x * y) % 2,
        "poly_str": "1 + y + x*y"
    },
    (1, 1, 0, 0): {
        "name": "Negation of x",
        "symbol": "¬x",
        "poly": lambda x, y: (1 + x) % 2,
        "poly_str": "1 + x"
    },
    (1, 1, 0, 1): {
        "name": "Implication (Conditional)",
        "symbol": "x → y",
        "poly": lambda x, y: (1 + x + x * y) % 2,
        "poly_str": "1 + x + x*y"
    },
    (1, 1, 1, 0): {
        "name": "NAND (Alternative Denial)",
        "symbol": "x ↑ y",
        "poly": lambda x, y: (1 + x * y) % 2,
        "poly_str": "1 + x*y"
    },
    (1, 1, 1, 1): {
        "name": "Tautology (Constant True)",
        "symbol": "1",
        "poly": lambda x, y: 1,
        "poly_str": "1"
    }
}

RESULTS = []

def record(label, expected, computed, passed=None):
    if passed is None:
        passed = (expected == computed)
    status = "PASS" if passed else "FAIL"
    RESULTS.append((label, expected, computed, status))
    mark = "[PASS]" if passed else "[FAIL]"
    if passed:
        print(f"  {mark} {label}")
    else:
        print(f"  {mark} {label}: expected {expected}, got {computed}")
    return passed

def main():
    print("=" * 80)
    print("  verify_s0_logic.py")
    print("  Verification of §5: S0 x S0 self-product mapping to 16 logic connectives")
    print("=" * 80)
    print()

    print("  Inputs (S0 x S0):")
    for i, inp in enumerate(INPUTS):
        print(f"    Input {i}: x={inp[0]}, y={inp[1]}")
    print()

    # Enumerate all 16 outputs and verify correctness of polynomial evaluation
    print(f"  {'Index':>5} | {'Vector':<12} | {'Symbolic':<8} | {'GF(2) Polynomial':<18} | {'Name':<35} | Verification")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*8}-+-{'-'*18}-+-{'-'*35}-+-------------")

    verified_count = 0
    # Enumerate binary patterns from 0 to 15 (which correspond to vectors)
    for idx in range(16):
        # build the 4-bit output vector from the index
        # Let idx = b3 b2 b1 b0 in binary, mapping to inputs:
        # (0,0) -> b0
        # (0,1) -> b1
        # (1,0) -> b2
        # (1,1) -> b3
        vector = (
            (idx >> 0) & 1,
            (idx >> 1) & 1,
            (idx >> 2) & 1,
            (idx >> 3) & 1
        )
        
        meta = LOGIC_METADATA[vector]
        poly_fn = meta["poly"]
        
        # Test poly_fn on all inputs
        poly_outputs = tuple(poly_fn(x, y) for x, y in INPUTS)
        match = (poly_outputs == vector)
        
        verify_str = "OK" if match else "FAIL"
        if match:
            verified_count += 1
            
        print(f"  {idx:>5} | {str(vector):<12} | {meta['symbol']:<8} | {meta['poly_str']:<18} | {meta['name']:<35} | {verify_str}")

    print()
    record("All 16 connectives successfully generated and verified in GF(2)", 16, verified_count)
    
    # Check index mapping math: 2^(2^n) where n=2 (number of variables)
    n_vars = 2
    expected_connectives = 2**(2**n_vars)
    record(f"Total binary connectives count formula: 2^(2^{n_vars}) = {expected_connectives}", 16, expected_connectives)

    # Print summary
    print()
    print("-" * 72)
    print("  SUMMARY")
    print("-" * 72)
    
    passed = sum(1 for (_, _, _, s) in RESULTS if s == "PASS")
    failed = sum(1 for (_, _, _, s) in RESULTS if s == "FAIL")
    total = len(RESULTS)

    print(f"  {'=' * 60}")
    print(f"  TOTAL: {total}   PASS: {passed}   FAIL: {failed}")
    print(f"  {'=' * 60}")
    if failed == 0:
        print("  ALL CLAIMS VERIFIED. S0 x S0 EXACTLY SPANS 16 BINARY CONNECTIVES.")
    else:
        print(f"  {failed} CLAIM(S) FAILED.")

if __name__ == '__main__':
    main()
