# verify_pending_5_ghz_fano.py
# Verification of Item 5: GHZ ↔ Fano cocycle
# We show that Mermin's XOR contradiction is isomorphic to the non-triviality of a cohomology class on the Fano plane.

import numpy as np

def main():
    print("=" * 70)
    print("VERIFYING PENDING 5: GHZ <-> FANO COCYCLE")
    print("=" * 70)

    # 1. 3-qubit GHZ state expectation values
    # |GHZ> = 1/sqrt(2) (|000> + |111>)
    # Define operators:
    # M1 = X1 X2 X3
    # M2 = X1 Y2 Y3
    # M3 = Y1 X2 Y3
    # M4 = Y1 Y2 X3
    
    # Define Pauli matrices
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    # Construct GHZ state vector
    GHZ = np.zeros(8, dtype=complex)
    GHZ[0] = 1.0 / np.sqrt(2)  # |000>
    GHZ[7] = 1.0 / np.sqrt(2)  # |111>
    
    # Construct Mermin operators as tensor products
    M1 = np.kron(X, np.kron(X, X))
    M2 = np.kron(X, np.kron(Y, Y))
    M3 = np.kron(Y, np.kron(X, Y))
    M4 = np.kron(Y, np.kron(Y, X))
    
    # Compute expectation values: <GHZ| M_i |GHZ>
    exp_vals = [
        np.real(np.vdot(GHZ, M1 @ GHZ)),
        np.real(np.vdot(GHZ, M2 @ GHZ)),
        np.real(np.vdot(GHZ, M3 @ GHZ)),
        np.real(np.vdot(GHZ, M4 @ GHZ))
    ]
    
    # Convert to F2 log-representation: +1 -> 0, -1 -> 1
    h_QM = np.array([0 if val > 0 else 1 for val in exp_vals])
    
    print("Quantum expectation values (computed from GHZ state):")
    print(f"  <M1> = {exp_vals[0]:+.1f} => h(X1 X2 X3) = {h_QM[0]}")
    print(f"  <M2> = {exp_vals[1]:+.1f} => h(X1 Y2 Y3) = {h_QM[1]}")
    print(f"  <M3> = {exp_vals[2]:+.1f} => h(Y1 X2 Y3) = {h_QM[2]}")
    print(f"  <M4> = {exp_vals[3]:+.1f} => h(Y1 Y2 X3) = {h_QM[3]}")
    print(f"  Sum of h_QM = {np.sum(h_QM) % 2} (mod 2)")

    # 2. Fano plane points and lines
    # Points in F2^3 (non-zero vectors):
    # p1 = (1, 0, 0)
    # p2 = (0, 1, 0)
    # p5 = (0, 0, 1)
    # p3 = p1 + p2 = (1, 1, 0)
    # p6 = p1 + p5 = (1, 0, 1)
    # p4 = p1 + p2 + p5 = (1, 1, 1)
    # p7 = p2 + p5 = (0, 1, 1)
    
    # Let's map points to measurement variables:
    # p1 -> X1
    # p2 -> X2
    # p3 -> X3
    # p4 -> Y1
    # p5 -> Y2
    # p6 -> Y3
    # p7 -> C (7th auxiliary point)
    
    # Fano lines are 3-subsets that sum to zero in F2^3:
    # L1 = {p1, p2, p3} = {X1, X2, X3} (Mermin M1)
    # L2 = {p1, p5, p6} = {X1, Y2, Y3} (Mermin M2)
    # L3 = {p4, p2, p6} = {Y1, X2, Y3} (Mermin M3)
    # L4 = {p4, p5, p3} = {Y1, Y2, X3} (Mermin M4)
    # L5 = {p2, p5, p7}
    # L6 = {p1, p4, p7}
    # L7 = {p3, p6, p7}
    
    # Let's verify that the 4 Mermin lines form a closed 2-cycle (boundary of a 3-cell)
    # by showing that every point is contained in an even number of these 4 lines:
    point_counts = {i: 0 for i in range(1, 8)}
    mermin_lines = [[1, 2, 3], [1, 5, 6], [4, 2, 6], [4, 5, 3]]
    for line in mermin_lines:
        for pt in line:
            point_counts[pt] += 1
            
    print("\nPoint counts in the 4 Mermin lines:")
    for pt, count in point_counts.items():
        print(f"  p{pt}: {count} times")
        # Every point must be contained an even number of times (2 or 0)
        assert count % 2 == 0, f"Point p{pt} has odd count {count}!"

    print("All points are contained an even number of times in the 4 Mermin lines.")
    print("This confirms that the sum of any local realistic assignment g(p) over these lines must be 0 mod 2:")
    print("  sum_{i=1}^4 (g(a_i) + g(b_i) + g(c_i)) = 2 * sum_p g(p) = 0 mod 2")
    
    # The quantum mechanical cochain sum is 1 mod 2.
    # Therefore, the cohomology class of h_QM on this 2-cycle is non-trivial (obstruction = 1).
    obstruction = np.sum(h_QM) % 2
    print(f"Cohomology class obstruction: {obstruction} (should be 1)")
    
    if obstruction == 1:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
