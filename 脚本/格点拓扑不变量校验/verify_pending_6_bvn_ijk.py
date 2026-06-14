# verify_pending_6_bvn_ijk.py
# Verification of Item 6: Birkhoff-von Neumann non-distributive lattice on H units i,j,k.
# We construct spin projection operators P_i, P_j, P_k and verify the failure of the distributive law.

import numpy as np

def proj_meet(P1, P2):
    # Meet (intersection): eigenspace of (P1 + P2)/2 with eigenvalue 1
    M = 0.5 * (P1 + P2)
    evals, evecs = np.linalg.eigh(M)
    # eigenvectors corresponding to eigenvalues very close to 1
    intersection_vecs = evecs[:, evals > 1.0 - 1e-9]
    if intersection_vecs.shape[1] == 0:
        return np.zeros((2, 2), dtype=complex)
    else:
        return intersection_vecs @ intersection_vecs.conj().T

def proj_join(P1, P2):
    # Join (sum): projection onto range of P1 + P2
    M = P1 + P2
    evals, evecs = np.linalg.eigh(M)
    # range is spanned by eigenvectors with eigenvalue > 1e-9
    range_vecs = evecs[:, evals > 1e-9]
    if range_vecs.shape[1] == 0:
        return np.zeros((2, 2), dtype=complex)
    else:
        return range_vecs @ range_vecs.conj().T

def main():
    print("=" * 70)
    print("VERIFYING PENDING 6: BVN NON-DISTRIBUTIVE LATTICE @ i,j,k")
    print("=" * 70)

    # Pauli matrices
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # 1. Construct projection operators P_i, P_j, P_k
    # Corresponding to imaginary units i, j, k as (I + sigma)/2
    P_i = 0.5 * (I + sigma_x)
    P_j = 0.5 * (I + sigma_y)
    P_k = 0.5 * (I + sigma_z)

    print("Projection operators:")
    print("  P_i (x-axis):\n", P_i)
    print("  P_j (y-axis):\n", P_j)
    print("  P_k (z-axis):\n", P_k)

    # 2. Evaluate Left-Hand Side: P_i /\ (P_j \/ P_k)
    P_j_join_k = proj_join(P_j, P_k)
    LHS = proj_meet(P_i, P_j_join_k)

    # 3. Evaluate Right-Hand Side: (P_i /\ P_j) \/ (P_i /\ P_k)
    P_i_meet_j = proj_meet(P_i, P_j)
    P_i_meet_k = proj_meet(P_i, P_k)
    RHS = proj_join(P_i_meet_j, P_i_meet_k)

    print("\nCalculated Subspace Operations:")
    print("  P_j \\/ P_k (Join of j and k):\n", P_j_join_k)
    print("  LHS = P_i /\\ (P_j \\/ P_k):\n", LHS)
    print("  P_i /\\ P_j:\n", P_i_meet_j)
    print("  P_i /\\ P_k:\n", P_i_meet_k)
    print("  RHS = (P_i /\\ P_j) \\/ (P_i /\\ P_k):\n", RHS)

    # Verify LHS != RHS
    lhs_rank = np.round(np.trace(LHS)).real
    rhs_rank = np.round(np.trace(RHS)).real
    print(f"\nSubspace Dimensions:")
    print(f"  LHS Rank: {lhs_rank}")
    print(f"  RHS Rank: {rhs_rank}")

    distributive_fails = not np.allclose(LHS, RHS)
    print(f"  Distributive law failed? {distributive_fails}")

    # LHS should be equal to P_i (rank 1), RHS should be 0 (rank 0)
    bvn_ok = np.allclose(LHS, P_i) and np.allclose(RHS, np.zeros((2,2)))
    print(f"  LHS == P_i and RHS == 0? {bvn_ok}")

    if distributive_fails and bvn_ok:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
