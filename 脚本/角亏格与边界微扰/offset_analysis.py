"""
Offset Analysis: dkᵀdk = ? for boundary operators at all levels.

Key question: In A + 3I = d₂ᵀd₂, is "3" universal for ∂₂,
or does it change with dimension/level?

Analysis:
1. BCC complex: check d₁ᵀd₁, d₂ᵀd₂, d₃ᵀd₃, d₄ᵀd₄
2. Single simplices of dimension 3,4,5,6: check all boundary operator levels
3. Derive the formula
"""

import sys
import io
import numpy as np
from scipy import sparse
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# =============================================================================
# Generic simplicial complex tools
# =============================================================================

def extract_k_simplices(top_simplices, k):
    """Extract all unique k-simplices from top-level simplices."""
    seen = {}
    result = []
    for s in top_simplices:
        for combo in combinations(s, k + 1):
            key = tuple(sorted(combo))
            if key not in seen:
                seen[key] = len(result)
                result.append(key)
    return result, seen


def build_boundary_operator(lower_simplices, lower_idx, upper_simplices, upper_idx):
    """
    Build boundary operator d_k: maps k-simplices to (k-1)-simplices.
    Matrix shape: (n_lower, n_upper).
    For a k-simplex [v0, v1, ..., vk] (sorted), boundary = sum (-1)^i [v0,...,v̂i,...,vk].
    """
    n_lower = len(lower_simplices)
    n_upper = len(upper_simplices)
    rows, cols, data = [], [], []

    for u_col, simplex in enumerate(upper_simplices):
        k = len(simplex) - 1  # dimension of upper simplex
        for i in range(k + 1):
            face = tuple(sorted(simplex[:i] + simplex[i+1:]))
            sign = (-1) ** i
            l_row = lower_idx[face]
            rows.append(l_row)
            cols.append(u_col)
            data.append(sign)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_lower, n_upper))


def analyze_dtd(dk, k_simplices, k_name, lower_name):
    """Analyze dkᵀdk and extract the offset and adjacency structure."""
    dtd = (dk.T @ dk).toarray().astype(float)
    n = dtd.shape[0]

    # Diagonal values
    diag = np.diag(dtd)
    diag_vals = np.unique(np.round(diag, 10))

    # Off-diagonal
    off_diag = dtd.copy()
    np.fill_diagonal(off_diag, 0)
    off_vals = np.unique(np.round(off_diag, 10))

    print(f"  {k_name}: dk shape={dk.shape}, dkᵀdk shape={dtd.shape}")
    print(f"    Diagonal values: {diag_vals}")
    print(f"    Off-diagonal values: {off_vals}")

    if len(diag_vals) == 1:
        offset = int(diag_vals[0])
        # Check if off-diagonal is a signed adjacency (values in {-1, 0, 1})
        is_signed_adj = all(v in [-1, 0, 1] for v in off_vals)
        print(f"    => OFFSET = {offset}, signed adjacency = {is_signed_adj}")
        return offset
    else:
        print(f"    => Non-uniform diagonal, no clean offset")
        return None


# =============================================================================
# Part 1: Single d-simplex analysis
# =============================================================================

def analyze_single_simplex(d):
    """
    For a single d-simplex (d+1 vertices), compute offsets at all boundary levels.
    """
    print(f"\n{'='*60}")
    print(f"SINGLE {d}-SIMPLEX ({d+1} vertices)")
    print(f"{'='*60}")

    vertices = list(range(d + 1))
    top_simplices = [tuple(vertices)]

    # Extract all k-simplices for k = 0, 1, ..., d
    all_simplices = {}
    all_idx = {}
    for k in range(d + 1):
        simps, idx = extract_k_simplices(top_simplices, k)
        all_simplices[k] = simps
        all_idx[k] = idx
        print(f"  {k}-simplices: {len(simps)}  = C({d+1},{k+1})")

    # Build and analyze boundary operators d_k for k = 1, ..., d
    print()
    offsets = {}
    for k in range(1, d + 1):
        dk = build_boundary_operator(
            all_simplices[k-1], all_idx[k-1],
            all_simplices[k], all_idx[k]
        )
        offset = analyze_dtd(dk, all_simplices[k], f"d_{k}", f"{k-1}-simplices")
        offsets[k] = offset

    return offsets


# =============================================================================
# Part 2: BCC complex analysis (all levels)
# =============================================================================

def build_bcc_open_complex():
    """Build the BCC complex with open boundary conditions (from existing code)."""
    sc_vertex_id = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                sc_vertex_id[(i, j, k)] = len(sc_vertex_id)
    n_sc = len(sc_vertex_id)

    bc_vertex_id = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_vertex_id[(i, j, k)] = n_sc + len(bc_vertex_id)
    n_vertices = n_sc + len(bc_vertex_id)

    tet_A_offsets = [(+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)]
    tet_B_offsets = [(-1, -1, -1), (-1, +1, +1), (+1, -1, +1), (+1, +1, -1)]

    simplices_4 = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bc_id = bc_vertex_id[(i, j, k)]
                cx, cy, cz = i + 0.5, j + 0.5, k + 0.5

                for offsets_list in [tet_A_offsets, tet_B_offsets]:
                    corner_ids = []
                    for dx, dy, dz in offsets_list:
                        sx = int(cx + dx * 0.5)
                        sy = int(cy + dy * 0.5)
                        sz = int(cz + dz * 0.5)
                        corner_ids.append(sc_vertex_id[(sx, sy, sz)])
                    simplices_4.append(tuple(sorted([bc_id] + corner_ids)))

    return n_vertices, simplices_4


def analyze_bcc_complex():
    """Analyze all boundary operator levels of the BCC complex."""
    print(f"\n{'='*60}")
    print(f"BCC 3x3x3 COMPLEX (54 four-simplices, 91 vertices)")
    print(f"{'='*60}")

    n_vertices, simplices_4 = build_bcc_open_complex()

    # Extract all k-simplices
    all_simplices = {}
    all_idx = {}
    for k in range(5):  # 0-simplices through 4-simplices
        simps, idx = extract_k_simplices(simplices_4, k)
        all_simplices[k] = simps
        all_idx[k] = idx
        print(f"  {k}-simplices: {len(simps)}")

    # Build and analyze all boundary operators
    print()
    offsets = {}
    for k in range(1, 5):
        dk = build_boundary_operator(
            all_simplices[k-1], all_idx[k-1],
            all_simplices[k], all_idx[k]
        )
        offset = analyze_dtd(dk, all_simplices[k], f"d_{k}", f"{k-1}-simplices")
        offsets[k] = offset

    return offsets


# =============================================================================
# Part 3: Also check dkdkᵀ (the "down" Laplacian)
# =============================================================================

def analyze_ddt(dk, k_name):
    """Analyze dk dkᵀ (maps lower simplices to lower simplices)."""
    ddt = (dk @ dk.T).toarray().astype(float)
    diag = np.diag(ddt)
    diag_vals = np.unique(np.round(diag, 10))
    off_diag = ddt.copy()
    np.fill_diagonal(off_diag, 0)
    off_vals = np.unique(np.round(off_diag, 10))
    print(f"  {k_name} dkdkᵀ: shape={ddt.shape}")
    print(f"    Diagonal values: {diag_vals}")
    print(f"    Off-diagonal values: {off_vals}")
    if len(diag_vals) == 1:
        offset = int(diag_vals[0])
        print(f"    => OFFSET = {offset}")
        return offset
    else:
        print(f"    => Non-uniform diagonal")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("OFFSET PATTERN ANALYSIS: dkᵀdk = Adj + c·I")
    print("What is c as a function of (simplex dim, boundary level)?")
    print("=" * 60)

    # Part 1: Single simplices of various dimensions
    results = {}
    for d in range(2, 7):  # 2-simplex through 6-simplex
        offsets = analyze_single_simplex(d)
        results[d] = offsets

    # Part 2: BCC complex
    bcc_offsets = analyze_bcc_complex()

    # Part 3: Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Offset c in dkᵀdk = Adj_signed + c·I")
    print(f"{'='*60}")
    print()
    print("Single simplex results:")
    print(f"{'Simplex dim d':>15} | ", end="")
    max_k = max(max(results[d].keys()) for d in results)
    for k in range(1, max_k + 1):
        print(f"  d_{k}  ", end="")
    print()
    print("-" * (15 + 3 + 8 * max_k))
    for d in sorted(results.keys()):
        print(f"{d:>15} | ", end="")
        for k in range(1, max_k + 1):
            if k in results[d]:
                v = results[d][k]
                print(f"  {v if v is not None else '?':>4}  ", end="")
            else:
                print(f"  {'':>4}  ", end="")
        print()

    print()
    print("BCC complex results:")
    for k, v in sorted(bcc_offsets.items()):
        print(f"  d_{k}: offset = {v}")

    # Part 4: Theoretical prediction
    print(f"\n{'='*60}")
    print("THEORETICAL ANALYSIS")
    print(f"{'='*60}")
    print()
    print("For a k-simplex [v0, ..., vk], the boundary has (k+1) faces.")
    print("dkᵀdk[i,i] = number of (k-1)-faces in the boundary of a k-simplex")
    print("           = k+1  (each k-simplex has exactly k+1 boundary faces)")
    print()
    print("Wait -- let's verify: for d_k, each column (k-simplex) has k+1 nonzero entries.")
    print("So (dkᵀdk)[i,i] = sum of squares of entries in column i = k+1 * 1^2 = k+1.")
    print()
    print("Prediction: offset for d_k = k+1")
    print()
    print("Check against results:")
    for d in sorted(results.keys()):
        for k in sorted(results[d].keys()):
            predicted = k + 1
            actual = results[d][k]
            match = "OK" if actual == predicted else "MISMATCH"
            print(f"  {d}-simplex, d_{k}: predicted={predicted}, actual={actual}  [{match}]")
    print()
    print("BCC complex check:")
    for k, v in sorted(bcc_offsets.items()):
        predicted = k + 1
        match = "OK" if v == predicted else "MISMATCH"
        print(f"  BCC d_{k}: predicted={predicted}, actual={v}  [{match}]")

    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print()
    print("The offset in dkᵀdk = A_signed + c*I is:")
    print()
    print("    c = k + 1")
    print()
    print("where k is the dimension of the simplices being related by dk.")
    print()
    print("For d_2 (mapping 2-simplices/faces to 1-simplices/edges):")
    print("  c = 2 + 1 = 3")
    print()
    print("This is NOT related to quaternionic structure.")
    print("It is purely combinatorial: each triangle has exactly 3 edges,")
    print("so each column of d_2 has exactly 3 nonzero entries (+/-1),")
    print("and (d2ᵀd2)[i,i] = sum of squares = 3.")
    print()
    print("The off-diagonal entries encode edge-sharing between faces:")
    print("Two k-simplices share a (k-1)-face iff their columns in dk")
    print("have a common nonzero row, and the product of the signs")
    print("gives the signed adjacency.")


if __name__ == "__main__":
    main()
