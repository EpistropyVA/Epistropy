"""
d6_zero_mode_irreps.py

6-simplex (7 vertices, 35 triangular faces) face-adjacency:
- Spectrum: -3(x14), 0(x14), 5(x6), 12(x1)
- Decompose the 14-dim zero-mode eigenspace under S_7 symmetry.

Method: compute character of the 14D representation by tracing
the permutation matrix for each conjugacy class of S_7.
"""

import io
import sys
import itertools
import numpy as np
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/d6_zero_mode_results.txt"
_LOG = []

def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


# Build 6-simplex face-adjacency
vertices = list(range(7))
faces = list(itertools.combinations(vertices, 3))
face_idx = {f: i for i, f in enumerate(faces)}
n = len(faces)
log(f"6-simplex: {len(vertices)} vertices, {n} faces")

A = np.zeros((n, n), dtype=int)
for i, f1 in enumerate(faces):
    for j, f2 in enumerate(faces):
        if i < j:
            shared = len(set(f1) & set(f2))
            if shared == 2:
                A[i, j] = A[j, i] = 1

vals, vecs = np.linalg.eigh(A)
log(f"\nEigenvalues: {np.round(sorted(set(np.round(vals, 4))), 4)}")

# Extract zero-mode subspace
zero_mask = np.abs(vals) < 0.01
zero_vecs = vecs[:, zero_mask]  # 35 x 14
log(f"Zero-mode dimension: {zero_vecs.shape[1]}")

# Projector onto zero-mode subspace
P0 = zero_vecs @ zero_vecs.T  # 35 x 35

# S_7 conjugacy classes (by cycle type)
# Cycle types of S_7: partitions of 7
from math import factorial

def cycle_type(perm):
    """Return sorted cycle lengths of a permutation (as tuple)."""
    visited = [False] * len(perm)
    cycles = []
    for i in range(len(perm)):
        if not visited[i]:
            length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                length += 1
            cycles.append(length)
    return tuple(sorted(cycles))

def perm_to_face_matrix(perm):
    """Given a permutation of 7 vertices, compute the 35x35 face permutation matrix."""
    M = np.zeros((n, n), dtype=int)
    for i, f in enumerate(faces):
        new_f = tuple(sorted(perm[v] for v in f))
        j = face_idx[new_f]
        M[j, i] = 1
    return M

log("\nComputing S_7 characters on zero-mode subspace...")
log("(Sampling one representative per conjugacy class)")

# Generate all permutations, group by cycle type, pick one representative
# S_7 has 15 conjugacy classes
conj_classes = defaultdict(list)

# For efficiency: generate all 5040 perms, classify
for perm in itertools.permutations(range(7)):
    ct = cycle_type(perm)
    if len(conj_classes[ct]) == 0:
        conj_classes[ct].append(perm)
    else:
        conj_classes[ct].append(None)  # just count

# Count class sizes
class_sizes = {}
class_reps = {}
for ct, members in conj_classes.items():
    class_sizes[ct] = len(members)
    class_reps[ct] = members[0]

log(f"\nS_7 conjugacy classes: {len(class_sizes)}")
log(f"Total: {sum(class_sizes.values())} (expect 5040={factorial(7)})")

# Compute character for each class
log(f"\n{'Cycle type':<20} {'Size':>6} {'chi(zero)':>10} {'chi(full)':>10}")
log("-" * 50)

characters_zero = {}
characters_full = {}

for ct in sorted(class_sizes.keys()):
    rep = class_reps[ct]
    M = perm_to_face_matrix(rep)
    # Character on zero-mode subspace = tr(P0 @ M)
    # Because P0 projects onto the subspace and M acts on it
    chi_zero = np.trace(P0 @ M)
    chi_full = np.trace(M)  # character on full 35-dim space
    characters_zero[ct] = chi_zero
    characters_full[ct] = chi_full
    log(f"{str(ct):<20} {class_sizes[ct]:>6} {chi_zero:>10.4f} {chi_full:>10.4f}")

# S_7 character table (irreps labeled by partitions)
# Dimensions of S_7 irreps:
# [7]:1, [6,1]:6, [5,2]:14, [5,1,1]:14, [4,3]:14, [4,2,1]:35, [4,1,1,1]:20,
# [3,3,1]:21, [3,2,2]:21, [3,2,1,1]:35, [3,1,1,1,1]:14, [2,2,2,1]:14,
# [2,2,1,1,1]:14, [2,1,1,1,1,1]:6, [1,1,1,1,1,1,1]:1

# To decompose: n_lambda = (1/|G|) sum_g class_size * chi_lambda(g)* * chi_rep(g)
# We need the full S_7 character table. Let me compute it from scratch using
# the Murnaghan-Nakayama rule or use a known table.

# Instead of hardcoding the full table, let's use a computational approach:
# The irrep characters can be computed from the Frobenius formula.
# But this is complex. Let me try a simpler approach:

# The 35-dim face representation of S_7 = the representation on 3-element subsets
# This is a well-known representation: it decomposes as
# V_{(7)} + V_{(5,2)} + V_{(4,3)} for the full 35-dim
# Wait, let me think... C(7,3) = 35 faces.

# The permutation representation on k-subsets of {1,...,n} decomposes as:
# Sum_{j=0}^{k} V_{(n-j, j)} (the "hook" representations)
# For n=7, k=3: V_{(7)} + V_{(6,1)} + V_{(5,2)} + V_{(4,3)}
# Dimensions: 1 + 6 + 14 + 14 = 35 ✓

log("\n\n=== Decomposition via known S_n representation theory ===")
log("\nThe permutation representation of S_7 on 3-subsets decomposes as:")
log("  V_(7) ⊕ V_(6,1) ⊕ V_(5,2) ⊕ V_(4,3)")
log("  dims:  1  +  6  +  14  +  14  = 35 ✓")
log("")
log("Our eigenvalue decomposition:")
log("  λ=12 (x1) + λ=5 (x6) + λ=0 (x14) + λ=-3 (x14) = 35")
log("")
log("Matching by dimension:")
log("  λ=12, deg=1  ↔  V_(7)   = trivial rep (fully symmetric)")
log("  λ=5,  deg=6  ↔  V_(6,1) = standard rep")
log("  λ=0,  deg=14 ↔  V_(5,2) or V_(4,3)")
log("  λ=-3, deg=14 ↔  V_(4,3) or V_(5,2)")

# To distinguish which 14-dim irrep is which, compute the character
# of V_(5,2) and V_(4,3) on a specific conjugacy class and compare.

# For the cycle type (1,1,1,1,1,1,1) = identity: both have chi=14.
# For cycle type (2,1,1,1,1,1) = one transposition:
# chi_{(5,2)}((2,1^5)) and chi_{(4,3)}((2,1^5))

# Known: chi_{(n-k,k)} on class with cycle type mu can be computed.
# For (5,2) on (2,1^5): use hook-length or MN rule.

# Actually, simpler: for the STANDARD representation V_(6,1) (dim 6),
# chi on transposition = 4 (trace of permutation matrix on {1..7} minus 1/7 * 7 = ...)
# V_(6,1): chi(identity)=6, chi(transposition)=4

# For k-subset reps: the character on a permutation sigma is
# = number of k-subsets fixed by sigma.

# For full 35-dim on transposition (12): fixed 3-subsets are those containing
# both 1,2 (then choose 1 from {3,4,5,6,7}: 5) or neither (choose 3 from {3,4,5,6,7}: 10)
# Total fixed = 15. So chi_full(transposition) = 15.

# Decomposition: chi_full = chi_{(7)} + chi_{(6,1)} + chi_{(5,2)} + chi_{(4,3)}
# 15 = 1 + chi_{(6,1)}(trans) + chi_{(5,2)}(trans) + chi_{(4,3)}(trans)

# chi_{(6,1)} on transposition: standard rep of S_7 = 7-dim natural minus trivial.
# Natural rep on transposition: fixes 5, swaps 2, so trace = 5.
# chi_{(6,1)} = 5 - 1 = 4.

# So: 15 = 1 + 4 + chi_{(5,2)} + chi_{(4,3)}
# chi_{(5,2)} + chi_{(4,3)} = 10

# Our computed characters on transposition class:
trans_ct = (1,1,1,1,1,2)
chi_zero_trans = characters_zero.get(trans_ct, None)
chi_minus3_val = None

# Also get character of lambda=-3 subspace on transposition
minus3_mask = vals < -2.5
minus3_vecs = vecs[:, minus3_mask]
P3 = minus3_vecs @ minus3_vecs.T

M_trans = perm_to_face_matrix(class_reps[trans_ct])
chi_zero_on_trans = np.trace(P0 @ M_trans)
chi_m3_on_trans = np.trace(P3 @ M_trans)

log(f"\nCharacter verification on transposition (2,1^5):")
log(f"  chi(zero-mode, trans) = {chi_zero_on_trans:.4f}")
log(f"  chi(lambda=-3, trans) = {chi_m3_on_trans:.4f}")
log(f"  chi(full, trans) = {characters_full.get(trans_ct, 'N/A')}")

# Known characters for S_7:
# V_(5,2) on (2,1^5): using MN rule or tables
# From standard S_7 character table:
# V_(5,2): chi(e)=14, chi(trans)=4
# V_(4,3): chi(e)=14, chi(trans)=6

# So if chi_zero = 4 → V_(5,2), if chi_zero = 6 → V_(4,3)

log(f"\nFrom S_7 character table:")
log(f"  chi_{{(5,2)}}(transposition) = 4")
log(f"  chi_{{(4,3)}}(transposition) = 6")

if abs(chi_zero_on_trans - 4) < 0.1:
    log(f"\n  → Zero modes (λ=0, dim=14) = V_(5,2)")
    log(f"  → Bottom band (λ=-3, dim=14) = V_(4,3)")
elif abs(chi_zero_on_trans - 6) < 0.1:
    log(f"\n  → Zero modes (λ=0, dim=14) = V_(4,3)")
    log(f"  → Bottom band (λ=-3, dim=14) = V_(5,2)")
else:
    log(f"\n  → UNEXPECTED: doesn't match either. Check computation.")

# Additional: verify on 3-cycle class
three_ct = (1,1,1,1,3)
M3 = perm_to_face_matrix(class_reps[three_ct])
chi_zero_3 = np.trace(P0 @ M3)
chi_m3_3 = np.trace(P3 @ M3)
log(f"\nVerification on 3-cycle (3,1^4):")
log(f"  chi(zero-mode) = {chi_zero_3:.4f}")
log(f"  chi(lambda=-3) = {chi_m3_3:.4f}")
# V_(5,2) on (3,1^4): 2, V_(4,3) on (3,1^4): 2 (both = 2, doesn't distinguish)

# Try 7-cycle
seven_ct = (7,)
M7 = perm_to_face_matrix(class_reps[seven_ct])
chi_zero_7 = np.trace(P0 @ M7)
chi_m3_7 = np.trace(P3 @ M7)
log(f"\nOn 7-cycle:")
log(f"  chi(zero-mode) = {chi_zero_7:.4f}")
log(f"  chi(lambda=-3) = {chi_m3_7:.4f}")
# V_(5,2) on (7): 0, V_(4,3) on (7): 0 (both 0)

# 4-cycle
four_ct = (1,1,1,4)
if four_ct in class_reps:
    M4 = perm_to_face_matrix(class_reps[four_ct])
    chi_zero_4 = np.trace(P0 @ M4)
    chi_m3_4 = np.trace(P3 @ M4)
    log(f"\nOn (4,1^3):")
    log(f"  chi(zero-mode) = {chi_zero_4:.4f}")
    log(f"  chi(lambda=-3) = {chi_m3_4:.4f}")

log("\n\n=== Complete eigenspace ↔ irrep correspondence ===")
log("6-simplex face-adjacency spectrum under S_7:")
log("")

# Compute all 4 eigenspace characters on all classes
eigenspaces = {
    'lambda=-3': minus3_mask,
    'lambda=0': zero_mask,
    'lambda=5': (np.abs(vals - 5) < 0.1),
    'lambda=12': (np.abs(vals - 12) < 0.1),
}

log(f"{'Cycle type':<20} {'Size':>6} | {'λ=-3':>8} {'λ=0':>8} {'λ=5':>8} {'λ=12':>8} | {'Sum':>6}")
log("-" * 80)

for ct in sorted(class_sizes.keys()):
    M = perm_to_face_matrix(class_reps[ct])
    chars = []
    for name, mask in eigenspaces.items():
        V_sub = vecs[:, mask]
        P_sub = V_sub @ V_sub.T
        chi = np.trace(P_sub @ M)
        chars.append(chi)
    total = sum(chars)
    log(f"{str(ct):<20} {class_sizes[ct]:>6} | {chars[0]:>8.2f} {chars[1]:>8.2f} {chars[2]:>8.2f} {chars[3]:>8.2f} | {total:>6.2f}")

log("\n\nExpected S_7 irrep dimensions:")
log("  V_(7):   dim=1   (trivial)")
log("  V_(6,1): dim=6   (standard)")
log("  V_(5,2): dim=14")
log("  V_(4,3): dim=14")

# Write results
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(_LOG))
log(f"\nResults written to {OUT_PATH}")
