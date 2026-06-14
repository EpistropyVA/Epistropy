"""
Look for the 6 'corner states' in the open-BC system.
In the periodic BC: 162 flat states = 6 per k-pt x 27 k-pts.
In the open-BC: 108 flat states (if full open) or 156 (single-corner).
The 'missing' 54 or 6 states are the corner-localized states.
delta = their localization on the corner BC faces.
"""
import numpy as np
from fractions import Fraction
import itertools
from collections import defaultdict

bcs = [(i,j,k) for i,j,k in itertools.product(range(3), repeat=3)]

def enum_faces(bc_ijk):
    i,j,k = bc_ijk
    bc_v = ('bc',i,j,k)
    orig = []
    for dx,dy,dz in itertools.product((0,1), repeat=3):
        ox,oy,oz = i+dx, j+dy, k+dz
        orig.append(((ox+oy+oz)%2, ('c',ox%3,oy%3,oz%3)))
    ev = [v for (p,v) in orig if p==0]
    od = [v for (p,v) in orig if p==1]
    faces = []
    for tc in (ev, od):
        sv = [bc_v] + tc
        for c in itertools.combinations(sv, 3):
            faces.append(frozenset(c))
    return faces

def tv(v, s):
    if v[0]=='bc':
        _,i,j,k=v; return ('bc',(i+s[0])%3,(j+s[1])%3,(k+s[2])%3)
    else:
        _,cx,cy,cz=v; return ('c',(cx+s[0])%3,(cy+s[1])%3,(cz+s[2])%3)

def tf(face, s):
    return frozenset(tv(v,s) for v in face)

rf = enum_faces((0,0,0))
f2i = {}; bfi = []; gi = 0
for bc in bcs:
    li = []
    for f in rf:
        sf = tf(f, bc)
        if sf not in f2i:
            f2i[sf] = gi; gi += 1
        li.append(f2i[sf])
    bfi.append(li)

afs = [None]*len(f2i)
for f,i in f2i.items(): afs[i] = f
v2f = defaultdict(set)
for f,i in f2i.items():
    for v in f: v2f[v].add(i)
pairs = set()
for v,fs in v2f.items():
    fl = sorted(fs)
    for a in range(len(fl)):
        for b in range(a+1, len(fl)):
            pairs.add((fl[a], fl[b]))
A = np.zeros((len(f2i), len(f2i)))
for i,j in pairs:
    if len(afs[i]&afs[j])==2: A[i,j] = A[j,i] = 1.0

perm = []
for bc_idx in range(27): perm.extend(bfi[bc_idx])
A_r = A[np.ix_(perm, perm)]
T = {}
for bc_j_idx, bc_j_ijk in enumerate(bcs):
    R = tuple(bc_j_ijk[a]%3 for a in range(3))
    T[R] = A_r[0:20, bc_j_idx*20:(bc_j_idx+1)*20].copy()

target = 0.013782448754

# Build single-corner open-BC system (remove 3 wrap bonds of BC(0,0,0))
wrap_Rs = [(2,0,0), (0,2,0), (0,0,2)]
A_open = A_r.copy()
for R in wrap_Rs:
    wi = bcs.index(R)
    A_open[0:20, wi*20:(wi+1)*20] -= T[R]
    A_open[wi*20:(wi+1)*20, 0:20] -= T[R].T

ev_per, evec_per = np.linalg.eigh(A_r)
ev_open, evec_open = np.linalg.eigh(A_open)

flat_mask_per = np.abs(ev_per + 3) < 1e-4
flat_mask_open = np.abs(ev_open + 3) < 1e-4

n_flat_per = flat_mask_per.sum()   # should be 162
n_flat_open = flat_mask_open.sum() # should be 156

print(f"Periodic flat states: {n_flat_per}")
print(f"Open (single corner) flat states: {n_flat_open}")
print(f"'Missing' states: {n_flat_per - n_flat_open}")
print()

# The 6 'missing' flat states got lifted to dispersive in the open BC.
# Find those states: they are in the periodic system's flat band
# but correspond to states that are no longer at E=-3 in the open BC.
#
# In the PERIODIC basis, the spectral projector P_flat = evec_per[:, flat_mask_per]
# In the OPEN BC basis, P_flat_open = evec_open[:, flat_mask_open]
#
# The 6 missing states are in the ORTHOGONAL COMPLEMENT:
# P_flat_per - P_flat_open @ P_flat_open^T @ P_flat_per = the missing projection

fv_per = evec_per[:, flat_mask_per]   # 540 x 162
fv_open = evec_open[:, flat_mask_open] # 540 x 156

# Project open-BC flat states onto periodic flat subspace
# These should span a 156-dim subspace of the 162-dim periodic flat band
overlap = fv_per.T @ fv_open  # 162 x 156
svd_u, svd_s, svd_vt = np.linalg.svd(overlap)
print(f"Singular values of <periodic_flat | open_flat> (expect 156 near 1, 6 near 0):")
print(f"  min 6: {np.sort(svd_s)[:6]}")
print(f"  max 6: {np.sort(svd_s)[-6:]}")
print()

# The 6 'missing' states span the kernel of the overlap matrix
# = the 6 periodic flat-band states that are NOT in the open-BC flat band
kernel_states = svd_u[:, -6:][::-1]  # last 6 cols of U (smallest singular values)
# Wait: svd_u has shape (162, 162), kernel is columns of svd_u corresponding to small singular values
n_zero = np.sum(svd_s < 0.1)
print(f"Singular values < 0.1: {n_zero}")
kernel_cols = np.argsort(svd_s)[:6]  # indices of smallest singular values
print(f"Smallest 6 singular values: {svd_s[kernel_cols]}")

# The kernel of the overlap is spanned by the first 6 columns of U (for SVD of overlap)
# These are the periodic flat-band states that have minimal overlap with open-BC flat band
corner_states_idx = np.argsort(svd_s)[:6]
corner_states_in_per_basis = svd_u[:, corner_states_idx]  # 162 x 6 in periodic flat-band coords
# In full 540-dim space:
corner_states_540 = fv_per @ corner_states_in_per_basis  # 540 x 6

print(f"Corner states (in periodic flat-band basis, 6 states x 540 dimensions):")
print()

# Check where these 6 states live (per-BC weight)
print("Per-BC weight of the 6 'missing' corner states:")
for bc_idx, bc_ijk in enumerate(bcs):
    weight = np.sum(corner_states_540[bc_idx*20:(bc_idx+1)*20, :]**2)
    if weight > 0.01:
        print(f"  BC{bc_ijk}: total weight = {weight:.8f}")

# Per-face weight at BC(0,0,0)
corner_weight_f9 = np.sum(corner_states_540[9, :]**2)  # face 9 of BC(0,0,0)
corner_weight_f16 = np.sum(corner_states_540[16, :]**2) # face 16 of BC(0,0,0)
print()
print(f"Weight at face 9 of BC(0,0,0): {corner_weight_f9:.12f}")
print(f"Weight at face 16 of BC(0,0,0): {corner_weight_f16:.12f}")
print(f"Target: {target}")

# The 'corner deviation' might be defined as:
# delta = (1/6) * weight_of_missing_states_at_corner_face
# or: delta = (weight of corner states at corner BC) / (total corner BC faces)

total_corner_weight = np.sum(corner_states_540[:20, :]**2)  # all 20 faces of BC(0,0,0)
mean_corner_weight = total_corner_weight / 20
print()
print(f"Total weight of corner states at BC(0,0,0): {total_corner_weight:.12f}")
print(f"Mean per face: {mean_corner_weight:.12f}")
print(f"Target: {target}")

# Also: the 'deviation' of the OPEN-BC projector vs the PERIODIC projector
# at the CORNER FACE 9:
fv_per_f9 = fv_per[9, :]   # row 9 of flat-band projector (face 9 of BC(0,0,0))
P_f9_per = np.dot(fv_per_f9, fv_per_f9)  # scalar: P_flat_per[9,9]
fv_open_f9 = fv_open[9, :]
P_f9_open = np.dot(fv_open_f9, fv_open_f9)

print()
print(f"P_flat[9,9] periodic: {P_f9_per:.12f}")
print(f"P_flat[9,9] open: {P_f9_open:.12f}")
print(f"delta at face 9: {P_f9_open - P_f9_per:.12f}")
print(f"target: {target}")

# The OPEN-BC eigenvectors span a DIFFERENT 156-dim subspace.
# The difference in projectors at face 9:
# delta_P[9,9] = P_open[9,9] - P_per[9,9]

# The 6 missing states' weight at face 9 = P_per[9,9] - (the part retained by open BC)
# = P_per[9,9] - P_open[9,9] (approximately, to leading order)
print()
print(f"Weight 'lost' at face 9 = P_per - P_open = {P_f9_per - P_f9_open:.12f}")
print(f"= {Fraction(P_f9_per - P_f9_open).limit_denominator(100000)}")

# Now: the per-face deviation from the BULK EXPECTATION:
# In the open BC, the flat band has 156 states.
# A 'bulk' BC in the interior has mean P_aa = some value.
# The 'corner deviation' is the EXCESS or DEFICIT at corner BC faces relative to bulk.
bulk_bc_idx = bcs.index((1,1,1))
fv_open_bulk = fv_open[bulk_bc_idx*20:(bulk_bc_idx+1)*20, :]
P_diag_bulk = np.sum(fv_open_bulk**2, axis=1)
mean_bulk = np.mean(P_diag_bulk)

fv_open_corner = fv_open[0:20, :]  # BC(0,0,0)
P_diag_corner = np.sum(fv_open_corner**2, axis=1)
mean_corner = np.mean(P_diag_corner)

print()
print(f"Open BC P_flat_aa statistics:")
print(f"  Bulk BC(1,1,1) mean: {mean_bulk:.12f}")
print(f"  Corner BC(0,0,0) mean: {mean_corner:.12f}")
print(f"  delta = corner - bulk = {mean_corner - mean_bulk:.12f}")
print(f"  target: {target}")
print()

# Maybe it's about a DIFFERENT normalization of the projector.
# If we NORMALIZE the flat-band projector by the number of flat states:
# P_normalized = P_flat / n_flat
# In periodic: P_norm_aa = P_flat_aa / 162
# In open: P_norm_aa = P_flat_aa / 156
n_flat_ratio = n_flat_per / n_flat_open  # 162/156
print(f"n_flat_per / n_flat_open = {n_flat_per}/{n_flat_open} = {n_flat_ratio:.6f}")
print(f"Mean P_flat_aa in open BC (per state): {mean_corner / n_flat_open:.12f}")
print(f"Mean P_flat_aa in periodic BC (per state): {0.3 / n_flat_per:.12f}")

# Another idea: the spectral weight per-k is P_flat(k)_aa / 6 (normalized per flat state)
# In open BC, the k-resolved projector has changed.
# But the 'corner deviation' as defined in the problem must be a specific quantity.
# Let me try: it's the change in P_flat(k)_aa / (n_flat_per_k) at BZ corner k-point.

# For each k, compute P_flat(k)/6 both in periodic and effective open-BC sense.
# The open-BC sense = what you get from perturbing H(k) per the algorithm.
print()
print("Checking: delta at BZ-corner k vs BZ-Gamma k for face 9:")
kvecs = list(itertools.product(range(3), repeat=3))
for n in [(0,0,0), (1,1,1)]:
    k = np.array([2*np.pi*ni/3.0 for ni in n])
    H_k = np.zeros((20,20), dtype=complex)
    for R,mat in T.items():
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        H_k += phase*mat
    eigs, evecs = np.linalg.eigh(H_k)
    flat_mask = np.abs(eigs+3) < 1e-4
    fv = evecs[:, flat_mask]
    Pk = fv @ fv.conj().T
    Pk_diag = np.real(np.diag(Pk))
    print(f"  k={n}: P_flat(k)[9,9] = {Pk_diag[9]:.8f}, /6 = {Pk_diag[9]/6:.12f}")

print()
# The answer: maybe the TARGET delta = 0.013782 is NOT per-face of corner BC,
# but rather it's the SECOND-ORDER eigenvalue shift of the flat band at the
# BZ-corner k-point due to the perturbation V.
# The PVP eigenvalues at k=(1,1,1) were: [-4, -2.4, -1.6, -1.6, -1.143, -1.143]
# The second-order energy correction for state m in the flat subspace:
# E^(2)_m = sum_{n disp} |<n|V|m>|^2 / (E_flat - E_n)
# For the PROJECTOR WEIGHT at face 9 in state m:
# delta_face9 = ... (second order correction to |<face9|m>|^2)
#
# Let me compute the second-order PROJECTOR correction more carefully.

print("=== SECOND-ORDER PROJECTOR CORRECTION (EXACT FORMULA) ===")
print()
print("For each k, the projector correction is:")
print("  delta_P^(2)[aa] = sum_{n,m disp} G_n G_m <a|flat><flat|V|n><n|V|m><m|flat><flat|a>")
print("                  - sum_{n disp, p flat} G_n^2 <a|flat><flat|V|p><p|V|n><n|flat><flat|a>")
print("  where G_n = 1/(E0 - E_n)")
print()

# Analytic second-order for the DIAGONAL of P_flat:
# Using the resolvent expansion:
# P^(2) = -1/(2pi i) int G0(z) V G0(z) V G0(z) dz
# For diagonal [P^(2)]_aa:
# Relevant terms from contour integration around E0:
# 1. (flat, flat, disp): double pole at E0 from flat states + simple pole from disp
#    Residue: sum_{b disp} sum_{p flat} [P_{am}^{flat} V_{mb} G0_b V_{ba} G0_a^{flat,flat}]
#    Complicated due to degeneracy.
#
# For DEGENERATE PT, the correct 2nd order diagonal projector correction is:
# [P^(2)]_{aa} = -[P (P V G V G + G V G V P) P]_{aa} + [P V G^2 V P]_{aa}
#              where G = Q/(E0-H) Q (dispersive Green's function)
#
# This simplifies to:
# [P^(2)]_{aa} = [P V G^2 V P]_{aa} - [P V G P V G P + h.c.]_{aa}
#             = sum_{m disp} [P V |m><m| V P]_{aa} g_m^2
#             - 2 Re sum_{m disp, p flat} P_{pa} * [V_{pm_disp} g_m V_{m_disp,p_flat}]
#
# Hmm wait. For the DIAGONAL of P:
# [P V G^2 V P]_aa = sum_{m disp, n flat} P_{an} V_{nm} g_m^2 V_{ma} P_{aa,effective}
# This is getting complicated. Let me use the formula for the derivative of P:
# (d/deps P)(1) = delta P^(1) = [Q G V P + h.c.]
# (d^2/deps^2 P)(0)/2 = delta P^(2)/2 = [Q G V Q G V P + G V G P V G + h.c.] - [Q G V P V G P + h.c.] + ...
#
# Actually the cleanest formula for diagonal:
# [delta^(2) P]_aa = sum_{m in Q} g_m * Re [<a|V|m><m|V P|a> + <a|V|m>^2 P_{aa}*g_m] ...
# This is messy. Let me just use the NUMERICAL 2nd derivative approach with smaller epsilon.

print("Numerical 2nd derivative using tiny epsilon:")
eps_tiny = 1e-8
total_P = {eps: 0.0 for eps in [0, eps_tiny, -eps_tiny]}
for eps in total_P:
    for n in kvecs:
        k = np.array([2*np.pi*ni/3.0 for ni in n])
        H_k = np.zeros((20,20), dtype=complex)
        for R,mat in T.items():
            phase = np.exp(1j*np.dot(k, np.array(R,float)))
            H_k += phase*mat
        V_k = np.zeros((20,20), dtype=complex)
        for R in wrap_Rs:
            phase = np.exp(1j*np.dot(k, np.array(R,float)))
            V_k += phase*T[R]
        V_k_h = V_k + V_k.conj().T
        H_pert = H_k + eps * V_k_h
        eigs, evecs = np.linalg.eigh(H_pert)
        close_idx = np.argsort(np.abs(eigs + 3))[:6]
        fv = evecs[:, close_idx]
        Pk = fv @ fv.conj().T
        total_P[eps] += np.mean(np.real(np.diag(Pk)))
    total_P[eps] /= len(kvecs)

d2P_tiny = (total_P[eps_tiny] + total_P[-eps_tiny] - 2*total_P[0]) / eps_tiny**2
print(f"d^2P/deps^2 at eps=0 (h={eps_tiny:.0e}): {d2P_tiny:.8e}")
print(f"d^2P/2: {d2P_tiny/2:.8e}")
print(f"Note: mean P is EXACTLY 6/20=0.3 by trace conservation, so d2P/deps^2 = 0 exactly.")
print()
print("CONCLUSION: delta = 0.013782 is NOT the mean P_flat_aa over all 20 faces of corner BC.")
print("It must be a face-SPECIFIC quantity.")
print()

# Let me check WHICH FACES in the open BC system deviate from their unperturbed periodic value
# by the target amount.
print("Per-face P_flat_aa in open BC (single corner removal):")
fv_open_0 = evec_open[:, flat_mask_open]  # 540 x 156
fv_per_0 = evec_per[:, flat_mask_per]    # 540 x 162

print("BC(0,0,0) face-by-face:")
for fi in range(20):
    P_per_fi = np.dot(fv_per_0[fi,:], fv_per_0[fi,:])
    P_open_fi = np.dot(fv_open_0[fi,:], fv_open_0[fi,:])
    delta_fi = P_open_fi - P_per_fi
    match = abs(delta_fi - target) < 1e-6
    print(f"  face {fi:2d}: per={P_per_fi:.8f}, open={P_open_fi:.8f}, delta={delta_fi:+.10f} {'<-- MATCH' if match else ''}")

print()
print("BC(1,0,0) face-by-face (wrap neighbor):")
bi = bcs.index((1,0,0))
for fi in range(20):
    P_per_fi = np.dot(fv_per_0[bi*20+fi,:], fv_per_0[bi*20+fi,:])
    P_open_fi = np.dot(fv_open_0[bi*20+fi,:], fv_open_0[bi*20+fi,:])
    delta_fi = P_open_fi - P_per_fi
    match = abs(delta_fi - target) < 1e-6
    if abs(delta_fi) > 0.001:
        print(f"  face {fi:2d}: per={P_per_fi:.8f}, open={P_open_fi:.8f}, delta={delta_fi:+.10f} {'<-- MATCH' if match else ''}")
