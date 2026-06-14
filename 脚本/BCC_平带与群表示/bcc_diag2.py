"""
Compute exact second-order projector correction for face 9.
The rough 2nd-order formula was wrong; use the correct Löwdin formula.
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

kvecs = list(itertools.product(range(3), repeat=3))
wrap_Rs = [(2,0,0), (0,2,0), (0,0,2)]
target = 0.013782448754

def bz_label(n):
    return ['G','E','F','C'][sum(1 for x in n if x!=0)]

print("=== SECOND-ORDER PROJECTOR CORRECTION (Kato-Bloch/Lowdin) ===")
print()
print("Physical setup:")
print("  Periodic: flat-band projector P = sum_k P_flat(k) / N_k")
print("  Perturbation V: remove 3 wrap bonds of BC(0,0,0)")
print("  V acts on (20-face corner BC) <-> (20-face wrap-neighbor BC) blocks")
print("  In k-space: V local to corner BC, mixes k-sectors")
print()
print("Approach: for each k independently, treat the 20-dim local H(k) problem.")
print("  V_k = Hermitian wrap-bond contribution at wavevector k")
print("  delta_P^(1) = G(k) V_k P_flat(k) + h.c.   [dispersive coupling]")
print("  delta_P^(2) = P_flat(k) V_k G(k)^2 V_k P_flat(k) - ...")
print()

# The correct second-order correction to the spectral projector P_flat(k):
# Using the Kato-Bloch perturbation expansion:
# P(lambda) = -1/(2*pi*i) * contour_integral dz (z-H)^{-1}
# P_perturbed = P + sum_n (1/n!) * [P_n^{(th) order term}]
#
# For a HERMITIAN perturbation V:
# delta_P^(1) = Q G(E0) V P + P V G(E0) Q   (where Q = 1-P, G = Q/(E0-H) Q)
# delta_P^(2) = Q G^2 V P V P + P V G^2 Q V P - P V G P V G P - P G V P V G^2 Q + ...
#
# For the DIAGONAL elements only (trace per face):
# [delta_P^(1)]_aa = 2 Re sum_{n in Q} <a|V|n><n|P_flat|a> / (E0 - En)
# [delta_P^(2)]_aa = sum_{n,m in Q} <a|V|n> <n|V|m><m|P_flat|a> / ((E0-En)(E0-Em))
#                 + sum_{n in Q, p in P} ... - sum_{n in Q} ... - ...
#
# Actually the simplest correct approach is the RESOLVENT expansion:
# delta_P = -1/(2*pi*i) * integral [(z-H-V)^{-1} - (z-H)^{-1}] dz around E0
# For small V, expand:
# = -1/(2*pi*i) * integral [G0(z) + G0(z)V G0(z) + G0(z)V G0(z)V G0(z) + ...] dz
# where G0(z) = (z-H)^{-1}
# delta_P^(1) = -1/(2*pi*i) * integral G0(z) V G0(z) dz
# delta_P^(2) = -1/(2*pi*i) * integral G0(z) V G0(z) V G0(z) dz
#
# For eigenvalues H|n> = En|n>:
# G0(z) = sum_n |n><n| / (z - En)
# delta_P^(1) = sum_{a in flat, b NOT in flat} |a><b| <a|V|b> / (E0-Eb) + h.c.
# [delta_P^(1)]_aa = 0  (the flat-to-flat and disp-to-disp diag terms cancel)
# Wait, diagonal elements:
# [delta_P^(1)]_aa = sum_{b not flat} 2 Re [<a flat|V|b disp> <b disp|a flat>] / (E0-Eb)
# = 2 Re sum_{n disp} <a|n><n|V|flat_states>|_a / (E0-En)
# = sum_m sum_{n disp} 2 Re [fv[a,m]* * (dv.T @ V @ fv)[n,m] * dv[a,n]] / (E0-En)
# = 2 Re sum_{n disp} [GVP]_{a,m} * fv[a,m]*   (sum over m flat)
# This is what I had before: dP1.

# For delta_P^(2) diagonal:
# [delta_P^(2)]_aa = -1/(2*pi*i) integral sum_{a,b,c} |a><c| <a|V|b><b|V|c> / ((z-Ea)(z-Eb)(z-Ec)) dz
# The contour picks up residues at E0 (for a or c in flat, b can be anything).
# For [delta_P^(2)]_aa diagonal:
# = sum over {(a flat, b disp, a flat), (a flat, b flat, c flat)} type terms
# The (flat, disp, flat) terms give:
# Residue at z=E0 (double pole): sum_{b disp} <a|V|b> <b|V|a> / (E0-Eb)^2  * (correction factor)
#
# The exact formula for the diagonal of delta_P^(2):
# [delta_P^(2)]_aa = - sum_{b not flat} |<a|V|b>|^2 / (E0-Eb)^2  (from the resolvent)
# Wait, let me be more careful.
#
# [delta_P^(2)]_aa = [P^{(2)}]_aa where
# P^{(2)} = P G V G V P - P G V P V G P - P V G P G V P + ...
#
# For the DIAGONAL:
# [P^{(2)}]_aa = sum_{b disp} sum_{c disp} fv[a] <flat|V|b><b|V|flat> fv[a]^T / ((E0-Eb)(E0-Ec)) * complex correction
#
# This is getting messy. Let me just use a direct numerical approach:
# Compute P_flat(H + epsilon*V) for small epsilon, take derivative to get delta_P.

epsilons = np.array([1e-4, 2e-4, 5e-4, 1e-3, -1e-4, -2e-4, -5e-4, -1e-3])

print("Numerical derivative approach:")
print("Computing d/depsilon P_flat(H + epsilon*V) at epsilon=0, face 9...")
print()

face_idx = 9
dP_face9 = {}

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

    # Compute P_flat(k) at several epsilon values
    P_vals = []
    for eps in epsilons:
        H_pert = H_k + eps * V_k_h
        eigs, evecs = np.linalg.eigh(H_pert)
        # Flat band: eigenvalues near E0 + epsilon*correction
        # Sort by proximity to -3 + eps*<V>
        E0_approx = -3.0
        flat_mask = np.abs(eigs - E0_approx) < 2.0  # generous tolerance
        # Take the 6 lowest eigenvalues (the flat band should stay near -3)
        sorted_idx = np.argsort(eigs)
        # Actually use the 6 states closest to E0_approx
        close_idx = np.argsort(np.abs(eigs - E0_approx))[:6]
        fv_pert = evecs[:, close_idx]
        Pk = fv_pert @ fv_pert.conj().T
        P_vals.append(np.real(Pk[face_idx, face_idx]))

    P_vals = np.array(P_vals)
    # Finite difference: d/deps at eps=0 using forward differences
    # Using 4-point stencil on eps = [+h, +2h, +5h] and [-h, -2h, -5h]
    # Simple: linear fit
    dP_face9[n] = P_vals

# Look at face-9 values across epsilon
print("k=(0,0,0) face 9 P_flat vs epsilon:")
n0 = (0,0,0)
for eps,v in zip(epsilons, dP_face9[n0]):
    print(f"  eps={eps:.0e}: P_face9 = {v:.10f}")

# Compute first derivative from finite diff
# d/deps at eps=0 ~ (P(+h) - P(-h)) / (2h)
# Use h=1e-4
P_plus = {}; P_minus = {}
for n in kvecs:
    P_plus[n] = dP_face9[n][0]   # eps=+1e-4
    P_minus[n] = dP_face9[n][4]  # eps=-1e-4

dP1_fd = {n: (P_plus[n] - P_minus[n]) / (2e-4) for n in kvecs}
dP2_fd = {n: (P_plus[n] + P_minus[n] - 2*dP_face9[n][0]*(P_plus[n]/P_plus[n])) for n in kvecs}
# Actually 2nd deriv: (P(+h) + P(-h) - 2*P(0)) / h^2
# Need P(0)
P0 = {}
for n in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n])
    H_k = np.zeros((20,20), dtype=complex)
    for R,mat in T.items():
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        H_k += phase*mat
    eigs, evecs = np.linalg.eigh(H_k)
    close_idx = np.argsort(np.abs(eigs + 3))[:6]
    fv0 = evecs[:, close_idx]
    Pk = fv0 @ fv0.conj().T
    P0[n] = np.real(Pk[face_idx, face_idx])

h = 1e-4
dP2_fd = {n: (P_plus[n] + P_minus[n] - 2*P0[n]) / h**2 for n in kvecs}

# Sum over k-points
dP1_total_f9 = sum(dP1_fd[n] for n in kvecs) / 27
dP2_total_f9 = sum(dP2_fd[n] for n in kvecs) / 27

print(f"\nFirst-order d/deps P_face9 (finite diff, h=1e-4): {dP1_total_f9:.12f}")
print(f"Second-order d^2/deps^2 P_face9 (finite diff): {dP2_total_f9:.12f}")
print(f"Target delta: {target}")
print()

# Also for face 0 (type-A face)
face_idx2 = 0
dP1_fd_f0 = {}
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
    P_vals_f0 = []
    for eps in epsilons[:2]:
        H_pert = H_k + eps * V_k_h
        eigs, evecs = np.linalg.eigh(H_pert)
        close_idx = np.argsort(np.abs(eigs + 3))[:6]
        fv_pert = evecs[:, close_idx]
        Pk = fv_pert @ fv_pert.conj().T
        P_vals_f0.append(np.real(Pk[face_idx2, face_idx2]))
    for eps in epsilons[4:5]:
        H_pert = H_k + eps * V_k_h
        eigs, evecs = np.linalg.eigh(H_pert)
        close_idx = np.argsort(np.abs(eigs + 3))[:6]
        fv_pert = evecs[:, close_idx]
        Pk = fv_pert @ fv_pert.conj().T
        P_vals_f0.append(np.real(Pk[face_idx2, face_idx2]))
    dP1_fd_f0[n] = (P_vals_f0[0] - P_vals_f0[2]) / (2e-4)

dP1_total_f0 = sum(dP1_fd_f0[n] for n in kvecs) / 27

print(f"First-order d/deps P_face0 (finite diff): {dP1_total_f0:.12f}")
print()

# Actually let me use the ANALYTIC first-order formula correctly
# and sum over all k-points to get the total per-face correction

print("=== ANALYTIC FIRST-ORDER CORRECTION (all faces) ===")
dP1_analytic = np.zeros(20, dtype=float)
for n in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n])
    H_k = np.zeros((20,20), dtype=complex)
    for R,mat in T.items():
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        H_k += phase*mat
    eigs, evecs = np.linalg.eigh(H_k)
    flat_mask = np.abs(eigs+3) < 1e-4
    disp_mask = ~flat_mask
    fv = evecs[:, flat_mask]
    dv = evecs[:, disp_mask]
    de = eigs[disp_mask]

    V_k = np.zeros((20,20), dtype=complex)
    for R in wrap_Rs:
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        V_k += phase*T[R]
    V_k_h = V_k + V_k.conj().T

    g_n = 1.0 / (-3.0 - de)
    V_dfl = dv.conj().T @ V_k_h @ fv
    GVP = dv @ (g_n[:,None] * V_dfl)
    dP1 = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
    dP1_analytic += dP1

dP1_analytic /= 27
print("Analytic first-order delta_P per face:")
for i,v in enumerate(dP1_analytic):
    f = Fraction(v).limit_denominator(100000)
    print(f"  face {i:2d}: {v:+.10f} ~ {f}")
print(f"\nFace 9: {dP1_analytic[9]:.12f}")
print(f"Target: {target}")
print(f"Ratio face9/target: {dP1_analytic[9]/target:.6f}")
print()

# Compare with finite-difference
print(f"Finite-diff face9: {dP1_total_f9:.12f}")
print(f"Analytic face9: {dP1_analytic[9]:.12f}")
print(f"Agreement: {abs(dP1_analytic[9] - dP1_total_f9) < 1e-4}")
