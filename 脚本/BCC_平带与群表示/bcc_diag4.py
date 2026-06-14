"""
Compute the second-order correction to mean P_flat_aa at the corner BC.
Using the Kato-Bloch formula:
delta^(2) P = -1/(2pi i) * contour integral G0(z) V G0(z) V G0(z) dz

For the DIAGONAL elements of P (trace over corner BC faces):
[delta^(2)P]_aa = sum_{b,c} [term from Cauchy integral evaluation]

The key terms are:
1. (flat, disp, flat): b=flat, c=disp or b=disp, c=flat -> zero for diagonal
2. (flat, flat, flat): gives [P_flat V P_flat V P_flat]_aa / 0 -> need L'Hopital -> handled by subtracting P projection
3. (disp, any, disp): handled by the Q G^2 V G^2 Q type terms

Actually the correct second-order projector correction is:
P^(2) = P G V G V P + (P G V Q V G P) type cancellations
= Q G V (P) V G Q  [the "folded-in" contribution]
  + P V G V G P * (sign) [wrong]

Use the STANDARD Brillouin-Wigner expansion (non-degenerate → degenerate):
For the FLAT BAND, which is degenerate, the correct 2nd order correction to P is:

P^(2)_aa = sum_{n disp} sum_{m disp} <a_flat| V |n_disp> <n_disp| V |m_disp> <m_disp|P_flat|a_flat> / ((E0-En)(E0-Em))
         + sum_{n disp} sum_{p flat} <a_flat| V |p_flat> <p_flat| V |n_disp> <n_disp|P_flat|a_flat> / ((E0-En)^2)   [Lowdin]
         - sum_{n disp} |<a_flat|V|n_disp>|^2 P_flat_aa / (E0-En)^2 [normalization correction]

Wait, this is for individual states, not the projector. For the PROJECTOR:

delta P^(2) = G0 V Q G0 V P + P V Q G0 V G0 + G0 V P V G0   (with P=P_flat, Q=1-P_flat)
where G0 = Q/(E0-H) Q = dispersive Green's function

For diagonal elements [delta P^(2)]_aa:
= sum_{n,m disp} fv[a]* V_nm G_m G_n V V + ...

This is getting complex. Let me use a different approach:
Compute P(epsilon) = P_flat(H + epsilon V) numerically for several epsilon values,
then use Richardson extrapolation to get d^2/deps^2 P at eps=0.
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
N_k = 27
target = 0.013782448754

# For the corner BC (0,0,0), compute P_flat(k, epsilon) for each k and several epsilon values
# then extract the MEAN P_flat over all 20 faces.

wrap_Rs = [(2,0,0), (0,2,0), (0,0,2)]

def compute_mean_P_flat_k(n_tuple, eps, wrap_Rs):
    """Compute mean P_flat(k, eps)_aa over all 20 faces."""
    k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
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
    return np.mean(np.real(np.diag(Pk)))

# Compute for BC(0,0,0) with multiple epsilon values
epsilons = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, -0.001, -0.002, -0.005, -0.01, -0.02, -0.05, -0.1]

print("Computing mean P_flat vs epsilon for BC(0,0,0)...")
mean_P_vs_eps = {}
for eps in epsilons:
    total = sum(compute_mean_P_flat_k(n, eps, wrap_Rs) for n in kvecs)
    mean_P_vs_eps[eps] = total / N_k
    if abs(eps) < 0.002 or eps == 0.1 or eps == -0.1:
        print(f"  eps={eps:+.4f}: mean P_flat = {mean_P_vs_eps[eps]:.12f}")

print()
# Compute derivative using finite differences
P0 = mean_P_vs_eps[0]
print(f"P(eps=0) = {P0:.12f}")

# First derivative (central difference)
h = 0.001
dP_1st = (mean_P_vs_eps[h] - mean_P_vs_eps[-h]) / (2*h)
print(f"dP/deps at eps=0 (h={h}): {dP_1st:.12f}")

# Second derivative
d2P = (mean_P_vs_eps[h] + mean_P_vs_eps[-h] - 2*P0) / h**2
print(f"d2P/deps2 at eps=0 (h={h}): {d2P:.12f}")

# Use multiple step sizes for Richardson extrapolation
h_vals = [0.001, 0.002, 0.005]
dP_1sts = [(mean_P_vs_eps[h] - mean_P_vs_eps[-h]) / (2*h) for h in h_vals]
d2Ps = [(mean_P_vs_eps[h] + mean_P_vs_eps[-h] - 2*P0) / h**2 for h in h_vals]
print()
print("First derivatives for various h:")
for h,d in zip(h_vals, dP_1sts): print(f"  h={h}: {d:.12f}")
print("Second derivatives for various h:")
for h,d in zip(h_vals, d2Ps): print(f"  h={h}: {d:.12f}")

print()
print(f"Target delta (eps=1): {target}")
print(f"P(eps=1) extrapolation:")
# P(eps) = P0 + dP*eps + d2P*eps^2/2 + ...
# At eps=1: delta = dP*1 + d2P/2
# But dP = 0 (first-order conserves trace), so delta ~ d2P/2 * 1^2
d2P_best = d2Ps[0]
dP_best = dP_1sts[0]
print(f"  P(1) ~ P0 + dP*1 + d2P*1^2/2 = {P0 + dP_best + d2P_best/2:.12f}")
print(f"  delta = P(1) - P0 ~ dP + d2P/2 = {dP_best + d2P_best/2:.12f}")
print(f"  2nd order only: d2P/2 = {d2P_best/2:.12f}")

# The actual perturbation at eps=1 IS V (full removal).
# So the physical delta should be P(eps=1) - P(eps=0) = delta P.
# At large eps, the expansion breaks down, but let's check the DIRECT computation.
print()
print("Direct computation at eps=1 (full removal):")
mean_P_full = sum(compute_mean_P_flat_k(n, 1.0, wrap_Rs) for n in kvecs) / N_k
print(f"  P(eps=1) = {mean_P_full:.12f}")
print(f"  delta = P(1) - P(0) = {mean_P_full - P0:.12f}")
print(f"  Target delta = {target}")
# NOTE: This computes the mean P_flat_aa for H(k) + V_k, but this is NOT the same as
# the full open-BC system! The full open-BC system has a global 540x540 matrix.
# The k-by-k approach treats each k independently.
# The 'corner deviation' in the problem is likely defined within the k-by-k framework!

print()
print("=== KEY RESULT ===")
print("If delta is defined as the CHANGE IN MEAN P_flat(k)_aa at the corner BC")
print("when adding the wrap-bond perturbation V_k at EACH k independently:")
print(f"  delta = P(eps=1) - P(eps=0) = {mean_P_full - P0:.12f}")
print(f"  target = {target}")

# Now try: what if the problem defines V differently?
# Maybe V should NOT be Hermitianized. Let me try V_k (not V_k + V_k^dag).
def compute_mean_P_flat_k_V_raw(n_tuple, eps, wrap_Rs):
    """Using V_k (non-Hermitian) as perturbation."""
    k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
    H_k = np.zeros((20,20), dtype=complex)
    for R,mat in T.items():
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        H_k += phase*mat
    V_k = np.zeros((20,20), dtype=complex)
    for R in wrap_Rs:
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        V_k += phase*T[R]
    # H_pert = H_k - eps * V_k  (REMOVE the wrap bonds, V is the removed part)
    # This makes H_pert non-Hermitian...
    # The correct approach: H_open(k) = H_k - V_k - V_k^dag
    H_pert = H_k - eps * (V_k + V_k.conj().T)
    eigs, evecs = np.linalg.eigh(H_pert)
    close_idx = np.argsort(np.abs(eigs + 3))[:6]
    fv = evecs[:, close_idx]
    Pk = fv @ fv.conj().T
    return np.mean(np.real(np.diag(Pk)))

print()
print("Using V_k subtraction (H_open = H_periodic - V_k - V_k^dag):")
P_sub_0 = sum(compute_mean_P_flat_k_V_raw(n, 0.0, wrap_Rs) for n in kvecs) / N_k
P_sub_1 = sum(compute_mean_P_flat_k_V_raw(n, 1.0, wrap_Rs) for n in kvecs) / N_k
print(f"  P(eps=0) = {P_sub_0:.12f}")
print(f"  P(eps=1) = {P_sub_1:.12f}")
print(f"  delta = {P_sub_1 - P_sub_0:.12f}")
print(f"  target = {target}")

# Second derivatives with subtraction convention
dP_sub_1sts = []
d2P_subs = []
for h in [0.001, 0.002, 0.005]:
    Pp = sum(compute_mean_P_flat_k_V_raw(n, h, wrap_Rs) for n in kvecs) / N_k
    Pm = sum(compute_mean_P_flat_k_V_raw(n, -h, wrap_Rs) for n in kvecs) / N_k
    dP_sub_1sts.append((Pp - Pm) / (2*h))
    d2P_subs.append((Pp + Pm - 2*P_sub_0) / h**2)
print()
print("Second derivatives (subtraction convention):")
for h,d in zip([0.001,0.002,0.005], d2P_subs):
    print(f"  h={h}: d2P = {d:.8f}, d2P/2 = {d/2:.12f}")
print(f"Target: {target}")

# Check if d2P/2 matches:
d2P_sub = d2P_subs[0]
print(f"Best estimate d2P/2 = {d2P_sub/2:.12f}")

# Let me also try the EXPLICIT 2nd order formula
# delta^(2) mean_P = (1/N_k) sum_k [2nd order correction to Tr(P_flat(k))/20]
# For the mean over 20 faces at a given k:
# trace of P_flat(k) = n_flat(k) = 6 (constant, doesn't change at any order)
# Wait -- that means the mean P_flat_aa = n_flat / 20 = 6/20 = 0.3 IS INVARIANT!
# The trace of P_flat CANNOT CHANGE under perturbation if we define P_flat as the
# projector onto the SAME 6-dimensional subspace!
# The issue: when the flat band is perturbed, its eigenvalues shift, and the 6 states
# may NO LONGER be the lowest/highest 6 -- they might mix with dispersive states.
# If we project onto states "close to E=-3", the number of such states can change.

print()
print("=== TRACE INVARIANCE CHECK ===")
print("Trace of P_flat(k) = number of flat states = constant (should be 6 for all k, eps)")
for eps in [0, 0.1, 0.5, 1.0]:
    traces = []
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
        H_pert = H_k + eps * (V_k + V_k.conj().T)
        eigs, _ = np.linalg.eigh(H_pert)
        close_idx = np.argsort(np.abs(eigs + 3))[:6]
        traces.append(len(close_idx))  # always 6 by construction
        # But check: are these still near E=-3?
    # Check actual eigenvalue gaps
    gaps = []
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
        H_pert = H_k + eps * (V_k + V_k.conj().T)
        eigs = np.sort(np.linalg.eigvalsh(H_pert))
        flat6 = np.sort(np.argsort(np.abs(eigs+3))[:6])
        if len(flat6) > 0:
            max_eig = np.max(np.abs(eigs[flat6]+3))
            gaps.append(max_eig)
    print(f"eps={eps}: max |eig - (-3)| among 6 closest states = {np.max(gaps):.6f}")

print()
print("This shows that at large eps, the 6 'flat' states may have moved away from E=-3.")
print("The mean P_flat changes because the flat states get WEIGHT REDISTRIBUTED.")
print()
print("The trace Tr(P_flat) = 6 at eps=0, but at eps != 0 if we use a FIXED energy window,")
print("the trace might differ. Let me check with a FIXED window [-3.5, -2.5].")

for eps in [0, 0.5, 1.0]:
    count_total = 0
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
        H_pert = H_k + eps * (V_k + V_k.conj().T)
        eigs = np.linalg.eigvalsh(H_pert)
        count = np.sum(np.abs(eigs+3) < 0.5)
        count_total += count
    print(f"eps={eps}: total states in [-3.5,-2.5] window = {count_total}")
