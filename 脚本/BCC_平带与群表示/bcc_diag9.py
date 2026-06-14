"""
Re-examine the "rough" second-order formula from bcc_diag.py that gave 0.013445.
That formula used:
    G2VP = dv @ (g2_n[:,None] * V_dfl)   -- V_dfl = V_{flat->disp}, shape 14x6 -> dv @ ... shape 20x6
    diag_g2 = diag(G2VP @ fv.conj().T)   -- = diag of (20x6) @ (6x20) = 20x20
    dP2 = -diag_g2

This is:
  G2VP = sum_n phi_n * (1/(E0-En)^2) * <n|V|flat>
  The diagonal element = <face_a| G^2 V |flat> @ <flat|
  = sum_n phi_n[a] * (1/(E0-En)^2) * sum_m <n|V|flat_m> fv[a,m]
  = [P_disp V G^2 (P_flat)]_aa -- this is wrong, it's [sum_n G^2 (keto) <n|V|flat> @ fv^T]_aa

Actually the formula is: G2VP_aa = sum_m sum_n dv[a,n] * g_n^2 * V_fd[n,m] * fv[a,m]^conj
= [dv @ diag(g_n^2) @ V_fd @ fv^T]_aa

Let me re-derive what the correct formula for the second-order correction to P is.

KATO-BLOCH FORMULA for projector P(eps) where H(eps) = H0 + eps*V:
P^(2) = - P^(1) P^(1)_0 - P^(0) P^(1) P^(1)_0 ...

Actually the cleaner way: use the resolvent formula:
  P = 1/(2pi*i) contour integral of G = (E - H)^{-1}

To second order in V:
  P^(2) = P^(0) V G0^2 P^(0) V G0 - G0 V P^(0) V G0^2 P^(0) - G0^2 V P^(0) V G0 P^(0)
  + G0 V G0 V P^(0) G0^2 + ... (lots of terms in this approach)

Let me use a simpler approach: the REDUCED RESOLVENT method.
  G = G0 (in flat-subspace complement)
  For a projector P onto eigenvalue E0 (degenerate):

  P(eps) = P^(0) + eps*P^(1) + eps^2*P^(2) + ...

  The key formula (for diagonal elements) is derived from:
  [P(eps)]_aa = sum_m |<a|phi_m(eps)>|^2
  where |phi_m(eps)> are the perturbed flat-band states.

  To 2nd order in eps:
  [P(eps)]_aa = [P^(0)]_aa + eps*[P^(1)]_aa + eps^2*[P^(2)]_aa + ...

  [P^(1)]_aa = 2 Re sum_m <flat_m^(0)|a><a|flat_m^(1)>
  where |flat_m^(1)> = G0 V |flat_m^(0)>  (dispersive-band part)
  [G0 V |flat_m^(0)>]_a = sum_n dv[a,n] * g_n * sum_b V[a_b] fv[b,m]

Actually I want to try a COMPLETELY DIFFERENT approach: just numerically differentiate the exact P_flat(k)
projector with respect to perturbation strength eps at eps=0. This gives the EXACT first-order correction
without any formula ambiguity.

And for the second-order correction, compute (P(eps) + P(-eps) - 2*P(0)) / (2*eps^2).

This tests the full projector correction at the LOCAL (20x20) level for each k.
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
wrap_Rs = [(2,0,0), (0,2,0), (0,0,2)]
target = 0.013782448754

def compute_Pflat_at_k_eps(k_vec, eps, debug=False):
    """Compute P_flat diagonal at given k with perturbation strength eps."""
    H_k = np.zeros((20,20), dtype=complex)
    for R,mat in T.items():
        phase = np.exp(1j*np.dot(k_vec, np.array(R,float)))
        H_k += phase*mat

    V_k = np.zeros((20,20), dtype=complex)
    for R in wrap_Rs:
        phase = np.exp(1j*np.dot(k_vec, np.array(R,float)))
        V_k += phase*T[R]
    V_k_h = V_k + V_k.conj().T  # Hermitian perturbation (removing bonds = -V, but let's check)

    # The perturbation is H_corner = H_per - V_wrap
    # So V in the formula is -V_k_h (we subtract wrap bonds)
    H_total = H_k - eps * V_k_h

    eigs, evecs = np.linalg.eigh(H_total)

    if debug:
        print(f"  Eigenvalues at eps={eps}: {np.sort(eigs)[:8]}")

    # For eps=0, flat band is exactly at -3
    # For small eps, we need to identify the "flat band" eigenstates
    # Use window around -3 + eps*(expected shift)
    # At 1st order, flat band shifts by O(eps), at 2nd order by O(eps^2)

    if abs(eps) < 1e-10:
        flat_mask = np.abs(eigs + 3) < 1e-4
    else:
        # Need to track 6 flattest states robustly
        # Sort by eigenvalue and pick the 6 lowest energy states near -3
        sorted_idx = np.argsort(np.abs(eigs + 3))
        flat_mask = np.zeros(20, dtype=bool)
        flat_mask[sorted_idx[:6]] = True

    fv = evecs[:, flat_mask]  # 20 x 6
    P_diag = np.real(np.sum(fv * fv.conj(), axis=1))  # 20
    return P_diag

def compute_dP1_dP2_numerical(k_vec, eps=1e-3, debug=False):
    """Numerically compute 1st and 2nd order corrections to P_flat diagonal."""
    P0 = compute_Pflat_at_k_eps(k_vec, 0.0, debug)
    P_plus = compute_Pflat_at_k_eps(k_vec, eps, debug)
    P_minus = compute_Pflat_at_k_eps(k_vec, -eps, debug)

    dP1 = (P_plus - P_minus) / (2*eps)
    dP2 = (P_plus + P_minus - 2*P0) / (eps**2)
    return P0, dP1, dP2

print("=== Numerical finite-difference P_flat derivative analysis ===")
print()
print("Testing at Gamma k=(0,0,0):")
k_G = np.zeros(3)
P0, dP1, dP2 = compute_dP1_dP2_numerical(k_G, eps=1e-4)
print(f"  P0[9] = {P0[9]:.8f} (unperturbed)")
print(f"  dP1[9] = {dP1[9]:.8f}")
print(f"  dP2[9] = {dP2[9]:.8f}")
print(f"  P0 mean = {np.mean(P0):.8f}")
print(f"  dP1 mean = {np.mean(dP1):.8f} (should be 0)")
print(f"  dP2 mean = {np.mean(dP2):.8f}")
print()

print("Testing at BZ corner k=(1,1,1):")
k_C = np.array([2*np.pi/3, 2*np.pi/3, 2*np.pi/3])
P0, dP1, dP2 = compute_dP1_dP2_numerical(k_C, eps=1e-4)
print(f"  P0[9] = {P0[9]:.8f}")
print(f"  dP1[9] = {dP1[9]:.8f}")
print(f"  dP2[9] = {dP2[9]:.8f}")
print(f"  P0 mean = {np.mean(P0):.8f}")
print(f"  dP1 mean = {np.mean(dP1):.8f}")
print(f"  dP2 mean = {np.mean(dP2):.8f}")
print()

# Now sum over all k-points
print("Summing over all 27 k-points:")
dP1_total = np.zeros(20)
dP2_total = np.zeros(20)
P0_total = np.zeros(20)
for n_tuple in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
    P0, dP1, dP2 = compute_dP1_dP2_numerical(k, eps=1e-4)
    dP1_total += dP1
    dP2_total += dP2
    P0_total += P0

dP1_total /= N_k
dP2_total /= N_k
P0_total /= N_k

print(f"  P0 mean over k [9]: {P0_total[9]:.8f} (should be ~0.3)")
print(f"  dP1 sum/N_k [9]: {dP1_total[9]:.12f}")
print(f"  dP2 sum/N_k [9]: {dP2_total[9]:.12f}")
print(f"  Target: {target}")
print()
print(f"  dP1+dP2 at face 9: {dP1_total[9]+dP2_total[9]:.12f}")
print()
print(f"  All faces dP1: {np.round(dP1_total, 6)}")
print(f"  All faces dP2: {np.round(dP2_total, 6)}")
print()
print(f"  Check: dP1 mean = {np.mean(dP1_total):.10f} (should be 0)")
print(f"  Check: dP2 mean = {np.mean(dP2_total):.10f}")

# Maybe delta = (1/N_k) * sum_k [correction to face-9 P_flat at the BZ-corner k?]
# Let me try: delta = dP1[face9] + dP2[face9] for k=(1,1,1) only
print()
k_C = np.array([2*np.pi/3, 2*np.pi/3, 2*np.pi/3])
P0_C, dP1_C, dP2_C = compute_dP1_dP2_numerical(k_C, eps=1e-4)
print(f"  At k=(1,1,1): dP1[9]={dP1_C[9]:.8f}, dP2[9]={dP2_C[9]:.8f}, total={dP1_C[9]+dP2_C[9]:.8f}")

# Also check per k-type
print()
def bz_label(n):
    return ['G','E','F','C'][sum(1 for x in n if x!=0)]

for label in ['G','E','F','C']:
    ks = [n for n in kvecs if bz_label(n)==label]
    dP1_sum = np.zeros(20)
    dP2_sum = np.zeros(20)
    for n in ks:
        k = np.array([2*np.pi*ni/3.0 for ni in n])
        P0, dP1, dP2 = compute_dP1_dP2_numerical(k, eps=1e-4)
        dP1_sum += dP1
        dP2_sum += dP2
    if len(ks) > 0:
        print(f"  {label} ({len(ks)} k-pts): dP1_sum[9]={dP1_sum[9]:.8f}, dP2_sum[9]={dP2_sum[9]:.8f}")
        print(f"    total[9]={dP1_sum[9]+dP2_sum[9]:.8f}, mean dP1[9]={dP1_sum[9]/len(ks):.8f}")

# Test: maybe delta = (sum_k dP2[9]) / 27 but with V defined as removing bonds (not adding back)
# Try flipping the sign of V
print("\n--- Trying V with OPPOSITE sign (adding bonds instead of removing) ---")
# Already done implicitly: H_corner = H_per - V, so V removes bonds.
# If we define V = +wrap bonds (adding back), then H_corner = H_per - V means same thing.
# But if delta is defined as H_open - H_per = +V (open has MORE bonds somewhere?),
# then we need to check what "corner body-center missing bonds" means.
print("V is defined as REMOVING the 3 wrap-around bonds, so H = H0 - eps*V_k_h")
print("This means at eps=1, the corner BC is simulated by removing those bonds.")
print("dP1 and dP2 are derivatives of P_flat with respect to eps in H = H0 - eps*V.")
print(f"At eps=1: P_flat changes by dP1 + dP2 + ... at each k.")
print(f"Sum over k of dP1/N_k = {dP1_total[9]:.8f} for face 9")
print(f"Sum over k of dP2/N_k = {dP2_total[9]:.8f} for face 9")
print(f"Sum: {dP1_total[9]+dP2_total[9]:.8f}")
print(f"Target: {target}")
