"""Diagnostic: search for delta=0.013782448754 in all projector quantities."""
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
keep_Rs = [(1,0,0), (0,1,0), (0,0,1)]
target = 0.013782448754

def bz_label(n):
    return ['G','E','F','C'][sum(1 for x in n if x!=0)]

# Per-k: compute P_flat(k) and various V projections
all_P_flat = {}  # k -> 20-array P_flat(k)_aa
all_V_flat_diag = {}  # k -> diag of V_flat(k) in 20-space
all_dP1 = {}  # k -> first-order correction

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

    # Flat projector diagonal
    Pk = fv @ fv.conj().T
    all_P_flat[n] = np.real(np.diag(Pk))

    # V_k (wrap-bond contribution to H(k))
    V_k = np.zeros((20,20), dtype=complex)
    for R in wrap_Rs:
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        V_k += phase*T[R]
    # Hermitian version
    V_k_h = V_k + V_k.conj().T

    # Diagonal of V in local 20-space (diagonal part of V_k_h)
    V_diag = np.real(np.diag(V_k_h))

    # Diagonal of P_flat V_k_h P_flat (first order within flat subspace)
    PVP = fv.conj().T @ V_k_h @ fv  # 6x6
    PVP_diag_in_full = np.real(np.diag(fv @ PVP @ fv.conj().T))  # 20-array
    all_V_flat_diag[n] = PVP_diag_in_full

    # First-order correction from dispersive: delta_P^(1) = G V P + P V G
    g_n = 1.0 / (-3.0 - de)
    V_dfl = dv.conj().T @ V_k_h @ fv  # (14x6)
    GVP = dv @ (g_n[:,None] * V_dfl)  # 20x6
    dP1 = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
    all_dP1[n] = dP1

# Now aggregate and search
print("=== SEARCHING FOR DELTA = 0.013782448754 ===")
print()

# 1. Mean over k of first-order correction at each face
dP1_mean = np.mean([all_dP1[n] for n in kvecs], axis=0)
print("Mean first-order correction per face:")
for i,v in enumerate(dP1_mean):
    if abs(v - target) < 1e-6:
        print(f"  MATCH face {i}: {v:.12f}")
print(f"  mean = {np.mean(dP1_mean):.8f}, max_abs = {np.max(np.abs(dP1_mean)):.8f}")

# 2. Mean over k of V_flat diagonal
Vfl_mean = np.mean([all_V_flat_diag[n] for n in kvecs], axis=0)
print("\nMean P_flat V P_flat diagonal per face:")
for i,v in enumerate(Vfl_mean):
    if abs(v - target) < 1e-6:
        print(f"  MATCH face {i}: {v:.12f}")
print(f"  mean = {np.mean(Vfl_mean):.8f}")

# 3. Per-face: for each BZ corner k, compute first-order correction
corner_ks = [n for n in kvecs if bz_label(n)=='C']
dP1_corner = np.mean([all_dP1[n] for n in corner_ks], axis=0)
print("\nFirst-order correction at BZ-corner k-pts, per face:")
for i,v in enumerate(dP1_corner):
    if abs(v - target) < 1e-6:
        print(f"  MATCH face {i}: {v:.12f}")

# 4. What if delta is (P_flat(k_corner) - P_flat(k_Gamma)) / some_normalization?
P_corner_k = np.mean([all_P_flat[n] for n in corner_ks], axis=0)
P_gamma_k = all_P_flat[(0,0,0)]
diff = P_corner_k - P_gamma_k
print("\nP_flat(corner k) - P_flat(Gamma) per face:")
for i,v in enumerate(diff):
    if abs(v - target) < 1e-6:
        print(f"  MATCH face {i}: {v:.12f}")
print(f"  values: {diff}")

# 5. What if delta = mean(P_flat(corner k) - overall mean per face)?
P_overall = np.mean([all_P_flat[n] for n in kvecs], axis=0)
diff2 = P_corner_k - P_overall
print("\nP_flat(corner k avg) - overall mean per face:")
for i,v in enumerate(diff2):
    if abs(v - target) < 1e-6:
        print(f"  MATCH face {i}: {v:.12f}")

# 6. Trace of P_{-3} restricted to corner BC in open-BC system
# We computed: 0.1557014238 (single corner removal)
# delta = 0.1557014238 - ??
single_corner_trace_mean = 0.1557014238
print(f"\nSingle corner removal mean P_aa: {single_corner_trace_mean}")
print(f"Difference from 0.3: {single_corner_trace_mean - 0.3}")

# 7. Maybe it's about the NUMBER OF FLAT STATES per BC in open system?
# Periodic: 6 flat states per BC (on average)
# After single corner removal: 156 flat states, 27 BCs -> 5.78 per BC
# But the corner BC has some specific number: check how much trace it holds
# trace at corner BC = 20 * 0.1557 = 3.114 flat states
# Average = 156/27 = 5.778
# delta = 3.114 / 5.778 = 0.5388... not matching

# 8. Try: 2nd order correction
# sum_k sum_{n disp} |<n|V|flat>|^2 / (E_flat - E_n)
# restricted to corner faces
print()
print("Computing second-order correction...")
dP2_per_face_per_k = {}
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
    g2_n = g_n**2
    V_dfl = dv.conj().T @ V_k_h @ fv
    G2VP = dv @ (g2_n[:,None] * V_dfl)

    # Second-order correction: -[P_flat V G^2 V P_flat]_aa diag
    # This is the negative of the derivative of the self-energy
    # For the PROJECTOR (not energy):
    # delta_P^(2) needs more care; skip for now
    # Just compute G^2 VP diag for reference
    diag_g2 = np.real(np.diag(G2VP @ fv.conj().T))
    dP2_per_face_per_k[n] = -diag_g2  # rough 2nd order

dP2_mean = np.mean([dP2_per_face_per_k[n] for n in kvecs], axis=0)
print(f"Rough 2nd order correction mean per face: {np.mean(dP2_mean):.8f}")
total_correction = dP1_mean + dP2_mean
print(f"Total (1st+2nd order) mean per face: {np.mean(total_correction):.8f}")
print(f"Face 9 total: {total_correction[9]:.12f}")
print(f"Mean total: {np.mean(total_correction):.12f}")
print(f"Target: {target}")

# 9. Check specific rational combinations of the per-face values at BZ types
from fractions import Fraction
# Face type B-special (9,16): G=1/2, E=69/160, F varies, C varies
# Let me compute the weighted VARIANCE of face 9 across k-types
f9_G = Fraction(1,2)
f9_E = Fraction(69,160)
# For face k: 8 k-pts give two values (151/330 x 4 and 138/385 x 4? Let me recount)
# Actually face k-pts: from earlier:
# k=(0,1,1): f9=151/330; k=(0,1,2): 138/385; k=(0,2,2): 151/330
# k=(1,0,1): 151/330; k=(1,0,2): 138/385; k=(1,1,0): 151/330; k=(1,2,0): 138/385
# k=(2,0,1): 138/385; k=(2,0,2): 151/330; k=(2,1,0): 138/385; k=(2,2,0): 151/330
# k=(0,2,1): 138/385 -- checking pattern
face_k_f9 = {}
for n in kvecs:
    if bz_label(n) == 'F':
        face_k_f9[n] = Fraction(all_P_flat[n][9]).limit_denominator(10000)

print("\nFace-9 at F-type k-pts:")
unique_vals = set(face_k_f9.values())
for v in sorted(unique_vals):
    count = sum(1 for x in face_k_f9.values() if x==v)
    print(f"  {v} = {float(v):.8f}: occurs {count} times")

# So among 12 face k-pts: 6 give 151/330 and 6 give 138/385
f9_F_hi = Fraction(151,330)
f9_F_lo = Fraction(138,385)
f9_F_avg = (6*f9_F_hi + 6*f9_F_lo) / 12

f9_C_hi = Fraction(3,5)
f9_C_lo = Fraction(8,21)
# Among 8 corner k-pts: 2 give 3/5 and 6 give 8/21
f9_C_avg = (2*f9_C_hi + 6*f9_C_lo) / 8

print(f"\nf9 summary:")
print(f"  G: {f9_G} = {float(f9_G):.8f}")
print(f"  E: {f9_E} = {float(f9_E):.8f}")
print(f"  F_avg: {f9_F_avg} = {float(f9_F_avg):.8f}")
print(f"  C_avg: {f9_C_avg} = {float(f9_C_avg):.8f}")

# Grand average of face 9:
f9_grand = (1*f9_G + 6*f9_E + 12*f9_F_avg + 8*f9_C_avg) / 27
print(f"  Grand avg: {f9_grand} = {float(f9_grand):.10f}")

# Deviation of C_avg from grand avg:
f9_dev = f9_C_avg - f9_grand
print(f"  C_avg - grand_avg: {f9_dev} = {float(f9_dev):.10f}")
print(f"  Target: {target}")
# How about face 9 at k=(1,1,1) (the special corner) minus grand avg?
f9_dev2 = f9_C_hi - f9_grand
print(f"  C_hi - grand_avg: {f9_dev2} = {float(f9_dev2):.10f}")
# (3/5 - grand):
# grand = (1/2 + 6*69/160 + 6*151/330 + 6*138/385 + 2*3/5 + 6*8/21) / 27
# Let me compute exactly
f9_grand_exact = (1*Fraction(1,2) + 6*Fraction(69,160) +
                  6*Fraction(151,330) + 6*Fraction(138,385) +
                  2*Fraction(3,5) + 6*Fraction(8,21)) / 27
print(f"  Grand avg exact: {f9_grand_exact} = {float(f9_grand_exact):.12f}")
print(f"  Match with computed mean: {abs(float(f9_grand_exact) - np.mean([all_P_flat[n][9] for n in kvecs])) < 1e-10}")

# Now: the 'corner deviation' in perturbation theory
# If the PERTURBATION selectively enhances weight at face 9 at BZ corner k-pts,
# and if delta is the contribution from those k-pts to the overall face-9 weight excess:
# delta_face9 = (1/27) * sum_k [P_flat(k)_aa_9 - grand_avg_9]
#             = (1/27) * sum_k P_flat(k)_aa_9 - grand_avg_9  = 0  (by definition)
# So that's zero.

# Let me try yet another angle: the problem says 'corner deviation on each of the 8 corner body-centers'
# and the algorithm involves projecting V into the flat band at each k.
#
# Perhaps: at BZ corner k=(1,1,1) (which corresponds to the real-space corner body-center (0,0,0)
# via the X-point / zone-boundary correspondence), the PERTURBATION V has a specific first-order
# effect on P_flat(k)_aa at the corner faces.
#
# Let me compute [P_flat(k) V_k P_flat(k)]_aa / E_correction for k_corner:
print("\n=== At BZ corner k=(1,1,1), V_flat matrix ===")
n = (1,1,1)
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

PVP = fv.conj().T @ V_k_h @ fv  # 6x6
print("PVP (6x6) at k=(1,1,1):")
print(np.round(np.real(PVP), 6))
ev_pvp, evec_pvp = np.linalg.eigh(PVP)
print(f"PVP eigenvalues: {ev_pvp}")

# The FIRST-ORDER correction to P_flat at EACH FACE is:
# delta_P^(1)_aa = (GVP)_aa (from dispersive coupling)
# but ALSO the INTRA-FLAT correction to P:
# The intra-flat correction to individual face weights comes from
# the rotation of flat-band eigenvectors induced by PVP.
# New flat eigenvectors = fv @ evec_pvp (rotated within flat subspace)
# This doesn't change P_flat = fv @ fv^dag (the projector is invariant under unitary rotation within subspace)

g_n = 1.0 / (-3.0 - de)
V_dfl = dv.conj().T @ V_k_h @ fv
GVP = dv @ (g_n[:,None] * V_dfl)
dP1_k111 = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
print(f"\ndP1 at k=(1,1,1) per face:")
for i,v in enumerate(dP1_k111):
    f=Fraction(v).limit_denominator(10000)
    print(f"  face {i}: {v:.8f} ~ {f}")
print(f"  face 9: {dP1_k111[9]:.12f}")
print(f"  target: {target}")
print(f"  face 9 / 27: {dP1_k111[9]/27:.12f}")
