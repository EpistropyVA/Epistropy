"""
Implement EXACTLY the algorithm from the problem statement:
1. For each k, find 6 flat-band eigenvectors of H(k)
2. Build V_local: the REMOVED-BOND perturbation for corner BC
3. V_flat(k) = P_flat(k) V_local P_flat(k) -- 6x6
4. Second-order: for each dispersive eigenvalue, compute G_n = 1/(-3 - lambda_n)
   sum_n |<n|V|flat>|^2 G_n -- this is a 6x6 matrix (effective H^(2) in flat subspace)
5. The corner deviation delta = something from these matrices, summed over k

Key insight I've been missing: the problem says V_flat(k) = P_flat(k) V P_flat(k)
is a 6x6 matrix, AND the second-order correction is ALSO projected back.
The "corner deviation" is the change in Tr[P_flat @ P_corner] where P_corner
is the projector onto "corner faces" (faces 9 and 16 specifically, or all type-B faces).

Let me compute:
- For each k: compute V_flat(k) eigenvalues (first order flat-band splitting)
- Compute H_eff(k) = V_flat(k) + H^(2)_flat(k) (total effective H in flat subspace)
- The NEW flat-band projector in the k-subspace is just projection onto all 6 eigenvectors
  of H_eff(k) -- but that's still 6 states, so P_flat is unchanged at first order in norm.
- The PHYSICAL quantity: the mean P_flat(k)_aa over the "corner-type" faces at each k,
  as a function of the perturbation strength.

Actually, re-reading: "Extract per-face diagonal elements and compute the corner deviation"
suggests that after projecting V into flat subspace and computing the second-order shift,
we evaluate Tr(delta P * face-projector) = sum_a delta P_aa where a ranges over corner faces.

The corner deviation = (1/N_k) * sum_k * (1/6) * Tr_flat[V_flat(k)] -- let me check this!
Tr_flat[V_flat] = Tr[P_flat V P_flat] = Tr[P_flat V] (cyclic property)
= Tr[V @ P_flat] = sum_a P_flat[aa] V[aa] + sum_{a!=b} P_flat[ab] V[ba]
This is the first-order energy shift of the flat band at k.
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

def bz_label(n):
    return ['G','E','F','C'][sum(1 for x in n if x!=0)]

print("=== Algorithm from problem statement ===")
print()
print("For each k:")
print("  1. Find 6 flat-band eigenvectors of H(k)")
print("  2. Build V_local = 20x20 Hermitian wrap-bond perturbation")
print("  3. V_flat(k) = P_flat(k) V_local P_flat(k) -- 6x6 projected")
print("  4. H^(2)_flat(k) = sum_n <flat|V|n><n|V|flat> / (-3 - lambda_n) -- second order Lowdin")
print("  5. H_eff(k) = V_flat(k) + H^(2)_flat(k) -- total effective in flat subspace")
print("  6. Diagonalize H_eff(k) to get perturbed flat states")
print("  7. Compute P_flat_perturbed(k) = new eigenvectors @ new eigenvectors.T")
print("  8. Corner deviation delta = change in P_flat(k)_aa at corner faces")
print()

# Key: when we diagonalize H_eff(k), the FLAT-BAND PROJECTOR P_flat does NOT change
# (we're just mixing within the 6-dim subspace). The FACE WEIGHTS change only if
# we look at SPECIFIC faces, not the mean.
#
# BUT: the problem says "corner deviation on each of the 8 corner body-centers"
# This suggests something IS computed per-BC.
#
# Let me try the exact algorithm: compute the sum over k of
# Tr_flat[H_eff(k)] * (weighting by face-type)
# This gives the first-order energy shift of the flat band.
#
# For the SPECTRAL PROJECTOR onto the flat band, the relevant quantity is:
# sum_k P_flat(k)_aa for face a
# = unperturbed value (0.3 for mean, varies by face)
# The CHANGE in this due to perturbation = 0 (trace invariant in flat subspace)
# UNLESS the perturbation mixes flat states with dispersive states,
# changing WHICH states are "flat".

# The problem must be computing something specific. Let me try:
# delta = (1/N_k) * sum_k * [Tr_flat(H^(2)_flat(k)) / 6 * (some face weight)]
# = the second-order energy correction averaged over flat states, weighted by face weights

# Actually: let me try computing the TRACE of H_eff(k) restricted to face 9 weight.
# This would be: sum_k (fv[:,:]^* @ H_eff(k) @ fv)[face9, face9] / N_k
# = sum_k [P_flat(k) H_eff(k)]_{9,9} / N_k
# = sum_k fv[9,:] @ H_eff_k @ fv[9,:]^* / N_k   (local contribution to face 9)

# The "second-order correction to the flat-band weight at face 9" =
# sum_k (1/N_k) sum_m [|<face9|flat_m(k)>|^2 * epsilon_m^(2)(k)]
# where epsilon_m^(2) = eigenvalue of H^(2)_flat(k) for flat state m.

# But this is the second-order energy correction to the flat band weighted by face-9 spectral weight.
# This is different from the spectral projector correction.

# Let me just compute this quantity directly:

delta_face9_energy = 0.0
delta_face9_Heff = 0.0
delta_mean_energy = 0.0
delta_mean_Heff = 0.0

H2_flat_by_k = {}
for n_tuple in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
    H_k = np.zeros((20,20), dtype=complex)
    for R,mat in T.items():
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        H_k += phase*mat
    eigs, evecs = np.linalg.eigh(H_k)
    flat_mask = np.abs(eigs+3) < 1e-4
    disp_mask = ~flat_mask
    fv = evecs[:, flat_mask]   # 20 x 6
    dv = evecs[:, disp_mask]   # 20 x 14
    de = eigs[disp_mask]       # 14

    V_k = np.zeros((20,20), dtype=complex)
    for R in wrap_Rs:
        phase = np.exp(1j*np.dot(k, np.array(R,float)))
        V_k += phase*T[R]
    V_k_h = V_k + V_k.conj().T  # Hermitian perturbation

    # V_flat(k) = P_flat V P_flat restricted to 6x6
    V_flat_6 = fv.conj().T @ V_k_h @ fv  # 6 x 6

    # H^(2)_flat(k) = sum_n <flat|V|n><n|V|flat> / (E0 - lambda_n)
    # V_fd = dv.T @ V_k_h @ fv  -- coupling flat to dispersive
    V_fd = dv.conj().T @ V_k_h @ fv  # 14 x 6
    g_n = 1.0 / (-3.0 - de)  # 14
    H2_flat = V_fd.conj().T @ (g_n[:,None] * V_fd)  # 6 x 6 (Lowdin second-order)
    # = sum_n |<n|V|flat>|^2 / (E0 - En) projected to 6x6 flat subspace
    # Actually: H^(2)[m,m'] = sum_n <flat_m|V|n> (1/(E0-En)) <n|V|flat_m'>
    # = V_fd.H @ diag(g_n) @ V_fd

    H_eff_k = np.real(V_flat_6 + H2_flat)  # total effective H in flat subspace (should be real)
    H2_flat_by_k[n_tuple] = H2_flat

    # The "corner deviation" at face 9:
    # P_flat(k)[9,9] = fv[9,:] @ fv[9,:].T = sum_m |fv[9,m]|^2 = unperturbed face-9 weight
    # After perturbation, the face weights get modified by the change in eigenvectors.
    # But within the flat subspace, the projector P_flat = fv @ fv.T is INVARIANT
    # under unitary rotations within the flat subspace.
    # The ONLY way face-9 weight changes is if flat states mix with dispersive states.
    # That's captured by the first-order correction dP1[9] = 0.022430 (computed earlier).
    # The second-order change in P comes from FURTHER mixing with dispersive states.
    #
    # HOWEVER, there's another contribution: the DIAGONAL of H_eff(k) restricted to face 9
    # gives the ENERGY SHIFT of face-9 spectral weight. This is:
    # [P_flat(k) H_eff(k)]_{9,9} = fv[9,:] @ H_eff_k @ fv[9,:].T
    # = the "face-9 weighted eigenvalue correction"
    P_flat_aa = np.real(np.diag(fv @ fv.conj().T))  # should be exactly the P_flat diagonal

    # The problem might be: delta = (1/N_k) * sum_k [P_flat(k)]_{9,9} * [Tr_flat(H^(2)) / 6]
    # = P_flat_mean * (sum_k Tr_flat(H^(2)) / (6*N_k))
    # = 0.3 * (mean second-order energy shift per flat state per k)
    H2_diag = np.real(np.diag(H2_flat))  # 6 values
    mean_H2_diag = np.mean(H2_diag)

    # Or: delta = (1/N_k) sum_k [P_flat(k)_{9,9} * mean_H2 / E_gap]?

    # [fv @ H_eff_k @ fv.T]_{9,9} = sum_{m,m'} fv[9,m]* H_eff[m,m'] fv[9,m']
    face9_Heff = np.real(fv[9,:].conj() @ H_eff_k @ fv[9,:])
    face9_H2 = np.real(fv[9,:].conj() @ H2_flat @ fv[9,:])

    delta_face9_energy += face9_H2
    delta_face9_Heff += face9_H2  # same thing

    # Mean over all 20 faces
    all_face_H2 = np.array([np.real(fv[a,:].conj() @ H2_flat @ fv[a,:]) for a in range(20)])
    delta_mean_energy += np.mean(all_face_H2)

delta_face9_energy /= N_k
delta_mean_energy /= N_k

print(f"sum_k [P_flat(k) H^(2)_flat(k)]_{{9,9}} / N_k = {delta_face9_energy:.12f}")
print(f"sum_k mean_face [P_flat(k) H^(2)_flat(k)]_{{aa}} / N_k = {delta_mean_energy:.12f}")
print(f"Target: {target}")
print()

# Also try: the trace of H^(2)_flat(k) summed over k and faces
print("For each k-type, mean face-9 H^(2) contribution:")
for label in ['G','E','F','C']:
    ks_of_type = [n for n in kvecs if bz_label(n)==label]
    vals = []
    for n_tuple in ks_of_type:
        k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
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
        V_fd = dv.conj().T @ V_k_h @ fv
        g_n = 1.0 / (-3.0 - de)
        H2_flat = V_fd.conj().T @ (g_n[:,None] * V_fd)
        face9_H2 = np.real(fv[9,:].conj() @ H2_flat @ fv[9,:])
        Tr_H2 = np.real(np.trace(H2_flat))
        face9_P = np.real(np.dot(fv[9,:].conj(), fv[9,:]))
        vals.append((n_tuple, face9_H2, Tr_H2, face9_P))
    print(f"\n{label} ({len(ks_of_type)} k-pts):")
    for n,f9,tr,p9 in vals[:3]:
        frac_f9 = Fraction(f9).limit_denominator(1000)
        frac_tr = Fraction(tr).limit_denominator(1000)
        print(f"  k={n}: H2_f9={f9:.8f}={frac_f9}, Tr_H2={tr:.6f}={frac_tr}, P_flat_f9={p9:.6f}")

# Check: what if delta = (sum_k P_flat_f9 * Tr_H2/6) / N_k?
delta_candidate = 0.0
for n_tuple in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
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
    V_fd = dv.conj().T @ V_k_h @ fv
    g_n = 1.0 / (-3.0 - de)
    H2_flat = V_fd.conj().T @ (g_n[:,None] * V_fd)
    Tr_H2 = np.real(np.trace(H2_flat))
    p9 = np.real(np.dot(fv[9,:].conj(), fv[9,:]))
    delta_candidate += p9 * Tr_H2 / 6.0
delta_candidate /= N_k
print(f"\nCandidate: sum_k P_flat_f9 * Tr_H2/6 / N_k = {delta_candidate:.12f}")
print(f"Target: {target}")

# Maybe it's just Tr_H2 / 20 summed over k?
Tr_H2_total = 0.0
for n_tuple in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
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
    V_fd = dv.conj().T @ V_k_h @ fv
    g_n = 1.0 / (-3.0 - de)
    H2_flat = V_fd.conj().T @ (g_n[:,None] * V_fd)
    Tr_H2_total += np.real(np.trace(H2_flat))

Tr_H2_total /= N_k
print(f"\nTr_H2 / N_k = {Tr_H2_total:.12f}")
print(f"Tr_H2 / N_k / 20 (per face) = {Tr_H2_total/20:.12f}")
print(f"Tr_H2 / N_k / 6 (per flat state) = {Tr_H2_total/6:.12f}")
print(f"Target: {target}")
print()
print(f"Ratio Tr_H2/20/target = {(Tr_H2_total/20)/target:.6f}")
print(f"Ratio Tr_H2/6/target = {(Tr_H2_total/6)/target:.6f}")
