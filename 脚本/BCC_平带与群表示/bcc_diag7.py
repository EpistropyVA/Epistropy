"""
Correct second-order projector formula and search for delta at face 9.
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

# For each k, compute first AND second order corrections to P_flat(k)_aa at face 9
# using the Kato-Bloch formula.
#
# delta^(1) P[a,a] = 2 Re sum_{n disp} <a|G(n)|flat> V <flat|a> G_n
# where I use: [G V P_flat + P_flat V G]_aa = 2 Re [sum_n G_n <a|n><n|V|P_flat|a>]
# = 2 Re sum_n G_n dv[a,n]* (dv[:,n].T @ V_k_h @ fv)[n,:] . fv[a,:]
#
# delta^(2) P[a,a] = [G V G V P + P V G V G]_aa - [G V P V G]_aa
# = 2 Re [sum_{n,m disp} G_n G_m <a|n><n|V|m><m|V|flat>flat[a]]  -- first term
#   - sum_{n disp} G_n^2 |<a|n>|^2 P_aa  -- normalization correction
# Wait, the standard formula (Kato):
# P^(2) = G V G V P + P V G V G - G V P V G
# [P^(2)]_aa = [G V G V P]_aa + [P V G V G]_aa - [G V P V G]_aa
# = 2 Re[G V G V P]_aa - [G V P V G]_aa

# [G V G V P]_aa = sum_{n,m disp} G_n G_m dv[a,n]* (dv[:,n].T V_k dv[:,m]) (dv[:,m].T V_k fv) fv[a,:]
# This is a (flat->disp->disp->flat->face) chain.

# [G V P V G]_aa = sum_{n,m disp} G_n G_m dv[a,n]* (dv[:,n].T V_k fv) (fv.T V_k dv[:,m]) dv[a,m]
# = sum_{n,m disp} G_n G_m conj(dv[a,n]) (V_dfl[n,:] * fv[a,:]) . (V_dfl[m,:] * fv[a,:]) * dv[a,m]
# = |<a|G V P|a>|^2 (scalar) -- no, that's not right either.

# Let me be more careful:
# G = sum_{n disp} |n><n| G_n
# [G V G V P]_aa = sum_n sum_m sum_p G_n G_m <a|n> <n|V|m> <m|V|p> <p|P|a>
#               = sum_{n,m disp, p flat} G_n G_m <a|n> <n|V|m> <m|V|p> <p|a>
#   (where <p|a> = fv[a,p] since |p> are flat-band eigenstates)
# [G V P V G]_aa = sum_n sum_p sum_m G_n G_m <a|n> <n|V|p> <p|P|a> -> wait need to be careful
# G V P V G = sum_{n,m disp} |n> G_n <n|V| P |V|m> G_m <m|
# [G V P V G]_aa = sum_{n,m disp} G_n G_m <a|n> <n| V P V |m> <m|a>
#               = sum_{n,m disp} G_n G_m <a|n> (V_k @ fv @ fv.T @ V_k)[n,m] <m|a>
#               where the indices n,m are in the dispersive sector.
# Let's define:
# A_disp = dv.T @ V_k_h @ fv @ fv.T @ V_k_h @ dv  (dispersive sector of G V P V G / G^2)
# [G V P V G]_aa = sum_{n,m disp} G_n G_m dv[a,n].conj() A_disp[n,m] dv[a,m]
# = (G_n * dv[a,n]).T @ A_disp @ (G_n * dv[a,n])
# where we combine n,m indices properly.

# This is the correct second-order formula.
# Let me implement this.

dP1_all_faces = np.zeros(20)
dP2_all_faces = np.zeros(20)

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
    V_k_h = V_k + V_k.conj().T

    g_n = 1.0 / (-3.0 - de)   # 14 (Green's function denominators)

    # V coupling matrices
    V_fd = dv.conj().T @ V_k_h @ fv   # dispersive-to-flat: 14 x 6
    V_dd = dv.conj().T @ V_k_h @ dv   # dispersive-to-dispersive: 14 x 14

    # G V P_flat:  sum_n G_n |n><n| V_k P_flat = dv @ diag(g_n) @ V_fd
    GVP = dv @ (g_n[:,None] * V_fd)   # 20 x 6

    # First-order: [delta P^(1)]_aa = 2 Re sum_m GVP[a,m] * fv[a,m]*
    dP1 = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))

    # [G V G V P]_aa:
    # = sum_{n,m disp, p flat} G_n G_m <a|n> <n|V_k|m> <m|V_k|p> <p|a>
    # = sum_a: (G * dv[a,:]) . V_dd . (G * V_fd) . fv[a,:]
    # Let me define:
    # GVGVP[a, p_flat] = sum_{n,m disp} G_n dv[a,n]* V_dd[n,m] G_m V_fd[m,p]
    # = [dv @ (G[:,None]*V_dd @ G[:,None]*V_fd)][a, p]   -- need to be careful
    # Actually:
    # sum_{n,m} dv[a,n].conj() G_n V_dd[n,m] G_m V_fd[m,p]
    # = (G_n * dv[a,:].conj()) . V_dd . (G_m * V_fd[:,p])
    # For all faces a and all flat p simultaneously:
    # GVGVP = conj(dv) @ diag(g) @ V_dd @ diag(g) @ V_fd  -- but this is (20 x 14) @ (14 x 14) @ (14 x 6)
    GVGVP = dv.conj() @ (g_n[:,None] * V_dd @ (g_n[:,None] * V_fd))  # 20 x 6
    # Wait: dv.conj() has shape (20 x 14), g_n * V_dd @ g_n * V_fd has shape (14 x 6)
    # Let me check dimensions:
    # g_n[:,None] * V_dd: 14 x 14 (broadcasting g_n as rows)
    # (g_n[:,None] * V_dd) @ (g_n[:,None] * V_fd): (14 x 14) @ (14 x 6) = 14 x 6
    # dv.conj() @ ...: (20 x 14) @ (14 x 6) = 20 x 6
    # [G V G V P]_aa = sum_p fv[a,p].conj() * GVGVP[a,p]
    term1 = np.sum(fv.conj() * GVGVP, axis=1)  # 20 (complex)
    # [P V G V G]_aa = conj([G V G V P]_aa) (since P is Hermitian and everything is Hermitian)
    # = 2 Re term1
    GVGVP_diag = 2.0 * np.real(term1)   # 20

    # [G V P V G]_aa:
    # = sum_{n,m disp} G_n G_m <a|n> <n|V P V|m> <m|a>
    # V P V in dispersive block: V_fd @ fv.T @ V_fd.T = V_fd @ V_fd.H (since fv.T @ V_fd.T = V_fd.H)
    # Wait: [V P V]_{nm} (in dispersive-dispersive block) = sum_p V_fd[n,p] * V_fd[m,p]*
    # = (V_fd @ V_fd.H)[n,m]  -- but V_fd is (14x6) so V_fd @ V_fd.H is 14x14
    VPV_dd = V_fd @ V_fd.conj().T   # 14 x 14

    # [G V P V G]_aa = sum_{n,m} G_n G_m dv[a,n].conj() VPV_dd[n,m] dv[a,m]
    # = (G_n * dv[a,:].conj()) . VPV_dd . (G_n * dv[a,:])  -- for each face a
    # = sum_n,m (g_n * dv[a,n]*) VPV_dd[n,m] (g_m * dv[a,m])
    # = diag of dv.conj() @ diag(g) @ VPV_dd @ diag(g) @ dv.T
    # For all faces simultaneously:
    # GVPVG_aa[a] = (g_n * dv[a,:].conj()) @ VPV_dd @ (g_n * dv[a,:])
    # = sum_a elementwise:
    GVdv = g_n[:,None] * dv.conj().T   # 14 x 20 (each row scaled by G_n)
    # VPV_dd @ GVdv: 14 x 20
    # Then dv.T @ (VPV_dd @ GVdv): sum over dispersive indices
    # [G V P V G]_aa = dv[a,:] . (VPV_dd . (g_n * dv[a,:].conj()))
    # Actually:
    # [G V P V G]_aa = sum_{n,m} G_n G_m dv[a,n]* VPV_dd[n,m] dv[a,m]
    # = (g * dv[a,:]).conj() @ VPV_dd @ (g * dv[a,:])
    # For all faces a at once:
    gdv = g_n[:,None] * dv.T   # 14 x 20 -- each column is g * dv[a,:]
    GVPVG_aa = np.real(np.sum(gdv.conj() * (VPV_dd @ gdv), axis=0))  # 20

    # P^(2) = G V G V P + P V G V G - G V P V G
    # [P^(2)]_aa = GVGVP_diag - GVPVG_aa
    dP2 = GVGVP_diag - GVPVG_aa   # 20

    dP1_all_faces += dP1
    dP2_all_faces += dP2

dP1_all_faces /= N_k
dP2_all_faces /= N_k

print("First-order correction dP1 per face (sum over all k, divided by N_k=27):")
for fi in range(20):
    v = dP1_all_faces[fi]
    f = Fraction(v).limit_denominator(100000)
    match = abs(v - target) < 1e-6
    print(f"  face {fi:2d}: {v:+.12f} ~ {f} {'<-- TARGET' if match else ''}")

print()
print("Second-order correction dP2 per face:")
for fi in range(20):
    v = dP2_all_faces[fi]
    f = Fraction(v).limit_denominator(100000)
    match = abs(v - target) < 1e-6
    print(f"  face {fi:2d}: {v:+.12f} ~ {f} {'<-- TARGET' if match else ''}")

print()
total = dP1_all_faces + dP2_all_faces
print("Total correction (1st + 2nd order) per face:")
for fi in range(20):
    v = total[fi]
    f = Fraction(v).limit_denominator(100000)
    match = abs(v - target) < 1e-6
    print(f"  face {fi:2d}: {v:+.12f} ~ {f} {'<-- TARGET' if match else ''}")

print()
print(f"Target delta = {target}")
print(f"Mean 1st order: {np.mean(dP1_all_faces):.12f}")
print(f"Mean 2nd order: {np.mean(dP2_all_faces):.12f}")
print(f"Mean total: {np.mean(total):.12f}")

# Breakdown by BZ type
for label in ['G', 'E', 'F', 'C']:
    ks = [n for n in kvecs if bz_label(n)==label]
    print(f"\n{label} ({len(ks)} k-pts): dP2 face 9 values:")
    for n in ks[:3]:
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
        V_fd = dv.conj().T @ V_k_h @ fv
        V_dd = dv.conj().T @ V_k_h @ dv
        GVGVP = dv.conj() @ (g_n[:,None] * (V_dd @ (g_n[:,None] * V_fd)))
        term1 = np.sum(fv.conj() * GVGVP, axis=1)
        GVGVP_diag = 2.0 * np.real(term1)
        VPV_dd = V_fd @ V_fd.conj().T
        gdv = g_n[:,None] * dv.T
        GVPVG_aa = np.real(np.sum(gdv.conj() * (VPV_dd @ gdv), axis=0))
        dP2_k = GVGVP_diag - GVPVG_aa
        print(f"  k={n}: dP2[9]={dP2_k[9]:.10f}, dP2[16]={dP2_k[16]:.10f}")
