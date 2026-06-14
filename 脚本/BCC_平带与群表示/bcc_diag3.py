"""
Explore the 8-corner structure and find delta = 0.013782448754.
Key insight: each of the 8 corner BCs removes a different combination of 3 bonds.
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

# Each corner BC removes 3 bonds, one per direction.
# For BC at position (p,q,r) with p,q,r in {0,2}:
# In x-direction: if p=0, remove -x bond (R=(2,0,0)); if p=2, remove +x bond (R=(1,0,0))
# Similarly for y,z.
corner_BCs = [(p,q,r) for p in (0,2) for q in (0,2) for r in (0,2)]
corner_removals = {}
for bc in corner_BCs:
    removed = []
    for axis in range(3):
        if bc[axis] == 0:
            R = [0,0,0]; R[axis] = 2; removed.append(tuple(R))
        else:
            R = [0,0,0]; R[axis] = 1; removed.append(tuple(R))
    corner_removals[bc] = removed

print("Corner BC removals:")
for bc in corner_BCs:
    print(f"  BC{bc}: remove {corner_removals[bc]}")

# For each corner BC, build the perturbation V_k at each k-point
# and compute first-order correction to face-9 weight
kvecs = list(itertools.product(range(3), repeat=3))
N_k = 27

print("\nFirst-order dP1 at face 9 per corner BC:")
dP1_face9_per_corner = {}
for bc in corner_BCs:
    bc_idx = bcs.index(bc)
    removed_Rs = corner_removals[bc]

    # Build perturbation matrix T blocks at this BC's position
    # The perturbation removes bonds from BC(bc) to its wrap neighbors.
    # T[R] is the hopping from BC(0,0,0) to BC(R). For BC at position bc_ijk,
    # the hopping to neighbor at bc_ijk + R_removed is T[R_removed].
    # BUT: T(R) is the hopping from the REFERENCE BC, so the actual blocks
    # for BC(bc_ijk) are T[R_removed] (by translation invariance).

    total_dP1_f9 = 0.0
    for n_tuple in kvecs:
        k = np.array([2*np.pi*ni/3.0 for ni in n_tuple])
        # H(k) for this k-point
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

        # V_k for this corner BC: sum of removed bonds
        V_k = np.zeros((20,20), dtype=complex)
        for R in removed_Rs:
            phase = np.exp(1j*np.dot(k, np.array(R,float)))
            V_k += phase*T[R]
        V_k_h = V_k + V_k.conj().T

        g_n = 1.0 / (-3.0 - de)
        V_dfl = dv.conj().T @ V_k_h @ fv
        GVP = dv @ (g_n[:,None] * V_dfl)
        dP1 = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
        total_dP1_f9 += dP1[9]

    total_dP1_f9 /= N_k
    dP1_face9_per_corner[bc] = total_dP1_f9
    print(f"  BC{bc} (remove {removed_Rs}): dP1_face9 = {total_dP1_f9:.10f}")

# Mean over 8 corners
mean_dP1_f9 = np.mean(list(dP1_face9_per_corner.values()))
print(f"\nMean dP1 at face 9 over 8 corner BCs: {mean_dP1_f9:.12f}")
print(f"Target: {target}")

# Also check face 16 (the other 'opposite' face)
print("\nFirst-order dP1 at face 16 per corner BC:")
dP1_face16_per_corner = {}
for bc in corner_BCs:
    removed_Rs = corner_removals[bc]
    total_dP1_f16 = 0.0
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
        for R in removed_Rs:
            phase = np.exp(1j*np.dot(k, np.array(R,float)))
            V_k += phase*T[R]
        V_k_h = V_k + V_k.conj().T
        g_n = 1.0 / (-3.0 - de)
        V_dfl = dv.conj().T @ V_k_h @ fv
        GVP = dv @ (g_n[:,None] * V_dfl)
        dP1 = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
        total_dP1_f16 += dP1[16]
    total_dP1_f16 /= N_k
    dP1_face16_per_corner[bc] = total_dP1_f16

mean_dP1_f16 = np.mean(list(dP1_face16_per_corner.values()))
print(f"Mean dP1 at face 16 over 8 corner BCs: {mean_dP1_f16:.12f}")

# Mean over faces 6-9, 16-19 (all type-B faces)
print("\nMean dP1 at type-B faces per corner BC:")
dP1_typeB_per_corner = {}
for bc in corner_BCs:
    removed_Rs = corner_removals[bc]
    total_dP1_typeB = np.zeros(8)  # 8 type-B faces
    typeB_faces = [6,7,8,9,16,17,18,19]
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
        for R in removed_Rs:
            phase = np.exp(1j*np.dot(k, np.array(R,float)))
            V_k += phase*T[R]
        V_k_h = V_k + V_k.conj().T
        g_n = 1.0 / (-3.0 - de)
        V_dfl = dv.conj().T @ V_k_h @ fv
        GVP = dv @ (g_n[:,None] * V_dfl)
        dP1_all = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
        for j,fi in enumerate(typeB_faces):
            total_dP1_typeB[j] += dP1_all[fi]
    total_dP1_typeB /= N_k
    mean_typeB = np.mean(total_dP1_typeB)
    dP1_typeB_per_corner[bc] = mean_typeB
    print(f"  BC{bc}: mean type-B dP1 = {mean_typeB:.10f}, per-face = {total_dP1_typeB}")

mean_all = np.mean(list(dP1_typeB_per_corner.values()))
print(f"\nMean type-B dP1 over 8 corner BCs: {mean_all:.12f}")
print(f"Target: {target}")

# Check: mean over all 20 faces per corner BC
print("\nMean dP1 over all 20 faces per corner BC:")
for bc in corner_BCs:
    removed_Rs = corner_removals[bc]
    total_dP1_all = np.zeros(20)
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
        for R in removed_Rs:
            phase = np.exp(1j*np.dot(k, np.array(R,float)))
            V_k += phase*T[R]
        V_k_h = V_k + V_k.conj().T
        g_n = 1.0 / (-3.0 - de)
        V_dfl = dv.conj().T @ V_k_h @ fv
        GVP = dv @ (g_n[:,None] * V_dfl)
        dP1_all = 2.0 * np.real(np.sum(GVP * fv.conj(), axis=1))
        total_dP1_all += dP1_all
    total_dP1_all /= N_k
    m = np.mean(total_dP1_all)
    print(f"  BC{bc}: mean dP1 = {m:.10f}")

print()
print("Summary of first-order contributions:")
print(f"  Face 9 at BC(0,0,0): {dP1_face9_per_corner.get((0,0,0), 'N/A'):.10f}")
print(f"  Face 9 at BC(2,2,2): {dP1_face9_per_corner.get((2,2,2), 'N/A'):.10f}")
print(f"  Mean face 9 over 8 corners: {mean_dP1_f9:.12f}")
print(f"  Target: {target}")
