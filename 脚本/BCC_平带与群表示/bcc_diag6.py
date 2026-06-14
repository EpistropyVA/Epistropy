"""
Focus: delta = 0.013782448754 must come from the FULL open-BC system.
In the full open-BC system: 108 flat states, compared to 162 periodic.
The 54 missing states are corner-localized.
Check if delta = (per-face P_flat deviation at corner BC) in the full system
is +0.013782 for some specific face or combination.
Also check: maybe delta is defined differently -- as P_flat(corner) / P_flat(bulk) - 1 or similar.
"""
import numpy as np
from fractions import Fraction
import itertools
from collections import defaultdict

# Build full open-BC system (no wrapping)
def build_full_open():
    bcs = [(i,j,k) for i,j,k in itertools.product(range(3), repeat=3)]
    all_faces = {}
    bc_face_indices = []
    gidx = 0
    for bc_ijk in bcs:
        i,j,k = bc_ijk
        bc_v = ('bc',i,j,k)
        corners_raw = []
        for dx,dy,dz in itertools.product((0,1), repeat=3):
            ox,oy,oz = i+dx, j+dy, k+dz
            corners_raw.append(((ox+oy+oz)%2, ('c',ox,oy,oz)))
        ev = [v for (p,v) in corners_raw if p==0]
        od = [v for (p,v) in corners_raw if p==1]
        li = []
        for tc in (ev, od):
            sv = [bc_v] + tc
            for combo in itertools.combinations(sv, 3):
                f = frozenset(combo)
                if f not in all_faces: all_faces[f] = gidx; gidx += 1
                li.append(all_faces[f])
        bc_face_indices.append(li)
    N = len(all_faces)
    afs = [None]*N
    for f,i in all_faces.items(): afs[i] = f
    v2f = defaultdict(set)
    for f,i in all_faces.items():
        for v in f: v2f[v].add(i)
    pairs = set()
    for v,fs in v2f.items():
        fl = sorted(fs)
        for a in range(len(fl)):
            for b in range(a+1, len(fl)):
                pairs.add((fl[a], fl[b]))
    A = np.zeros((N,N))
    for i,j in pairs:
        if len(afs[i]&afs[j])==2: A[i,j] = A[j,i] = 1.0
    return A, bcs, bc_face_indices

A_open, bcs_open, bfi_open = build_full_open()
ev_open, evec_open = np.linalg.eigh(A_open)
mask_open = np.abs(ev_open + 3) < 1e-4
n_flat = mask_open.sum()
print(f"Full open-BC flat states: {n_flat}")

fv_open = evec_open[:, mask_open]  # 540 x 108

# Per-face P_flat_aa
target = 0.013782448754

print("\nLooking for faces with P_flat delta = +0.013782448754")
print("(compared to periodic value of 0.3 = 162/540)")
P_periodic = 0.3  # mean P_flat_aa in periodic BC

# The full open-BC system has N=540 faces. Per-face P_flat_aa:
P_flat_open = np.sum(fv_open**2, axis=1)  # 540 values

# Check each face in each BC
hits = []
for bc_idx, bc_ijk in enumerate(bcs_open):
    for fi_local in range(20):
        global_fi = bfi_open[bc_idx][fi_local]
        p_val = P_flat_open[global_fi]
        delta = p_val - P_periodic
        if abs(delta - target) < 1e-6:
            hits.append((bc_ijk, fi_local, p_val, delta))

if hits:
    print(f"FOUND {len(hits)} matches!")
    for bc, fi, p, d in hits[:20]:
        print(f"  BC{bc} face {fi}: P_aa={p:.12f}, delta={d:.12f}")
else:
    print("No direct match found.")

# Show deviation from 0.3 for each face grouped by BC type
print("\nDeviation from 0.3 per face per BC type:")
for bc_type_label, bc_type_selector in [
    ("corner (3 boundary dims)", lambda ijk: sum(1 for x in ijk if x in (0,2)) == 3),
    ("edge (2 boundary dims)", lambda ijk: sum(1 for x in ijk if x in (0,2)) == 2),
    ("face (1 boundary dim)", lambda ijk: sum(1 for x in ijk if x in (0,2)) == 1),
    ("bulk (0 boundary dims)", lambda ijk: sum(1 for x in ijk if x in (0,2)) == 0),
]:
    type_bcs = [(idx, ijk) for idx, ijk in enumerate(bcs_open) if bc_type_selector(ijk)]
    if not type_bcs:
        continue
    # Collect all face deltas for this type
    all_deltas = []
    for bc_idx, bc_ijk in type_bcs:
        for fi_local in range(20):
            global_fi = bfi_open[bc_idx][fi_local]
            p_val = P_flat_open[global_fi]
            all_deltas.append((fi_local, p_val - P_periodic))

    unique_deltas = {}
    for fi, d in all_deltas:
        if fi not in unique_deltas:
            unique_deltas[fi] = []
        unique_deltas[fi].append(d)

    print(f"\n{bc_type_label} ({len(type_bcs)} BCs):")
    for fi in sorted(unique_deltas.keys()):
        vals = unique_deltas[fi]
        mean_d = np.mean(vals)
        std_d = np.std(vals)
        f = Fraction(mean_d).limit_denominator(100000)
        match = abs(mean_d - target) < 1e-6
        print(f"  face {fi:2d}: mean delta = {mean_d:+.10f} ~ {f} {'<-- TARGET MATCH!' if match else ''}")

# Also check: maybe delta is defined using the NORMALIZED projector P / n_flat
print(f"\nNormalized projector (P / n_flat):")
print(f"  Periodic: P_aa_norm = 0.3 / 162 = {0.3/162:.12f}")
print(f"  Full open mean at corner BC: {np.mean(P_flat_open[:20]) / n_flat:.12f}")
print(f"  Difference: {np.mean(P_flat_open[:20]) / n_flat - 0.3/162:.12f}")
print(f"  Target: {target}")

# What about: delta = P_corner(open) / n_flat_open - P_bulk(periodic) / n_flat_per?
# = (mean at corner / 108) - (0.3 / 162)
corner_normalized = np.mean(P_flat_open[:20]) / n_flat
periodic_normalized = 0.3 / 162
print(f"\ncorner_normalized - periodic_normalized = {corner_normalized - periodic_normalized:.12f}")
print(f"target = {target}")

# What about the RATIO of spectral weights?
ratio = np.mean(P_flat_open[:20]) / P_periodic
print(f"\nRatio corner/periodic mean P_aa = {ratio:.12f}")

# Let me also try the SPECIFIC faces at the corner BC in the full open-BC
print("\nPer-face P_flat_aa at corner BC (0,0,0) in full open-BC:")
for fi in range(20):
    global_fi = bfi_open[0][fi]
    p_val = P_flat_open[global_fi]
    delta = p_val - P_periodic
    match = abs(delta - target) < 1e-6
    f = Fraction(p_val).limit_denominator(100000)
    print(f"  face {fi:2d}: P={p_val:.10f}, delta={delta:+.10f} {'<-- MATCH' if match else ''}")

# Also check corner BC (2,2,2) for comparison
corner_222_idx = bcs_open.index((2,2,2))
print("\nPer-face P_flat_aa at corner BC (2,2,2) in full open-BC:")
for fi in range(20):
    global_fi = bfi_open[corner_222_idx][fi]
    p_val = P_flat_open[global_fi]
    delta = p_val - P_periodic
    match = abs(delta - target) < 1e-6
    f = Fraction(p_val).limit_denominator(100000)
    print(f"  face {fi:2d}: P={p_val:.10f}, delta={delta:+.10f} {'<-- MATCH' if match else ''}")
