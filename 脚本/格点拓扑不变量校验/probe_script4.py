# probe_script4_structure.py
# Probe the internal structure of the Wilson loop to understand
# what "1-to-1 correspondence" actually means.

import itertools
import numpy as np

# --- reuse lattice construction ---
def build_bcc_lattice_periodic():
    return list(itertools.product(range(3), repeat=3))

def enumerate_simplex_faces(bc_ijk):
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)
    orig_corners = []
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        ox, oy, oz = i + dx, j + dy, k + dz
        wx, wy, wz = ox % 3, oy % 3, oz % 3
        orig_corners.append(((ox + oy + oz) % 2, ('c', wx, wy, wz)))
    even_corners = [v for (par, v) in orig_corners if par == 0]
    odd_corners  = [v for (par, v) in orig_corners if par == 1]
    faces = []
    for tet_corners in (even_corners, odd_corners):
        simplex_verts = [bc_v] + tet_corners
        for combo in itertools.combinations(simplex_verts, 3):
            faces.append(frozenset(combo))
    return faces

def translate_vertex(v, s):
    si, sj, sk = s
    if v[0] == 'bc':
        _, i, j, k = v
        return ('bc', (i+si)%3, (j+sj)%3, (k+sk)%3)
    else:
        _, cx, cy, cz = v
        return ('c', (cx+si)%3, (cy+sj)%3, (cz+sk)%3)

def translate_face(face, s):
    return frozenset(translate_vertex(v, s) for v in face)

def build_all_faces(body_centers):
    ref_faces = enumerate_simplex_faces((0,0,0))
    face_to_idx = {}; bc_face_indices = []; g = 0
    for bc_idx, bc_ijk in enumerate(body_centers):
        local = []
        for ref_face in ref_faces:
            sf = translate_face(ref_face, bc_ijk)
            if sf not in face_to_idx:
                face_to_idx[sf] = g; g += 1
            local.append(face_to_idx[sf])
        bc_face_indices.append(local)
    return face_to_idx, bc_face_indices, ref_faces

def build_adjacency_matrix(face_to_idx):
    N = len(face_to_idx)
    A = np.zeros((N, N))
    all_faces = [None]*N
    for f, i in face_to_idx.items(): all_faces[i] = f
    vtf = {}
    for f, i in face_to_idx.items():
        for v in f:
            vtf.setdefault(v, set()).add(i)
    pairs = set()
    for v, fs in vtf.items():
        fl = sorted(fs)
        for a in range(len(fl)):
            for b in range(a+1, len(fl)):
                pairs.add((fl[a], fl[b]))
    for i, j in pairs:
        if len(all_faces[i] & all_faces[j]) == 2:
            A[i,j] = A[j,i] = 1.0
    return A

body_centers = build_bcc_lattice_periodic()
face_to_idx, bc_face_indices, ref_faces = build_all_faces(body_centers)
A = build_adjacency_matrix(face_to_idx)

perm = []
for bc_idx in range(27):
    perm.extend(bc_face_indices[bc_idx])
perm = np.array(perm)
A_reord = A[np.ix_(perm, perm)]

T = {}
for j, bc in enumerate(body_centers):
    R = tuple(bc[a]%3 for a in range(3))
    T[R] = A_reord[0:20, j*20:(j+1)*20].copy()

kvecs = list(itertools.product(range(3), repeat=3))
vecs_all = {}
for n in kvecs:
    k = np.array([2*np.pi*ni/3.0 for ni in n])
    H = np.zeros((20,20), dtype=complex)
    for R, mat in T.items():
        H += np.exp(1j*np.dot(k, np.array(R, dtype=float))) * mat
    _, evecs = np.linalg.eigh(H)
    vecs_all[n] = evecs

# --- Key probe: per-loop pi-eigenvalue count ---
print("=" * 70)
print("PROBE: Per-Wilson-loop pi-eigenvalue breakdown")
print("=" * 70)

axes = ['x','y','z']
loop_results = {}
for ax_i, ax_n in enumerate(axes):
    for o1 in range(3):
        for o2 in range(3):
            loop_k = []
            for na in range(3):
                nt = [0,0,0]
                nt[ax_i] = na
                oa = [i for i in range(3) if i != ax_i]
                nt[oa[0]] = o1; nt[oa[1]] = o2
                loop_k.append(tuple(nt))
            loop_results[(ax_n, o1, o2)] = loop_k

deg_bands = list(range(6))
print(f"\nLoop -> (num_pi_phases, all 6 Wilson eigenvalue phases/pi)")
pi_count_per_loop = {}
for loop_key, loop_k in sorted(loop_results.items()):
    nk = len(loop_k)
    W = np.eye(6, dtype=complex)
    for j in range(nk):
        kc = loop_k[j]; kn = loop_k[(j+1)%nk]
        S = np.zeros((6,6), dtype=complex)
        for a, ab in enumerate(deg_bands):
            for b, bb in enumerate(deg_bands):
                S[a,b] = np.vdot(vecs_all[kc][:,ab], vecs_all[kn][:,bb])
        W = W @ S
    evals = np.linalg.eigvals(W)
    phases = np.angle(evals)
    n_pi = sum(1 for p in phases if abs(abs(p) - np.pi) < 0.1)
    pi_count_per_loop[loop_key] = n_pi
    phases_over_pi = sorted(phases / np.pi)
    print(f"  {loop_key}: n_pi={n_pi}, phases/pi={[f'{p:.3f}' for p in phases_over_pi]}")

all_counts = list(pi_count_per_loop.values())
print(f"\nAll loops have exactly 2 pi-eigenvalues? {all(c==2 for c in all_counts)}")
print(f"Total pi-modes: {sum(all_counts)} = 27 * 2 = 54? {sum(all_counts)==54}")

# --- Probe: parity structure of pi-eigenvectors ---
# For each Wilson loop, check if the 2 pi-eigenvectors
# have support predominantly on even-tet faces (0-9) vs odd-tet faces (10-19).
print(f"\n{'='*70}")
print("PROBE: Parity (even/odd tet) structure of pi-eigenvectors")
print("=" * 70)

parity_clean_count = 0
parity_mixed_count = 0
for loop_key, loop_k in sorted(loop_results.items()):
    nk = len(loop_k)
    W = np.eye(6, dtype=complex)
    for j in range(nk):
        kc = loop_k[j]; kn = loop_k[(j+1)%nk]
        S = np.zeros((6,6), dtype=complex)
        for a, ab in enumerate(deg_bands):
            for b, bb in enumerate(deg_bands):
                S[a,b] = np.vdot(vecs_all[kc][:,ab], vecs_all[kn][:,bb])
        W = W @ S
    evals, evecs = np.linalg.eig(W)
    phases = np.angle(evals)
    pi_idx = [i for i,p in enumerate(phases) if abs(abs(p)-np.pi) < 0.1]
    
    for idx in pi_idx:
        v = evecs[:, idx]
        # This v is in the 6-band subspace. Project to 20-dim:
        psi = vecs_all[loop_k[0]][:, deg_bands] @ v
        # psi is 20-dim: first 10 = even tet faces, last 10 = odd tet faces
        even_weight = np.sum(np.abs(psi[:10])**2)
        odd_weight = np.sum(np.abs(psi[10:])**2)
        total = even_weight + odd_weight
        if total > 1e-15:
            even_frac = even_weight / total
            odd_frac = odd_weight / total
            if even_frac > 0.99 or odd_frac > 0.99:
                parity_clean_count += 1
            else:
                parity_mixed_count += 1

print(f"Pi-modes with clean parity (>99% on one tet): {parity_clean_count}")
print(f"Pi-modes with mixed parity: {parity_mixed_count}")
print(f"Total: {parity_clean_count + parity_mixed_count}")

# --- Probe: overlap rank explanation ---
print(f"\n{'='*70}")
print("PROBE: Overlap rank 19 = 1+6+12 decomposition check")
print("=" * 70)

# Build simplex vectors and pi-mode vectors, compute SVD
simplex_vectors = np.zeros((54, 540))
for bc_idx in range(27):
    gf = bc_face_indices[bc_idx]
    for li in range(10):
        simplex_vectors[2*bc_idx, gf[li]] = 1.0
    for li in range(10, 20):
        simplex_vectors[2*bc_idx+1, gf[li]] = 1.0

pi_modes_540 = np.zeros((54, 540), dtype=complex)
mi = 0
for loop_key, loop_k in sorted(loop_results.items()):
    nk = len(loop_k)
    W = np.eye(6, dtype=complex)
    for j in range(nk):
        kc = loop_k[j]; kn = loop_k[(j+1)%nk]
        S = np.zeros((6,6), dtype=complex)
        for a, ab in enumerate(deg_bands):
            for b, bb in enumerate(deg_bands):
                S[a,b] = np.vdot(vecs_all[kc][:,ab], vecs_all[kn][:,bb])
        W = W @ S
    evals, evecs = np.linalg.eig(W)
    phases = np.angle(evals)
    pi_idx = [i for i,p in enumerate(phases) if abs(abs(p)-np.pi) < 0.1]
    for idx in pi_idx:
        v = evecs[:,idx]
        psi = vecs_all[loop_k[0]][:, deg_bands] @ v
        k_vec = np.array([2*np.pi*ni/3.0 for ni in loop_k[0]])
        for bc_idx, bc_ijk in enumerate(body_centers):
            R = np.array(bc_ijk, dtype=float)
            phase = np.exp(1j*np.dot(k_vec, R))
            gf = bc_face_indices[bc_idx]
            for li in range(20):
                pi_modes_540[mi, gf[li]] = psi[li] * phase
        mi += 1

overlap = simplex_vectors @ pi_modes_540.conj().T
svals = np.linalg.svd(overlap, compute_uv=False)
print("Singular values (rounded):")
for i, sv in enumerate(svals):
    if sv > 1e-8:
        print(f"  sigma_{i+1} = {sv:.6f}")

# Group by approximate value
from collections import Counter
rounded = [round(sv, 2) for sv in svals if sv > 1e-5]
mult = Counter(rounded)
print(f"\nMultiplicity structure: {dict(mult)}")
print(f"Total rank: {len(rounded)}")
