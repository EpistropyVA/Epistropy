# -*- coding: utf-8 -*-
# j_axis_sign_flip.py
#
# Compare two coupling matrix definitions on the same 22-orbit BCC graph.
# Identify the 3 eigenvector directions that flip sign between:
#   C_F (Frobenius norm)  signature 13+/9-
#   C_E (Edge count)      signature 10+/1⁰/11-
# These flipping directions are candidates for the "j-axis".

import itertools
import numpy as np
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Section 1: Build BCC lattice faces (copied from verify_t3_orbit22_coupling.py)
# ---------------------------------------------------------------------------

def get_geom_pos(v):
    if v[0] == 'bc':
        _, i, j, k = v
        return np.array([i + 0.5, j + 0.5, k + 0.5])
    else:
        _, x, y, z = v
        return np.array([float(x), float(y), float(z)])


def build_faces_open():
    face_set = {}
    face_list = []
    bc_face_map = {}

    for i, j, k in itertools.product(range(3), repeat=3):
        bc_v = ('bc', i, j, k)
        even_corners = []
        odd_corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            ox, oy, oz = i + dx, j + dy, k + dz
            parity = (ox + oy + oz) % 2
            cv = ('c', ox, oy, oz)
            if parity == 0:
                even_corners.append(cv)
            else:
                odd_corners.append(cv)

        bc_faces = []
        for tet_corners in (even_corners, odd_corners):
            simplex_verts = [bc_v] + tet_corners
            for combo in itertools.combinations(simplex_verts, 3):
                f = frozenset(combo)
                if f not in face_set:
                    face_set[f] = len(face_list)
                    face_list.append(f)
                bc_faces.append(face_set[f])
        bc_face_map[(i, j, k)] = bc_faces

    return face_list, face_set, bc_face_map


def build_adjacency_open(face_list, face_set):
    N = len(face_list)
    A = np.zeros((N, N), dtype=float)

    vertex_to_faces = defaultdict(set)
    for idx, f in enumerate(face_list):
        for v in f:
            vertex_to_faces[v].add(idx)

    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a in range(len(flist)):
            for b in range(a + 1, len(flist)):
                candidate_pairs.add((flist[a], flist[b]))

    for i, j in candidate_pairs:
        if len(face_list[i] & face_list[j]) == 2:
            A[i, j] = 1.0
            A[j, i] = 1.0

    return A


# ---------------------------------------------------------------------------
# Section 2: O_h symmetry group
# ---------------------------------------------------------------------------

CENTER = np.array([1.5, 1.5, 1.5])


def generate_oh_group():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            M = np.zeros((3, 3), dtype=float)
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            mats.append(M)
    assert len(mats) == 48
    return mats


def apply_oh_to_vertex(v, M):
    pos = get_geom_pos(v) - CENTER
    new_pos = M @ pos + CENTER
    return new_pos


def pos_to_vertex(pos):
    half = pos - 0.5
    if np.allclose(half, np.round(half), atol=1e-9):
        i, j, k = int(round(half[0])), int(round(half[1])), int(round(half[2]))
        if 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2:
            return ('bc', i, j, k)
    if np.allclose(pos, np.round(pos), atol=1e-9):
        x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
        if 0 <= x <= 3 and 0 <= y <= 3 and 0 <= z <= 3:
            return ('c', x, y, z)
    return None


def apply_oh_to_face(face, M):
    new_verts = []
    for v in face:
        new_pos = apply_oh_to_vertex(v, M)
        new_v = pos_to_vertex(new_pos)
        if new_v is None:
            return None
        new_verts.append(new_v)
    return frozenset(new_verts)


def build_oh_orbits(face_list, face_set, oh_group):
    N = len(face_list)
    assigned = [-1] * N
    orbits = []
    orbit_of = [-1] * N

    print("  Building O_h action table (48 x 540)...")
    action_table = np.full((48, N), -1, dtype=int)
    unmapped = 0
    for sym_idx, M in enumerate(oh_group):
        for face_idx, face in enumerate(face_list):
            img = apply_oh_to_face(face, M)
            if img is not None and img in face_set:
                action_table[sym_idx][face_idx] = face_set[img]
            else:
                unmapped += 1

    print(f"  Unmapped: {unmapped}")
    assert unmapped == 0

    for start in range(N):
        if assigned[start] != -1:
            continue
        orbit_idx = len(orbits)
        orbit_members = set()
        queue = [start]
        orbit_members.add(start)
        while queue:
            cur = queue.pop()
            for sym_idx in range(48):
                img = action_table[sym_idx][cur]
                if img not in orbit_members:
                    orbit_members.add(img)
                    queue.append(img)
        for f in orbit_members:
            assigned[f] = orbit_idx
            orbit_of[f] = orbit_idx
        orbits.append(sorted(orbit_members))

    return orbits, orbit_of, action_table


# ---------------------------------------------------------------------------
# Section 3: Compute both coupling matrices
# ---------------------------------------------------------------------------

def compute_both_couplings(A, orbits):
    """
    Returns four 22x22 matrices:
      C_F_raw   : Frobenius norm of block
      C_F_norm  : Frobenius / sqrt(|Oi||Oj|)
      C_E_raw   : sum of entries (edge count)
      C_E_norm  : edge count / sqrt(|Oi||Oj|)
    Also returns block_stats dict: (i,j) -> (edge_count, frob, variance_proxy, block_size)
    """
    n = len(orbits)
    C_F_raw  = np.zeros((n, n))
    C_F_norm = np.zeros((n, n))
    C_E_raw  = np.zeros((n, n))
    C_E_norm = np.zeros((n, n))
    block_stats = {}

    for i in range(n):
        for j in range(i, n):
            block = A[np.ix_(orbits[i], orbits[j])]
            edge_count  = block.sum()
            frob        = np.linalg.norm(block, 'fro')
            bsize       = len(orbits[i]) * len(orbits[j])
            var_proxy   = frob**2 - edge_count**2 / bsize

            norm_factor = np.sqrt(len(orbits[i]) * len(orbits[j]))

            C_F_raw[i, j]  = frob
            C_F_raw[j, i]  = frob
            C_F_norm[i, j] = frob / norm_factor
            C_F_norm[j, i] = frob / norm_factor

            C_E_raw[i, j]  = edge_count
            C_E_raw[j, i]  = edge_count
            C_E_norm[i, j] = edge_count / norm_factor
            C_E_norm[j, i] = edge_count / norm_factor

            block_stats[(i, j)] = (edge_count, frob, var_proxy, bsize)
            if i != j:
                block_stats[(j, i)] = (edge_count, frob, var_proxy, bsize)

    return C_F_raw, C_F_norm, C_E_raw, C_E_norm, block_stats


# ---------------------------------------------------------------------------
# Section 4: Eigenvector comparison
# ---------------------------------------------------------------------------

def match_eigenvectors(evecs_F, evecs_E):
    """
    Greedy matching of eigenvectors by maximum |inner product|.
    evecs_F, evecs_E: columns are eigenvectors (from eigh, ascending order).
    Returns list of (F_col, E_col, abs_ip) in matched pairs, sorted by F_col.
    """
    n = evecs_F.shape[1]
    ip_mat = np.abs(evecs_F.T @ evecs_E)  # shape (n, n): ip_mat[f, e]

    used_e = set()
    matches = []  # (F_col, E_col, abs_ip)

    for f_col in range(n):
        row = ip_mat[f_col].copy()
        for e_col in used_e:
            row[e_col] = -1.0
        e_col = int(np.argmax(row))
        matches.append((f_col, e_col, ip_mat[f_col, e_col]))
        used_e.add(e_col)

    return matches


def identify_sign_flips(evals_F, evals_E, matches, threshold=1e-8):
    """
    For each matched pair, determine if there's a sign flip or zero-crossing.
    Returns list of (F_col, E_col, eval_F, eval_E, flip_type) for flipping pairs.
    flip_type: 'pos->neg', 'neg->pos', 'pos->zero', 'neg->zero', 'zero->pos', 'zero->neg'
    """
    flips = []
    for f_col, e_col, abs_ip in matches:
        ef = evals_F[f_col]
        ee = evals_E[e_col]
        sign_f = 1 if ef > threshold else (-1 if ef < -threshold else 0)
        sign_e = 1 if ee > threshold else (-1 if ee < -threshold else 0)
        if sign_f != sign_e:
            if sign_f == 1 and sign_e == -1:
                ftype = 'pos->neg'
            elif sign_f == -1 and sign_e == 1:
                ftype = 'neg->pos'
            elif sign_f == 1 and sign_e == 0:
                ftype = 'pos->zero'
            elif sign_f == -1 and sign_e == 0:
                ftype = 'neg->zero'
            elif sign_f == 0 and sign_e == 1:
                ftype = 'zero->pos'
            else:
                ftype = 'zero->neg'
            flips.append((f_col, e_col, ef, ee, ftype, abs_ip))
    return flips


# ---------------------------------------------------------------------------
# Section 5: Anatomy helpers
# ---------------------------------------------------------------------------

def participation_ratio(vec):
    """PR = (sum |v_i|^2)^2 / sum |v_i|^4. For a unit vector: 1/sum |v_i|^4."""
    v2 = vec**2
    v4 = vec**4
    return v2.sum()**2 / v4.sum()


def orbit_size_class(size):
    if size == 48:
        return 'L48'
    elif size == 24:
        return 'M24'
    elif size == 12:
        return 'S12'
    else:
        return 'T8'


def cross_size_content(vec, orbit_sizes):
    """
    Fraction of energy (|v_i|^2) in orbits whose size differs from
    the orbit with the largest |component|.
    """
    dominant_idx = int(np.argmax(np.abs(vec)))
    dominant_size = orbit_sizes[dominant_idx]
    energy_other = sum(vec[i]**2 for i in range(len(vec)) if orbit_sizes[i] != dominant_size)
    return energy_other / (vec**2).sum()


def variance_content(vec, block_stats, n_orbits):
    """
    Total 'variance content' of an eigenvector:
    sum over (i,j) of |v_i * v_j| * variance_proxy(i,j)
    """
    total = 0.0
    for i in range(n_orbits):
        for j in range(n_orbits):
            vp = block_stats[(i, j)][2] if (i, j) in block_stats else 0.0
            total += abs(vec[i] * vec[j]) * vp
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("J-AXIS SIGN FLIP ANALYSIS")
    print("BCC 3x3x3 lattice: Frobenius vs Edge-count coupling comparison")
    print("=" * 72)

    # --- Build lattice ---
    print("\n[SETUP] Building BCC lattice and O_h orbits...")
    face_list, face_set, bc_face_map = build_faces_open()
    assert len(face_list) == 540
    print(f"  Faces: {len(face_list)}")

    A = build_adjacency_open(face_list, face_set)
    print(f"  Adjacency matrix: {A.shape}, edges: {int(A.sum()) // 2}")

    oh_group = generate_oh_group()
    orbits, orbit_of, action_table = build_oh_orbits(face_list, face_set, oh_group)
    orbits = sorted(orbits, key=lambda o: -len(o))
    orbit_sizes = [len(o) for o in orbits]
    n_orbits = len(orbits)

    size_census = Counter(orbit_sizes)
    print(f"  Orbits: {n_orbits}, census: {dict(sorted(size_census.items(), reverse=True))}")
    assert n_orbits == 22

    # --- Compute coupling matrices ---
    print("\n[SECTION 1] Computing coupling matrices...")
    C_F_raw, C_F_norm, C_E_raw, C_E_norm, block_stats = compute_both_couplings(A, orbits)
    print("  Done.")

    # --- Eigenvalue analysis ---
    print("\n[SECTION 2] Eigenvalue analysis")
    evals_F, evecs_F = np.linalg.eigh(C_F_norm)  # ascending
    evals_E, evecs_E = np.linalg.eigh(C_E_norm)  # ascending

    def sig(evals, thr=1e-8):
        n_pos  = int(np.sum(evals >  thr))
        n_neg  = int(np.sum(evals < -thr))
        n_zero = len(evals) - n_pos - n_neg
        return n_pos, n_zero, n_neg

    sp_F = sig(evals_F)
    sp_E = sig(evals_E)

    print(f"\n  C_F_norm signature: {sp_F[0]}+ / {sp_F[1]}zero / {sp_F[2]}-")
    print(f"  C_E_norm signature: {sp_E[0]}+ / {sp_E[1]}zero / {sp_E[2]}-")

    print(f"\n  {'Rank':>4}  {'eval_F':>12}  {'eval_E':>12}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}")
    for i in range(n_orbits - 1, -1, -1):
        print(f"  {n_orbits-i:>4}  {evals_F[i]:>+12.6f}  {evals_E[i]:>+12.6f}")

    # --- Match eigenvectors ---
    print("\n[SECTION 3] Eigenvector matching (greedy by max |inner product|)")
    matches = match_eigenvectors(evecs_F, evecs_E)

    print(f"\n  {'F-rank':>6}  {'E-rank':>6}  {'eval_F':>12}  {'eval_E':>12}  {'|IP|':>8}  sign?")
    print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  -----")
    for f_col, e_col, abs_ip in sorted(matches, key=lambda x: -evals_F[x[0]]):
        ef = evals_F[f_col]
        ee = evals_E[e_col]
        sign_f = '+' if ef > 1e-8 else ('-' if ef < -1e-8 else '0')
        sign_e = '+' if ee > 1e-8 else ('-' if ee < -1e-8 else '0')
        flip_mark = " <-- FLIP" if sign_f != sign_e else ""
        f_rank = n_orbits - f_col
        e_rank = n_orbits - e_col
        print(f"  {f_rank:>6}  {e_rank:>6}  {ef:>+12.6f}  {ee:>+12.6f}  {abs_ip:>8.5f}  {sign_f}->{sign_e}{flip_mark}")

    # --- Identify sign flips ---
    flips = identify_sign_flips(evals_F, evals_E, matches)
    print(f"\n  Identified {len(flips)} sign-flipping direction(s):")
    for f_col, e_col, ef, ee, ftype, abs_ip in flips:
        print(f"    F-col {f_col} (rank {n_orbits-f_col}): eval_F={ef:+.6f}, eval_E={ee:+.6f}, "
              f"type={ftype}, |IP|={abs_ip:.5f}")

    # --- Anatomy of flipping directions ---
    print("\n" + "=" * 72)
    print("[SECTION 4] Anatomy of sign-flipping eigenvectors")
    print("=" * 72)

    # Size class labels for all 22 orbits
    size_labels = [orbit_size_class(s) for s in orbit_sizes]
    label_order = ['L48', 'M24', 'S12', 'T8']

    for flip_idx, (f_col, e_col, ef, ee, ftype, abs_ip) in enumerate(flips):
        vF = evecs_F[:, f_col]
        vE = evecs_E[:, e_col]

        print(f"\n--- Flip direction {flip_idx+1}: F-col={f_col}, E-col={e_col}, {ftype} ---")
        print(f"    eval_F = {ef:+.8f}  eval_E = {ee:+.8f}  |IP| = {abs_ip:.6f}")

        # Eigenvector components
        print(f"\n    Eigenvector components (F then E), by orbit index:")
        print(f"    {'Orbit':>5}  {'Size':>5}  {'Class':>5}  {'vF':>10}  {'vE':>10}  {'|vF|^2':>10}")
        print(f"    {'-'*5}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}")
        for i in range(n_orbits):
            print(f"    {i:>5}  {orbit_sizes[i]:>5}  {size_labels[i]:>5}  "
                  f"{vF[i]:>+10.5f}  {vE[i]:>+10.5f}  {vF[i]**2:>10.5f}")

        # Participation ratio
        pr_F = participation_ratio(vF)
        pr_E = participation_ratio(vE)
        print(f"\n    Participation ratio (PR): F={pr_F:.3f}, E={pr_E:.3f}  "
              f"(22=fully delocalized, 1=fully localized)")

        # Energy by size class
        print(f"\n    Energy (|v_i|^2) distribution by orbit-size class:")
        for cls in label_order:
            idxs = [i for i, l in enumerate(size_labels) if l == cls]
            energy_F = sum(vF[i]**2 for i in idxs)
            energy_E = sum(vE[i]**2 for i in idxs)
            print(f"      {cls}: F={energy_F:.5f}  E={energy_E:.5f}  (n_orbits={len(idxs)})")

        # Cross-size content
        csc_F = cross_size_content(vF, orbit_sizes)
        csc_E = cross_size_content(vE, orbit_sizes)
        print(f"\n    Cross-size content: F={csc_F:.5f}  E={csc_E:.5f}")
        print(f"    (fraction of energy in orbits with size != dominant orbit's size)")

        # Which blocks drive the sign flip (C_F[i,j] - C_E[i,j] weighted by vF_i * vF_j)
        print(f"\n    Top-10 block contributions to (C_F - C_E) quadratic form [vF^T(C_F-C_E)vF]:")
        contrib_list = []
        for i in range(n_orbits):
            for j in range(n_orbits):
                diff_ij = C_F_norm[i, j] - C_E_norm[i, j]
                contrib = vF[i] * vF[j] * diff_ij
                if abs(contrib) > 1e-12:
                    contrib_list.append((i, j, diff_ij, contrib))
        contrib_list.sort(key=lambda x: -abs(x[3]))
        print(f"    {'i':>3}  {'j':>3}  {'sz_i':>5}  {'sz_j':>5}  {'C_F-C_E':>10}  {'contrib':>12}")
        print(f"    {'-'*3}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*10}  {'-'*12}")
        for i, j, diff_ij, contrib in contrib_list[:10]:
            print(f"    {i:>3}  {j:>3}  {orbit_sizes[i]:>5}  {orbit_sizes[j]:>5}  "
                  f"{diff_ij:>+10.5f}  {contrib:>+12.7f}")

        # Variance content
        vc = variance_content(vF, block_stats, n_orbits)
        print(f"\n    Variance content (sum |v_i*v_j| * var_proxy(i,j)): {vc:.4f}")

    # --- Distribution vs Total decomposition ---
    print("\n" + "=" * 72)
    print("[SECTION 5] Distribution vs Total: variance proxy analysis")
    print("=" * 72)

    # Collect all (i<=j) block stats
    all_blocks = []
    for i in range(n_orbits):
        for j in range(i, n_orbits):
            ec, fr, vp, bs = block_stats[(i, j)]
            all_blocks.append((i, j, ec, fr, vp, bs))

    # Sort by variance_proxy descending
    all_blocks.sort(key=lambda x: -x[4])

    print(f"\n  Top 10 blocks by variance_proxy = frob^2 - edge_count^2 / block_size:")
    print(f"  {'i':>3}  {'j':>3}  {'sz_i':>5}  {'sz_j':>5}  {'edge_ct':>8}  "
          f"{'frob':>8}  {'var_proxy':>10}  {'block_sz':>9}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}")
    for i, j, ec, fr, vp, bs in all_blocks[:10]:
        print(f"  {i:>3}  {j:>3}  {orbit_sizes[i]:>5}  {orbit_sizes[j]:>5}  "
              f"{ec:>8.1f}  {fr:>8.4f}  {vp:>10.4f}  {bs:>9}")

    # High-variance block mask (top 10 blocks)
    top_pairs = set((b[0], b[1]) for b in all_blocks[:10])
    # Also symmetric
    top_pairs_sym = set()
    for i, j in top_pairs:
        top_pairs_sym.add((i, j))
        top_pairs_sym.add((j, i))

    # Project sign-flipping eigenvectors onto high-variance subspace
    # "High variance subspace" = orbits that appear in top-10 high-variance blocks
    hv_orbit_set = set()
    for i, j in top_pairs:
        hv_orbit_set.add(i)
        hv_orbit_set.add(j)
    hv_orbits = sorted(hv_orbit_set)

    print(f"\n  Orbits appearing in top-10 high-variance blocks: {hv_orbits}")
    print(f"  Their sizes: {[orbit_sizes[i] for i in hv_orbits]}")

    print(f"\n  Projection of sign-flipping eigenvectors onto high-variance orbit subspace:")
    for flip_idx, (f_col, e_col, ef, ee, ftype, abs_ip) in enumerate(flips):
        vF = evecs_F[:, f_col]
        hv_energy = sum(vF[i]**2 for i in hv_orbits)
        print(f"    Flip {flip_idx+1} (F-col={f_col}, {ftype}): "
              f"HV-subspace energy = {hv_energy:.5f} / 1.0000")

    # All eigenvectors for comparison
    print(f"\n  HV-subspace energy for ALL 22 eigenvectors (F basis, descending by eigenvalue):")
    print(f"  {'rank':>5}  {'eval_F':>12}  {'HV_energy':>10}  {'flip?':>6}")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*6}")
    flip_f_cols = set(f for f, e, ef, ee, ft, ip in flips)
    for col in range(n_orbits - 1, -1, -1):
        vF = evecs_F[:, col]
        hv_energy = sum(vF[i]**2 for i in hv_orbits)
        rank = n_orbits - col
        mark = "YES" if col in flip_f_cols else ""
        print(f"  {rank:>5}  {evals_F[col]:>+12.6f}  {hv_energy:>10.5f}  {mark:>6}")

    # --- j-axis summary ---
    print("\n" + "=" * 72)
    print("[SECTION 6] j-axis summary")
    print("=" * 72)

    if not flips:
        print("\n  No sign-flipping directions found.")
        return

    print(f"\n  Found {len(flips)} candidate j-axis direction(s):\n")
    for flip_idx, (f_col, e_col, ef, ee, ftype, abs_ip) in enumerate(flips):
        vF = evecs_F[:, f_col]
        pr_F = participation_ratio(vF)
        csc_F = cross_size_content(vF, orbit_sizes)
        vc = variance_content(vF, block_stats, n_orbits)
        hv_energy = sum(vF[i]**2 for i in hv_orbits)

        # Dominant size class
        class_energy = {}
        for cls in label_order:
            idxs = [i for i, l in enumerate(size_labels) if l == cls]
            class_energy[cls] = sum(vF[i]**2 for i in idxs)
        dominant_cls = max(class_energy, key=lambda c: class_energy[c])

        print(f"  Direction {flip_idx+1}:")
        print(f"    Type:             {ftype}  (eval_F={ef:+.6f}, eval_E={ee:+.6f})")
        print(f"    Eigenvec |IP|:    {abs_ip:.6f}  (1.0 = same direction)")
        print(f"    Participation:    PR={pr_F:.2f} / 22  ({'delocalized' if pr_F > 10 else 'localized'})")
        print(f"    Dominant class:   {dominant_cls} ({class_energy[dominant_cls]:.4f} energy)")
        print(f"    Cross-size frac:  {csc_F:.4f}  ({'high cross-class coupling' if csc_F > 0.4 else 'within-class'})")
        print(f"    HV-subspace:      {hv_energy:.4f} energy in high-variance blocks")
        print(f"    Variance content: {vc:.4f}  (F distribution weight)")

    # Cross-class coupling summary
    print(f"\n  Cross-class coupling check for flipping directions:")
    print(f"  (Do they preferentially couple orbits of DIFFERENT sizes?)")
    for flip_idx, (f_col, e_col, ef, ee, ftype, abs_ip) in enumerate(flips):
        vF = evecs_F[:, f_col]
        same_class_energy = 0.0
        cross_class_energy = 0.0
        for i in range(n_orbits):
            for j in range(n_orbits):
                w = abs(vF[i] * vF[j])
                if orbit_sizes[i] == orbit_sizes[j]:
                    same_class_energy += w
                else:
                    cross_class_energy += w
        total = same_class_energy + cross_class_energy
        print(f"    Direction {flip_idx+1}: same-class={same_class_energy/total:.4f}, "
              f"cross-class={cross_class_energy/total:.4f}")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == '__main__':
    main()
