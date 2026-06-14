# verify_t3_orbit22_coupling.py
# Verification: 3x3x3 BCC lattice complex -- 22 O_h orbit decomposition of 540 triangular faces
# and corrected coupling spectrum analysis.
#
# Background: The earlier Gemini claim of 15 O_h orbits was falsified.
# True decomposition: 22 orbits (4×48 + 12×24 + 3×12 + 3×8 = 540)
# O_h group (order 48) acts about center (1.5, 1.5, 1.5).
#
# This script:
#   1. Builds the BCC lattice faces in OPEN (non-periodic) geometric coordinates
#   2. Applies all 48 O_h symmetries to classify faces into orbits
#   3. Verifies 22 orbits with the correct size census
#   4. Computes 22x22 orbit coupling matrix from the 540x540 face adjacency matrix A
#   5. Structural readings on the corrected spectrum
#   6. Chiral full-rank check (Laplacian rank, twisted variant)
#   7. Crown graph check per orbit

import itertools
import numpy as np
from scipy import stats
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Section 1: Build BCC lattice faces in OPEN geometric coordinates
# ---------------------------------------------------------------------------
# Body-centers: ('bc', i, j, k) at geometric position (i+0.5, j+0.5, k+0.5), i,j,k in {0,1,2}
# SC corners:   ('c', x, y, z) at geometric position (x, y, z), x,y,z in {0,1,2,3}
# (NOT wrapped mod 3 -- open boundary)

def get_geom_pos(v):
    if v[0] == 'bc':
        _, i, j, k = v
        return np.array([i + 0.5, j + 0.5, k + 0.5])
    else:
        _, x, y, z = v
        return np.array([float(x), float(y), float(z)])


def build_faces_open():
    """Build all triangular faces of the BCC 3x3x3 complex in open (non-periodic) coords.
    Each BC at (i,j,k) has 8 corners; they split into 2 parity classes giving 2 tetrahedra;
    each tetrahedron has 4 triangular faces (C(4,3)=4); but we actually use C(5,3)=10 per tet
    (1 BC + 4 corners = 5 vertices; each face is a 3-subset). Wait -- let's recheck:
    the original code uses simplex_verts = [bc_v] + tet_corners (5 vertices total),
    then C(5,3) = 10 faces per tet, 20 faces per BC, 20*27 = 540 faces total.
    We replicate that here with OPEN coords (no mod 3 wrapping on corners).
    """
    face_set = {}  # frozenset -> index
    face_list = []
    bc_face_map = {}  # bc_ijk -> list of face indices

    for i, j, k in itertools.product(range(3), repeat=3):
        bc_v = ('bc', i, j, k)
        # 8 corners, split by parity (WITHOUT mod 3)
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
            simplex_verts = [bc_v] + tet_corners  # 5 vertices
            for combo in itertools.combinations(simplex_verts, 3):
                f = frozenset(combo)
                if f not in face_set:
                    face_set[f] = len(face_list)
                    face_list.append(f)
                bc_faces.append(face_set[f])
        bc_face_map[(i, j, k)] = bc_faces

    return face_list, face_set, bc_face_map


def build_adjacency_open(face_list, face_set):
    """Build 540x540 adjacency matrix: two faces are adjacent iff they share exactly 2 vertices."""
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
# Section 2: O_h symmetry group (48 elements) about center (1.5, 1.5, 1.5)
# ---------------------------------------------------------------------------
# O_h = all signed permutation matrices (3x3 matrices with exactly one nonzero
# entry per row and column, that entry being +1 or -1)

def generate_oh_group():
    """Generate all 48 signed permutation matrices."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            M = np.zeros((3, 3), dtype=float)
            for row, col in enumerate(perm):
                M[row, col] = signs[row]
            mats.append(M)
    assert len(mats) == 48
    return mats


CENTER = np.array([1.5, 1.5, 1.5])


def apply_oh_to_vertex(v, M):
    """Apply O_h matrix M (about CENTER) to a vertex, return transformed position."""
    pos = get_geom_pos(v) - CENTER
    new_pos = M @ pos + CENTER
    return new_pos


def pos_to_vertex(pos):
    """Convert geometric position back to vertex label, or None if out of range."""
    # Check if it's a BC (half-integer coordinates)
    half = pos - 0.5
    if np.allclose(half, np.round(half), atol=1e-9):
        i, j, k = int(round(half[0])), int(round(half[1])), int(round(half[2]))
        if 0 <= i <= 2 and 0 <= j <= 2 and 0 <= k <= 2:
            return ('bc', i, j, k)
    # Check if it's a corner (integer coordinates)
    if np.allclose(pos, np.round(pos), atol=1e-9):
        x, y, z = int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))
        if 0 <= x <= 3 and 0 <= y <= 3 and 0 <= z <= 3:
            return ('c', x, y, z)
    return None


def apply_oh_to_face(face, M):
    """Apply O_h matrix M to a face (frozenset of 3 vertices). Return transformed frozenset or None."""
    new_verts = []
    for v in face:
        new_pos = apply_oh_to_vertex(v, M)
        new_v = pos_to_vertex(new_pos)
        if new_v is None:
            return None
        new_verts.append(new_v)
    return frozenset(new_verts)


def build_oh_orbits(face_list, face_set, oh_group):
    """Partition the 540 faces into O_h orbits."""
    N = len(face_list)
    assigned = [-1] * N
    orbits = []
    orbit_of = [-1] * N

    # Build action table: action_table[sym_idx][face_idx] = image_face_idx (or -1)
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

    print(f"  Unmapped (face -> outside lattice): {unmapped}")
    assert unmapped == 0, f"Expected 0 unmapped faces, got {unmapped}"

    for start in range(N):
        if assigned[start] != -1:
            continue
        orbit_idx = len(orbits)
        # BFS over orbit
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
# Section 3: Verify 22 orbits with correct size census
# ---------------------------------------------------------------------------

def verify_orbits(orbits):
    sizes = sorted([len(o) for o in orbits])
    size_counts = Counter(sizes)
    print(f"\n  Number of orbits: {len(orbits)}")
    print(f"  Size distribution: {dict(sorted(size_counts.items()))}")
    print(f"  Total faces covered: {sum(sizes)}")

    expected_total = 540
    expected_num = 22
    expected_census = {48: 4, 24: 12, 12: 3, 8: 3}

    ok = True
    if len(orbits) != expected_num:
        print(f"  FAIL: expected {expected_num} orbits, got {len(orbits)}")
        ok = False
    else:
        print(f"  PASS: 22 orbits confirmed")

    if size_counts != Counter(expected_census):
        print(f"  FAIL: size census mismatch. Expected {expected_census}, got {dict(sorted(size_counts.items()))}")
        ok = False
    else:
        print(f"  PASS: size census 4×48 + 12×24 + 3×12 + 3×8 = 540 confirmed")

    if sum(sizes) != expected_total:
        print(f"  FAIL: total faces {sum(sizes)} != {expected_total}")
        ok = False
    else:
        print(f"  PASS: total faces = 540")

    return ok


# ---------------------------------------------------------------------------
# Section 4: 22x22 orbit coupling matrix
# ---------------------------------------------------------------------------

def compute_orbit_coupling(A, orbits):
    """Compute the 22x22 orbit coupling matrix from the 540x540 adjacency matrix A.

    Two conventions:
      - Raw Frobenius norm of the block A[orbit_i, orbit_j]
      - Normalized by sqrt(|orbit_i| * |orbit_j|)
    """
    n_orbits = len(orbits)
    C_raw = np.zeros((n_orbits, n_orbits))
    C_norm = np.zeros((n_orbits, n_orbits))

    for i in range(n_orbits):
        for j in range(i, n_orbits):
            block = A[np.ix_(orbits[i], orbits[j])]
            frob = np.linalg.norm(block, 'fro')
            C_raw[i, j] = frob
            C_raw[j, i] = frob
            norm_factor = np.sqrt(len(orbits[i]) * len(orbits[j]))
            C_norm[i, j] = frob / norm_factor
            C_norm[j, i] = frob / norm_factor

    return C_raw, C_norm


def analyze_coupling_matrix(C, label):
    """Compute and print eigenvalue analysis of symmetric coupling matrix."""
    evals = np.linalg.eigvalsh(C)
    evals_sorted = np.sort(evals)[::-1]  # descending

    print(f"\n  {label} -- eigenvalues (descending):")
    for i, ev in enumerate(evals_sorted):
        print(f"    [{i+1:2d}] {ev:+.6f}")

    n_pos = np.sum(evals > 1e-10)
    n_neg = np.sum(evals < -1e-10)
    n_zero = len(evals) - n_pos - n_neg
    print(f"\n  Signature: {n_pos} positive, {n_zero} zero, {n_neg} negative")
    print(f"  Range: [{evals_sorted[-1]:.6f}, {evals_sorted[0]:.6f}]")

    # Degeneracy check
    rounded = np.round(evals_sorted, 8)
    degen = Counter(rounded)
    degens = {v: cnt for v, cnt in degen.items() if cnt > 1}
    if degens:
        print(f"  Degeneracies: {degens}")
    else:
        print(f"  No exact degeneracies detected")

    return evals_sorted


def top_eigenvec_vs_orbit_sizes(C_norm, orbits):
    """Pearson correlation of top eigenvector amplitudes vs orbit sizes."""
    evals, evecs = np.linalg.eigh(C_norm)
    top_vec = evecs[:, -1]  # largest eigenvalue
    sizes = np.array([len(o) for o in orbits])
    amplitudes = np.abs(top_vec)

    r, p = stats.pearsonr(amplitudes, sizes)
    print(f"\n  Top eigenvector (norm-conv) amplitude vs orbit size: Pearson r={r:.4f}, p={p:.4e}")
    return r, p


# ---------------------------------------------------------------------------
# Section 5: Chiral full-rank check
# ---------------------------------------------------------------------------

def chiral_rank_check(A, face_list, bc_face_map):
    """
    Each face's parent BC gives its sign: sigma(i,j,k) = (-1)^(i+j+k).
    Build twisted matrix W_ij = sigma(parent_i) * sigma(parent_j) * A_ij.
    Check rank of graph Laplacian L = D - A and twisted Laplacian L_W = D_W - W.
    Claim: rank(L) = 539 (1 zero mode), rank(L_W) = 540 (0 zero modes).
    """
    # Map face index -> parent BC ijk
    face_to_bc = {}
    for bc_ijk, face_indices in bc_face_map.items():
        for fi in face_indices:
            face_to_bc[fi] = bc_ijk

    # BC sign
    def bc_sign(ijk):
        i, j, k = ijk
        return (-1) ** (i + j + k)

    N = len(face_list)
    sigma = np.array([bc_sign(face_to_bc[fi]) for fi in range(N)])

    # Twisted adjacency
    W = A * np.outer(sigma, sigma)

    # Standard graph Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Twisted Laplacian
    D_W = np.diag(np.abs(W).sum(axis=1))
    L_W = D_W - W

    print(f"\n  Computing ranks (using threshold 1e-8 on singular values)...")
    sv_L = np.linalg.svd(L, compute_uv=False)
    sv_LW = np.linalg.svd(L_W, compute_uv=False)

    rank_L = int(np.sum(sv_L > 1e-8))
    rank_LW = int(np.sum(sv_LW > 1e-8))

    print(f"  Laplacian L rank:         {rank_L} (expect 539, i.e., 1 zero mode)")
    print(f"  Twisted Laplacian L_W rank: {rank_LW} (expect 540, i.e., 0 zero modes / full rank)")

    zero_modes_L = N - rank_L
    zero_modes_LW = N - rank_LW
    print(f"  Zero modes of L:   {zero_modes_L} (expect 1)")
    print(f"  Zero modes of L_W: {zero_modes_LW} (expect 0)")

    pass_L = (zero_modes_L == 1)
    pass_LW = (zero_modes_LW == 0)
    print(f"  {'PASS' if pass_L else 'FAIL'}: untwisted Laplacian has 1 zero mode")
    print(f"  {'PASS' if pass_LW else 'FAIL'}: twisted Laplacian has 0 zero modes (full rank 540)")

    return rank_L, rank_LW


# ---------------------------------------------------------------------------
# Section 6: Crown graph check per orbit
# ---------------------------------------------------------------------------

def check_orbit_internal_structure(A, orbits):
    """
    For each orbit, extract orbit-internal adjacency sub-block and analyze structure.
    Crown graph K_{m,m} minus perfect matching: 2m vertices, m-regular, bipartite,
    each vertex non-adjacent to exactly one vertex in opposite part.
    Also check size-8 orbits for (8, 3, 2, 0) = 2 disjoint K_4.
    """
    size_classes = defaultdict(list)
    for i, o in enumerate(orbits):
        size_classes[len(o)].append(i)

    for size in sorted(size_classes.keys()):
        orbit_indices = size_classes[size]
        print(f"\n  Orbit size {size} ({len(orbit_indices)} orbits):")
        for oi in orbit_indices:
            o = orbits[oi]
            sub = A[np.ix_(o, o)]
            degrees = sub.sum(axis=1).astype(int)
            deg_counts = Counter(degrees)
            is_regular = (len(deg_counts) == 1)
            reg_deg = degrees[0] if is_regular else None

            # Bipartiteness check via BFS coloring
            color = [-1] * len(o)
            color[0] = 0
            queue = [0]
            bipartite = True
            parts = [[], []]
            parts[0].append(0)
            while queue:
                cur = queue.pop()
                for nb in range(len(o)):
                    if sub[cur, nb] > 0:
                        if color[nb] == -1:
                            color[nb] = 1 - color[cur]
                            parts[color[nb]].append(nb)
                            queue.append(nb)
                        elif color[nb] == color[cur]:
                            bipartite = False
                            break

            # Crown graph check: bipartite, regular, and each vertex has exactly 1 non-neighbor in opp part
            crown_ok = False
            if bipartite and is_regular and len(parts[0]) == len(parts[1]):
                m = len(parts[0])
                # Each vertex in part0 should be adjacent to all of part1 except one
                crown_ok = True
                for v in parts[0]:
                    non_adj_in_opp = [u for u in parts[1] if sub[v, u] == 0]
                    if len(non_adj_in_opp) != 1:
                        crown_ok = False
                        break
                if crown_ok:
                    for v in parts[1]:
                        non_adj_in_opp = [u for u in parts[0] if sub[v, u] == 0]
                        if len(non_adj_in_opp) != 1:
                            crown_ok = False
                            break

            # Size-8: check for 2 disjoint K_4 pattern (strongly regular (8,3,2,0))
            k4_check = None
            if size == 8:
                # SRG(8,3,2,0): 8 vertices, 3-regular, any two adjacent vertices have 2 common neighbors,
                # any two non-adjacent vertices have 0 common neighbors
                if is_regular and reg_deg == 3:
                    A2 = sub @ sub
                    srg_ok = True
                    for u in range(size):
                        for v in range(size):
                            if u == v:
                                continue
                            cn = A2[u, v]
                            if sub[u, v] == 1:
                                if int(round(cn)) != 2:
                                    srg_ok = False
                            else:
                                if int(round(cn)) != 0:
                                    srg_ok = False
                    k4_check = srg_ok

            print(f"    Orbit {oi}: degree_seq={dict(deg_counts)}, regular={is_regular}"
                  f"{'(deg='+str(reg_deg)+')' if is_regular else ''}, bipartite={bipartite}")
            if crown_ok:
                m = size // 2
                print(f"      -> Crown graph K_{{{m},{m}}} minus perfect matching: CONFIRMED")
            elif bipartite:
                print(f"      -> Bipartite but not crown graph pattern")
            else:
                print(f"      -> Not bipartite (crown graph ruled out)")
            if k4_check is not None:
                print(f"      -> SRG(8,3,2,0) = 2 disjoint K_4: {'CONFIRMED' if k4_check else 'NOT confirmed'}")

            # Also compute eigenvalue spectrum of sub-block
            evals_sub = np.sort(np.linalg.eigvalsh(sub))[::-1]
            print(f"      Eigenvalues: [{', '.join(f'{e:.3f}' for e in evals_sub)}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("VERIFY T3 ORBIT-22 COUPLING")
    print("3x3x3 BCC lattice -- 22 O_h orbit decomposition and coupling spectrum")
    print("=" * 72)

    # --- Build faces ---
    print("\n[SECTION 1] Building BCC faces (open, non-periodic coordinates)")
    face_list, face_set, bc_face_map = build_faces_open()
    N = len(face_list)
    print(f"  Total faces: {N} (expect 540)")
    assert N == 540, f"Expected 540 faces, got {N}"
    print("  PASS: 540 faces")

    # --- Build adjacency ---
    print("\n[SECTION 2] Building 540x540 face adjacency matrix")
    A = build_adjacency_open(face_list, face_set)
    print(f"  A shape: {A.shape}, symmetric: {np.allclose(A, A.T)}")
    print(f"  Total edges (non-zero off-diagonal / 2): {int(A.sum()) // 2}")
    degrees = A.sum(axis=1)
    print(f"  Degree stats: min={degrees.min():.0f}, max={degrees.max():.0f}, mean={degrees.mean():.2f}")

    # --- O_h orbits ---
    print("\n[SECTION 3] Computing O_h orbits about center (1.5, 1.5, 1.5)")
    oh_group = generate_oh_group()
    print(f"  O_h group: {len(oh_group)} elements")
    orbits, orbit_of, action_table = build_oh_orbits(face_list, face_set, oh_group)
    orbit_ok = verify_orbits(orbits)

    # Sort orbits by size (descending) for cleaner output
    orbits_sorted = sorted(orbits, key=lambda o: -len(o))
    orbit_sizes = [len(o) for o in orbits_sorted]

    # --- Coupling matrix ---
    print("\n[SECTION 4] 22x22 orbit coupling matrix")
    C_raw, C_norm = compute_orbit_coupling(A, orbits_sorted)

    print("\n  Orbit sizes (sorted descending):")
    print("  " + " ".join(f"{s:3d}" for s in orbit_sizes))

    print("\n  [4a] Raw Frobenius norm convention:")
    evals_raw = analyze_coupling_matrix(C_raw, "Raw Frobenius")

    print("\n  [4b] Normalized (Frobenius / sqrt(|Oi||Oj|)) convention:")
    evals_norm = analyze_coupling_matrix(C_norm, "Normalized")

    print("\n  [4c] Cross-check against prior range claims:")
    print(f"  Raw range: [{evals_raw[-1]:.4f}, {evals_raw[0]:.4f}]")
    print(f"  Normalized range: [{evals_norm[-1]:.4f}, {evals_norm[0]:.4f}]")
    print(f"  Prior independent run reported normalized range ~[-0.21, +0.79]")
    print(f"  Falsified 15-orbit range was [-1.10, +8.56]")

    # Top eigenvec vs orbit sizes
    print("\n  [4d] Top eigenvector correlation with orbit sizes:")
    r, p = top_eigenvec_vs_orbit_sizes(C_norm, orbits_sorted)

    # --- Structural readings ---
    print("\n[SECTION 5] Structural readings on corrected spectrum (normalized conv)")
    evals_n, evecs_n = np.linalg.eigh(C_norm)
    evals_n_sorted = np.sort(evals_n)[::-1]

    n_pos = int(np.sum(evals_n > 1e-10))
    n_neg = int(np.sum(evals_n < -1e-10))
    n_zero = 22 - n_pos - n_neg
    print(f"  Signature: ({n_pos}+, {n_zero}0, {n_neg}-)")

    # Gap structure
    print("\n  Gap structure (consecutive eigenvalue differences > 0.01):")
    for i in range(len(evals_n_sorted) - 1):
        gap = evals_n_sorted[i] - evals_n_sorted[i+1]
        if gap > 0.01:
            print(f"    Between eigenvalue {i+1} ({evals_n_sorted[i]:.4f}) and "
                  f"{i+2} ({evals_n_sorted[i+1]:.4f}): gap = {gap:.4f}")

    # Isolated modes (large gap on both sides)
    print("\n  Isolated modes (gap > 0.05 on both sides):")
    isolated = []
    for i in range(len(evals_n_sorted)):
        gap_above = evals_n_sorted[i-1] - evals_n_sorted[i] if i > 0 else float('inf')
        gap_below = evals_n_sorted[i] - evals_n_sorted[i+1] if i < len(evals_n_sorted)-1 else float('inf')
        if gap_above > 0.05 and gap_below > 0.05:
            isolated.append((i, evals_n_sorted[i]))
            print(f"    Eigenvalue {i+1}: {evals_n_sorted[i]:.4f} (gaps: {gap_above:.4f}/{gap_below:.4f})")
    if not isolated:
        print("    None found at threshold 0.05")

    # --- Connected components diagnostic ---
    print("\n[SECTION 5b] Connected components of face adjacency graph (open coords)")
    from collections import defaultdict as _dd
    adj_dict = _dd(set)
    for _i in range(N):
        for _j in range(_i + 1, N):
            if A[_i, _j] > 0:
                adj_dict[_i].add(_j)
                adj_dict[_j].add(_i)
    visited_c = [False] * N
    components = []
    for _start in range(N):
        if not visited_c[_start]:
            comp = []
            _q = [_start]
            visited_c[_start] = True
            while _q:
                _cur = _q.pop()
                comp.append(_cur)
                for _nb in adj_dict[_cur]:
                    if not visited_c[_nb]:
                        visited_c[_nb] = True
                        _q.append(_nb)
            components.append(len(comp))
    print(f"  Number of connected components: {len(components)}")
    print(f"  Component sizes: {sorted(components, reverse=True)}")
    if len(components) == 2:
        print(f"  NOTE: Open construction separates even-tet from odd-tet faces")
        print(f"  (no shared edges cross parity in open coords -> 2 equal components of 270).")
        print(f"  This is why both Laplacians have 2 zero modes, not 1 and 0.")
        print(f"  The 1-zero-mode claim applies to the PERIODIC construction (wrapping")
        print(f"  creates cross-parity adjacencies). Open coords: 2 components, 2 zero modes.")

    # --- Chiral rank check ---
    print("\n[SECTION 6] Chiral full-rank check")
    rank_L, rank_LW = chiral_rank_check(A, face_list, bc_face_map)

    # --- Crown graph check ---
    print("\n[SECTION 7] Crown graph / internal structure check per orbit size class")
    check_orbit_internal_structure(A, orbits_sorted)

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  22-orbit confirmation: {'PASS' if orbit_ok else 'FAIL'}")
    print(f"  540 faces: PASS")
    print(f"  Raw eigenvalue range: [{evals_raw[-1]:.4f}, {evals_raw[0]:.4f}]")
    print(f"  Normalized eigenvalue range: [{evals_norm[-1]:.4f}, {evals_norm[0]:.4f}]")
    print(f"  Laplacian rank: {rank_L} (expect 539)")
    print(f"  Twisted Laplacian rank: {rank_LW} (expect 540)")


if __name__ == "__main__":
    main()
