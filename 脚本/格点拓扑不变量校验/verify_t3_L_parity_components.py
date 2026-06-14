# verify_t3_L_parity_components.py
#
# Test prediction: in a BCC L×L×L lattice, the face-adjacency graph on triangular
# simplex faces has parity-driven connectivity that depends on L mod 2.
#
# PREDICTION:
#   OPEN (non-periodic): always 2 components (even-parity faces / odd-parity faces)
#   PERIODIC:
#     L odd  (L=3): 1 component  (wrap by L flips corner parity → parities mix)
#     L even (L=2,4): 2 components (wrap preserves parity → decoupling persists)
#
# Construction:
#   BCs at (i,j,k), i,j,k in range(L).
#   Each BC spawns 2 tetrahedra split by corner parity = (i+dx + j+dy + k+dz) mod 2
#   BEFORE wrapping (unwrapped coordinate sum).
#   Vertices identified AFTER wrapping: ('bc',i%L,j%L,k%L) and ('c',x%L,y%L,z%L).
#   Face = frozenset of 3 vertices (triangular face of 4-simplex = choose 3 from 5).
#   Adjacent = share exactly 2 vertices.
#
# Each simplex has C(5,3)=10 triangular faces; a BC spawns 2 simplices → 20 faces.
# Upper bound on distinct faces: 20 * L^3. At small L identification may reduce this.
#
# Additionally: faces are tagged with parity (0=even, 1=odd) inherited from which
# tet they came from. Parity of a face = parity of the corners in its tet.
# (bc vertex is shared; a face may be {bc,c1,c2} with both c1,c2 from the same-parity
# tet, or {c1,c2,c3} from same tet, or {bc,c1,c2} where c1 is from a tet.)
# The key claim: in OPEN mode, faces from different tet parities share at most 1 vertex
# (only bc can be shared), so no cross-parity edge in the adjacency graph.

import itertools
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def build_faces_for_L(L, periodic):
    """
    Build all triangular simplex faces for BCC L×L×L.
    Returns:
        face_list: list of frozensets (each a set of vertex tuples)
        face_parity: list of ints (0=even-tet, 1=odd-tet) for each face
        degenerate_count: number of face candidates skipped (< 3 distinct vertices)
    """
    face_to_parity = {}   # frozenset -> parity (first assignment wins; checked consistent)
    degenerate_count = 0

    for i, j, k in itertools.product(range(L), repeat=3):
        bc_raw = (i, j, k)
        if periodic:
            bc_v = ('bc', i % L, j % L, k % L)
        else:
            bc_v = ('bc', i, j, k)

        # Enumerate 8 corners with their UNWRAPPED parity
        orig_corners = []
        for dx, dy, dz in itertools.product((0, 1), repeat=3):
            ox, oy, oz = i + dx, j + dy, k + dz
            parity = (ox + oy + oz) % 2  # BEFORE wrapping
            if periodic:
                cv = ('c', ox % L, oy % L, oz % L)
            else:
                cv = ('c', ox, oy, oz)
            orig_corners.append((parity, cv))

        even_corners = [v for (p, v) in orig_corners if p == 0]
        odd_corners  = [v for (p, v) in orig_corners if p == 1]

        for tet_parity, tet_corners in enumerate([even_corners, odd_corners]):
            simplex_verts = [bc_v] + tet_corners  # 5 vertices
            # Generate all C(5,3)=10 triangular faces
            for combo in itertools.combinations(simplex_verts, 3):
                face_set = frozenset(combo)
                # Check for degeneracy (< 3 distinct vertices after wrapping)
                if len(face_set) < 3:
                    degenerate_count += 1
                    continue
                if face_set not in face_to_parity:
                    face_to_parity[face_set] = tet_parity
                # Note: if a face appears in both even and odd tets (only possible
                # if both bc+even_corners and bc+odd_corners contain the same triple),
                # we keep the first assignment. We'll track mixed faces separately.

    face_list = list(face_to_parity.keys())
    face_parity = [face_to_parity[f] for f in face_list]
    return face_list, face_parity, degenerate_count


def build_adjacency_and_components(face_list):
    """
    Build face-adjacency graph (shared exactly 2 vertices) and count components.
    Returns: (n_components, labels)
    """
    N = len(face_list)
    face_idx = {f: i for i, f in enumerate(face_list)}

    # Build vertex -> face index map
    vertex_to_faces = defaultdict(set)
    for idx, face in enumerate(face_list):
        for v in face:
            vertex_to_faces[v].add(idx)

    # Find adjacent pairs
    rows, cols = [], []
    candidate_pairs = set()
    for v, fset in vertex_to_faces.items():
        flist = sorted(fset)
        for a in range(len(flist)):
            for b in range(a + 1, len(flist)):
                candidate_pairs.add((flist[a], flist[b]))

    for i, j in candidate_pairs:
        shared = len(face_list[i] & face_list[j])
        if shared == 2:
            rows.extend([i, j])
            cols.extend([j, i])

    # Build sparse adjacency
    data = [1] * len(rows)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    n_comp, labels = connected_components(A, directed=False, return_labels=True)
    return n_comp, labels


def analyze_parity_mixing(face_parity, labels, n_comp):
    """
    For each component, check if it contains both even and odd parity faces.
    Returns: list of bools (True if component is mixed)
    """
    comp_parities = defaultdict(set)
    for face_idx, (par, comp) in enumerate(zip(face_parity, labels)):
        comp_parities[comp].add(par)
    mixed = [len(comp_parities[c]) > 1 for c in range(n_comp)]
    return mixed


def run_for_L(L):
    """Run both OPEN and PERIODIC analyses for given L. Return result dict."""
    results = {}
    for mode in ['open', 'periodic']:
        periodic = (mode == 'periodic')
        face_list, face_parity, degen = build_faces_for_L(L, periodic)
        n_faces = len(face_list)
        n_comp, labels = build_adjacency_and_components(face_list)
        mixed = analyze_parity_mixing(face_parity, labels, n_comp)
        n_mixed = sum(mixed)
        parity_mixed = (n_mixed > 0)
        results[mode] = {
            'n_faces': n_faces,
            'degenerate': degen,
            'n_components': n_comp,
            'parity_mixed': parity_mixed,
            'n_mixed_components': n_mixed,
        }
    return results


def main():
    print("=" * 72)
    print("VERIFY T3 L-PARITY COMPONENTS: BCC L×L×L Face-Adjacency Graph")
    print("=" * 72)
    print()
    print("PREDICTION:")
    print("  OPEN: always 2 components (even-face / odd-face disconnected)")
    print("  PERIODIC odd L (L=3): 1 component (wrap flips parity → parities mix)")
    print("  PERIODIC even L (L=2,4): 2 components (wrap preserves parity)")
    print()

    Ls = [2, 3, 4]
    all_results = {}

    for L in Ls:
        print(f"--- L = {L} ---")
        r = run_for_L(L)
        all_results[L] = r
        for mode in ['open', 'periodic']:
            m = r[mode]
            print(f"  [{mode.upper():8s}] faces={m['n_faces']:5d}  "
                  f"degen={m['degenerate']:3d}  "
                  f"components={m['n_components']}  "
                  f"parity_mixed={'YES' if m['parity_mixed'] else 'NO '}")
        print()

    # Summary table
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    header = f"{'L':>3}  {'Open comp':>10}  {'Periodic comp':>14}  {'Parity mixed?':>14}  {'Face count (P)':>15}  {'Degen (P)':>10}"
    print(header)
    print("-" * len(header))
    for L in Ls:
        r = all_results[L]
        o = r['open']
        p = r['periodic']
        print(f"{L:>3}  {o['n_components']:>10}  {p['n_components']:>14}  "
              f"{'YES' if p['parity_mixed'] else 'NO':>14}  "
              f"{p['n_faces']:>15}  {p['degenerate']:>10}")

    print()
    print("=" * 72)
    print("PASS/FAIL EVALUATION")
    print("=" * 72)

    all_pass = True

    # OPEN: always 2 components
    for L in Ls:
        o = all_results[L]['open']
        ok = (o['n_components'] == 2)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] L={L} OPEN: {o['n_components']} components (expected 2)")
        if not ok:
            all_pass = False

    print()

    # PERIODIC predictions
    periodic_predictions = {
        2: {'expected_comp': 2, 'expected_mixed': False},
        3: {'expected_comp': 1, 'expected_mixed': True},
        4: {'expected_comp': 2, 'expected_mixed': False},
    }
    for L in Ls:
        p = all_results[L]['periodic']
        pred = periodic_predictions[L]
        comp_ok = (p['n_components'] == pred['expected_comp'])
        mix_ok = (p['parity_mixed'] == pred['expected_mixed'])
        ok = comp_ok and mix_ok
        status = "PASS" if ok else "FAIL"
        exp_mix = "YES" if pred['expected_mixed'] else "NO"
        got_mix = "YES" if p['parity_mixed'] else "NO"
        print(f"[{status}] L={L} PERIODIC: {p['n_components']} components "
              f"(expected {pred['expected_comp']}), "
              f"parity_mixed={got_mix} (expected {exp_mix})")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("OVERALL: PASS — parity prediction confirmed for all L in {2, 3, 4}")
    else:
        print("OVERALL: FAIL — parity prediction NOT confirmed for at least one L")

    # Extra diagnostics: for each L, report component parity breakdown
    print()
    print("=" * 72)
    print("COMPONENT PARITY BREAKDOWN (PERIODIC)")
    print("=" * 72)
    for L in Ls:
        periodic = True
        face_list, face_parity, degen = build_faces_for_L(L, periodic)
        n_comp, labels = build_adjacency_and_components(face_list)
        from collections import Counter
        comp_parities = defaultdict(Counter)
        for idx, (par, comp) in enumerate(zip(face_parity, labels)):
            comp_parities[comp][par] += 1
        print(f"L={L}: {n_comp} component(s)")
        for c in range(n_comp):
            ev = comp_parities[c].get(0, 0)
            od = comp_parities[c].get(1, 0)
            total = ev + od
            mixed_tag = " [MIXED]" if (ev > 0 and od > 0) else ""
            print(f"  Component {c}: {total} faces  (even={ev}, odd={od}){mixed_tag}")


if __name__ == "__main__":
    main()
