# -*- coding: utf-8 -*-
# cross_dim_census.py
# Cross-dimensional census table for the d-cascade (0D through 8D).
# Tabulates combinatorial invariants at each level.
# Goal: identify what 5D uniquely contributes, and what combinatorial
# quantity must appear there to preserve cross-dimensional identities.

import math
import sys
import io
from collections import OrderedDict

# Force UTF-8 stdout to handle Unicode symbols on Windows (GBK terminal)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------------------------------------------------------------
# Dimension data
# Each entry: dict with all known/computed invariants.
# Fields:
#   name            -- key structure
#   vertices        -- number of 0-cells
#   edges           -- number of 1-cells
#   faces           -- number of 2-cells
#   cells3          -- number of 3-cells (None if not applicable)
#   euler           -- Euler characteristic chi
#   sym_name        -- symmetry group name
#   sym_order       -- |Aut| or |G|
#   bott_pi         -- pi_n(O) in stable range (Bott periodicity)
#   absorbs         -- which algebraic axis this dimension absorbs
#   notes           -- extra structural notes
# ---------------------------------------------------------------------------

DIM_DATA = OrderedDict([
    (0, dict(
        name        = "S0 = {+/-1}",
        vertices    = 2,
        edges       = 0,
        faces       = 0,
        cells3      = 0,
        euler       = 2,            # chi(S0) = 2 (two points)
        sym_name    = "Z_2",
        sym_order   = 2,
        bott_pi     = "Z_2",        # pi_0(O) = Z_2
        absorbs     = "distinction operator (S0)",
        notes       = "Binary distinction. Identity seed of all higher structure.",
    )),
    (1, dict(
        name        = "Connectivity / 1-simplex",
        vertices    = 2,
        edges       = 1,
        faces       = 0,
        cells3      = 0,
        euler       = 1,            # chi(interval) = 1
        sym_name    = "Z_2",
        sym_order   = 2,
        bott_pi     = "0",          # pi_1(O) = 0
        absorbs     = "pi_1 (irreplaceable connectivity)",
        notes       = "Links as 1-simplices. Irreplaceable by higher structure.",
    )),
    (2, dict(
        name        = "Fano plane PG(2,2)",
        vertices    = 7,
        edges       = 7,
        faces       = 1,            # 1 projective plane
        cells3      = 0,
        euler       = 1,            # projective plane: chi=1
        sym_name    = "GL(3,F_2) ~= PSL(2,7)",
        sym_order   = 168,
        bott_pi     = "0",          # pi_2(O) = 0 (stable)
        absorbs     = "octonion imaginary units (7 points = G_2 root seed)",
        notes       = "7 points, 7 lines, 3 pts/line, 3 lines/pt. Fano incidence.",
    )),
    (3, dict(
        name        = "Simple Cubic (SC) lattice Z^3",
        vertices    = 64,           # 4^3 corners in 3x3x3 open: 64
        edges       = None,         # coordination 6 -> not a single finite graph
        faces       = None,
        cells3      = 27,           # 27 unit cells in 3x3x3
        euler       = None,
        sym_name    = "O_h",
        sym_order   = 48,
        bott_pi     = "Z",          # pi_3(O) = Z (Bott)
        absorbs     = "quaternion real part (1-axis)",
        notes       = "Coordination 6. Lattice Z^3. O_h symmetry order 48.",
    )),
    (4, dict(
        name        = "BCC 3x3x3 (27 body-centers)",
        vertices    = 27,           # 27 BC centers
        edges       = None,         # tetrahedral structure
        faces       = 540,          # 20 faces per BC x 27 = 540 triangular faces
        cells3      = None,
        euler       = None,
        sym_name    = "O_h",
        sym_order   = 48,
        bott_pi     = "0",          # pi_4(O) = 0 (Bott)
        absorbs     = "quaternion i-axis",
        notes       = (
            "20 triangular faces per BC (C(5,3)=10 per tet, 2 tets per BC). "
            "540 = 20x27. Stella octangula structure. "
            "22 O_h orbits: 4x48 + 12x24 + 3x12 + 3x8 = 540."
        ),
    )),
    (5, dict(
        name        = "22 O_h orbits -- 5D coupling layer",
        vertices    = 22,           # 22 orbit classes = 22 nodes in orbit space
        edges       = None,         # 22x22 coupling matrix non-zero off-diag
        faces       = None,
        cells3      = None,
        euler       = None,
        sym_name    = "O_h (inherited)",
        sym_order   = 48,
        bott_pi     = "0",          # pi_5(O) = 0 (Bott)
        absorbs     = "quaternion j-axis (second-order relations, cross-terms)",
        notes       = (
            "UNKNOWN -- target of investigation. "
            "22 orbits live here as second-order structure. "
            "j-axis = relations BETWEEN relations (orbit coupling). "
            "Spectral signature convention-dependent (NOT a fixed invariant): "
            "(10+,1,11-) under Frobenius/geomean norm (orbit22_coupling_pinned.py), "
            "(13+,0,9-) under edge-count norm (j_axis_sign_flip.py). "
            "54 = 27x2 = oriented BCs? Or 54 = 540/10? "
            "Or 54 = 6x9 = coordination x orbit-size-class?"
        ),
    )),
    (6, dict(
        name        = "G_2 zero-mode / Fano-octonion layer",
        vertices    = 7,            # 7 imaginary octonion units = G_2 generators
        edges       = 7,            # 7 lines of Fano = G_2 root structure
        faces       = None,
        cells3      = None,
        euler       = None,
        sym_name    = "G_2",
        sym_order   = 12096,        # |G_2| compact exceptional group
        bott_pi     = "0",          # pi_6(O) = 0 (Bott)
        absorbs     = "quaternion k-axis (ij composite / octonion associator)",
        notes       = (
            "G_2 zero-mode appears. 7 points of Fano = 7 imaginary octonion units. "
            "G_2 is the automorphism group of octonions."
        ),
    )),
    (7, dict(
        name        = "Bott return Z / pi_7(O)=Z",
        vertices    = None,
        edges       = None,
        faces       = None,
        cells3      = None,
        euler       = None,
        sym_name    = "O (Bott period 7 return)",
        sym_order   = None,
        bott_pi     = "Z",          # pi_7(O) = Z (same as pi_3)
        absorbs     = "octonion associativity defect (3D echo)",
        notes       = "Bott homotopy returns Z like dim 3. Mirror to 3D.",
    )),
    (8, dict(
        name        = "Bott period closes / 8D cobordism",
        vertices    = None,
        edges       = None,
        faces       = None,
        cells3      = None,
        euler       = None,
        sym_name    = "Spin(8) / triality",
        sym_order   = None,
        bott_pi     = "Z_2",        # pi_8(O) = Z_2 (Bott table row 0 repeats)
        absorbs     = "cobordism resolution of 7D defect",
        notes       = (
            "Bott period 8 closes. "
            "pi_8(O) = Z_2 (Bott table: Z_2,Z_2,0,Z,0,0,0,Z then repeats). "
            "8D resolves 7D defect via cobordism."
        ),
    )),
])

# Stable homotopy groups of O: pi_n(O) for n mod 8
# 0:Z_2, 1:Z_2, 2:0, 3:Z, 4:0, 5:0, 6:0, 7:Z
BOTT_SEQUENCE = {
    0: "Z_2",
    1: "Z_2",
    2: "0",
    3: "Z",
    4: "0",
    5: "0",
    6: "0",
    7: "Z",
}

def bott_corrected(n):
    return BOTT_SEQUENCE.get(n % 8, "?")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(v, width=8, na="--"):
    if v is None:
        return na.center(width)
    if isinstance(v, float):
        return f"{v:.4f}".rjust(width)
    return str(v).rjust(width)

def fmt_str(v, width=24, na="--"):
    if v is None:
        return na.ljust(width)
    s = str(v)
    if len(s) > width:
        s = s[:width-1] + "~"
    return s.ljust(width)

def section(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)

def subsection(title):
    bar = "-" * max(0, 74 - len(title))
    print()
    print(f"--- {title} {bar}")

# ---------------------------------------------------------------------------
# Section 1: Main census table
# ---------------------------------------------------------------------------

def print_census_table():
    section("CROSS-DIMENSIONAL CENSUS TABLE: d-CASCADE 0D -> 8D")

    header = (
        f"{'Dim':>3}  "
        f"{'Structure':<28}  "
        f"{'Verts':>6}  "
        f"{'Edges':>6}  "
        f"{'Faces':>6}  "
        f"{'3-cells':>7}  "
        f"{'chi':>4}  "
        f"{'|Sym|':>7}  "
        f"{'pi_n(O)':<8}  "
        f"{'Absorbs':<32}"
    )
    print()
    print(header)
    print("-" * len(header))

    for dim, d in DIM_DATA.items():
        bott = bott_corrected(dim)
        row = (
            f"{dim:>3}  "
            f"{fmt_str(d['name'], 28)}  "
            f"{fmt(d['vertices'], 6)}  "
            f"{fmt(d['edges'], 6)}  "
            f"{fmt(d['faces'], 6)}  "
            f"{fmt(d['cells3'], 7)}  "
            f"{fmt(d['euler'], 4)}  "
            f"{fmt(d['sym_order'], 7)}  "
            f"{bott:<8}  "
            f"{fmt_str(d['absorbs'], 32)}"
        )
        print(row)

    print()
    print("  Notes:")
    print("  chi = Euler characteristic. -- = not applicable or unknown.")
    print("  pi_n(O): stable homotopy groups of O (Bott periodicity period 8).")
    print("  Bott sequence (n mod 8): 0->Z_2, 1->Z_2, 2->0, 3->Z, 4->0, 5->0, 6->0, 7->Z")

# ---------------------------------------------------------------------------
# Section 2: Cross-dimensional ratios
# ---------------------------------------------------------------------------

def compute_ratios():
    subsection("CROSS-DIMENSIONAL RATIOS")

    print()
    print("  Vertex ratios (dim n / dim n-1):")
    v_prev = None
    for dim, d in DIM_DATA.items():
        v = d['vertices']
        if v is not None and v_prev is not None:
            ratio = v / v_prev
            print(f"    dim {dim-1}->{dim}: {v_prev} -> {v}  ratio = {ratio:.4f}")
        elif v is not None:
            print(f"    dim {dim}: {v} (baseline)")
        else:
            print(f"    dim {dim}: -- (unknown)")
        if v is not None:
            v_prev = v

    print()
    print("  Face ratios (dim n / dim n-1):")
    f_prev = None
    for dim, d in DIM_DATA.items():
        f = d['faces']
        if f is not None and f_prev is not None:
            if f_prev == 0:
                print(f"    dim {dim-1}->{dim}: {f_prev} -> {f}  ratio = N/A (prev=0)")
            else:
                ratio = f / f_prev
                print(f"    dim {dim-1}->{dim}: {f_prev} -> {f}  ratio = {ratio:.4f}")
        elif f is not None:
            print(f"    dim {dim}: {f} (baseline)")
        else:
            print(f"    dim {dim}: -- (unknown)")
        if f is not None:
            f_prev = f

    print()
    print("  Symmetry order ratios:")
    s_prev = None
    for dim, d in DIM_DATA.items():
        s = d['sym_order']
        if s is not None and s_prev is not None:
            ratio = s / s_prev
            print(f"    dim {dim-1}->{dim}: |G|={s_prev} -> {s}  ratio = {ratio:.4f}")
        elif s is not None:
            print(f"    dim {dim}: |G|={s} (baseline)")
        else:
            print(f"    dim {dim}: -- (unknown)")
        if s is not None:
            s_prev = s

# ---------------------------------------------------------------------------
# Section 3: 54 = 27x2 coincidence analysis
# ---------------------------------------------------------------------------

def factorize(n):
    """Return all factorizations of n as list of (a,b) with a<=b."""
    result = []
    for a in range(1, int(math.isqrt(n)) + 1):
        if n % a == 0:
            result.append((a, n // a))
    return result

def prime_factorization(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def analyze_54():
    subsection("54 COINCIDENCE ANALYSIS")

    candidates = [54, 27, 22, 20, 540, 48, 168, 12096]
    interps = {
        54: [
            "27 BCs x 2 (oriented) = 54 oriented body-centers",
            "540 faces / 10 = 54  (10 = C(5,3)/2 per tet per face-type)",
            "6 x 9 = coordination x 3^2",
            "9 x 6 = 9 orbit-size-classes x O_h-index?",
            "3^3 x 2 = 54",
            "540 / (22 - 12) = 540/10 = 54  (removing one size class)",
        ],
        540: [
            "20 faces/BC x 27 BCs = 540",
            "C(5,3) x 2 tets x 27 = 10 x 2 x 27 = 540",
            "12 x 45 = 540",
            "4 x 135 = 540",
            "54 x 10 = 540",
        ],
        22: [
            "4x48 + 12x24 + 3x12 + 3x8 = 192+288+36+24=540,  4+12+3+3=22 orbits",
            "22 = number of O_h orbits of 540 faces",
            "22 = dim of orbit coupling matrix (5D layer)",
        ],
    }

    for n in candidates:
        facts = factorize(n)
        pf = " x ".join(str(p) for p in prime_factorization(n))
        print(f"\n  {n} = {pf}")
        print(f"  Factorizations: {facts}")
        if n in interps:
            print(f"  Interpretations:")
            for interp in interps[n]:
                print(f"    * {interp}")

# ---------------------------------------------------------------------------
# Section 4: What is missing at 5D?
# ---------------------------------------------------------------------------

def analyze_5d_gap():
    subsection("5D GAP ANALYSIS: What is uniquely absent/present?")

    print("""
  Structure at neighboring dimensions:
    4D: 540 faces organized by 27 BCs (i-axis: first-order quaternion)
        Each BC -> 20 faces -> 2 tetrahedra x 10 triangles
        Coupling = adjacency within the 540-face complex

    5D (unknown): 22 O_h orbits of those 540 faces
        Orbit = equivalence class under O_h symmetry
        Coupling matrix = 22x22 block sums of 540x540 adjacency
        j-axis = relations BETWEEN orbits (second-order)

    6D: G_2 zero-mode
        G_2 = Aut(octonions)
        7 imaginary units = 7 Fano points
        Fano plane appears as combinatorial skeleton

  What 4D has that 5D must inherit and transform:
    4D gives:  540 faces with shared-edge adjacency
    5D gives:  quotient of 540 by O_h -> 22-dim orbit space
               The coupling matrix C_{22x22} IS the 5D structure
               Its spectral signature (13+, 9-) is a new invariant

  What 5D has that no other dimension has:
    -> The 22x22 coupling matrix (spectral signature convention-dependent, see notes)
    -> j-axis = second-order relational content
    -> The Fiedler value of the orbit graph Laplacian
    -> Cross-orbit coupling (inter-class vs intra-class ratio)

  What is ABSENT at 5D that appears at 2D, 4D, 6D:
    2D: Fano incidence structure (7 pts x 7 lines, 3/3 regularity)
    4D: BCC face count 540 (exact product 20x27)
    6D: G_2 zero-mode (exceptional Lie group)
    5D: No analogous "famous combinatorial structure" known
        -> This is the gap. 5D is purely relational (coupling of relations).

  Combinatorial quantity that MUST appear at 5D:
    The 22 orbits have size partition: {48:4, 24:12, 12:3, 8:3}
    -> 4 size classes. This is 2^2 = 4 -- quaternion basis dimension?
    -> 22 orbits span a 22-dim space; spectral signature is NOT convention-invariant
       (see orbit22_coupling_pinned.py / j_axis_sign_flip.py) -- the (13,9) reading
       below held only for one normalization choice and should not be treated as fixed:
         13 + 9 = 22  (check, under edge-count norm only)
         13 - 9 = 4   <- this is 2^2 = |quaternion basis|? (convention-dependent)
         13 / 9 ~= 1.44 <- near sqrt(2)? (convention-dependent)

  Key 54 readings:
    54 = 27 x 2: 27 BCs, oriented (+/-) = 54 oriented body-centers
    54 = 540 / 10: 10 = triangles per tetrahedron type
    54 = 6 x 9: 6 coordination x 9 (= 3^2) -- dimensional cascade
    54 appears as the bridge between 4D (face count 540) and 5D (orbit 22):
       540 / 54 = 10 = triangles per tetrahedron type
       But: 540 / 22 ~= 24.5 ~= |orbit size 24| (median orbit size!)
    """)

# ---------------------------------------------------------------------------
# Section 5: Bott periodicity alignment
# ---------------------------------------------------------------------------

def analyze_bott():
    subsection("BOTT PERIODICITY ALIGNMENT")

    print()
    print(f"  {'Dim':>3}  {'pi_n(O)':>8}  {'mod 8':>5}  {'Pattern'}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*5}  {'-'*40}")
    bott_names = {
        "Z_2": "finite (torsion)",
        "0":   "trivial",
        "Z":   "infinite cyclic (stable charge)",
    }
    for dim in range(9):
        b = bott_corrected(dim)
        pattern = bott_names.get(b, "?")
        print(f"  {dim:>3}  {b:>8}  {dim%8:>5}  {pattern}")

    print("""
  Bott pattern relevance to d-cascade:
    dim 3 (Z):  SC lattice -- infinite cyclic charge, links to 3D Poincare
    dim 4 (0):  BCC faces -- trivial stable charge (purely geometric)
    dim 5 (0):  orbit coupling -- trivial stable charge (also purely geometric)
    dim 6 (0):  G_2 layer -- trivial stable charge (still geometric)
    dim 7 (Z):  Bott returns -- infinite cyclic, mirrors dim 3

  Implication for 5D: pi_5(O) = 0 means 5D carries NO stable topological charge.
  This is consistent with 5D being a COUPLING layer (relational, not generative).
  5D does not create new charges -- it mediates between charges created at 3D and 4D.
    """)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  CROSS-DIMENSIONAL CENSUS: d-CASCADE 0D -> 8D")
    print("  Epistropy framework -- 5D reconstruction analysis")
    print("=" * 80)

    print_census_table()
    compute_ratios()
    analyze_54()
    analyze_5d_gap()
    analyze_bott()

    section("CONCLUSION")
    print("""
  1. CONSTANT ACROSS DIMENSIONS:
     - O_h symmetry (order 48) reappears at 3D, 4D, 5D -- O_h is the lattice invariant.
     - Bott period 8 is the meta-rhythm.
     - "20 per BC" and "540 total" are 4D-specific; no analogue found at 3D or 5D.

  2. WHAT 5D UNIQUELY CONTRIBUTES:
     - Second-order relational structure: coupling matrix over orbit space.
     - A spectral signature exists but is convention-dependent (not yet a fixed
       conserved quantity -- (10+,1,11-) vs (13+,0,9-) depending on normalization).
     - j-axis content: eigenvectors that mix orbit size classes.
     - 5D is the only dimension without a named "famous" geometric object --
       it is purely emergent from the 4D->5D quotient operation.

  3. WHAT MUST APPEAR AT 5D TO CLOSE THE CASCADE:
     - The 22x22 coupling matrix with non-trivial spectral structure.
     - A Fiedler value > 0 (algebraic connectivity: 5D is connected).
     - Cross-size-class eigenvectors encoding j-axis content.
     - A convention-independent spectral signature -- not yet established; current
       readings (n_pos - n_neg = 4) hold only under specific normalizations.

  4. 54 READING:
     54 = 27 x 2 = oriented BCs (most natural reading).
     54 = 540 / 10 = faces per tet-type-per-BC (bridge 4D->5D).
     The cleanest statement: 54 counts the minimal relational units
     (27 BCs x 2 parity classes) that the 22 orbits must mediate.
""")

if __name__ == '__main__':
    main()
