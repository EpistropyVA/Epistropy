"""
F2-homology of P^2(F_q) for q = 2, 3, 4.

Computes:
  - Points and lines of P^2(F_q)
  - Collinear-triple simplicial complex
  - Boundary matrices over F_2
  - Betti numbers beta_0, beta_1, beta_2
  - Euler characteristic
  - |PGL(3, F_q)|
"""

from itertools import combinations, product
from functools import reduce

# ---------------------------------------------------------------------------
# Generic F_q machinery
# ---------------------------------------------------------------------------

def make_Fq(q):
    """
    Return (elements, add, mul, inv_add, inv_mul) for F_q.
    q must be a prime power: 2, 3, 4.
    For prime q: elements = range(q), operations are mod q.
    For q=4: F_4 = F_2[x]/(x^2+x+1), represented as (a,b) in F_2^2.
    """
    if q == 2:
        elems = list(range(2))
        add = lambda a, b: (a + b) % 2
        mul = lambda a, b: (a * b) % 2
        neg = lambda a: a
        def inv(a):
            if a == 0: raise ZeroDivisionError
            return 1
        return elems, add, mul, neg, inv

    elif q == 3:
        elems = list(range(3))
        add = lambda a, b: (a + b) % 3
        mul = lambda a, b: (a * b) % 3
        neg = lambda a: (-a) % 3
        def inv(a):
            if a == 0: raise ZeroDivisionError
            return pow(a, -1, 3)
        return elems, add, mul, neg, inv

    elif q == 4:
        # F_4 = F_2[omega]/(omega^2+omega+1)
        # Elements: 0=(0,0), 1=(1,0), omega=(0,1), omega^2=(1,1)
        elems = [(0,0),(1,0),(0,1),(1,1)]
        def add(a, b):
            return ((a[0]+b[0])%2, (a[1]+b[1])%2)
        def mul(a, b):
            # (a0+a1*w)(b0+b1*w) = a0b0 + (a0b1+a1b0)*w + a1b1*w^2
            # w^2 = w+1 => a1b1*(w+1)
            r0 = (a[0]*b[0] + a[1]*b[1]) % 2
            r1 = (a[0]*b[1] + a[1]*b[0] + a[1]*b[1]) % 2
            return (r0, r1)
        def neg(a): return a  # char 2
        def inv(a):
            if a == (0,0): raise ZeroDivisionError
            for e in elems:
                if e != (0,0) and mul(a, e) == (1,0):
                    return e
            raise ValueError(f"No inverse for {a}")
        # Verify w^2+w+1=0
        w = (0,1)
        one = (1,0)
        zero = (0,0)
        assert add(add(mul(w,w), w), one) == zero, "F_4 polynomial check failed"
        return elems, add, mul, neg, inv
    else:
        raise ValueError(f"q={q} not supported")


def projective_plane(q):
    """
    Enumerate points and lines of P^2(F_q).
    Returns (points, lines) where each is a list of frozensets.
    points[i] = canonical rep as tuple (x,y,z)
    lines[j] = frozenset of point indices
    """
    elems, add, mul, neg, inv = make_Fq(q)
    zero = elems[0]
    one = elems[1]

    # All nonzero vectors in F_q^3
    all_vecs = [(x,y,z) for x in elems for y in elems for z in elems
                if (x,y,z) != (zero,zero,zero)]

    # Nonzero scalars = F_q*
    nonzero = [e for e in elems if e != zero]

    def scale(lam, v):
        return (mul(lam,v[0]), mul(lam,v[1]), mul(lam,v[2]))

    # Canonical representative: first nonzero coordinate = 1
    def canonical(v):
        for i in range(3):
            if v[i] != zero:
                lam = inv(v[i])
                return scale(lam, v)
        raise ValueError

    # Deduplicate
    seen = set()
    points = []
    for v in all_vecs:
        c = canonical(v)
        if c not in seen:
            seen.add(c)
            points.append(c)
    points.sort()  # deterministic order

    pt_idx = {p: i for i, p in enumerate(points)}

    # Inner product over F_q
    def dot(a, b):
        return add(add(mul(a[0],b[0]), mul(a[1],b[1])), mul(a[2],b[2]))

    # Lines: same enumeration as points (dual projective plane)
    lines_reps = list(points)  # same canonical forms as covectors
    lines = []
    for lrep in lines_reps:
        pts_on = frozenset(pt_idx[p] for p in points if dot(lrep, p) == zero)
        lines.append(pts_on)

    return points, lines


# ---------------------------------------------------------------------------
# Simplicial complex from collinear triples
# ---------------------------------------------------------------------------

def build_complex(points, lines):
    """
    Vertices: indices 0..len(points)-1
    Edges: ALL pairs (complete graph K_n)
    Faces: collinear triples (subsets of size 3 of each line)
    """
    n = len(points)
    vertices = list(range(n))
    edges = list(combinations(range(n), 2))  # all pairs
    edge_idx = {e: i for i, e in enumerate(edges)}

    # Collinear triples
    face_set = set()
    for line in lines:
        line_pts = sorted(line)
        for tri in combinations(line_pts, 3):
            face_set.add(tri)
    faces = sorted(face_set)

    return vertices, edges, edge_idx, faces


# ---------------------------------------------------------------------------
# Boundary matrices over F_2
# ---------------------------------------------------------------------------

def boundary1(vertices, edges):
    """d1: C_1 -> C_0, columns=edges, rows=vertices"""
    n_v = len(vertices)
    n_e = len(edges)
    mat = [[0]*n_e for _ in range(n_v)]
    for j, (u,v) in enumerate(edges):
        mat[u][j] = 1
        mat[v][j] = 1
    return mat


def boundary2(edges, edge_idx, faces):
    """d2: C_2 -> C_1, columns=faces, rows=edges"""
    n_e = len(edges)
    n_f = len(faces)
    mat = [[0]*n_f for _ in range(n_e)]
    for k, tri in enumerate(faces):
        a, b, c = tri
        for edge in [(a,b),(a,c),(b,c)]:
            e = tuple(sorted(edge))
            j = edge_idx[e]
            mat[j][k] = 1
    return mat


# ---------------------------------------------------------------------------
# Gaussian elimination over F_2
# ---------------------------------------------------------------------------

def rank_F2(mat):
    """Rank of matrix over F_2 via row reduction. mat is list of lists."""
    if not mat or not mat[0]:
        return 0
    nrows = len(mat)
    ncols = len(mat[0])
    # Work on a copy
    M = [row[:] for row in mat]
    pivot_row = 0
    for col in range(ncols):
        # Find pivot
        found = -1
        for r in range(pivot_row, nrows):
            if M[r][col] == 1:
                found = r
                break
        if found == -1:
            continue
        M[pivot_row], M[found] = M[found], M[pivot_row]
        for r in range(nrows):
            if r != pivot_row and M[r][col] == 1:
                for c in range(ncols):
                    M[r][c] ^= M[pivot_row][c]
        pivot_row += 1
    return pivot_row


# ---------------------------------------------------------------------------
# Betti numbers
# ---------------------------------------------------------------------------

def betti(vertices, edges, edge_idx, faces):
    d1 = boundary1(vertices, edges)
    d2 = boundary2(edges, edge_idx, faces)

    rank_d1 = rank_F2(d1)
    rank_d2 = rank_F2(d2)

    n0 = len(vertices)
    n1 = len(edges)
    n2 = len(faces)

    # beta_0 = n0 - rank(d1)
    beta0 = n0 - rank_d1
    # beta_1 = n1 - rank(d1) - rank(d2) = ker(d1) - im(d2)
    beta1 = n1 - rank_d1 - rank_d2
    # beta_2 = n2 - rank(d2)
    beta2 = n2 - rank_d2

    chi = n0 - n1 + n2

    return {
        'n0': n0, 'n1': n1, 'n2': n2,
        'rank_d1': rank_d1, 'rank_d2': rank_d2,
        'beta0': beta0, 'beta1': beta1, 'beta2': beta2,
        'chi': chi
    }


# ---------------------------------------------------------------------------
# |PGL(3, F_q)|
# ---------------------------------------------------------------------------

def pgl3_order(q):
    """
    |GL(3,q)| = (q^3-1)(q^3-q)(q^3-q^2)
    |PGL(3,q)| = |GL(3,q)| / (q-1)
    """
    gl = (q**3 - 1) * (q**3 - q) * (q**3 - q**2)
    return gl // (q - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(q, label=None):
    if label is None:
        label = f"P^2(F_{q})"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    points, lines = projective_plane(q)
    n_pts = len(points)
    n_lines = len(lines)
    print(f"Points: {n_pts}   (expected {(q**3-1)//(q-1)})")
    print(f"Lines:  {n_lines}   (expected {(q**3-1)//(q-1)})")
    print(f"Points per line: {len(lines[0])}   (expected {q+1})")
    print(f"Lines per point: ", end="")
    # count lines through point 0
    lpp = sum(1 for L in lines if 0 in L)
    print(f"{lpp}   (expected {q+1})")

    vertices, edges, edge_idx, faces = build_complex(points, lines)

    # Count collinear triples per line
    collinear_per_line = len(list(combinations(range(q+1), 3)))  # C(q+1,3)
    print(f"\nCollinear triples per line: {collinear_per_line}")
    print(f"Total collinear triples (with dup check): {len(faces)}")

    # Check duplicates
    raw_total = n_lines * collinear_per_line
    print(f"  Raw (n_lines * C(q+1,3)): {raw_total}")
    print(f"  Duplicates removed: {raw_total - len(faces)}")

    print(f"\nEdges (K_{n_pts}): {len(edges)}")
    print(f"Faces (collinear triples): {len(faces)}")

    # Non-collinear triples
    total_triples = len(list(combinations(range(n_pts), 3)))
    non_collinear = total_triples - len(faces)
    print(f"Total triples C({n_pts},3): {total_triples}")
    print(f"Non-collinear triples: {non_collinear}")

    r = betti(vertices, edges, edge_idx, faces)
    print(f"\n--- F_2 Homology ---")
    print(f"rank(d1) = {r['rank_d1']},  rank(d2) = {r['rank_d2']}")
    print(f"beta_0 = {r['beta0']}")
    print(f"beta_1 = {r['beta1']}")
    print(f"beta_2 = {r['beta2']}")
    print(f"chi    = {r['chi']}   (V - E + F = {r['n0']} - {r['n1']} + {r['n2']})")

    aut = pgl3_order(q)
    print(f"\n|PGL(3,F_{q})| = {aut}")

    return {
        'label': label, 'q': q,
        'V': r['n0'], 'E': r['n1'], 'F': r['n2'],
        'beta0': r['beta0'], 'beta1': r['beta1'], 'beta2': r['beta2'],
        'chi': r['chi'],
        'aut': aut,
        'collinear_triples': len(faces),
        'non_collinear': non_collinear,
    }


def fano_reference():
    """Fano plane P^2(F_2): known values."""
    return {
        'label': 'P^2(F_2) Fano',
        'q': 2,
        'V': 7, 'E': 21, 'F': 7,
        'beta0': 1, 'beta1': 8, 'beta2': 0,
        'chi': -13,
        'aut': pgl3_order(2),
        'collinear_triples': 7,
        'non_collinear': 28,
    }


def print_table(results):
    print(f"\n{'='*70}")
    print("  Comparison Table")
    print(f"{'='*70}")
    labels = [r['label'] for r in results]
    hdr = f"{'':25}" + "".join(f"{l:>15}" for l in labels)
    print(hdr)
    print("-"*70)
    rows = [
        ('V', 'V'),
        ('E', 'E'),
        ('F (collinear triples)', 'F'),
        ('beta_0', 'beta0'),
        ('beta_1', 'beta1'),
        ('beta_2', 'beta2'),
        ('chi', 'chi'),
        ('|Aut|', 'aut'),
    ]
    for name, key in rows:
        row = f"{name:25}" + "".join(f"{r[key]:>15}" for r in results)
        print(row)
    print(f"{'='*70}")


if __name__ == '__main__':
    print("F_2-Homology of Projective Planes P^2(F_q)")
    print("Using collinear-triple simplicial complex + complete-graph edges")

    # P^2(F_2) — verify against known Fano
    print("\n--- Verifying P^2(F_2) (Fano plane) ---")
    r2 = analyze(2, "P^2(F_2) [Fano]")

    # P^2(F_3)
    r3 = analyze(3, "P^2(F_3)")

    # P^2(F_4)
    r4 = analyze(4, "P^2(F_4)")

    # Comparison table
    results = [r2, r3, r4]
    print_table(results)

    print("\nDone.")
