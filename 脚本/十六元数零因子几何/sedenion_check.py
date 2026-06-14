# sedenion_check.py — Cayley-Dickson construction check (R→C→H→O→S)
# UTF-8 encoding
# Optimized: precomputed 16×16 structure-constants table for O(1) products

import numpy as np
import sys

# ── Structure-constants table (built once via recursive Cayley-Dickson) ────────

def _build_table(dim):
    """
    Build structure-constants table T[i,j] = (sign, k) such that e_i * e_j = sign * e_k.
    Uses recursive Cayley-Dickson on indices.
    Returns sign_table (dim×dim int), index_table (dim×dim int).
    """
    if dim == 1:
        # e0 * e0 = e0
        return np.ones((1, 1), dtype=np.int8), np.zeros((1, 1), dtype=np.int16)

    half = dim // 2
    s_h, k_h = _build_table(half)  # smaller algebra

    sign_T = np.zeros((dim, dim), dtype=np.int8)
    idx_T  = np.zeros((dim, dim), dtype=np.int16)

    # Cayley-Dickson: basis split into (a-half, b-half)
    # e_(i) for i<half  → first component
    # e_(i) for i>=half → second component (index i-half in b-slot)
    #
    # Product rule for (a,b)*(c,d) = (a*c - conj(d)*b, d*a + b*conj(c))
    # In basis terms:
    #   (i,0) * (j,0) → (i*j, 0)            = same as half-algebra
    #   (i,0) * (0,j) → (0, j*i)            [d*a part; d=e_j, a=e_i → e_j*e_i in b-slot]
    #   (0,i) * (j,0) → (0, e_0 * e_j ... ) [b*conj(c) part; b=e_i, conj(c)=conj(e_j)=±e_j]
    #   (0,i) * (0,j) → (-conj(e_j)*e_i, 0) [real part; conj(e_j)=±e_j depending on j]
    #
    # Precisely, for basis vectors:
    # conj(e_j) = e_j if j==0 else -e_j   (Cayley-Dickson conjugate on half-algebra basis)
    #
    # Let A=e_i (i<half), B=e_j (j<half) denote half-algebra basis elements.
    # (A, 0)*(C, 0) = (A*C, 0)
    # (A, 0)*(0, D) = (0, D*A)            [b=0, so b*conj(c)=0; d*a = D*A]
    # (0, B)*(C, 0) = (0, B*conj(C))      [a=0, so a*c=0; d=0 so d*a=0; b*conj(c) = B*conj(C)]
    # (0, B)*(0, D) = (-conj(D)*B, 0)     [a*c=0; d*a=0; -conj(d)*b = -conj(D)*B; b*conj(c)=0]

    # Fill (i<half) * (j<half) → same as half
    sign_T[:half, :half] = s_h
    idx_T[:half, :half]  = k_h

    # Fill (i<half) * (j>=half): (A,0)*(0,D) = (0, D*A)
    # result index = k_h[j-half, i] + half, sign = s_h[j-half, i]
    for i in range(half):
        for j in range(half, dim):
            d = j - half
            k = k_h[d, i]
            s = s_h[d, i]
            sign_T[i, j] = s
            idx_T[i, j]  = k + half

    # Fill (i>=half) * (j<half): (0,B)*(C,0) = (0, B*conj(C))
    # conj(e_j) for j in half-algebra: if j==0 → +e_0, else → -e_j
    # so B*conj(C): if j==0 → B*e_0 = B → sign=1, k=i-half
    #               if j>0  → B*(-e_j) = -(B*e_j)
    for i in range(half, dim):
        for j in range(half):
            b = i - half
            if j == 0:
                # B * conj(e_0) = B * e_0 = B
                sign_T[i, j] = 1
                idx_T[i, j]  = i  # (0, e_b) → index i
            else:
                # B * conj(e_j) = B * (-e_j) = -(B*e_j)
                k = k_h[b, j]
                s = s_h[b, j]
                sign_T[i, j] = -s
                idx_T[i, j]  = k + half

    # Fill (i>=half) * (j>=half): (0,B)*(0,D) = (-conj(D)*B, 0)
    # conj(D)=conj(e_d): if d==0 → +e_0, else → -e_d
    # conj(D)*B: if d==0 → e_0*B = B  → sign=1, k=b
    #            if d>0  → (-e_d)*B = -(e_d*B)
    # Result: -(conj(D)*B)
    for i in range(half, dim):
        for j in range(half, dim):
            b = i - half
            d = j - half
            if d == 0:
                # conj(e_0)*e_b = e_b → sign=1
                # result: -(1*e_b) at real slot b
                sign_T[i, j] = -1
                idx_T[i, j]  = b
            else:
                # conj(e_d) = -e_d → (-e_d)*e_b = -(e_d*e_b)
                # result: -( -(e_d*e_b) ) = +(e_d*e_b)
                k = k_h[d, b]
                s = s_h[d, b]
                sign_T[i, j] = s
                idx_T[i, j]  = k

    return sign_T, idx_T


# Build tables for dim=8 and dim=16 at module load
_SIGN8, _IDX8   = _build_table(8)
_SIGN16, _IDX16 = _build_table(16)


def _mul_vec_vec(x, y, sign_T, idx_T):
    """Multiply two vectors using structure-constants table. O(n^2) but all numpy."""
    dim = len(x)
    # result[k] = sum_{i,j: idx_T[i,j]==k} sign_T[i,j] * x[i] * y[j]
    # Vectorized: compute all i*j contributions at once
    xy_outer = np.outer(x, y)                     # (dim, dim)
    contributions = sign_T * xy_outer              # signed contributions
    # Scatter into result by index
    result = np.zeros(dim)
    np.add.at(result, idx_T.ravel(), contributions.ravel())
    return result


def _mul_batch_vec(X, y, sign_T, idx_T):
    """Multiply batch of vectors X (shape N×dim) by single vector y. Returns N×dim."""
    # contributions[i,j] = sign_T[i,j] * y[j]  → broadcast over batch dimension
    # result[n, k] = sum_{i,j: idx[i,j]==k} sign[i,j] * X[n,i] * y[j]
    dim = len(y)
    N = X.shape[0]
    # weight[i,j] = sign_T[i,j] * y[j]
    weight = sign_T * y[np.newaxis, :]            # (dim, dim)
    # For each sample n: result[n,:] = X[n,:] @ weight scattered by idx_T
    # = einsum('ni,ij->nk') where k=idx_T[i,j]
    # Flatten: X[n,i] * weight[i,j] → shape (N, dim*dim), scatter to (N, dim)
    contrib = X[:, :, np.newaxis] * weight[np.newaxis, :, :]  # (N, dim, dim)
    result = np.zeros((N, dim))
    flat_idx = idx_T.ravel()                       # (dim*dim,)
    contrib_flat = contrib.reshape(N, dim * dim)   # (N, dim*dim)
    np.add.at(result, (slice(None), flat_idx), contrib_flat)
    return result


def _mul_vec_batch(x, Y, sign_T, idx_T):
    """Multiply single vector x (dim,) by batch Y (N×dim). Returns N×dim."""
    dim = len(x)
    N = Y.shape[0]
    # weight[i,j] = sign_T[i,j] * x[i]
    weight = sign_T * x[:, np.newaxis]            # (dim, dim)
    contrib = weight[np.newaxis, :, :] * Y[:, np.newaxis, :]  # (N, dim, dim)
    result = np.zeros((N, dim))
    flat_idx = idx_T.ravel()
    contrib_flat = contrib.reshape(N, dim * dim)
    np.add.at(result, (slice(None), flat_idx), contrib_flat)
    return result


# ── Public API (keeps backward-compat for sanity_check / T3 / T4) ─────────────

def cd_mul(x, y):
    """Cayley-Dickson multiplication using structure-constants table."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dim = len(x)
    if dim == 16:
        return _mul_vec_vec(x, y, _SIGN16, _IDX16)
    elif dim == 8:
        return _mul_vec_vec(x, y, _SIGN8, _IDX8)
    elif dim <= 4:
        # Build on the fly for small dims (rare)
        s, k = _build_table(dim)
        return _mul_vec_vec(x, y, s, k)
    else:
        raise ValueError(f"Unsupported dim={dim}")


def cd_conj(x):
    """Cayley-Dickson conjugate: e_0 → +e_0, e_i → -e_i for i>0."""
    c = -x.copy()
    c[0] = x[0]
    return c


def cd_norm(x):
    return np.sqrt(np.dot(x, x))


def basis_vec(dim, i):
    v = np.zeros(dim)
    v[i] = 1.0
    return v


# ── Sanity gate ───────────────────────────────────────────────────────────────

def sanity_check():
    errors = []

    # dim 2: e1^2 = -1
    e1 = basis_vec(2, 1)
    r = cd_mul(e1, e1)
    expected = basis_vec(2, 0) * (-1)
    if not np.allclose(r, expected, atol=1e-14):
        errors.append(f"dim2: e1^2 = {r}, expected {expected}")

    # dim 4: e1*e2 = ±e3 and e1*e2 != e2*e1
    e1_4 = basis_vec(4, 1)
    e2_4 = basis_vec(4, 2)
    p12 = cd_mul(e1_4, e2_4)
    p21 = cd_mul(e2_4, e1_4)
    e3_4 = basis_vec(4, 3)
    if not (np.allclose(p12, e3_4, atol=1e-14) or np.allclose(p12, -e3_4, atol=1e-14)):
        errors.append(f"dim4: e1*e2 = {p12}, expected ±e3")
    if np.allclose(p12, p21, atol=1e-14):
        errors.append(f"dim4: e1*e2 == e2*e1, expected non-commutative")

    # dim 8: norm multiplicativity on random samples
    rng = np.random.default_rng(42)
    for _ in range(20):
        x = rng.standard_normal(8)
        y = rng.standard_normal(8)
        xy = cd_mul(x, y)
        lhs = cd_norm(xy)
        rhs = cd_norm(x) * cd_norm(y)
        if abs(lhs - rhs) > 1e-10 * max(rhs, 1e-15):
            errors.append(f"dim8: norm multiplicativity failed: |xy|={lhs:.6f}, |x||y|={rhs:.6f}")
            break

    if errors:
        print("SANITY GATE FAILED:")
        for e in errors:
            print(" ", e)
        sys.exit(1)
    else:
        print("Sanity gate: PASSED (e1^2=-1 in C, e1*e2=±e3≠e2*e1 in H, norm-mult in O)")


# ── T1: Norm multiplicativity ─────────────────────────────────────────────────

def test_t1():
    print("\n── T1: Norm multiplicativity ───────────────────────────────────────")
    rng = np.random.default_rng(0)
    dims = [2, 4, 8, 16]
    all_pass = True
    for dim in dims:
        if dim == 16:
            sign_T, idx_T = _SIGN16, _IDX16
        elif dim == 8:
            sign_T, idx_T = _SIGN8, _IDX8
        else:
            sign_T, idx_T = _build_table(dim)

        N = 1000
        X = rng.standard_normal((N, dim))
        Y = rng.standard_normal((N, dim))

        # Batch multiply: result[n] = X[n] * Y[n]
        XY = np.array([_mul_vec_vec(X[n], Y[n], sign_T, idx_T) for n in range(N)])
        lhs = np.sqrt((XY * XY).sum(axis=1))
        rhs = np.sqrt((X * X).sum(axis=1)) * np.sqrt((Y * Y).sum(axis=1))
        denom = np.where(rhs > 1e-15, rhs, 1e-15)
        devs = np.abs(lhs - rhs) / denom
        max_dev = devs.max()

        expected_pass = dim <= 8
        passed = (max_dev < 1e-10) if expected_pass else (max_dev > 1e-6)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  dim={dim:2d}: max rel deviation = {max_dev:.3e}  [{status}]")
    print(f"T1 overall: {'PASS' if all_pass else 'FAIL'}")
    print("  Expected: ~0 for dims 2,4,8 (composition algebras); clearly >0 for dim 16 (sedenions fail norm-mult)")
    return all_pass


# ── T2: Zero divisor search ───────────────────────────────────────────────────

def test_t2():
    print("\n── T2: Zero divisor search ─────────────────────────────────────────")
    results = {}

    for dim in [8, 16]:
        if dim == 16:
            sign_T, idx_T = _SIGN16, _IDX16
        else:
            sign_T, idx_T = _SIGN8, _IDX8

        # Build all candidate vectors: e_i + s*e_j for i<j, s in {±1}
        # Shape: (num_candidates, dim)
        pairs = [(i, j, s)
                 for i in range(1, dim)
                 for j in range(i + 1, dim)
                 for s in [1.0, -1.0]]
        num = len(pairs)

        # Build X matrix
        X = np.zeros((num, dim))
        for idx, (i, j, s) in enumerate(pairs):
            X[idx, i] = 1.0
            X[idx, j] = s

        Y = X.copy()  # same candidate set

        # For each x in X, compute x*y for all y in Y using vectorized ops
        # Result shape: (num_x, num_y, dim)
        # Do it in blocks to avoid huge memory
        count = 0
        examples = []

        # Precompute: for each x, the left-mul matrix L_x (dim×dim)
        # Then XY[x,y] = L_x @ Y[y]
        # But building all L_x is num*dim*dim floats — fine for dim=16, num~1920
        # L_x[:, j] = x * e_j  (column j is x multiplied by basis e_j)

        # Build all left-mul matrices at once
        # L_all[n, :, j] = X[n] * e_j  → shape (num, dim, dim)
        E = np.eye(dim)  # basis vectors as rows
        # L_all[n, k, j] = (X[n] * e_j)[k]
        # _mul_vec_batch(x, Y, ...) gives (num_Y, dim) for single x
        # We need L_all[n] = left_mul_matrix(X[n])
        # Vectorize: for each basis column j, compute X * e_j for all X simultaneously
        # → use _mul_batch_vec(X, e_j, ...) shape (num, dim)

        L_all = np.zeros((num, dim, dim))
        for j in range(dim):
            ej = E[j]
            # col j of each L matrix
            L_all[:, :, j] = _mul_batch_vec(X, ej, sign_T, idx_T)

        # Products: P[n, m] = L_all[n] @ Y[m]  → shape (num, num, dim)
        # = einsum('nkj,mj->nmk', L_all, Y)
        # Use matmul: (num, dim, dim) @ (dim, num) = (num, dim, num) → transpose
        P = np.einsum('nkj,mj->nmk', L_all, Y)  # (num_x, num_y, dim)

        # Norm squared of each product
        norm_sq = (P * P).sum(axis=2)  # (num_x, num_y)

        # Zero divisors: norm_sq < threshold AND x != y (to avoid trivial x=0 cases)
        # X and Y are all nonzero by construction
        mask = norm_sq < 1e-24

        # Count unique unordered pairs {x, y} — avoid double-counting
        # Use upper triangle only (x_idx < y_idx)
        # Also exclude x == y (same index → same vector)
        triu_mask = np.triu(mask, k=1)
        indices = np.argwhere(triu_mask)
        count = len(indices)

        for xi, yi in indices[:3]:
            x = X[xi]
            y = Y[yi]
            xy = P[xi, yi]
            examples.append((x.copy(), y.copy(), xy.copy()))

        results[dim] = (count, examples)

    for dim in [8, 16]:
        count, examples = results[dim]
        expected_pass = (count == 0) if dim == 8 else (count > 0)
        passed = expected_pass
        status = "PASS" if passed else "FAIL"
        print(f"  dim={dim}: zero-divisor pairs found = {count}  [{status}]")
        if dim == 16 and examples:
            print(f"  Example zero-divisor pairs at dim 16:")
            for idx, (x, y, xy) in enumerate(examples):
                xi_nz = np.nonzero(x)[0]
                yi_nz = np.nonzero(y)[0]
                print(f"    pair {idx+1}: x={dict(zip(xi_nz.tolist(), x[xi_nz].tolist()))}  "
                      f"y={dict(zip(yi_nz.tolist(), y[yi_nz].tolist()))}  "
                      f"‖x*y‖={cd_norm(xy):.2e}")

    t2_pass = (results[8][0] == 0) and (results[16][0] > 0)
    print(f"T2 overall: {'PASS' if t2_pass else 'FAIL'}")
    print("  Expected: 0 zero-divisors at dim 8 (octonions are a division algebra), >0 at dim 16")
    return t2_pass, results[16][1]


# ── T3: Square roots, idempotents, quadratic identity ────────────────────────

def test_t3():
    print("\n── T3: Square roots of unity / idempotents at dim 16 ───────────────")
    dim = 16
    all_pass = True

    # (a) e_i^2 = -1 for all i > 0
    neg_one = np.zeros(dim); neg_one[0] = -1.0
    ei_sq_ok = True
    for i in range(1, dim):
        ei = basis_vec(dim, i)
        r = cd_mul(ei, ei)
        if not np.allclose(r, neg_one, atol=1e-14):
            print(f"  (a) FAIL: e_{i}^2 = {r}, expected -1")
            ei_sq_ok = False
    if ei_sq_ok:
        print(f"  (a) e_i^2 = -1 for all i=1..15: PASS")
    else:
        all_pass = False

    # (b) numerical search for u with u^2 = 1, u != ±1
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except ImportError:
        have_scipy = False

    def f_obj(u_flat):
        u = u_flat.reshape(dim)
        u2 = cd_mul(u, u)
        target = np.zeros(dim); target[0] = 1.0
        diff = u2 - target
        return float(np.dot(diff, diff))

    def f_grad(u_flat):
        eps = 1e-7
        g = np.zeros_like(u_flat)
        f0 = f_obj(u_flat)
        for k in range(len(u_flat)):
            u2 = u_flat.copy(); u2[k] += eps
            g[k] = (f_obj(u2) - f0) / eps
        return g

    rng = np.random.default_rng(123)
    found_non_trivial = []
    for _ in range(300):
        u0 = rng.standard_normal(dim)
        u0 /= cd_norm(u0) + 1e-15

        if have_scipy:
            res = minimize(f_obj, u0, jac=f_grad, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-30, 'gtol': 1e-15})
            u_conv = res.x.reshape(dim)
            fval = res.fun
        else:
            u = u0.copy()
            lr = 0.01
            for step in range(5000):
                g = f_grad(u)
                u -= lr * g
                if np.dot(g, g) < 1e-30:
                    break
            u_conv = u
            fval = f_obj(u)

        if fval < 1e-16:
            dist_plus = cd_norm(u_conv - basis_vec(dim, 0))
            dist_minus = cd_norm(u_conv + basis_vec(dim, 0))
            if min(dist_plus, dist_minus) > 0.1:
                found_non_trivial.append((u_conv.copy(), fval))

    if found_non_trivial:
        print(f"  (b) Non-trivial u with u^2=1 found: {len(found_non_trivial)} — UNEXPECTED FAIL")
        all_pass = False
    else:
        print(f"  (b) No non-trivial u with u^2=1 found (300 starts): PASS (expected)")

    # (c) quadratic identity: x^2 - 2*x0*x + ||x||^2 * 1 = 0
    one_vec = basis_vec(dim, 0)
    max_res = 0.0
    for _ in range(1000):
        x = rng.standard_normal(dim)
        x0 = x[0]
        x2 = cd_mul(x, x)
        norm2 = np.dot(x, x)
        lhs = x2 - 2 * x0 * x + norm2 * one_vec
        res = cd_norm(lhs)
        if res > max_res:
            max_res = res
    qi_pass = max_res < 1e-10
    if not qi_pass:
        all_pass = False
    print(f"  (c) Quadratic identity max residual norm = {max_res:.3e}: {'PASS' if qi_pass else 'FAIL'}")
    print(f"      Expected: ≈0 (holds for all Cayley-Dickson algebras)")

    print(f"T3 overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ── T4: Invertibility and determinant ────────────────────────────────────────

def test_t4(zd_examples):
    print("\n── T4: Invertibility via left-multiplication matrix ─────────────────")
    dim = 16

    def left_mul_matrix(x):
        M = np.zeros((dim, dim))
        for j in range(dim):
            ej = basis_vec(dim, j)
            M[:, j] = cd_mul(x, ej)
        return M

    # Zero divisor case
    if not zd_examples:
        print("  No zero-divisor example available from T2 — skipping zero-div det check")
        zd_ok = False
    else:
        x_zd = zd_examples[0][0]
        L_zd = left_mul_matrix(x_zd)
        det_zd = abs(np.linalg.det(L_zd))
        zd_ok = det_zd < 1e-6
        print(f"  Zero-divisor x = e_{np.nonzero(x_zd)[0].tolist()}-combo:")
        print(f"    |det(L_x)| = {det_zd:.6e}  [{'PASS' if zd_ok else 'FAIL'} — expected ≈0]")

    # Non-zero-divisor case: random sedenion, retry until non-degenerate
    rng = np.random.default_rng(77)
    nzd_ok = False
    for attempt in range(50):
        x_rand = rng.standard_normal(dim)
        is_zd = False
        for i in range(1, dim):
            for j in range(i + 1, dim):
                for s in [1.0, -1.0]:
                    y = basis_vec(dim, i) + s * basis_vec(dim, j)
                    if cd_norm(cd_mul(x_rand, y)) < 1e-10:
                        is_zd = True
                        break
                if is_zd:
                    break
            if is_zd:
                break
        if not is_zd:
            L_rand = left_mul_matrix(x_rand)
            det_rand = abs(np.linalg.det(L_rand))
            nzd_ok = det_rand > 1e-6
            print(f"  Non-zero-divisor random x (attempt {attempt+1}):")
            print(f"    |det(L_x)| = {det_rand:.6e}  [{'PASS' if nzd_ok else 'FAIL'} — expected clearly >0]")
            break

    t4_pass = zd_ok and nzd_ok
    print(f"T4 overall: {'PASS' if t4_pass else 'FAIL'}")
    print("  Expected: det≈0 for zero-divisor (singular L_x), det>>0 for generic element")
    return t4_pass


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("Cayley-Dickson Algebra Check: R → C → H → O → S (sedenions)")
    print("=" * 68)

    try:
        import numpy
        print(f"numpy version: {numpy.__version__}")
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        import numpy

    try:
        import scipy
        print(f"scipy version: {scipy.__version__}")
    except ImportError:
        print("scipy not available — T3(b) will use gradient descent fallback")

    print()
    sanity_check()

    r1 = test_t1()
    r2, zd_examples = test_t2()
    r3 = test_t3()
    r4 = test_t4(zd_examples)

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for label, result in [("T1 norm multiplicativity", r1),
                           ("T2 zero divisors", r2),
                           ("T3 square roots / quadratic id", r3),
                           ("T4 invertibility / det", r4)]:
        print(f"  {label:35s}: {'PASS' if result else 'FAIL'}")
    overall = all([r1, r2, r3, r4])
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 68)
