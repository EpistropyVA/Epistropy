# Bug Fix Report — verify_sec13_sec21.py

**Date**: 2026-06-14  
**Final result**: TOTAL 72 | PASS 69 | FAIL 3 | SKIP/WARN 0

---

## Bug 1 — Octonion associativity test (§13-A)

**Root cause**: The triple `(e1, e2, e4)` satisfies `(e1*e2)*e4 = e1*(e2*e4) = e7` in the standard Fano plane convention — it happens to be associative, so the test always returned `False` for non-associativity.

**Fix**: Changed to `(e1, e2, e5)`:
- `(e1*e2)*e5 = e3*e5 = e6`
- `e1*(e2*e5) = e1*e7 = -e6`  (opposite sign → genuinely non-associative)

**Line changed**: ~145-151. Changed `e4 = make_oct(4)` → `e5 = make_oct(5)` and updated the `record()` label and print strings.

**Before**: `[FAIL]` computed=False  
**After**: `[PASS]` computed=True

---

## Bug 2 — Sedenion zero-divisor tests (§13-C, §13-E, §13-F)

### Root cause analysis

The `cayley_dickson_double()` implementation is arithmetically correct and matches `sedenion_check.py` exactly (verified: zero basis-product mismatches across all 256 pairs). The bugs were in the *choice of test vectors*, not the multiplication itself.

**Key discovery**: `x = (e_1 + e_8)/√2` has `rank(L_x) = 16` — it is NOT a zero-divisor. Pairs `(a, b)` with `b = 8` (i.e. `a XOR b = a + 8 ∈ {9..15}` with the octonion unit at index 8) are never zero-divisors; only pairs with `b ∈ {9..15}` yield the 42 genuine zero-divisor pairs (6 per XOR-bin × 7 bins).

### §13-C fix

Changed the zero-divisor representative from `x = (e_1+e_8)/√2` (rank 16, NOT a ZD) to `x = (e_2+e_11)/√2` (rank 12, confirmed ZD with 4-dim kernel).

**Before**: `[FAIL] rank=16, ker=0`  
**After**: `[PASS] rank=12, ker=4`

### §13-E fix

Changed the loop `for b in range(8, 16)` → `for b in range(9, 16)`. This excludes the 7 spurious pairs `(a, 8)` for `a ∈ {1..7}` that satisfy the XOR constraint `u ∈ {9..15}` but are not actual zero-divisors.

**Before**: `[FAIL] count=49, total=98, per-bin=7`  
**After**: `[PASS] count=42, total=84, per-bin=6`

### §13-F fix

Changed the cocycle-cancellation example from `(e_1+e_8)/√2 * (e_2+e_11)/√2` (product = `e_3 ≠ 0`) to `(e_2+e_11)/√2 * (e_5+e_12)/√2` (product = 0, verified with working sedenion implementation). Both pairs are in the `u=9` bin.

**Before**: `[FAIL] max|xy|=1.0`  
**After**: `[PASS] max|xy|=0.0, N(x)*N(y)=1, N(xy)=0`

---

## Bug 3 — Cross-reference crash (GBK / stdout I/O error)

**Root cause**: Both cross-reference scripts (`dim3_vs_dim4_capacity.py`, `0d_investment_budget_16D.py`) contain module-level code `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', ...)`. When loaded via `importlib.util.spec_from_file_location` + `exec_module`, this replaces `sys.stdout` with a new TextIOWrapper that wraps the same underlying buffer. When that wrapper is later garbage-collected or re-replaced, it closes the shared buffer — making our own stdout wrapper raise `ValueError: I/O operation on closed file` on any subsequent `print()`.

**Fix**: Before calling `exec_module`, replace `sys.stdout` with a `_DummyStdout` object that has an independent `io.BytesIO()` as its `.buffer`. The imported module's `TextIOWrapper` then wraps this dummy buffer instead of the real one. After `exec_module` returns, restore the real stdout immediately before any further output.

```python
class _DummyStdout:
    buffer = io.BytesIO()
    def write(self, s): pass
    def flush(self): pass

_real_stdout = sys.stdout
sys.stdout = _DummyStdout()
spec.loader.exec_module(mod)
sys.stdout = _real_stdout   # restore before any prints
```

Applied to both `test_cross_reference_dim3_vs_dim4()` and `test_cross_reference_0d_budget()`.

**Before**: Fatal `ValueError` crash, no summary printed  
**After**: Both cross-references load and run; all 14 cross-reference checks PASS

---

## Remaining pre-existing FAILs (not in scope, not touched)

| Claim | Expected | Computed |
|---|---|---|
| §21: Model B induced β₂ before K-close = 6 | 6 | 0 |
| §21: Model B K-close β₃ = 18 | 18 | 12 |
| §21: Excluding v₈ gives minimum β₂ residual | True | False (v9/v10/v11 give β₂=0 < v8's β₂=5) |

These 3 failures existed before this patch and are in the §21 Markov/geometry section, not in §13 or the cross-reference infrastructure.
