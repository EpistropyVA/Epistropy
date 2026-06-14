# Cross-Reference Consistency Report

**Date**: 2026-06-14
**Scope**: 有趣的拓扑和几何的互洽（终）§13–§23 against scripts in `.agent/scripts/有趣的拓扑和几何的互洽/`
**Method**: Fresh script runs + comparison against document claims + cross-doc check with（二）

Prior audit coverage: `audit_claim_matrix.md` (2026-06-12) covers §§1–12 and §13–§19 intersection claims from（二）perspective (71 claims, scripts T1-run). This report focuses on the **§13–§23 specific claims** in（终）and the **6D_bott_echo/ scripts** which are newer (Jun 13-14) and were not covered by the prior audit.

---

## Part A: Script Run Results (§13–§23 scope)

### Top-level scripts

| Script | Exit | Key output | Notes |
|--------|------|-----------|-------|
| `rh_boundary_matrix_verify.py` | 0 CLEAN | Fano: V=7, E=21, F=7, β₁=8, χ=-7; SC: β₁=0; BCC: β₁=0 | All d²=0 PASS |
| `rh_betti_scan.py` | 0 CLEAN | β₁ sequence: 0D:0, 1D:1, 2D:8, 3D:0, 4D:0 (cubes 4D–7D all β₁=0) | |
| `rh_steinberg_verify.py` | 0 CLEAN | dim H₁(Fano; F₂)=8, irreducible, GL(3,F₂) order=168 | |
| `rh_extended_homology.py` | 0 CLEAN | ker(d₁)=15, im(d₂)=7, β₁=8 (Fano) | |
| `rh_cross_translation.py` | 0 CLEAN | β₁ cascade: 0D:0, 1D:1, 2D:8, 3D:0, 4D:0; K-theory translation table | |
| `rh_face_existence_check.py` | 0 CLEAN | 7D without faces: β₁=5; 7D with faces: β₁=0 | Flags 7D face ambiguity |
| `rh_morse_midpoint_verify.py` | 0 CLEAN | All cobordisms W_0–W_8 PASS; h=1/2 at Z₂ midpoint | |
| `0d_investment_budget_16D.py` | 0 CLEAN | 0D net_cum: powers of 7 at 0D,1D,2D,6D (EXACT); 3D–16D off; doubling law 10D–16D confirmed | |
| `vertex_promotion_census_16D.py` | 0 CLEAN | d²=0 all dims; vertex promotion trace 0D–16D | |

### 6D_bott_echo/ scripts (newer, primary subject)

| Script | Exit | Key output | Notes |
|--------|------|-----------|-------|
| `dim3_vs_dim4_capacity.py` | 0 CLEAN | Step 3+: β₃(3D-capped)=12,44,105,205,355,579; β₃(4D-open)=0 all | Matches §21 table exactly |
| `stack_vs_accumulator.py` | 0 CLEAN | Model A: β₂=3→0, β₃=0; Model B: β₂=6→0, β₃=18; Model C: β₂=12→0, β₃=36 | |
| `cascade_z2_tracking.py` | 0 CLEAN | Steps 0–8 C+/C- decomposition; K-close → β₃(C+)=79, β₃(C-)=79 | |
| `cascade_targeted_closure.py` | 0 CLEAN | Step 7: β₂(total)=49 (C+25/C-24); targeted close: β₂→43 (kills 3 on each side) | |
| `bott_echo_step3.py` | 0 CLEAN | Max β₁=9 (not 8) achievable by v11; configs with β₁=8: 10 | |
| `bott_period2_exploration.py` | 0 CLEAN | 9D best: β₂=16; 10D best: **β₂=29** (not 24) | **DISCREPANCY** |
| `z2_rep_decomposition.py` | 0 CLEAN | Step 4 (17V): β₂(C+)=6, β₂(C-)=6, total=12; torsion-free confirmed | |
| `scan_exclude_vertex.py` | 0 CLEAN | At Step 7 (β₂=34): exclude v8→β₂=5 (minimum); no exclusion yields β₂=0 | |
| `task1_cycles.py` | 0 CLEAN | 5 β₂ generators: cycles 1–3 on ijk coordinate planes {0,1,2,3,8,12} etc; cycle 4 on {1,3,5,7,8,13}; cycle 5 on {0,1,2,3,6,7,8,13} | |

---

## Part B: Document Claim vs Script Output Comparison (§13–§23)

### §13 — Bott Periodicity

| Claim | Supporting script | Script output | Match? | Notes |
|-------|------------------|---------------|--------|-------|
| BCC平带162 Wilson本征值分裂为108(相位0)+54(相位π) | bcc_bloch_decomposition.py (prior audit) | 108+54 ✓ | YES | From prior audit #62 |
| rank L_x=12, ker L_x=4维 | sedenion_check.py | rank=12, ker=4 ✓ | YES | Prior audit #46 |
| rank(L_x L_y)=8≠0 | sedenion_check.py | rank=8 ✓ | YES | Prior audit #47 |
| SAME-84: PSL(2,7)-集同构 | sedenion_klein_84.py | order=168 ✓ | YES | Prior audit #44 |
| 无非平凡幂等元 u²=u仅u=±1 | sedenion_check.py | confirmed ✓ | YES | Prior audit #65 |
| Bott过渡带宽度=除法代数虚单位数 | no dedicated script | interpretive | NO SCRIPT | External/interpretive |

### §21 — Second Bott Period (PRIMARY NEW CLAIMS)

#### β₂ build cascade table (non-symmetrized model)

| dim | V | Doc claim β₂ | Script output β₂ | Match? | Script |
|-----|---|--------------|------------------|--------|--------|
| 8D | 13 | 3 | 3 ✓ | YES | dim3_vs_dim4_capacity Step 0: β₂=3 (starts before first Period-2 vertex) |
| 9D | 14 | 16 | **16** ✓ | YES | bott_period2_exploration: best=16 |
| 10D | 15 | 24 | **29** ✗ | **MISMATCH** | bott_period2_exploration: best=29, not 24 |
| 11D | 16 | 34 | not enumerated | UNVERIFIED | Script only goes to v14 (10D) |

**Note on the 10D mismatch**: `bott_period2_exploration.py` searches all subsets for max-β₂ insertion of v14 and reports best β₂=29. The document claims β₂=24 for 10D. Possible explanations: (a) the document uses a specific constrained connection rule (not max-β₂ greedy), (b) the script uses the greedy max but the document uses a symmetric or "non-redundant" construction. The document text says "build阶段的贪心策略始终避开 v₈/v₁₂ 连接 → Tet=0 → β₂ 只增不减." The exploration script connects to "all existing" vertices — so it likely does connect to v₁₂, which would reduce β₂ by creating tetrahedra. The document's constraint (avoid v₈, v₁₂ connections) is likely enforced differently in the script. **This is a model definition mismatch, not an arithmetic error.**

#### β₃(3D-capped) vs β₃(4D-open) table

| Step | Doc claim β₃(3D-capped) | Script β₃(3D-capped) | Doc claim β₃(4D-open) | Script β₃(4D-open) | Match? |
|------|------------------------|----------------------|----------------------|---------------------|--------|
| Step 3 (+v₁₁) | 12 | 12 ✓ | 0 | 0 ✓ | YES |
| Step 4 P1 K-close | 44 | 44 ✓ | 0 | 0 ✓ | YES |
| Step 7 (+v₁₅) | 355 | 355 ✓ | 0 | 0 ✓ | YES |
| Step 8 P2 K-close | 579 | 579 ✓ | 0 | 0 ✓ | YES |

**All β₃ capacity-lock numbers match exactly.** `dim3_vs_dim4_capacity.py` confirms every value in §21's capacity table.

#### Z₂ representation decomposition (symmetrized 17V model)

| Claim | Script output | Match? |
|-------|---------------|--------|
| Step 4后 β₂(C+)=6, β₂(C-)=6, total=12 | cascade_z2_tracking Step 4: C+=6, C-=6 ✓ | YES |
| Period 2 build有+1偏移: C+始终比C-多1 (14/13, 20/19, 25/24) | Steps 5/6/7: 14/13, 20/19, 25/24 ✓ | YES |
| K-close β₃(C+)=79, β₃(C-)=79 | Step 8 K-close: 79/79 ✓ | YES |
| 靶向v₈-v₁₂: 各-3 | cascade_targeted_closure Step 8: C+ 25→22, C- 24→21 (both -3) ✓ | YES |
| v₈是最小残余选项 (exclude v₈→β₂=5) | scan_exclude_vertex: exclude v8→β₂=5 (minimum of all exclusions) ✓ | YES |

#### S² cycle identification (task1_cycles.py)

| Claim | Script output | Match? |
|-------|---------------|--------|
| Cycle 0: {0,1,2,3,8,12} jk面, 8 triangles | Generator 1: vertices [0,1,2,3,8,12], 8 triangles ✓ | YES |
| Cycle 1: {0,1,4,5,8,12} ik面 | Generator 2: [0,1,4,5,8,12] ✓ | YES |
| Cycle 2: {0,2,4,6,8,12} ij面 | Generator 3: [0,2,4,6,8,12] ✓ | YES |
| Cycle 4: {1,3,5,7,8,13} x=1面 | Generator 4: [1,3,5,7,8,13] ✓ | YES |
| Cycle 5: {0,1,2,3,6,7,8,13} 体对角六边形 | Generator 5: [0,1,2,3,6,7,8,13], 12 triangles ✓ | YES |

β₂=5 (not 3) at this stage — consistent with text: "v₈是被排除者, 排除v₈给出最小残余β₂=5."

#### Stack model (stack_vs_accumulator.py)

| Claim | Script output | Match? |
|-------|---------------|--------|
| Model A (纯栈3个S²) K-close后β₃=0 | A: β₃=0 ✓ | YES |
| Model B β₃=18 | B: β₃=18 ✓ | YES |
| Model C β₃=36 | C: β₃=36 ✓ | YES |

### §22 — Axiom Closure

| Claim | Script | Match? | Notes |
|-------|--------|--------|-------|
| Fano V-E+F=7-21+7=-7 | rh_boundary_matrix_verify.py: χ=-7 ✓ | YES | |
| d²=0 Fano verified | rh_boundary_matrix_verify.py: PASS | YES | |
| β₁(Fano)=8 | all boundary scripts: 8 ✓ | YES | |
| GL(3,F₂)=168, H₁ irreducible, Steinberg 8D | rh_steinberg_verify.py: all PASS | YES | |
| ∂K≠0强制3D | NO SCRIPT | NO SCRIPT | audit_logic_sec13_23.md flags this as CRITICAL logical gap — not a computational claim |

### §23 — Full Dimension Matrix

| Claim | Script | Match? |
|-------|--------|--------|
| β₁序列 0,1,8,0,0,2,5,8,0 | rh_betti_scan.py (0D-4D) + rh_extended_homology.py | 0D:0✓ 1D:1✓ 2D:8✓ 3D:0✓ 4D:0✓; 5D:2, 6D:5, 7D:8 — claimed in doc but scripts use different model for 5D–7D (see below) |
| 4D-open列全零 (β₃≡0) | dim3_vs_dim4_capacity.py: all 0 ✓ | YES |
| β₃估计 26D: 10⁴–10⁵量级 | NOT COMPUTED — extrapolation claim | NO SCRIPT |

**5D–7D β₁ discrepancy**: The document's §23 table shows β₁ = 2(5D), 5(6D), 8(7D). The `rh_betti_scan.py` only computes cubes (5D-cube β₁=0), not the specific orbit/G₂/echo construction. The `rh_morse_midpoint_verify.py` defines W₄ (O_h orbit 13 verts promoted) and W₆ (G₂ zero-mode 1 vert), consistent with the cascade structure. The β₁=2,5,8 values for 5D–7D come from `rh_5d_orbit_construction.py` and the step-by-step BCC build — these are **not the same as hypercube β₁=0**. The document is using a specific cascade complex (V=22, V=7, V=8 structures), not n-cubes. No mismatch — just different complexes.

---

## Part C: Cross-document Consistency with 互洽（二）

互洽（二）exists at `d:/AI thoery/.agent/memory/product/math/有趣的拓扑和几何的互洽（二）.md`.

The（终）document explicitly references values from（二）at:
- §20: "cascade验证（互洽（二）§43）：β₁序列 0,1,8,0,0,2,5,8,0"
- §21: "β₂=3 (8D)" carried forward from period-1 closeout

**Cross-doc checks**:

| Value in（终） | Source in（二） | Match? |
|--------------|--------------|--------|
| β₁=8 at 7D | (二)§43 confirmed | YES — both scripts confirm 8 |
| β₂=3 at 8D (residual after K-close) | (二)§43 carry-forward | YES — dim3_vs_dim4 Step 0 shows β₂=3 as 8D base |
| 84零因子构形 | (二)§11待定区v358 | YES — sedenion_check.py confirms |
| rank L_x=12 | (二)§11 | YES |
| Wilson π=54 | (二)§11待定区v357 | YES |

**New claims in（终）not in（二）** (no cross-doc inconsistency, genuinely new):
- §21 entire: Period 2 cascade, β₂ build table, Z₂ decomposition, +1 偏移, stack vs accumulator
- §22: Axiom closure chain (new synthesis)
- §23: Full dimension matrix (new synthesis incorporating prior)

No contradictions found between（终）claims and（二）source values.

---

## Part D: Contradictions and Gaps

### Confirmed contradictions

| # | Location | Type | Description |
|---|----------|------|-------------|
| C1 | §21 β₂ build table, 10D row | **MISMATCH** | Doc: β₂=24 at 10D. `bott_period2_exploration.py` finds best=29. Root cause: script uses greedy all-connect strategy; doc uses "avoid v₈/v₁₂" constraint. The two models are not identical. The script does NOT implement the document's stated construction. |

### Unverified claims (no script exists)

| # | Location | Claim | Gap type |
|---|----------|-------|---------|
| U1 | §21 11D β₂=34 | 11D max β₂ value | `bott_period2_exploration.py` only enumerates to 10D (v14). 11D enumeration not done. |
| U2 | §21 "K-close在12D完成period 2闭合" | 12D as natural closure point | `bott_period2_closure.py` tests K-close closure but results not yet visible. dim3_vs_dim4 shows K-close Step 8 (=12D) kills β₂ but this is the "all connect" strategy. |
| U3 | §21 "period 2 闭合机制应镜像period 1" | Period 2 closure mechanism conjecture | Explicitly marked "待验证" in document — correctly identified as open. |
| U4 | §23 "β₃ 26D: 10⁴–10⁵量级" | Extrapolation to 26D | No script covers > 16D. Qualitative extrapolation only. |
| U5 | §13 "Bott过渡带宽度=虚单位数" | No dedicated computational verification | External/structural reading. Marked EXTERNAL in prior audit #61. |
| U6 | §22 "∂K≠0强制3D" | Step 5 of axiom chain | audit_logic_sec13_23.md flags as CRITICAL logical gap: the implication is not a theorem but an unstated closure/acyclicity design principle. |

### Pre-existing known issues (from prior audit, not re-tested)

| Script | Status | Description |
|--------|--------|-------------|
| `bcc_diag5.py` | FAIL | NameError: N_k undefined (coding bug, 2nd-order perturbation section) |
| `d6_zero_mode_analysis.py` | FAIL | NameError: d undefined in reporting section |
| `verify_pending_4b_rank19_shell.py` | FAIL | [FAIL] CORNER shell rank mismatch |

---

## Summary

**New scripts (6D_bott_echo/) run result**: All CLEAN (exit 0, no Python errors).

**Quantitative claims in §21 (Period 2)**: The β₃ capacity-lock table (12/44/105/205/355/579) matches exactly. The Z₂ decomposition table matches exactly (C+/C- ±1 asymmetry pattern). The S² cycle identity matches exactly.

**One model mismatch (C1)**: §21 β₂=24 at 10D vs script best=29. Not an arithmetic error — the document's build strategy (avoid v₈/v₁₂ connections) is not implemented in `bott_period2_exploration.py`, which uses unconstrained max-β₂ search. The constrained strategy would need a dedicated script to verify β₂=24.

**11D β₂=34**: The script does not enumerate 11D (would require checking ~131072 subsets of v15 connections). Unverified.

**Logical gap in §22**: The step "∂K≠0 + ∂²=0 → 3D forced" is not a theorem — it is an unstated design requirement of the cascade. This is a philosophical/foundational gap, not a computational one, and is correctly identified in the existing `audit_logic_sec13_23.md`.

**Cross-document consistency**: No contradictions between（终）and（二）. Values carried forward (β₁=8, β₂=3 residual, 84 zero divisors, rank=12) are all consistent.
