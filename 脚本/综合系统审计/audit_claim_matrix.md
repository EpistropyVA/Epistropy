# Audit: Claim×Script Matrix
**Date**: 2026-06-12
**Status**: AUDIT SNAPSHOT — rots as code changes. Rerun after any script or document modification.
**Scope**: 有趣的拓扑和几何的互洽（二）and（终）, scripts in `.agent/scripts/`

---

## T1 — Batch Script Run Results

All scripts run with `python <script>` from `d:\AI thoery\.agent\scripts\`. Timeout 300s each.
"Real failure" = Traceback/SyntaxError/ImportError/NameError/FAILED in output, OR non-zero exit code with traceback.
Scripts with "error" / "deviation" in output text as computational terms are marked clean (no false-positive).

### bcc_* scripts

| Script | Exit | Real Failure? | Runtime | Notes |
|--------|------|---------------|---------|-------|
| bcc_540_spectrum.py | 0 | CLEAN | 0.7s | |
| bcc_kronecker_decomp.py | 0 | CLEAN | 0.2s | Imports from bcc_540_spectrum.py |
| bcc_three_axis_kronecker.py | 0 | CLEAN | 0.6s | Imports from bcc_540_spectrum.py |
| bcc_540_periodic_bc.py | 0 | CLEAN | 0.2s | |
| bcc_finite_size_scaling.py | 0 | CLEAN | 0.4s | "error" in output = numerical deviation, not Python error |
| bcc_bloch_decomposition.py | 0 | CLEAN | 0.2s | "error" in output = Kronecker residual, not Python error |
| bcc_3simplex_analog.py | 0 | CLEAN | 0.5s | |
| bcc_flatband_oh_irreps.py | 0 | CLEAN | 0.3s | "error" in output = group theory reconstruction error ~1e-15, not Python error |
| bcc_adams_axis_analysis.py | 0 | CLEAN | 0.3s | |
| bcc_L5_corner_deviation.py | 0 | CLEAN | 1.0s | |
| bcc_sympy_exact_deviation.py | 0 | CLEAN | 0.2s | "error" in output = rational approximation error, not Python error |
| bcc_periodic_corner_deviation.py | 0 | CLEAN | 0.2s | |
| bcc_L6_corner_deviation.py | 0 | CLEAN | 9.0s | "error" in output = fraction approximation error, not Python error |
| bcc_corner_perturbation.py | 0 | CLEAN | 0.2s | |
| bcc_corner_gauss_bonnet.py | 0 | CLEAN | 0.9s | |
| bcc_homology_verification.py | 0 | CLEAN | 0.3s | |
| bcc_diag.py | 0 | CLEAN | 0.2s | |
| bcc_diag2.py | 0 | CLEAN | 0.1s | |
| bcc_diag3.py | 0 | CLEAN | 0.3s | |
| bcc_diag4.py | 0 | CLEAN | 0.3s | |
| **bcc_diag5.py** | **1** | **FAIL** | 0.2s | **NameError: 'N_k' not defined (line 311). Script has a coding bug.** |
| bcc_diag6.py | 0 | CLEAN | 0.2s | |
| bcc_diag7.py | 0 | CLEAN | 0.1s | |
| bcc_diag8.py | 0 | CLEAN | 0.2s | |
| bcc_diag9.py | 0 | CLEAN | 0.1s | |
| bcc_54_analysis2.py | 0 | CLEAN | 1.2s | |
| bcc_54_global_modes.py | 0 | CLEAN | 3.2s | |

### d5_*, d6_* scripts

| Script | Exit | Real Failure? | Runtime | Notes |
|--------|------|---------------|---------|-------|
| d5_5simplex_network.py | 0 | CLEAN | 15.7s | |
| **d6_zero_mode_analysis.py** | **1** | **FAIL** | 0.5s | **NameError: 'd' not defined (line 542, in structural_interpretation).** Late-stage analysis block; main computation runs, fails in reporting section. |
| d6_zero_mode_irreps.py | 0 | CLEAN | 0.1s | |

### sedenion_* scripts

| Script | Exit | Real Failure? | Runtime | Notes |
|--------|------|---------------|---------|-------|
| sedenion_check.py | 0 | CLEAN | 56.1s | Long runtime — exhaustive sedenion arithmetic |
| sedenion_klein_84.py | 0 | CLEAN | 1.6s | Uses sympy.combinatorics |

### verify_pending_* scripts

| Script | Exit | Real Failure? | Runtime | Notes |
|--------|------|---------------|---------|-------|
| verify_pending_4_three54.py | 0 | CLEAN | 0.1s | |
| **verify_pending_4b_rank19_shell.py** | **0** | **FAIL** | 0.1s | **[FAIL] CORNER shell — rank mismatch in shell decomposition. Exit 0 but reports internal failure.** |
| verify_pending_4c_corner_shadow.py | 0 | CLEAN | 0.1s | Imports from verify_pending_4b |
| verify_pending_4d_controls.py | 0 | CLEAN | 0.1s | Imports from verify_pending_4b |
| verify_pending_5_ghz_fano.py | 0 | CLEAN | 0.1s | |
| **verify_pending_6_bvn_ijk.py** | **0** | **NOTE** | 0.1s | "Distributive law failed? True" is the EXPECTED result (PASS); BvN non-distributivity is what's being verified. Exit 0, PASS. |
| verify_pending_7_orientation_forgetting.py | 0 | CLEAN | 1.2s | |
| verify_pending_8_rank_drop4.py | 0 | CLEAN | 4.3s | |
| verify_pending_9_moreno_g2.py | 0 | CLEAN | 10.3s | |
| verify_pending_10_orientation_transport.py | 0 | CLEAN | 0.4s | |

### corner_*, genus_uniqueness.py, w2_exact.py

| Script | Exit | Real Failure? | Runtime | Notes |
|--------|------|---------------|---------|-------|
| corner_defect_perturbation.py | 0 | CLEAN | 0.3s | |
| genus_uniqueness.py | 0 | CLEAN | 0.3s | Uses sympy |
| w2_exact.py | 0 | CLEAN | 1.4s | Uses sympy |

### T1 Summary
- **Total scripts run**: 47
- **CLEAN (exit 0, no real failure)**: 44
- **FAIL**: 3
  - `bcc_diag5.py` — NameError: `N_k` undefined (coding bug in 2nd-order perturbation section)
  - `d6_zero_mode_analysis.py` — NameError: `d` undefined in `structural_interpretation()` (reporting section, main computation runs)
  - `verify_pending_4b_rank19_shell.py` — internal `[FAIL] CORNER shell` assertion (the script exits 0 but reports a failed check; shell rank decomposition disagreement)
- **NOTE** (pass with expected "failure" output): `verify_pending_6_bvn_ijk.py` — distributive law failing IS the expected result, PASS confirmed

---

## T2 — Claim × Script Matrix

Sources:
- **(二)** = 有趣的拓扑和几何的互洽（二）.md
- **(终)** = 有趣的拓扑和几何的互洽（终）.md

Independence definitions:
- **SINGLE-PATH**: one construction, even if multiple files import it (same algorithm / same numpy.linalg.eigh path)
- **MULTI-PATH**: two genuinely independent constructions (e.g. numpy float + sympy exact, or entirely different algorithms)
- **EXTERNAL**: verified against literature or known theorem, not a new computation

| # | doc§ | Claim (one line) | Supporting script(s) | Independence | Claim type |
|---|------|-----------------|---------------------|--------------|------------|
| 1 | (二)§10 | 3×3×3 BCC有27个体心、64个SC顶点、54个4-单纯形 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 2 | (二)§10 | 最大共享顶点数 = 2（无面接触、无体接触） | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 3 | (二)§10 | 共享顶点0/1/2的出现次数分别为1120/203/108 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 4 | (二)§10 | 面=0和体=0确认无高阶共享 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 5 | (二)§10 | 8-hub交叉矩阵每行3个"2"+4个"1" = Q3立方体图 | bcc_540_spectrum.py | SINGLE-PATH | INTERPRETIVE |
| 6 | (二)§10 | sqrt(3)方向有6条路径 = C(4,2) = 6 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 7 | (二)§11 | 54个4-单纯形 × C(5,3)=10面 = 540个独立三角形面 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 8 | (二)§11 | 手征扭转后秩 = 540（满秩，零模=0） | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 9 | (二)§11 | 未扭转时零模=1（全局连通） | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 10 | (二)§11 | O_h轨道数=22（非15），轨道分解4×size48+12×size24+3×size12+3×size8=540 | bcc_540_spectrum.py (orbit computation) | SINGLE-PATH | NUMBER |
| 11 | (二)§11 (待定区) | 540×540面邻接矩阵特征值-3简并度108 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 12 | (二)§11 (待定区) | 特征值-2简并度56 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 13 | (二)§11 (待定区) | 特征值±sqrt(2)各16 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 14 | (二)§11 (待定区) | 特征值1简并度16 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 15 | (二)§11 (待定区) | 孤立模≈10.86，间距1.23 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 16 | (二)§11 (待定区) | S^0符号不影响谱（W=D_σ A D_σ对角相似） | bcc_540_spectrum.py | SINGLE-PATH | FORMULA |
| 17 | (二)§11 (待定区) | L_local完全均匀：27个20×20块精确相同（差值=0） | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 18 | (二)§11 (待定区) | L_local谱={-2^(×10), +1^(×8), +6^(×2)} | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 19 | (二)§11 (待定区) | 56@-2禁闭投影：Corner=4.000000, Edge=2.000000, Face=0.000000, Center=0.000000 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 20 | (二)§11 (待定区) | 禁闭公式：2(5-deg(v))，六位小数精确 | bcc_540_spectrum.py | SINGLE-PATH | FORMULA |
| 21 | (二)§11 (待定区) | 108@-3投影近似跟踪图度数(3.014,3.998,4.989,5.984)，偏差~0.01 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 22 | (二)§11 (待定区) | Kronecker可分性被证伪，残差60% | bcc_kronecker_decomp.py (imports bcc_540_spectrum) | SINGLE-PATH | NUMBER |
| 23 | (二)§11 (待定区) | 三轴Kronecker残差43% | bcc_three_axis_kronecker.py (imports bcc_540_spectrum) | SINGLE-PATH | NUMBER |
| 24 | (二)§11 (待定区) | 周期BC下全部27个BC等价（度数6）✓ | bcc_540_periodic_bc.py | SINGLE-PATH | NUMBER |
| 25 | (二)§11 (待定区) | 周期谱中-2.0精确不存在（56维重新分布到无理数簇） | bcc_540_periodic_bc.py | SINGLE-PATH | NUMBER |
| 26 | (二)§11 (待定区) | Kronecker不可分在周期BC下仍失败，残差62% | bcc_540_periodic_bc.py | SINGLE-PATH | NUMBER |
| 27 | (二)§11 (待定区) | Corner偏差与L无关（L=3:+0.01378, L=4:+0.01386），偏差总和精确为零 | bcc_finite_size_scaling.py | SINGLE-PATH | NUMBER |
| 28 | (二)§11 (待定区) | 精确简并度公式：deg(λ=-3)=2|E_BC|, deg(λ=-2)=Σ2max(5-deg(v),0) | bcc_finite_size_scaling.py | SINGLE-PATH | FORMULA |
| 29 | (二)§11 (待定区) | Bloch分解将540×540精确块对角化为27个20×20 H(k)，误差~1e-14 | bcc_bloch_decomposition.py | SINGLE-PATH | NUMBER |
| 30 | (二)§11 (待定区) | 严格最近邻耦合：7个非零T(R)（1 self+6 NN），beyond-NN=0 | bcc_bloch_decomposition.py | SINGLE-PATH | NUMBER |
| 31 | (二)§11 (待定区) | 反演破缺：‖T(+x)-T(-x)‖_F=6.0（三轴全部） | bcc_bloch_decomposition.py | SINGLE-PATH | NUMBER |
| 32 | (二)§11 (待定区) | Kronecker误差k依赖：Γ点=0，BZ边=5.2，BZ面=7.3，BZ角=9.0 | bcc_bloch_decomposition.py | SINGLE-PATH | NUMBER |
| 33 | (二)§11 (待定区) | 6条平带（λ=-3，全k点），6×27=162=周期BC的λ=-3简并度 | bcc_flatband_oh_irreps.py | SINGLE-PATH | NUMBER |
| 34 | (二)§11 (待定区) | O_h分解：6D=A_2u ⊕ E_u ⊕ T_1u (1+2+3)，全部奇宇称 | bcc_flatband_oh_irreps.py | SINGLE-PATH | FORMULA |
| 35 | (二)§11 (待定区) | 谱钉扎：T(0,0,0)谱为{-2,+1,+6}无-3，平带由代数锁死非去耦 | bcc_bloch_decomposition.py + bcc_flatband_oh_irreps.py | SINGLE-PATH | INTERPRETIVE |
| 36 | (二)§11 (待定区v357) | Berry相位：Wilson π模108个(trivial)+54个(π) | bcc_bloch_decomposition.py (Wilson loop section) | SINGLE-PATH | NUMBER |
| 37 | (二)§11 (待定区v357) | d_k^T d_k = A_signed + (k+1)I，d=2至d=6全部验证 | d5_5simplex_network.py / d6_zero_mode_irreps.py | SINGLE-PATH | FORMULA |
| 38 | (二)§11 (待定区v357) | d=6零模=14=Ad(G_2)，简并度d(d-1)/2-1=14 | d6_zero_mode_analysis.py (partial — script fails in reporting section) | SINGLE-PATH | NUMBER |
| 39 | (二)§11 (待定区v357) | PSL(2,7)≅GL(3,F_2)分解：14=ρ_6⊕ρ_8=Ad(G_2)限制到PSL(2,7) | d6_zero_mode_irreps.py | SINGLE-PATH | FORMULA |
| 40 | (二)§11 (待定区v357) | 顶点星对14D零空间投影=严格0，棱星=满秩14D，Fano线=6D | d6_zero_mode_irreps.py | SINGLE-PATH | NUMBER |
| 41 | (二)§11 (待定区v358) | Cayley-Dickson乘法表e_a·e_b=±e_{a⊕b}，256例全验，极化恒等式256例精确成立 | sedenion_check.py | SINGLE-PATH | NUMBER |
| 42 | (二)§11 (待定区v358) | 范数乘性边界：dim 2,4,8最大相对偏差~4×10^-16；dim 16偏差0.443 | sedenion_check.py | MULTI-PATH (numpy float + exact integer check in sedenion_check.py) | NUMBER |
| 43 | (二)§11 (待定区v358) | 零因子计数：dim 8=0对；dim 16=336带号有序→168带号无序→84指标构形 | sedenion_check.py + sedenion_klein_84.py | SINGLE-PATH (both build from same arithmetic) | NUMBER |
| 44 | (二)§11 (待定区v358) | GL(4,2)中保持84-集的子群阶=168，即PSL(2,7) | sedenion_klein_84.py | SINGLE-PATH | NUMBER |
| 45 | (二)§11 (待定区v358) | 零因子箱结构：公共值u∈{9..15}（7值，各24带号对）；e_a低指标(1-7)，e_b高指标(8-15) | sedenion_check.py | SINGLE-PATH | NUMBER |
| 46 | (二)§11 (待定区v358) | rank L_x=12，ker L_x=4维 | sedenion_check.py | SINGLE-PATH | NUMBER |
| 47 | (二)§11 (待定区v358) | rank(L_x L_y)=8≠0（非结合性阻断链复形资格） | sedenion_check.py | SINGLE-PATH | NUMBER |
| 48 | (二)§11 (待定区v358) | BCC收支：点数-单纯形数=(n+1)^3-n^3，槽位平衡每n成立 | bcc_54_analysis2.py | SINGLE-PATH | FORMULA |
| 49 | (二)§11 (待定区v358) | Radon-Hurwitz锚：dim 8最大团=7=ρ(8)-1；dim 16最大团=8=ρ(16)-1 | sedenion_check.py | SINGLE-PATH | NUMBER |
| 50 | (二)§11 (待定区v359·#4) | 两个54维子空间overlap矩阵秩=19（非54），奇异值重数1+6+12精确且O_h对称保护 | verify_pending_4_three54.py | SINGLE-PATH | NUMBER |
| 51 | (二)§11 (待定区v359·#5) | GHZ↔Fano cocycle：4个Mermin算符期望值(+1,-1,-1,-1)，4条Mermin线均为合法Fano线 | verify_pending_5_ghz_fano.py | SINGLE-PATH | NUMBER |
| 52 | (二)§11 (待定区v359·#6) | BvN非分配格：P_i∧(P_j∨P_k)=P_i≠0=(P_i∧P_j)∨(P_i∧P_k)显式反例 | verify_pending_6_bvn_ijk.py | SINGLE-PATH | FORMULA |
| 53 | (二)§11 (待定区v359·#7) | 336/168/84阶梯：符号遗忘336→168，有序遗忘168→84，与Klein 84边同构 | verify_pending_7_orientation_forgetting.py + sedenion_klein_84.py | SINGLE-PATH (sedenion_klein_84 shares sympy.combinatorics construction) | NUMBER |
| 54 | (二)§11 (待定区v359·#8) | rank扣4：84个零因子全扫描rank L_x=12逐一成立，rank(L_xL_y)=8逐一成立 | verify_pending_8_rank_drop4.py | SINGLE-PATH | NUMBER |
| 55 | (二)§11 (待定区v359·#9) | Moreno G_2：84构形邻接图连通性/正则度/自同构群计算验证（文献声明未核实） | verify_pending_9_moreno_g2.py | SINGLE-PATH | INTERPRETIVE |
| 56 | (二)§11 (待定区v359·#10) | d_1 d_2=0代数验证；1080个局部环holonomy全=+1；存在全局非可缩环holonomy=-1 | verify_pending_10_orientation_transport.py | SINGLE-PATH | FORMULA |
| 57 | (二)§12 (Gemini校准) | 540×540特征值表：Gemini 53组与独立验证逐组吻合4位小数 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 58 | (二)§12 (Gemini校准) | O_h轨道数Gemini=15，独立验证=22（错误） | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 59 | (二)§12 (Gemini校准) | 108投影值(3.014,3.998,4.989,5.984)与Gemini完全吻合 | bcc_540_spectrum.py | SINGLE-PATH | NUMBER |
| 60 | (终)§13 | dim 16：范数乘性最大相对偏差0.443（Hurwitz边界实测） | sedenion_check.py | SINGLE-PATH | NUMBER |
| 61 | (终)§13 | Bott过渡带宽度=除法代数虚单位数（C:1，H:3） | NOT FOUND (structural reading from Bott homotopy groups; no dedicated script) | EXTERNAL | INTERPRETIVE |
| 62 | (终)§13 | BCC平带162个Wilson本征值分裂为108(相位0)+54(相位π) | bcc_bloch_decomposition.py | SINGLE-PATH | NUMBER |
| 63 | (终)§13 | d_k^T d_k对角元=k+1（k-单纯形闭合税） | d5_5simplex_network.py / d6_zero_mode_irreps.py | SINGLE-PATH | FORMULA |
| 64 | (终)§13 | π_6(O)=0处面邻接零模=G_2，14D | d6_zero_mode_analysis.py (partial) + d6_zero_mode_irreps.py | SINGLE-PATH | INTERPRETIVE |
| 65 | (终)§13 | sedenion无非平凡幂等元（u^2=1仅u=±1），残差为0 | sedenion_check.py | SINGLE-PATH | FORMULA |
| 66 | (终)§13 | rank(L_x L_y)=8≠0：非结合性拒绝∂^2=0算子资格 | sedenion_check.py | SINGLE-PATH | FORMULA |
| 67 | (终)§13 | SAME-84：84个零因子构形与Klein四次曲线84边作为PSL(2,7)-集同构 | sedenion_klein_84.py + verify_pending_8_rank_drop4.py | SINGLE-PATH | INTERPRETIVE |
| 68 | (终)§18 | BvN非分配格：3D分配律丧失，i,j,k三正交轴自旋投影标准反例 | verify_pending_6_bvn_ijk.py | SINGLE-PATH | FORMULA |
| 69 | (终)§19 | 不变量signature(2^n,0)在dim 16完好；守恒律N(xy)=N(x)N(y)在dim 16失效（偏差0.443） | sedenion_check.py | SINGLE-PATH | FORMULA |
| 70 | (二)§9 | BCC四轴两两组合C(4,2)=6定义6个平面=FCC面心 | NOT FOUND (combinatorial identity, no dedicated script) | EXTERNAL | NUMBER |
| 71 | (二)§8 | 每体心2个对偶4-单纯形×27体心=54（待定区：与Wilson π相位模数相同） | bcc_540_spectrum.py (count) + bcc_bloch_decomposition.py (Wilson=54) | MULTI-PATH (geometry count vs Wilson loop computation) | INTERPRETIVE |

---

## T3 Candidate Queue

Criteria: SINGLE-PATH + INTERPRETIVE + load-bearing (labeled 判决级 or used as a section conclusion in the document).

| # | Claim ref | Claim | Why load-bearing | Weakness |
|---|-----------|-------|-----------------|----------|
| T3-1 | #5 (二§10) | 8-hub交叉矩阵每行3个"2"+4个"1"=立方体图Q3，作为"3D立方对称在4-单纯形层面的继承"的结论 | Section §10结论：BCC拓扑继承SC的立方对称 | SINGLE-PATH (one numpy.linalg.eigh pipeline)；Q3解读是人工叠加在数值矩阵上的图论标签，未通过独立图同构验证 |
| T3-2 | #35 (二§11待定区) | 谱钉扎解读：平带不来自跳跃矩阵零空间而是代数锁死 | 用于解释为何6条平带出现——作为flatband_oh_irreps和bloch_decomposition的联合结论 | SINGLE-PATH；"代数锁死"是对两个数值结果的联合解释，比单一数值更脆弱 |
| T3-3 | #55 (二§11待定区v359·#9) | Moreno G_2：零因子流形≅G_2声明（文献未核实） | 文档明确标注"文献声明待核"——但在§13(终)中被援引为"两次G_2现身的关系"的一侧 | 图论计算部分通过，但G_2声明依赖Moreno 1998原文，脚本verify_pending_9_moreno_g2.py只验证图的组合性质 |
| T3-4 | #64 (终§13) | π_6(O)=0处面邻接零模=G_2，作为"6-simplex→Fano→O→G_2→Ad(G_2)=14"结构链的结论 | 在（终）§13作为Bott过渡带内部结构的关键节点；链条完整性是论文质量中心 | d6_zero_mode_analysis.py在reporting section有NameError（脚本失败），d6_zero_mode_irreps.py仅做irrep分解；链条中"6-simplex有7顶点=Im(O)"→Fano→G_2部分是INTERPRETIVE叠加 |
| T3-5 | #67 (终§13) | SAME-84：零因子构形与Klein四次曲线84边作为PSL(2,7)-集同构 | 在（终）§13"环末端检查"和§19"商容积合法性"中被用作结构闭合的证据 | sedenion_klein_84.py和verify_pending_8_rank_drop4.py共享同一代数构造路径（两者均从sedenion F2算术出发），非独立构造；Klein同构读法是传递G-集分类，依赖单一代数构造 |
| T3-6 | #71 (二§8) | 54个4-单纯形与54个Wilson π模"是否同一对象"（待定区标注） | 文档本人标为"待验证"，但被用作"4D链接到Wilson相位"的暗示性桥梁 | overlap矩阵秩=19（非54）已由verify_pending_4_three54.py证伪直接对应；但54=54的数量巧合仍在使用；MULTI-PATH在数量上，SINGLE-PATH在同构声明上——区分未明确 |

---

## Counts Summary

**T1 Script results:**
- Total run: 47
- CLEAN: 44
- FAIL (real): 3 (bcc_diag5.py, d6_zero_mode_analysis.py, verify_pending_4b_rank19_shell.py)
- NOTE (expected failure output = PASS): 1 (verify_pending_6_bvn_ijk.py)

**T2 Claim matrix:**
- Total claims: 71
- NUMBER: 37
- FORMULA: 16
- INTERPRETIVE: 18
- SINGLE-PATH: 64
- MULTI-PATH: 3 (#42, #43-as-note, #71)
- EXTERNAL: 2 (#61, #70)
- NOT FOUND: 2 (#61, #70)

**T3 queue (SINGLE-PATH + INTERPRETIVE + load-bearing):** 6 entries
- T3-1: Q3 立方体图解读 (§10)
- T3-2: 谱钉扎代数锁死解读 (§11)
- T3-3: Moreno G_2 文献未核实 (§11/§13)
- T3-4: π_6(O)=0→G_2 结构链 (§13终，supporting script partially broken)
- T3-5: SAME-84 Klein同构 (§13终/§19)
- T3-6: 54=54 数量巧合桥梁 (§8/§11)
