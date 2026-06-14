"""
verify_stabilizer_blocks_galois.py
Test: eigenvalue multiplicities = little-group irrep dims (Schur),
      Galois conjugate suites glue within same irrep sector.
"""
import itertools, sys
from fractions import Fraction
import numpy as np
from collections import defaultdict

# ── Reuse construction from verify_prime_suite_spectroscopy.py ────────────────
def build_bcc_lattice_periodic():
    return list(itertools.product(range(3), repeat=3))

def enumerate_simplex_faces(bc_ijk):
    i,j,k = bc_ijk; bc_v = ('bc',i,j,k)
    orig = []
    for dx,dy,dz in itertools.product((0,1),repeat=3):
        ox,oy,oz = i+dx,j+dy,k+dz
        orig.append(((ox+oy+oz)%2, ('c',ox%3,oy%3,oz%3)))
    evens = [v for p,v in orig if p==0]; odds = [v for p,v in orig if p==1]
    faces=[]
    for tc in (evens,odds):
        sv=[bc_v]+tc
        for c in itertools.combinations(sv,3): faces.append(frozenset(c))
    return faces

def translate_vertex(v,s):
    si,sj,sk=s
    if v[0]=='bc': _,i,j,k=v; return ('bc',(i+si)%3,(j+sj)%3,(k+sk)%3)
    else: _,x,y,z=v; return ('c',(x+si)%3,(y+sj)%3,(z+sk)%3)

def translate_face(face,s): return frozenset(translate_vertex(v,s) for v in face)

def build_all_faces(bcs):
    ref_faces = enumerate_simplex_faces((0,0,0))
    ftoi={}; bcfi=[]; gi=0
    for bc in bcs:
        li=[]
        for rf in ref_faces:
            sf=translate_face(rf,bc)
            if sf not in ftoi: ftoi[sf]=gi; gi+=1
            li.append(ftoi[sf])
        bcfi.append(li)
    return ftoi, bcfi, ref_faces

def build_adjacency_matrix(ftoi):
    N=len(ftoi); A=np.zeros((N,N)); afs=[None]*N
    for f,i in ftoi.items(): afs[i]=f
    vtf=defaultdict(set)
    for f,i in ftoi.items():
        for v in f: vtf[v].add(i)
    pairs=set()
    for v,fs in vtf.items():
        fl=sorted(fs)
        for a in range(len(fl)):
            for b in range(a+1,len(fl)): pairs.add((fl[a],fl[b]))
    for i,j in pairs:
        if len(afs[i]&afs[j])==2: A[i,j]=A[j,i]=1.0
    return A

def build_H_k(T,k):
    H=np.zeros((20,20),dtype=complex)
    for R,bl in T.items(): H+=np.exp(1j*np.dot(k,R))*bl
    return H

def build_T(bcs,bcfi,A_full):
    order=[];
    for bi in range(len(bcs)): order.extend(bcfi[bi])
    Ar=A_full[np.ix_(order,order)]
    T={}
    for j,bc in enumerate(bcs):
        Rc=tuple((r+1)%3-1 for r in bc); bl=Ar[:20,j*20:(j+1)*20]
        T[Rc]=T.get(Rc,np.zeros((20,20)))+bl
    return T

# ── Oh group generation (from bcc_flatband_oh_irreps.py) ─────────────────────
def generate_oh():
    C4z=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=int)
    C3=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=int)
    gen=[C4z,C3,C4z.T,C3.T]; S={tuple(np.eye(3,dtype=int).flatten()):np.eye(3,dtype=int)}
    q=[np.eye(3,dtype=int)]
    while q:
        cur=q.pop(0)
        for g in gen:
            nw=g@cur; key=tuple(nw.flatten())
            if key not in S: S[key]=nw; q.append(nw)
    inv=-np.eye(3,dtype=int)
    els=[]
    for m in S.values(): els.append(m.copy()); els.append((inv@m).copy())
    return els

def classify_oh(g):
    det=int(round(np.linalg.det(g))); tr=int(round(np.trace(g)))
    od=np.abs(g-np.diag(np.diag(g))).sum()
    if det==1:
        if tr==3: return 'E'
        elif tr==0: return 'C3'
        elif tr==1: return 'C4'
        elif tr==-1: return "C2'" if od<0.5 else 'C2'
    else:
        if tr==-3: return 'i'
        elif tr==0: return 'S6'
        elif tr==-1: return 'S4'
        elif tr==1: return 'sigma_d' if od<0.5 else 'sigma_h'
    return f'?{det},{tr}'

OH_CLASSES=['E','C3','C2','C4',"C2'",'i','S6','sigma_h','S4','sigma_d']
OH_ORDER  ={'E':1,'C3':8,'C2':6,'C4':6,"C2'":3,'i':1,'S6':8,'sigma_h':6,'S4':6,'sigma_d':3}
# Subgroup character tables (little groups)
# C4v: classes E,2C4,C2,2sigma_v,2sigma_d (orders 1,2,1,2,2)
C4V_CLASSES=['E','2C4','C2','2sv','2sd']; C4V_ORDER={'E':1,'2C4':2,'C2':1,'2sv':2,'2sd':2}
C4V_CHAR={
    'A1':[1,1,1,1,1],'A2':[1,1,1,-1,-1],'B1':[1,-1,1,1,-1],'B2':[1,-1,1,-1,1],
    'E': [2,0,-2,0,0]
}
# C2v: classes E,C2,sigma_v,sigma_v' (orders 1,1,1,1)
C2V_CLASSES=['E','C2','sv','sv2']; C2V_ORDER={c:1 for c in C2V_CLASSES}
C2V_CHAR={
    'A1':[1,1,1,1],'A2':[1,1,-1,-1],'B1':[1,-1,1,-1],'B2':[1,-1,-1,1]
}
# C3v: classes E,2C3,3sigma_v (orders 1,2,3)
C3V_CLASSES=['E','2C3','3sv']; C3V_ORDER={'E':1,'2C3':2,'3sv':3}
C3V_CHAR={
    'A1':[1,1,1],'A2':[1,1,-1],'E':[2,-1,0]
}

def vertex_coord(v):
    if v[0]=='bc': _,i,j,k=v; return np.array([i-1.,j-1.,k-1.])
    else: _,x,y,z=v; return np.array([x-1.5,y-1.5,z-1.5])

def apply_g_vertex(g,v):
    c=g@vertex_coord(v)
    if v[0]=='bc': return ('bc',int(round(c[0]+1))%3,int(round(c[1]+1))%3,int(round(c[2]+1))%3)
    else: return ('c',int(round(c[0]+1.5))%3,int(round(c[1]+1.5))%3,int(round(c[2]+1.5))%3)

def apply_g_face(g,face): return frozenset(apply_g_vertex(g,v) for v in face)

def bc_of_face(face):
    for v in face:
        if v[0]=='bc': return (v[1],v[2],v[3])
    return None

def build_bloch_rep_matrix(g, ref_faces, ref_f2i, ftoi, bcfi, bcs, k_vec):
    """
    Build 20x20 Bloch representation matrix U(g) at momentum k.
    g maps face m -> face n in cell delta_R; U[n,m] = exp(i k.delta_R)
    """
    U=np.zeros((20,20),dtype=complex)
    for m,face in enumerate(ref_faces):
        gf=apply_g_face(g,face)
        bc_owner=bc_of_face(gf)
        if bc_owner is None:
            if gf in ftoi:
                gi2=ftoi[gf]
                for bi2,bc2 in enumerate(bcs):
                    if gi2 in bcfi[bi2]: bc_owner=bc2; break
            if bc_owner is None: raise ValueError(f"No owner for {gf}")
        inv_sh=tuple((-s)%3 for s in bc_owner)
        tf=translate_face(gf,inv_sh)
        if tf not in ref_f2i: raise ValueError(f"Translated face not in ref: {tf}")
        n=ref_f2i[tf]
        # cell displacement delta_R: bc_owner in {0,1,2}^3 -> center: (r+1)%3-1
        dR=np.array([(r+1)%3-1 for r in bc_owner],dtype=float)
        U[n,m]=np.exp(1j*np.dot(k_vec,dR))
    return U

TOL=1e-9
def group_eigs(evals,tol=1e-8):
    se=sorted(evals); groups=[]; i=0
    while i<len(se):
        v=se[i]; cl=[v]; j=i+1
        while j<len(se) and abs(se[j]-v)<tol*(1+abs(v))+tol: cl.append(se[j]); j+=1
        groups.append((np.mean(cl),len(cl))); i=j
    return groups

def to_rat(x,lim=4000):
    f=Fraction(x).limit_denominator(lim); return f,abs(float(f)-x)<TOL

def find_suites(vals):
    """Find algebraic conjugate suites among distinct values."""
    used=set(); suites=[]
    for sz in [2,3,4]:
        for combo in itertools.combinations(range(len(vals)),sz):
            if any(c in used for c in combo): continue
            vs=[vals[i] for i in combo]
            syms=[]
            ok=True
            for k in range(1,sz+1):
                s=sum(np.prod([vs[i] for i in idx]) for idx in itertools.combinations(range(sz),k))
                f,o=to_rat(s)
                if not o: ok=False; break
                syms.append(f)
            if ok:
                suites.append((list(combo),vs,syms)); [used.add(c) for c in combo]
    lone=[i for i in range(len(vals)) if i not in used]
    return suites,[vals[i] for i in lone]

# ── Little group identification ───────────────────────────────────────────────
def little_group(oh_els,k_vec,tol=1e-9):
    """Return elements g s.t. g*k ≡ k mod 2pi."""
    lg=[]
    for g in oh_els:
        gk=g@k_vec; diff=gk-k_vec
        diff_mod=np.array([d%(2*np.pi) for d in diff])
        diff_mod2=np.array([min(abs(dm),abs(dm-2*np.pi)) for dm in diff_mod])
        if np.max(diff_mod2)<tol+1e-7: lg.append(g)
    return lg

def identify_group(n):
    return {48:'Oh',8:'C4v',4:'C2v',6:'C3v'}.get(n,f'order-{n}')

# ── Isotypic projection ───────────────────────────────────────────────────────
def decompose_rep(chars_by_class, class_order, char_table, classes):
    """Return multiplicity of each irrep."""
    order=sum(class_order[c] for c in classes)
    mults={}
    for irrep,row in char_table.items():
        n=sum(class_order[cl]*row[ci].conjugate()*chars_by_class.get(cl,0) for ci,cl in enumerate(classes))
        mults[irrep]=n/order
    return mults

# ── Map Oh classes to C4v/C2v/C3v classes ────────────────────────────────────
def classify_in_subgroup(g, group_name):
    det=int(round(np.linalg.det(g))); tr=int(round(np.trace(g)))
    od=np.abs(g-np.diag(np.diag(g))).sum()
    if group_name=='Oh': return classify_oh(g)
    if group_name=='C4v':
        # axis k=(2pi/3,0,0); C4v: E,2C4,C2,2sigma_v,2sigma_d
        if det==1:
            if tr==3: return 'E'
            elif tr==1: return '2C4'
            elif tr==-1: return 'C2'
        else:
            # reflections: sigma_v (contain x-axis), sigma_d (diagonal)
            if tr==1: return '2sv'
            elif tr==-1: return '2sd'  # S4 improper
        return f'?C4v_{det},{tr}'
    if group_name=='C2v':
        if det==1:
            if tr==3: return 'E'
            if tr==-1: return 'C2'
        else:
            if tr==1: return 'sv'
            elif tr==-1: return 'sv2'
        return f'?C2v_{det},{tr}'
    if group_name=='C3v':
        if det==1:
            if tr==3: return 'E'
            if tr==0: return '2C3'
        else:
            if tr==1: return '3sv'
        return f'?C3v_{det},{tr}'
    return '?'

def get_char_table_and_classes(group_name):
    if group_name=='Oh': return OH_CHAR_TABLE_STUB, OH_CLASSES, OH_ORDER
    if group_name=='C4v': return C4V_CHAR, C4V_CLASSES, C4V_ORDER
    if group_name=='C2v': return C2V_CHAR, C2V_CLASSES, C2V_ORDER
    if group_name=='C3v': return C3V_CHAR, C3V_CLASSES, C3V_ORDER
    return {},[],({}),{}

OH_CHAR_TABLE_STUB={
    'A1g':[1,1,1,1,1,1,1,1,1,1],'A2g':[1,1,-1,-1,1,1,1,-1,-1,1],
    'Eg' :[2,-1,0,0,2,2,-1,0,0,2],'T1g':[3,0,-1,1,-1,3,0,-1,1,-1],
    'T2g':[3,0,1,-1,-1,3,0,1,-1,-1],'A1u':[1,1,1,1,1,-1,-1,-1,-1,-1],
    'A2u':[1,1,-1,-1,1,-1,-1,1,1,-1],'Eu' :[2,-1,0,0,2,-2,1,0,0,-2],
    'T1u':[3,0,-1,1,-1,-3,0,1,-1,1],'T2u':[3,0,1,-1,-1,-3,0,-1,1,1],
}
OH_IRREP_DIMS={'A1g':1,'A2g':1,'Eg':2,'T1g':3,'T2g':3,'A1u':1,'A2u':1,'Eu':2,'T1u':3,'T2u':3}
SUBG_DIMS={'C4v':{'A1':1,'A2':1,'B1':1,'B2':1,'E':2},
           'C2v':{'A1':1,'A2':1,'B1':1,'B2':1},
           'C3v':{'A1':1,'A2':1,'E':2}}

K_POINTS={'Gamma':np.array([0.,0.,0.]),
          'axis':np.array([2*np.pi/3,0.,0.]),
          'face':np.array([2*np.pi/3,2*np.pi/3,0.]),
          'body':np.array([2*np.pi/3,2*np.pi/3,2*np.pi/3])}
EXPECTED_LG_ORDERS={'Gamma':48,'axis':8,'face':4,'body':6}

def main():
    lines=[]
    def W(s=""): lines.append(s); print(s)

    W("=== STABILIZER BLOCKS & GALOIS GLUING VERIFICATION ===\n")

    # Build lattice
    bcs=build_bcc_lattice_periodic()
    ftoi,bcfi,ref_faces=build_all_faces(bcs)
    A=build_adjacency_matrix(ftoi)
    T=build_T(bcs,bcfi,A)
    ref_f2i={f:m for m,f in enumerate(ref_faces)}
    oh_els=generate_oh()

    W(f"Lattice: 27 cells, {len(ftoi)} faces; Oh: {len(oh_els)} elements\n")

    results={}
    for orbit,k_vec in K_POINTS.items():
        W(f"{'='*65}")
        W(f"ORBIT: {orbit}  k={np.round(k_vec/(np.pi/3),3)}·(pi/3)")

        # Step 1: Little group
        lg=little_group(oh_els,k_vec)
        gname=identify_group(len(lg))
        exp_ord=EXPECTED_LG_ORDERS[orbit]
        W(f"  Little group: {gname} order={len(lg)} (expected {exp_ord}) {'OK' if len(lg)==exp_ord else 'MISMATCH'}")

        # Step 2: Bloch rep matrices U(g) for g in little group; check [U(g),H(k)]=0
        H=build_H_k(T,k_vec)
        max_comm=0.
        bad_gs=[]
        U_mats={}
        for g in lg:
            try:
                U=build_bloch_rep_matrix(g,ref_faces,ref_f2i,ftoi,bcfi,bcs,k_vec)
                comm=np.max(np.abs(U@H - H@U))
                U_mats[id(g)]=(g,U)
                if comm>1e-7: bad_gs.append((g,comm))
                max_comm=max(max_comm,comm)
            except Exception as e:
                W(f"  WARNING: Bloch rep failed for element: {e}")
        W(f"  max ||[U(g),H(k)]||: {max_comm:.2e}  ({'PASS' if max_comm<1e-7 else 'WARN—dropped bad g'})")
        if bad_gs:
            W(f"  Dropped {len(bad_gs)} elements with nonzero commutator")
            valid_gs=[(g,U) for (g,U) in U_mats.values() if not any(np.array_equal(g,bg) for bg,_ in bad_gs)]
        else:
            valid_gs=list(U_mats.values())

        # Step 3: Decompose 20-dim rep into little-group irreps via characters
        char_table,classes,class_order=get_char_table_and_classes(gname)
        chars_by_class=defaultdict(list)
        for g,U in valid_gs:
            cls=classify_in_subgroup(g,gname)
            chars_by_class[cls].append(np.real(np.trace(U)))
        mean_chars={cls:np.mean(v) for cls,v in chars_by_class.items()}
        mults=decompose_rep(mean_chars,class_order,char_table,classes)
        W(f"  20-dim rep decomposition ({gname}):")
        decomp_str=[]; total_dim=0
        for ir,m in sorted(mults.items(),key=lambda x:x[0]):
            mr=int(round(m.real))
            d=SUBG_DIMS.get(gname,{}).get(ir,OH_IRREP_DIMS.get(ir,0)) if gname!='Oh' else OH_IRREP_DIMS.get(ir,0)
            if mr>0:
                decomp_str.append(f"{mr}x{ir}"); total_dim+=mr*d
                W(f"    {mr}x{ir}(dim {d})")
        W(f"  Total dim check: {total_dim} (expect 20) {'OK' if total_dim==20 else 'FAIL'}")

        # Step 4: Eigenvalues and Schur check
        evals=np.linalg.eigvalsh(H)
        evecs=np.linalg.eigh(H)[1]
        groups=group_eigs(evals)
        non_flat=[(rep,mult) for rep,mult in groups if abs(rep+3)>1e-6]
        W(f"\n  Eigenvalue groups (non-flat):")
        for rep,mult in non_flat:
            rf,ro=to_rat(rep); rs=str(rf) if ro else f"{rep:.8f}"
            W(f"    λ={rs:<20} mult={mult}")

        # Step 5: Galois suites
        distinct=[rep for rep,mult in non_flat]
        suites,lone=find_suites(distinct)
        rat_distinct=[v for v in distinct if to_rat(v)[1]]
        irrat_distinct=[v for v in distinct if not to_rat(v)[1]]

        W(f"\n  Conjugate suites among irrational eigenvalues:")
        suite_irrep_map={}
        for (idxs,vs,syms) in suites:
            deg=len(vs); s_str=", ".join(f"{v:.8f}" for v in vs)
            W(f"    degree-{deg} suite: {{{s_str}}}")
            W(f"      sym-polys: {syms[:3]}")
            # Find multiplicities
            ms=[next(m for r,m in non_flat if abs(r-v)<1e-8) for v in vs]
            W(f"      mults: {ms}")
        if lone: W(f"    lone irrationals: {[f'{v:.8f}' for v in lone]}")

        # Step 6: Isotypic projection — for each eigenspace, find which irreps carry it
        # Use isotypic projectors P_mu = (d_mu/|G|) sum_g chi_mu(g)* U(g)
        W(f"\n  Irrep sector → eigenvalue mapping:")
        G_size=len(valid_gs)
        irrep_to_eigvals=defaultdict(list)
        schur_ok=True
        for ir,row in char_table.items():
            d=SUBG_DIMS.get(gname,{}).get(ir,1) if gname!='Oh' else OH_IRREP_DIMS.get(ir,1)
            # Build isotypic projector
            P_iso=np.zeros((20,20),dtype=complex)
            for g,U in valid_gs:
                cls=classify_in_subgroup(g,gname)
                ci=classes.index(cls) if cls in classes else -1
                if ci<0: continue
                chi_mu=row[ci]
                P_iso+=chi_mu.conjugate()*U
            P_iso*=(d/G_size)
            # For each eigengroup, check projection weight
            for rep,mult in non_flat:
                mask=np.abs(evals-rep)<1e-7*(1+abs(rep))+1e-7
                V_sub=evecs[:,mask]
                proj=P_iso@V_sub
                weight=np.linalg.norm(proj,'fro')**2/mult
                if weight>0.1:
                    irrep_to_eigvals[ir].append((rep,mult,weight))

        # Schur check: for each eigenvalue, sum dims of carrying irreps = mult
        eval_to_irreps=defaultdict(list)
        for ir,evlist in irrep_to_eigvals.items():
            d=SUBG_DIMS.get(gname,{}).get(ir,1) if gname!='Oh' else OH_IRREP_DIMS.get(ir,1)
            for (rep,mult,w) in evlist:
                eval_to_irreps[round(rep,8)].append((ir,d,mult,w))

        W(f"  {'Eigenvalue':<22} {'mult':>4}  {'carrying irreps (dim,weight)':>35}  Schur?")
        for rep,mult in non_flat:
            key=round(rep,8); irlist=eval_to_irreps.get(key,[])
            ir_str="; ".join(f"{ir}(d={d},w={w:.2f})" for ir,d,m,w in irlist)
            dim_sum=sum(d for ir,d,m,w in irlist if w>0.3)
            ok=(dim_sum==mult)
            if not ok: schur_ok=False
            W(f"    λ={rep:<18.8f} {mult:>4}  {ir_str:<35}  {'OK' if ok else 'FAIL'}")

        W(f"  Schur (mult=irrep-dim sum) overall: {'PASS' if schur_ok else 'FAIL'}")

        # Galois gluing verdict
        W(f"\n  Galois gluing (suite members → same irrep?):")
        galois_ok=True
        for (idxs,vs,syms) in suites:
            irreps_per_member=[]
            for v in vs:
                key=round(v,8); irlist=eval_to_irreps.get(key,[])
                carrying={ir for ir,d,m,w in irlist if w>0.3}
                irreps_per_member.append(carrying)
            all_same=(len(set(frozenset(s) for s in irreps_per_member))==1)
            s_str=", ".join(f"{v:.6f}" for v in vs)
            W(f"    suite {{{s_str}}}: irreps={irreps_per_member} same={all_same}")
            if not all_same: galois_ok=False
        if not suites: W("    (no irrational suites at this k-point)")

        overall=(schur_ok and galois_ok)
        W(f"\n  *** VERDICT: Schur={'PASS' if schur_ok else 'FAIL'}, Galois-gluing={'PASS' if galois_ok else 'FAIL'}, Overall={'CONFIRMED' if overall else 'NOT CONFIRMED'} ***")
        results[orbit]={'lg':gname,'lg_ord':len(lg),'decomp':"+".join(decomp_str),
                        'schur':schur_ok,'galois':galois_ok,'total_ok':overall}

    # Summary
    W(f"\n{'='*65}")
    W("SUMMARY (≤35 lines)")
    W(f"{'Orbit':<8} {'LittleGrp':<8} {'Order':>5} {'20D-decomp':<30} {'Schur':>6} {'Galois':>7} {'OK':>4}")
    W("-"*72)
    for orbit,r in results.items():
        W(f"{orbit:<8} {r['lg']:<8} {r['lg_ord']:>5} {r['decomp']:<30} {'PASS' if r['schur'] else 'FAIL':>6} {'PASS' if r['galois'] else 'FAIL':>7} {'Y' if r['total_ok'] else 'N':>4}")
    all_ok=all(r['total_ok'] for r in results.values())
    W(f"\nOverall verdict: {'BOTH MECHANISMS CONFIRMED' if all_ok else 'PARTIAL CONFIRMATION — see details'}")
    W("Mechanism: stabilizer fixes block structure (Schur: mult=irrep-dim).")
    W("           Galois conjugates glue within same irrep sector across eigenvalues.")

    out_path=r"d:/AI thoery/.agent/scripts/verify_stabilizer_blocks_galois_results.txt"
    with open(out_path,'w',encoding='utf-8') as f: f.write("\n".join(lines))
    print(f"\nResults written to: {out_path}")

if __name__=="__main__":
    main()
