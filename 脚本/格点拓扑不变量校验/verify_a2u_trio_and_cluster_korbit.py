"""
verify_a2u_trio_and_cluster_korbit.py

TEST 1: A2u trio under Z3^3 translations — do translations connect the three A2u
        vectors across singular-value clusters (sizes 1, 6, 12)?
TEST 2: cluster ↔ k-orbit — does the 1-cluster live at Γ, 6-cluster on axis orbit,
        12-cluster on face-diagonal orbit, body-diagonal absent from rank-19 range?
"""

import sys, io, itertools
import numpy as np
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT = "d:/AI thoery/.agent/scripts/verify_a2u_trio_and_cluster_korbit_results.txt"
_LOG = []
def log(m=""): print(m); _LOG.append(str(m))

# ── reuse: lattice + face construction ──────────────────────────────────────

def build_bcc_lattice_periodic():
    return list(itertools.product(range(3), repeat=3))

def enumerate_simplex_faces(bc_ijk):
    i, j, k = bc_ijk
    bc_v = ('bc', i, j, k)
    orig = []
    for dx, dy, dz in itertools.product((0,1), repeat=3):
        ox,oy,oz = i+dx, j+dy, k+dz
        orig.append(((ox+oy+oz)%2, ('c', ox%3, oy%3, oz%3)))
    even = [v for (p,v) in orig if p==0]
    odd  = [v for (p,v) in orig if p==1]
    faces = []
    for corners in (even, odd):
        verts = [bc_v]+corners
        for combo in itertools.combinations(verts,3):
            faces.append(frozenset(combo))
    return faces

def translate_vertex(v, s):
    si,sj,sk = s
    if v[0]=='bc': _,i,j,k=v; return ('bc',(i+si)%3,(j+sj)%3,(k+sk)%3)
    else:          _,cx,cy,cz=v; return ('c',(cx+si)%3,(cy+sj)%3,(cz+sk)%3)

def translate_face(face, s):
    return frozenset(translate_vertex(v,s) for v in face)

def build_all_faces(body_centers):
    ref_faces = enumerate_simplex_faces((0,0,0))
    face_to_idx = {}; bc_face_indices = []; idx = 0
    for bc_ijk in body_centers:
        local = []
        for rf in ref_faces:
            sf = translate_face(rf, bc_ijk)
            if sf not in face_to_idx: face_to_idx[sf]=idx; idx+=1
            local.append(face_to_idx[sf])
        bc_face_indices.append(local)
    return face_to_idx, bc_face_indices

def build_adjacency_matrix(face_to_idx):
    N = len(face_to_idx)
    A = np.zeros((N,N))
    faces_list = [None]*N
    for f,i in face_to_idx.items(): faces_list[i]=f
    v2f = defaultdict(set)
    for f,i in face_to_idx.items():
        for v in f: v2f[v].add(i)
    pairs = set()
    for fset in v2f.values():
        fl = sorted(fset)
        for a in range(len(fl)):
            for b in range(a+1,len(fl)): pairs.add((fl[a],fl[b]))
    for i,j in pairs:
        if len(faces_list[i]&faces_list[j])==2: A[i,j]=A[j,i]=1.
    return A

def build_pi_modes_and_simplices(body_centers, face_to_idx, bc_face_indices):
    A = build_adjacency_matrix(face_to_idx)
    perm = []
    for bc_idx in range(27): perm.extend(bc_face_indices[bc_idx])
    perm = np.array(perm,dtype=int)
    A_r = A[np.ix_(perm,perm)]
    def blk(A_r,i,j,s=20): return A_r[i*s:(i+1)*s, j*s:(j+1)*s].copy()
    T = {}
    for bc_j_idx,bc_j_ijk in enumerate(body_centers):
        T[tuple(x%3 for x in bc_j_ijk)] = blk(A_r,0,bc_j_idx)
    kvecs = list(itertools.product(range(3),repeat=3))
    vecs_all = {}
    for nt in kvecs:
        k = np.array([2*np.pi*n/3. for n in nt])
        H = sum(np.exp(1j*np.dot(k,np.array(R,dtype=float)))*mat for R,mat in T.items())
        ev,evec = np.linalg.eigh(H)
        assert np.allclose(ev[:6],-3.,atol=1e-10)
        vecs_all[nt] = evec
    loops={}
    for ax_i,ax in enumerate(['x','y','z']):
        for o1 in range(3):
            for o2 in range(3):
                lk=[]
                for na in range(3):
                    nt=[0,0,0]; nt[ax_i]=na
                    oa=[i for i in range(3) if i!=ax_i]
                    nt[oa[0]]=o1; nt[oa[1]]=o2
                    lk.append(tuple(nt))
                loops[(ax,o1,o2)]=lk
    pi_modes=[]
    for lk in loops.values():
        W=np.eye(6,dtype=complex)
        for j in range(3):
            kc,kn=lk[j],lk[(j+1)%3]
            S=np.array([[np.vdot(vecs_all[kc][:,a],vecs_all[kn][:,b])
                         for b in range(6)] for a in range(6)])
            W=W@S
        ev,evec=np.linalg.eig(W)
        for idx,(p,v) in enumerate(zip(np.angle(ev),evec.T)):
            if abs(abs(p)-np.pi)<0.1:
                Psi=vecs_all[lk[0]][:,:6]@v
                pi_modes.append((lk[0],Psi))
    assert len(pi_modes)==54
    pi540=np.zeros((54,540),dtype=complex)
    for m,(ks,Psi) in enumerate(pi_modes):
        kv=np.array([2*np.pi*n/3. for n in ks])
        for bc_idx,bc_ijk in enumerate(body_centers):
            ph=np.exp(1j*np.dot(kv,np.array(bc_ijk,dtype=float)))
            for li in range(20):
                pi540[m,bc_face_indices[bc_idx][li]]=Psi[li]*ph
    simp=np.zeros((54,540))
    for bc_idx in range(27):
        gf=bc_face_indices[bc_idx]
        for li in range(10):  simp[2*bc_idx,  gf[li]]=1.
        for li in range(10,20): simp[2*bc_idx+1,gf[li]]=1.
    return pi540, simp, pi_modes

# ── O_h for A2u extraction ───────────────────────────────────────────────────

OH_CLASSES=['E','C3','C2','C4',"C2'",'i','S6','sigma_h','S4','sigma_d']
OH_CLASS_ORDER={'E':1,'C3':8,'C2':6,'C4':6,"C2'":3,'i':1,'S6':8,'sigma_h':6,'S4':6,'sigma_d':3}
OH_CHAR_TABLE={
    'A2u':[ 1, 1,-1,-1, 1,-1,-1, 1, 1,-1],
}

def generate_oh_elements():
    C4z=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=int)
    C3=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=int)
    def key(m): return tuple(m.flatten())
    O={}; q=[np.eye(3,dtype=int)]; O[key(np.eye(3,dtype=int))]=np.eye(3,dtype=int)
    for g in [C4z,C3,C4z.T,C3.T]:
        pass
    queue=[np.eye(3,dtype=int)]
    while queue:
        cur=queue.pop(0)
        for g in [C4z,C3,C4z.T,C3.T]:
            nw=g@cur; k=key(nw)
            if k not in O: O[k]=nw; queue.append(nw)
    assert len(O)==24
    inv=-np.eye(3,dtype=int)
    oh=[]
    for m in O.values(): oh.append(m.copy()); oh.append((inv@m).copy())
    assert len(oh)==48
    return oh

def classify(g):
    det=int(round(np.linalg.det(g))); tr=int(round(np.trace(g)))
    if det==1:
        if tr==3: return 'E'
        elif tr==0: return 'C3'
        elif tr==1: return 'C4'
        elif tr==-1:
            return "C2'" if np.abs(g-np.diag(np.diag(g))).sum()<0.5 else 'C2'
    else:
        if tr==-3: return 'i'
        elif tr==0: return 'S6'
        elif tr==-1: return 'S4'
        elif tr==1:
            return 'sigma_d' if np.abs(g-np.diag(np.diag(g))).sum()<0.5 else 'sigma_h'

def vertex_coord(v):
    if v[0]=='bc': _,i,j,k=v; return np.array([i-1.,j-1.,k-1.])
    else: _,cx,cy,cz=v; return np.array([cx-1.5,cy-1.5,cz-1.5])

def apply_oh_face(g,face):
    def xfm(v):
        c=vertex_coord(v); nc=g@c
        if v[0]=='bc': return ('bc',int(round(nc[0]+1.))%3,int(round(nc[1]+1.))%3,int(round(nc[2]+1.))%3)
        else:          return ('c', int(round(nc[0]+1.5))%3,int(round(nc[1]+1.5))%3,int(round(nc[2]+1.5))%3)
    return frozenset(xfm(v) for v in face)

def build_row_of(g, face_to_idx):
    N=len(face_to_idx); faces=[None]*N
    for f,i in face_to_idx.items(): faces[i]=f
    row_of=np.zeros(N,dtype=int)
    for i,f in enumerate(faces): row_of[i]=face_to_idx[apply_oh_face(g,f)]
    return row_of

def a2u_projector_on_cluster(U_c, oh_elements, all_row_of, elem_class):
    """Project U_c (d×540) rows onto A2u isotypic component."""
    d=len(U_c)
    # A2u character: chi(g)
    chi_A2u={cls:OH_CHAR_TABLE['A2u'][OH_CLASSES.index(cls)] for cls in OH_CLASSES}
    # Projector: P = (1/48)*sum_g chi(g)* R(g)
    # R(g)_{ab} = U_c[a,:][row_of] . U_c[b,:].conj() = U_c[:,row_of] @ U_c.conj().T
    P=np.zeros((d,d),dtype=complex)
    for g_idx,g in enumerate(oh_elements):
        ro=all_row_of[g_idx]
        Rg=U_c[:,ro]@U_c.conj().T
        P+=chi_A2u[elem_class[g_idx]]*Rg
    P/=48.
    # Eigenvector of P with eigenvalue ~1
    ev,evec=np.linalg.eigh(P)
    # A2u is 1D so rank(P)=1; pick the eigenvector with largest eigenvalue
    idx=np.argmax(ev)
    if ev[idx]<0.5:
        return None, ev
    v=evec[:,idx]
    # lift to R^540: sum_j v[j]*U_c[j,:]
    v540=v@U_c
    v540/=np.linalg.norm(v540)
    return v540, ev

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    log("="*70)
    log("A2u TRIO + CLUSTER-KORBIT VERIFICATION")
    log("="*70)

    body_centers=build_bcc_lattice_periodic()
    face_to_idx,bc_face_indices=build_all_faces(body_centers)
    assert len(face_to_idx)==540
    log(f"Faces: 540 OK")

    pi540,simp,pi_modes=build_pi_modes_and_simplices(body_centers,face_to_idx,bc_face_indices)
    overlap=simp@pi540.T.conj()
    U_full,s_full,Vh_full=np.linalg.svd(overlap)
    rank19=int(np.sum(s_full>1e-5))
    assert rank19==19
    log(f"Overlap rank: {rank19} (OK)")

    # Cluster assignment
    s19=s_full[:19]
    means=sorted(set(np.round(s19,4)))  # crude: use gap detection
    gaps=np.abs(np.diff(np.sort(s19)[::-1]))
    big_gaps=[i for i,g in enumerate(gaps) if g>1e-8]
    # Build cluster membership
    sv_sorted_idx=np.argsort(s19)[::-1]  # descending
    splits=[0]+[b+1 for b in big_gaps]+[19]
    clusters_raw=[]
    for a,b in zip(splits,splits[1:]): clusters_raw.append(list(sv_sorted_idx[a:b]))
    clusters_by_size=sorted(clusters_raw,key=len)  # [1-dim, 6-dim, 12-dim]
    assert [len(c) for c in clusters_by_size]==[1,6,12]
    cl1,cl6,cl12=clusters_by_size
    log(f"Clusters: 1+6+12 verified. Sizes: {[len(c) for c in clusters_by_size]}")
    log(f"  1-dim sv: {s19[cl1[0]]:.6f}")
    log(f"  6-dim sv mean: {np.mean(s19[cl6]):.6f}")
    log(f"  12-dim sv mean: {np.mean(s19[cl12]):.6f}")

    # Lift left SVs to R^540 via simplex rows
    L=(U_full[:,:19].conj().T@simp)  # (19,540)
    norms=np.linalg.norm(L,axis=1,keepdims=True); L/=norms

    # O_h machinery
    oh_elements=generate_oh_elements()
    elem_class=[classify(g) for g in oh_elements]
    all_row_of=[build_row_of(g,face_to_idx) for g in oh_elements]

    # ── TEST 1: A2u trio ──────────────────────────────────────────────────────
    log("\n── TEST 1: A2u trio under Z3^3 translations ──")
    a2u_vecs=[]
    for cl_label,cl_idx in [('1-dim',cl1),('6-dim',cl6),('12-dim',cl12)]:
        U_c=L[cl_idx,:]
        # re-orthogonalize rows
        u_tmp,_,vh_tmp=np.linalg.svd(U_c,full_matrices=False); U_c=u_tmp@vh_tmp
        v540,ev=a2u_projector_on_cluster(U_c,oh_elements,all_row_of,elem_class)
        log(f"  {cl_label}: A2u projector eigenvalues top-3: {sorted(np.real(ev))[-3:]}")
        if v540 is None:
            log(f"  WARNING: no A2u component found in {cl_label}")
            a2u_vecs.append(np.zeros(540,dtype=complex))
        else:
            a2u_vecs.append(v540)
            log(f"  {cl_label}: A2u vector extracted (norm={np.linalg.norm(v540):.6f})")

    v1,v6,v12=a2u_vecs

    # Build Z3^3 translation unitaries: T_R acts on 540 face index space by
    # permuting faces: face f -> translate_face(f, R)
    # We store as row_of arrays like O_h (same mechanism).
    def build_translation_row_of(R):
        N=len(face_to_idx); faces=[None]*N
        for f,i in face_to_idx.items(): faces[i]=f
        row_of=np.zeros(N,dtype=int)
        for i,f in enumerate(faces): row_of[i]=face_to_idx[translate_face(f,R)]
        return row_of

    translations=[(1,0,0),(1,1,0),(1,1,1)]
    trans_row_of=[build_translation_row_of(R) for R in translations]

    # Also build overlap matrix S=simp@simp.T.conj() projected onto 54D:
    # The "overlap operator" in 54-dim is overlap_matrix itself.
    # Commutator [T, S] in 54-dim simplex space:
    # T acts on simplex rows: T*simp[i,:] = simp[i,trans_row_of]
    # In 54-dim: T_54[i,j] = <e_i | T | e_j> = dot(simp[i,:], simp[j,ro])
    # S_54 = overlap = simp@pi540.T.conj()  -- not square in same space...
    # Use the 54x54 Gram matrix of simplex vectors G=simp@simp.T as the "overlap in 54D"
    G=simp@simp.T  # (54,54) real, the natural inner product matrix
    # T on simplex basis: T_R[i,j] = (simp[i,ro]*simp[j,:]).sum() / (norms)
    # But simp rows are indicator vectors (not normalized), so T_R as operator on R^54:
    # (T_R)_{ij} = simp[i, ro] . simp[j, :] -- NO, T_R should map simp[j] -> simp[j][ro]
    # More precisely: T_R maps the simplex subspace to itself (since translation is a symmetry).
    # Represent T_R on the 54-dim space: use ONB from SVD of simp.
    # simp = U_s @ S_s @ Vh_s; rows of Vh_s are ONB in R^540.
    _,sv_s,Vh_s=np.linalg.svd(simp,full_matrices=False)  # Vh_s (54,540)
    T54_mats=[]
    for ro in trans_row_of:
        # T maps basis vector Vh_s[a,:] -> Vh_s[a,ro] (permuted)
        # R_T[a,b] = Vh_s[a,ro] @ Vh_s[b,:].conj().T = dot(Vh_s[a,ro], Vh_s[b])
        T54=Vh_s[:,ro]@Vh_s.T  # (54,54)
        T54_mats.append(T54)

    # The overlap operator in the 54-dim simplex ONB space:
    # S matrix acting on Vh_s: S_{ab} = Vh_s[a,:] @ overlap_matrix_in_540 @ Vh_s[b,:].conj()
    # But overlap_matrix was (54×54) in the simp/pi basis, not 540-dim.
    # For commutator test: work in R^540 projected to simp subspace:
    # S540 = simp.T @ inv(simp@simp.T) @ simp  is the projector; the "overlap operator" O540
    # = pi540.T@pi540 restricted... Instead use the natural one:
    # S_op in simp subspace = (Vh_s @ pi540.T) @ (pi540 @ Vh_s.T) = (Vh_s@pi540.T)@conj (54x54)
    S_op54=(Vh_s@pi540.T.conj())@(pi540@Vh_s.T)  # (54,54) complex
    comms=[]
    for R,T54 in zip(translations,T54_mats):
        comm=T54@S_op54-S_op54@T54
        comms.append(np.linalg.norm(comm,'fro'))
        log(f"  ||[T_{R}, S]||_F = {comms[-1]:.4e}")

    # 3x3 matrices M_R[i,j] = |<v_i|T_R|v_j>| for i,j in {v1,v6,v12}
    log("\n  3x3 |M_R| matrices (|<A2u_i | T_R | A2u_j>|), rows/cols = [1,6,12]:")
    for R,ro in zip(translations,trans_row_of):
        M=np.zeros((3,3))
        for ri,vi in enumerate([v1,v6,v12]):
            for rj,vj in enumerate([v1,v6,v12]):
                M[ri,rj]=abs(np.dot(vi.conj(),vj[ro]))
        log(f"  T_{R}:")
        for row_i,row in enumerate(M):
            log(f"    [{' '.join(f'{x:.4f}' for x in row)}]")
    # Verdict
    # Check off-diagonal magnitudes
    max_offdiag=max(abs(np.dot(v1.conj(),v6[ro]))+abs(np.dot(v1.conj(),v12[ro]))+
                    abs(np.dot(v6.conj(),v1[ro]))+abs(np.dot(v6.conj(),v12[ro]))+
                    abs(np.dot(v12.conj(),v1[ro]))+abs(np.dot(v12.conj(),v6[ro]))
                    for ro in trans_row_of)
    connected = max_offdiag > 0.1
    log(f"\n  Max total off-diagonal: {max_offdiag:.4f}")
    log(f"  TEST 1 VERDICT: A2u vectors are {'CONNECTED (off-diag)' if connected else 'ISOLATED (block-diagonal)'} under translations")

    # ── TEST 2: cluster ↔ k-orbit ─────────────────────────────────────────────
    log("\n── TEST 2: cluster ↔ k-orbit correspondence ──")
    # 27 k-points, 2 tet labels per cell → 54 basis labels (cell_idx, tet_parity)
    # Fourier transform: F[k, (cell,tet)] = (1/sqrt(27)) * exp(i k.R_cell)
    # Weight of singular direction u (in 54-dim simplex space) at k-point n_tuple:
    # First work in the simplex cell-tet basis directly.
    # For each of 19 singular directions (left SV in simplex space U_full[:, :19]),
    # compute power per k-point.
    #
    # Decompose simplex index (0..53) into cell_idx = index//2, tet = index%2.
    # F matrix (27 k-points × 27 cells) for the cell degree of freedom:
    kvecs=list(itertools.product(range(3),repeat=3))  # 27 k-points
    F=np.zeros((27,27),dtype=complex)
    for ki,nt in enumerate(kvecs):
        kv=np.array([2*np.pi*n/3. for n in nt])
        for ci,cell in enumerate(body_centers):
            F[ki,ci]=np.exp(1j*np.dot(kv,np.array(cell,dtype=float)))/np.sqrt(27)

    # Expand U_full (54×54) columns to cell-tet basis: shape (27,2) per vector
    # U_full[:, i] = left SV i in simplex-pair space (54 dim, indexed by [cell*2+tet])
    # k-weight of SV i at k-point ki:
    #   w[i,ki] = sum_{tet in {0,1}} |sum_cell F[ki,cell] * U_full[cell*2+tet, i]|^2

    U19=U_full[:,:19]  # (54,19)
    # Reshape to (27,2,19): [cell, tet, sv]
    U_ct=U19.reshape(27,2,19)
    # FT over cell: (27,2,19) -> sum over cell with phase F[ki,cell]
    # Utilde[ki,tet,sv] = sum_cell F[ki,cell]*U_ct[cell,tet,sv]
    Utilde=np.einsum('kc,ctm->ktm',F,U_ct)  # (27,2,19)
    # weight[sv,ki] = sum_tet |Utilde[ki,tet,sv]|^2
    weight=np.einsum('ktm,ktm->mk',np.abs(Utilde)**2,np.ones_like(np.abs(Utilde)**2))
    # weight shape (19, 27) -- each row is a singular direction, each col is a k-point

    # k-orbit classification
    def korbit(nt):
        nz=sum(1 for x in nt if x!=0)
        if nz==0: return 'Gamma'
        if nz==1: return 'axis'
        if nz==2: return 'face-diag'
        return 'body-diag'
    korbit_labels=[korbit(nt) for nt in kvecs]
    orbit_names=['Gamma','axis','face-diag','body-diag']
    orbit_sizes={'Gamma':1,'axis':6,'face-diag':12,'body-diag':8}

    # For each cluster, aggregate weight over orbits
    # cluster indices in L (and U_full): cl1,cl6,cl12 are indices into the 19
    # But cluster indices are indices into s_full[:19] ordering.
    # U_full[:, cl1[0]] is left SV for cluster 1, etc.
    # weight[i,:] uses the ordering of U_full columns (0..18 in descending sv order).
    # cl1,cl6,cl12 are already indices into s19 (which is s_full[:19]).
    log(f"\n  Weight table (cluster × orbit), summed weights (3 decimals):")
    log(f"  {'Cluster':10s}  {'Gamma':8s}  {'axis':8s}  {'face-diag':10s}  {'body-diag':10s}  {'sum':6s}")
    table={}
    for cl_label,cl_idx in [('1-dim',cl1),('6-dim',cl6),('12-dim',cl12)]:
        w_cluster=weight[cl_idx,:]  # (d, 27)
        row={}
        for orb in orbit_names:
            k_mask=[i for i,lb in enumerate(korbit_labels) if lb==orb]
            row[orb]=float(np.sum(w_cluster[:,k_mask]))
        total=sum(row.values())
        log(f"  {cl_label:10s}  {row['Gamma']:8.3f}  {row['axis']:8.3f}  {row['face-diag']:10.3f}  {row['body-diag']:10.3f}  {total:6.3f}")
        table[cl_label]=row

    # Verdicts
    t2_pass=(table['1-dim']['Gamma']>0.9*sum(table['1-dim'].values()) and
             table['6-dim']['axis']>0.9*sum(table['6-dim'].values()) and
             table['12-dim']['face-diag']>0.9*sum(table['12-dim'].values()) and
             table['1-dim']['body-diag']<0.01 and
             table['6-dim']['body-diag']<0.01 and
             table['12-dim']['body-diag']<0.01)

    # Check body-diag absent from all 19 dims
    bd_mask=[i for i,lb in enumerate(korbit_labels) if lb=='body-diag']
    bd_total_weight=float(np.sum(weight[:,bd_mask]))
    log(f"\n  Total weight on body-diagonal orbit across all 19 SVs: {bd_total_weight:.6f}")
    log(f"  TEST 2 VERDICT: {'PASS' if t2_pass else 'FAIL'} — 1→Γ, 6→axis, 12→face-diag, body-diag absent (19=27-8)")

    log("\n"+"="*70)
    log("SUMMARY (≤30 lines)")
    log("="*70)
    log(f"Rank-19 overlap, clusters 1+6+12 verified.")
    log(f"||[T_R, S]||_F: "+", ".join(f"R={R}:{c:.2e}" for R,c in zip(translations,comms)))
    log(f"\nTEST 1 — 3x3 |M_R| matrices:")
    for R,ro in zip(translations,trans_row_of):
        M=np.zeros((3,3))
        for ri,vi in enumerate([v1,v6,v12]):
            for rj,vj in enumerate([v1,v6,v12]):
                M[ri,rj]=abs(np.dot(vi.conj(),vj[ro]))
        log(f"  T_{R}: diag=[{M[0,0]:.3f},{M[1,1]:.3f},{M[2,2]:.3f}]  off=[{M[0,1]:.3f},{M[0,2]:.3f},{M[1,2]:.3f}]")
    log(f"VERDICT T1: A2u vectors {'CONNECTED across clusters' if connected else 'ISOLATED (block-diagonal)'} under Z3^3 translations")
    log(f"\nTEST 2 — cluster×orbit weight table:")
    log(f"  {'Cluster':10s}  {'Gamma':8s}  {'axis':8s}  {'face-diag':10s}  {'body-diag':10s}")
    for cl_label,cl_idx in [('1-dim',cl1),('6-dim',cl6),('12-dim',cl12)]:
        row=table[cl_label]
        log(f"  {cl_label:10s}  {row['Gamma']:8.3f}  {row['axis']:8.3f}  {row['face-diag']:10.3f}  {row['body-diag']:10.3f}")
    log(f"  Body-diagonal total weight: {bd_total_weight:.6f}")
    log(f"VERDICT T2: {'PASS' if t2_pass else 'FAIL'} — 1-cluster@Γ, 6-cluster@axis, 12-cluster@face-diag, body-diag absent")

if __name__=="__main__":
    main()
    with open(OUT,"w",encoding="utf-8") as f:
        f.write("\n".join(_LOG)+"\n")
    print(f"[results written to {OUT}]")
