"""
verify_pi_modes_eu_irrep.py

Question: Is the 2D pi-phase eigenspace of each Wilson loop (along z-axis) an Eu irrep?

Strategy:
1. Reuse build_bloch_and_flatband(L=3) from verify_t9_L4_wilson_pi_mode.py
2. Reuse Oh group machinery from bcc_flatband_oh_irreps.py
3. For transverse k = (0,0) and (1,0) [= (2pi/3, 0)]:
   a. Compute Wilson loop W_z (6x6 in flat-band frame)
   b. Extract 2D pi-eigenspace in 20D face space
   c. Build D4h subgroup (little group of z-axis at k_perp=0: C4v x {i,S4})
   d. Project symmetry operators into 2D space, compute characters
   e. Compare with Eu vs Eg character rows
"""

import sys, io, itertools
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_PATH = "d:/AI thoery/.agent/scripts/verify_pi_modes_eu_irrep_results.txt"
_LOG = []
def log(msg=""):
    print(msg); _LOG.append(str(msg))

# ── reuse from verify_t9_L4_wilson_pi_mode.py ──────────────────────────────
from collections import defaultdict

def build_bcc_lattice_periodic(L):
    return list(itertools.product(range(L), repeat=3))

def enumerate_simplex_faces(bc_ijk, L):
    i,j,k = bc_ijk
    bc_v = ('bc',i,j,k)
    orig_corners = []
    for dx,dy,dz in itertools.product((0,1),repeat=3):
        ox,oy,oz = i+dx,j+dy,k+dz
        wx,wy,wz = ox%L,oy%L,oz%L
        parity = (ox+oy+oz)%2
        orig_corners.append((parity,('c',wx,wy,wz)))
    even_corners = [v for p,v in orig_corners if p==0]
    odd_corners  = [v for p,v in orig_corners if p==1]
    faces=[]
    for tet in (even_corners,odd_corners):
        for combo in itertools.combinations([bc_v]+tet,3):
            f = frozenset(combo)
            if len(f)==3: faces.append(f)
    return faces

def translate_face(face,shift,L):
    si,sj,sk=shift
    def tv(v):
        if v[0]=='bc': _,i,j,k=v; return ('bc',(i+si)%L,(j+sj)%L,(k+sk)%L)
        else: _,x,y,z=v; return ('c',(x+si)%L,(y+sj)%L,(z+sk)%L)
    return frozenset(tv(v) for v in face)

def build_all_faces(bcs,L):
    ref_faces = enumerate_simplex_faces((0,0,0),L)
    f2i,bc_fi={},[]
    gi=0
    for bc in bcs:
        li=[]
        for rf in ref_faces:
            sf=translate_face(rf,bc,L)
            if sf not in f2i: f2i[sf]=gi; gi+=1
            li.append(f2i[sf])
        bc_fi.append(li)
    return f2i,bc_fi,ref_faces

def build_adjacency(f2i):
    N=len(f2i); A=np.zeros((N,N))
    fs=[None]*N
    for f,i in f2i.items(): fs[i]=f
    v2f=defaultdict(set)
    for f,i in f2i.items():
        for v in f: v2f[v].add(i)
    pairs=set()
    for s in v2f.values():
        sl=sorted(s)
        for a in range(len(sl)):
            for b in range(a+1,len(sl)): pairs.add((sl[a],sl[b]))
    for i,j in pairs:
        if len(fs[i]&fs[j])==2: A[i,j]=A[j,i]=1
    return A

def build_bloch(L):
    bcs=build_bcc_lattice_periodic(L)
    f2i,bc_fi,ref_faces=build_all_faces(bcs,L)
    N=len(f2i); n20=len(ref_faces)
    A=build_adjacency(f2i)
    perm=np.array([i for bc_idx in range(len(bcs)) for i in bc_fi[bc_idx]])
    Ar=A[np.ix_(perm,perm)]
    T={}
    for j,bc in enumerate(bcs):
        R=tuple(bc); blk=Ar[:n20,j*n20:(j+1)*n20].copy()
        T[R]=T.get(R,np.zeros((n20,n20)))+blk
    kvecs=list(itertools.product(range(L),repeat=3))
    vecs_all={}
    for nt in kvecs:
        k=np.array([2*np.pi*n/L for n in nt])
        H=np.zeros((n20,n20),dtype=complex)
        for R,m in T.items(): H+=np.exp(1j*np.dot(k,np.array(R,float)))*m
        ev,evec=np.linalg.eigh(H)
        # flat-band: 6 lowest (eigenvalue -3)
        vecs_all[nt]=evec
    return vecs_all, n20, f2i, bc_fi, ref_faces

# ── reuse Oh machinery from bcc_flatband_oh_irreps.py ──────────────────────
def generate_oh_elements():
    C4z=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=int)
    C3=np.array([[0,0,1],[1,0,0],[0,1,0]],dtype=int)
    def k(m): return tuple(m.flatten())
    O={k(np.eye(3,dtype=int)):np.eye(3,dtype=int)}
    q=[np.eye(3,dtype=int)]; gens=[C4z,C3,C4z.T,C3.T]
    while q:
        c=q.pop(0)
        for g in gens:
            nw=g@c; key=k(nw)
            if key not in O: O[key]=nw; q.append(nw)
    inv=-np.eye(3,dtype=int)
    els=[]
    for m in O.values(): els.append(m.copy()); els.append((inv@m).copy())
    return els

def apply_oh_to_vertex(g,v,L=3):
    center = {'bc':1.0,'c':1.5}
    if v[0]=='bc':
        _,i,j,k=v; coord=np.array([i-1.,j-1.,k-1.])
        nc=g@coord
        return ('bc',int(round(nc[0]+1))%L,int(round(nc[1]+1))%L,int(round(nc[2]+1))%L)
    else:
        _,x,y,z=v; coord=np.array([x-1.5,y-1.5,z-1.5])
        nc=g@coord
        return ('c',int(round(nc[0]+1.5))%L,int(round(nc[1]+1.5))%L,int(round(nc[2]+1.5))%L)

def build_perm20(g,ref_faces,rf2li,f2i,bc_fi,bcs,L=3):
    P=np.zeros((20,20))
    for m,face in enumerate(ref_faces):
        gf=frozenset(apply_oh_to_vertex(g,v,L) for v in face)
        # find BC owner
        bc_owner=None
        for v in gf:
            if v[0]=='bc': bc_owner=(v[1],v[2],v[3]); break
        if bc_owner is None:
            if gf in f2i:
                gi=f2i[gf]
                for bi,bc in enumerate(bcs):
                    if gi in bc_fi[bi]: bc_owner=bc; break
        inv=tuple((-s)%L for s in bc_owner)
        tf=frozenset(apply_oh_to_vertex(np.eye(3,dtype=int),v,L) if False else
                     ('bc',(v[1]+inv[0])%L,(v[2]+inv[1])%L,(v[3]+inv[2])%L) if v[0]=='bc'
                     else ('c',(v[1]+inv[0])%L,(v[2]+inv[1])%L,(v[3]+inv[2])%L)
                     for v in gf)
        if tf not in rf2li:
            raise ValueError(f"translated face not in ref: {tf}")
        n=rf2li[tf]
        P[n,m]=1.
    return P

# ── D4h character table (little group of z-axis) ───────────────────────────
# D4h = {E, 2C4(z), C2(z), 2C2'(x,y), 2C2''(diag), i, 2S4, sigma_h, 2sigma_v, 2sigma_d}
# For Eu vs Eg: the key difference is sign under i and S4.
# Eu: χ(E)=2, χ(C4)=0, χ(C2z)=-2... wait, let me use standard D4h table:
# D4h irreps (Mulliken):
#   E  2C4  C2z  2C2'  2C2''  i  2S4  sigma_h  2sigma_v  2sigma_d
# A1g  1   1    1    1     1   1   1     1       1         1
# A2g  1   1    1   -1    -1   1   1     1      -1        -1
# B1g  1  -1    1    1    -1   1  -1     1       1        -1
# B2g  1  -1    1   -1     1   1  -1     1      -1         1
# Eg   2   0   -2    0     0   2   0    -2       0         0
# A1u  1   1    1    1     1  -1  -1    -1      -1        -1
# A2u  1   1    1   -1    -1  -1  -1    -1       1         1
# B1u  1  -1    1    1    -1  -1   1    -1      -1         1
# B2u  1  -1    1   -1     1  -1   1    -1       1        -1
# Eu   2   0   -2    0     0  -2   0     2       0         0
D4H_CLASSES = ['E','2C4','C2z','2C2p','2C2pp','i','2S4','sh','2sv','2sd']
D4H_CHARS = {
    'A1g': [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'A2g': [ 1, 1, 1,-1,-1, 1, 1, 1,-1,-1],
    'B1g': [ 1,-1, 1, 1,-1, 1,-1, 1, 1,-1],
    'B2g': [ 1,-1, 1,-1, 1, 1,-1, 1,-1, 1],
    'Eg' : [ 2, 0,-2, 0, 0, 2, 0,-2, 0, 0],
    'A1u': [ 1, 1, 1, 1, 1,-1,-1,-1,-1,-1],
    'A2u': [ 1, 1, 1,-1,-1,-1,-1,-1, 1, 1],
    'B1u': [ 1,-1, 1, 1,-1,-1, 1,-1,-1, 1],
    'B2u': [ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1],
    'Eu' : [ 2, 0,-2, 0, 0,-2, 0, 2, 0, 0],
}
D4H_SIZE = [1,2,1,2,2,1,2,1,2,2]  # conjugacy class sizes, total=16

def classify_d4h(g):
    """Classify a 3x3 int matrix as D4h conjugacy class (z-axis group)."""
    det=int(round(np.linalg.det(g))); tr=int(round(np.trace(g)))
    # Check if g fixes z: g@[0,0,1] = ±[0,0,1]
    z=np.array([0,0,1]); gz=g@z
    if det==1:  # proper
        if tr==3: return 'E'
        if tr==1: return '2C4'   # C4 or C4^3, fixes z-axis
        if tr==-1:
            # C2: check if it fixes z (C2z) or is a horizontal C2
            if np.allclose(gz,z): return 'C2z'
            # C2' vs C2'': axis in xy plane
            # C2' along x or y axis: g has -1 on two diag entries
            # C2'' along [1,1,0] type: off-diagonal
            od=np.abs(g-np.diag(np.diag(g))).sum()
            if od<0.5: return '2C2p'   # diagonal matrix -> axis along x or y
            return '2C2pp'             # off-diagonal -> [1,1,0] axis
    else:  # improper
        if tr==-3: return 'i'
        if tr==-1: return '2S4'
        if tr==1:
            # sigma_h (z-plane) vs sigma_v vs sigma_d
            if np.allclose(gz,-z): return 'sh'  # reflects z -> -z: sigma_h
            od=np.abs(g-np.diag(np.diag(g))).sum()
            if od<0.5: return '2sv'    # diagonal, fixes x or y axis
            return '2sd'               # off-diagonal
    return 'unknown'

def get_d4h_elements(oh_els):
    """Extract the 16 D4h elements (little group of z) from Oh."""
    z=np.array([0,0,1])
    d4h=[]
    for g in oh_els:
        gz=g@z
        if np.allclose(np.abs(gz),z): d4h.append(g)
    return d4h

# ── Wilson loop and pi-eigenspace extraction ────────────────────────────────
def wilson_z_loop(kx,ky,L,vecs_all):
    """Wilson loop along z at fixed (kx,ky). Returns W (6x6) and the 6 flat-band vecs at each k."""
    loop=[]; fb_vecs=[]
    for kz in range(L):
        kt=(kx,ky,kz)
        ev_k=vecs_all[kt]
        # flat band: first 6 eigenvectors (eigenvalue -3)
        fb=ev_k[:,:6]
        loop.append(fb); fb_vecs.append(fb)
    W=np.eye(6,dtype=complex)
    for j in range(L):
        fa=loop[j]; fb=loop[(j+1)%L]
        S=fa.conj().T@fb  # 6x6 overlap
        W=W@S
    return W, fb_vecs

def pi_eigenspace_in_20d(W, fb_kstart):
    """
    Find pi-phase eigenvectors of W (6x6) and lift them to 20D face space
    via fb_kstart (20x6 flat-band frame at the start k-point).
    Returns (n_pi, vecs_20d) where vecs_20d has shape (20, n_pi).
    """
    evals,evecs=np.linalg.eig(W)
    phases=np.angle(evals)
    pi_mask=np.abs(np.abs(phases)-np.pi)<0.05
    n_pi=pi_mask.sum()
    if n_pi==0: return 0,None
    v6=evecs[:,pi_mask]  # 6 x n_pi (in flat-band frame)
    v20=fb_kstart@v6     # 20 x n_pi (lift to face space)
    # orthonormalize
    q,_=np.linalg.qr(v20)
    return n_pi, q[:,:n_pi]

# ── character computation in the 2D pi-space ────────────────────────────────
def compute_2d_chars(v2d, d4h_els, ref_faces, rf2li, f2i, bc_fi, bcs):
    """
    For each D4h element g, build 20x20 perm P(g), project into 2D pi-space,
    compute trace. Returns dict {class: chi}.
    """
    chars={}
    cls_list=[]
    traces=[]
    for g in d4h_els:
        P=build_perm20(g,ref_faces,rf2li,f2i,bc_fi,bcs,L=3)
        # 2x2 rep matrix in pi-space: R = v2d^dag P v2d
        R=v2d.conj().T@P@v2d
        chi=np.real(np.trace(R))
        cls=classify_d4h(g)
        cls_list.append(cls); traces.append(chi)
    # average per class
    from collections import defaultdict
    cc=defaultdict(list)
    for c,t in zip(cls_list,traces): cc[c].append(t)
    return {c:float(np.mean(v)) for c,v in cc.items()}

def match_irrep(chi_dict):
    """Compare character dict with D4h irreps, return best match and decomposition."""
    # build character vector in D4H_CLASSES order
    chi_vec=np.array([chi_dict.get(c,0.0) for c in D4H_CLASSES])
    sizes=np.array(D4H_SIZE,dtype=float)
    order=16.0
    results={}
    for irrep,row in D4H_CHARS.items():
        row_v=np.array(row,dtype=float)
        n=np.dot(sizes*row_v,chi_vec)/order
        results[irrep]=n
    return results

# ── main ────────────────────────────────────────────────────────────────────
def analyze_kperp(kx,ky,vecs_all,d4h_els,ref_faces,rf2li,f2i,bc_fi,bcs,L=3):
    log(f"\n── k_perp = ({kx},{ky}) (= ({kx*2}π/{L}, {ky*2}π/{L})) ──")
    W,fb_vecs=wilson_z_loop(kx,ky,L,vecs_all)
    evals=np.linalg.eigvals(W); phases=np.angle(evals)
    log(f"  Wilson phases/π: {np.sort(np.real(phases/np.pi))}")
    n_pi,v2d=pi_eigenspace_in_20d(W,fb_vecs[0])
    log(f"  π-eigenspace dimension: {n_pi}")
    if n_pi!=2:
        log(f"  WARNING: expected 2 pi-modes, got {n_pi}. Skipping irrep check.")
        return None
    chi_dict=compute_2d_chars(v2d,d4h_els,ref_faces,rf2li,f2i,bc_fi,bcs)
    log(f"  Character row (D4h classes):")
    for c in D4H_CLASSES:
        log(f"    {c:8s}: {chi_dict.get(c,0.0):+.4f}")
    mults=match_irrep(chi_dict)
    log(f"  Irrep decomposition:")
    nonzero=[]
    for irr in ['A1g','A2g','B1g','B2g','Eg','A1u','A2u','B1u','B2u','Eu']:
        n=mults[irr]; nr=int(round(n))
        if abs(n)>0.1: log(f"    {irr}: {n:.4f} ~ {nr}"); nonzero.append((irr,nr))
    # verdict
    eu_n=round(mults.get('Eu',0)); eg_n=round(mults.get('Eg',0))
    if eu_n==1 and eg_n==0 and sum(nr for _,nr in nonzero)==1:
        verdict="Eu (confirmed)"
    elif eg_n==1 and eu_n==0 and sum(nr for _,nr in nonzero)==1:
        verdict="Eg (not Eu)"
    elif len(nonzero)==2 and all(d==1 for _,d in nonzero):
        verdict=f"reducible: {nonzero[0][0]}+{nonzero[1][0]}"
    else:
        verdict=f"ambiguous: {nonzero}"
    log(f"  VERDICT: {verdict}")
    return verdict, chi_dict, mults

def main():
    log("="*72)
    log("VERIFY: π-eigenspace of Wilson loop (z-axis) = Eu irrep?")
    log("="*72)
    L=3
    log(f"\nBuilding L={L} BCC Bloch Hamiltonian...")
    vecs_all,n20,f2i,bc_fi,ref_faces=build_bloch(L)
    bcs=build_bcc_lattice_periodic(L)
    rf2li={f:m for m,f in enumerate(ref_faces)}
    log(f"  n_faces_per_BC={n20}, total faces={len(f2i)}")

    log("\nBuilding Oh and extracting D4h (little group of z-axis)...")
    oh_els=generate_oh_elements()
    d4h_els=get_d4h_elements(oh_els)
    log(f"  Oh: {len(oh_els)} elements, D4h: {len(d4h_els)} elements (expect 16)")

    # verify D4h class counts
    from collections import Counter
    cls_counts=Counter(classify_d4h(g) for g in d4h_els)
    log(f"  D4h class counts: {dict(cls_counts)}")

    # analyze two k_perp values
    r1=analyze_kperp(0,0,vecs_all,d4h_els,ref_faces,rf2li,f2i,bc_fi,bcs)
    r2=analyze_kperp(1,0,vecs_all,d4h_els,ref_faces,rf2li,f2i,bc_fi,bcs)

    log("\n"+"="*72)
    log("SUMMARY")
    log("="*72)
    log(f"  k_perp=(0,0):  {r1[0] if r1 else 'N/A'}")
    log(f"  k_perp=(1,0):  {r2[0] if r2 else 'N/A'}")
    log("\nD4h character table reference (Eu vs Eg):")
    log(f"  {'':8s}  E  2C4  C2z  2C2'  2C2''   i  2S4  sh  2sv  2sd")
    log(f"  Eg    :  2    0   -2    0     0     2    0   -2    0    0")
    log(f"  Eu    :  2    0   -2    0     0    -2    0    2    0    0")
    log("  Key: Eu has χ(i)=-2, χ(sh)=+2;  Eg has χ(i)=+2, χ(sh)=-2")
    if r1:
        chi=r1[1]; inv_chi=chi.get('i',None); sh_chi=chi.get('sh',None)
        log(f"\n  k_perp=(0,0): χ(i)={inv_chi:.3f}, χ(σh)={sh_chi:.3f}")
    if r2:
        chi=r2[1]; inv_chi=chi.get('i',None); sh_chi=chi.get('sh',None)
        log(f"  k_perp=(1,0): χ(i)={inv_chi:.3f}, χ(σh)={sh_chi:.3f}")

if __name__=="__main__":
    try:
        main()
    finally:
        with open(OUT_PATH,"w",encoding="utf-8") as f:
            f.write("\n".join(_LOG)+"\n")
        print(f"\n[results → {OUT_PATH}]")
