import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from itertools import permutations, product
from collections import Counter
from math import lcm

INF = 'inf'

def mul3(v, M):
    return tuple((sum(M[i][j]*v[j] for j in range(3))) % 3 for i in range(3))

def signed_perm_matrices():
    mats = []
    for perm in permutations(range(3)):
        for signs in product([1,-1], repeat=3):
            M = [[0]*3 for _ in range(3)]
            for i, pp in enumerate(perm):
                M[i][pp] = signs[i]
            mats.append(tuple(tuple(r) for r in M))
    return mats

def mat_det(M):
    a,b,c=M[0];d,e,f=M[1];g,h,k=M[2]
    return a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g)

all48 = signed_perm_matrices()
rot24 = [M for M in all48 if mat_det(M)==1]

def group_action(pts, group):
    idx = {p:i for i,p in enumerate(pts)}
    return [tuple(idx[mul3(p,M)] for p in pts) for M in group]

def close_group(perms, n):
    s = set(perms)
    changed = True
    while changed:
        changed = False
        new = set()
        for a in list(s):
            for b in list(s):
                c = tuple(a[b[i]] for i in range(n))
                if c not in s: new.add(c); changed=True
        s |= new
    return s

def perm_compose(a, b): return tuple(a[b[i]] for i in range(len(a)))

p = 11
p1 = list(range(p))+[INF]

def mobius(a,b,c,d,x,p):
    if x==INF:
        if c%p==0: return INF
        return a*pow(c,-1,p)%p
    num=(a*x+b)%p; den=(c*x+d)%p
    if den==0: return INF
    return num*pow(den,-1,p)%p

def apply_mob_perm(a,b,c,d,p):
    img = tuple(mobius(a,b,c,d,x,p) for x in p1)
    return tuple(p1.index(v) for v in img)

def perm_order_list(perm):
    n=len(perm); visited=[False]*n; o=1
    for i in range(n):
        if not visited[i]:
            c=0; j=i
            while not visited[j]: visited[j]=True; j=perm[j]; c+=1
            o=lcm(o,c)
    return o

def to_std(t0,t1,t2,p):
    if t2==INF:
        if t0==INF or t1==INF: return None
        return ((t1-t0)%p,t0%p,0,1)
    if t0==INF:
        return (t2%p,(t1-t2)%p,1,0)
    if t1==INF:
        return (t2%p,(-t0)%p,1,(-1)%p)
    t0,t1,t2=t0%p,t1%p,t2%p
    den=(t2-t1)%p
    if den==0: return None
    c=(t1-t0)*pow(den,-1,p)%p
    return (t2*c%p,t0,c,1)

def mm(A,B,p):
    a1,b1,c1,d1=A;a2,b2,c2,d2=B
    return ((a1*a2+b1*c2)%p,(a1*b2+b1*d2)%p,(c1*a2+d1*c2)%p,(c1*b2+d1*d2)%p)

def minv(A,p):
    a,b,c,d=A; det=(a*d-b*c)%p
    if det==0: return None
    inv=pow(det,-1,p)
    return (d*inv%p,(-b*inv)%p,(-c*inv)%p,(a*inv)%p)

def mobius_from_3pts(s0,s1,s2,d0,d1,d2,p):
    T1=to_std(s0,s1,s2,p); T2=to_std(d0,d1,d2,p)
    if T1 is None or T2 is None: return None
    T1i=minv(T1,p)
    if T1i is None: return None
    return mm(T2,T1i,p)

def cycle_type_list(perm):
    n=len(perm); visited=[False]*n; cycles=[]
    for i in range(n):
        if not visited[i]:
            c=0; j=i
            while not visited[j]: visited[j]=True; j=perm[j]; c+=1
            cycles.append(c)
    return tuple(sorted(cycles,reverse=True))

face_pts = sorted(set(mul3((1,1,0), M) for M in all48))
n=12
perms24_face = group_action(face_pts, rot24)
img24_face = list(close_group(perms24_face, n))

# Find S4 in PGL2(F_11) with matching cycle type distribution
ord4_set = set(); ord2_set = set()
for a in range(p):
    for b in range(p):
        for c in range(p):
            for d in range(p):
                if (a*d-b*c)%p == 0: continue
                perm = apply_mob_perm(a,b,c,d,p)
                o = perm_order_list(perm)
                if o==4: ord4_set.add(perm)
                elif o==2: ord2_set.add(perm)

S4_pgl2 = None
for g4 in list(ord4_set)[:30]:
    if S4_pgl2: break
    for g2 in list(ord2_set)[:200]:
        grp = close_group({g4,g2}, 12)
        if len(grp)==24:
            ct = Counter(perm_order_list(g) for g in grp)
            if dict(ct)=={1:1,2:9,3:8,4:6}:
                ct2 = Counter(cycle_type_list(g) for g in grp if perm_order_list(g)==2)
                if ct2.get((2,2,2,2,2,2),0)==3 and ct2.get((2,2,2,2,2,1,1),0)==6:
                    S4_pgl2 = set(grp); break

print(f"Found S4 in PGL2: {S4_pgl2 is not None}, size={len(S4_pgl2) if S4_pgl2 else 0}")

if not S4_pgl2:
    print("ERROR: could not find S4 in PGL2(F_11)")
    sys.exit(1)

# Find group isomorphism face_img24 -> S4_pgl2
# Generators of face img24
g4_face = next(g for g in img24_face if perm_order_list(g)==4)
g2_face = next(g for g in img24_face if perm_order_list(g)==2
               and len(close_group({g4_face, g}, 12))==24)
print(f"Generators of face img24: g4={g4_face}, g2={g2_face}")

S4_pgl2_list = list(S4_pgl2)

found_phi = None
for g4_mob in [g for g in S4_pgl2_list if perm_order_list(g)==4]:
    if found_phi: break
    for g2_mob in [g for g in S4_pgl2_list if perm_order_list(g)==2]:
        # Build iso by closure: g4_face->g4_mob, g2_face->g2_mob
        iso = {}
        id12 = tuple(range(12))
        iso[id12] = id12
        iso[g4_face] = g4_mob
        iso[g2_face] = g2_mob

        valid = True
        changed = True
        while changed:
            changed = False
            for a, fa in list(iso.items()):
                for b, fb in list(iso.items()):
                    ab = perm_compose(a,b)
                    fab = perm_compose(fa,fb)
                    if ab not in iso:
                        iso[ab] = fab; changed=True
                    elif iso[ab] != fab:
                        valid = False; break
                if not valid: break
            if not valid: break

        if not valid or len(iso)!=24: continue

        # Derive sigma for each choice of sigma(0)
        for sigma0_idx in range(12):
            sigma = [None]*12
            sigma[0] = p1[sigma0_idx]

            consistent = True
            for g_face, g_mob in iso.items():
                i = g_face[0]
                val = p1[g_mob[sigma0_idx]]
                if sigma[i] is not None and sigma[i] != val:
                    consistent = False; break
                sigma[i] = val

            if not consistent or None in sigma or len(set(sigma))!=12:
                continue

            # Verify Mobius property
            all_ok = True
            for g in img24_face:
                mob = mobius_from_3pts(sigma[0],sigma[1],sigma[2],
                                       sigma[g[0]],sigma[g[1]],sigma[g[2]],p)
                if mob is None: all_ok=False; break
                am,bm,cm,dm = mob
                for i in range(12):
                    if mobius(am,bm,cm,dm,sigma[i],p)!=sigma[g[i]]:
                        all_ok=False; break
                if not all_ok: break

            if all_ok:
                found_phi = sigma[:]
                print("FOUND phi for face/O:")
                for i, pt in enumerate(face_pts):
                    print(f"  {pt} -> {sigma[i]}")
                break
        if found_phi: break

if not found_phi:
    print("Could not find phi for face/O via isomorphism method.")
    # Debug: how many iso maps found?
    print("Checking a few iso maps...")
