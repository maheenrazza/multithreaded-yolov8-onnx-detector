import numpy as np

def normalize_points_2d(P):  # P: Nx2
    mu = P.mean(axis=0)
    Q = P - mu
    mean_dist = np.mean(np.sqrt(np.sum(Q**2, axis=1)))
    s = np.sqrt(2) / mean_dist
    T = np.array([[s,0,-s*mu[0]],
                  [0,s,-s*mu[1]],
                  [0,0,1]])
    Pn = (T @ np.c_[P, np.ones(len(P))].T).T
    return Pn[:, :2], T

def dlt_homography(src, dst):        # src,dst: Nx2 (N>=4, non-collinear)
    src_n, Ts = normalize_points_2d(src)
    dst_n, Td = normalize_points_2d(dst)

    N = src_n.shape[0]
    A = []
    for i in range(N):
        x,y = src_n[i]
        u,v = dst_n[i]
        A.append([x,y,1, 0,0,0, -u*x, -u*y, -u])
        A.append([0,0,0, x,y,1, -v*x, -v*y, -v])
    A = np.asarray(A)

    # Solve Ah=0 by SVD (right singular vector with smallest singular value)
    _,_,Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3,3)

    # Denormalize: H = Td^{-1} * Hn * Ts
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2,2]

def apply_H(H, pts):  # pts: Nx2 → Nx2
    P = np.c_[pts, np.ones(len(pts))].T
    Q = H @ P
    Q = (Q[:2]/Q[2]).T
    return Q

def reproj_errors(H, src, dst):
    pred = apply_H(H, src)
    e = np.linalg.norm(pred - dst, axis=1)
    return e, pred

# Replace these with your measured coordinates:
src = np.array([[-2,3],[4,-5],[8,7],[-6,6],[10,-8],[-4,2]], float)
dst = np.array([[-12,6],[16,-10],[30,20],[-24,12],[40,-24],[-20,8]], float)

# Use first 4 to estimate, last 2 to validate:
H = dlt_homography(src[:4], dst[:4])

# Training error (optional)
e_train, pred_train = reproj_errors(H, src[:4], dst[:4])
# Validation on remaining 2
e_val,  pred_val  = reproj_errors(H, src[4:],  dst[4:])

print("H =\n", H)
print("Train RMS err:", np.sqrt((e_train**2).mean()))
print("Val   RMS err:", np.sqrt((e_val**2).mean()))
