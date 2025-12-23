try:
    from jxglm_rs import glmi8,mlmi8,mlmpf32
except Exception as e:
    print(f"{e}\nPlease build jxglm_rs for glmrc. Source code is in ext/glm_rs")
import numpy as np   


def FEM(y:np.ndarray,X:np.ndarray,M:np.ndarray,chunksize:int=50_000,threads:int=1,):
    '''
    # fastGLM for dtype int8
    
    :param y: trait vector (n,1)
    :type y: np.ndarray
    :param X: indice matrix of fixed effects (n,p)
    :type X: np.ndarray
    :param M: SNP matrix (m,n)
    :type M: np.ndarray
    :param chunksize: chunksize per step
    :type chunksize: int
    :param threads: number of threads
    :type threads: int
    '''
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    ixx = np.ascontiguousarray(np.linalg.pinv(X.T @ X), dtype=np.float64)
    M = np.ascontiguousarray(M, dtype=np.int8)
    if M.ndim != 2:
        raise ValueError("M must be 2D array with shape (m, n)")
    if M.shape[1] != y.shape[0]:
        raise ValueError(f"M must be shape (m, n). Got M.shape={M.shape}, but n=len(y)={y.shape[0]}")
    result:np.ndarray = glmi8(y,X,ixx,M,chunksize,threads)
    return result

def fastMLM(y, X, UT, iUXUX=None, vgs=1.0, G=None, step=10000, threads=0):
    """
    # Per-marker MLM test (G is int8, marker rows).
    
    :param y: (n,) float64
    :param X: (n,q0) float64
    :param UT: (n,n) float64  = U.T
    :param iUXUX: (q0,q0) float64  = pinv((UT@X).T @ (UT@X))
    :param vgs: scalar
    :param G: (m,n) int8
    
    :return: (m,3) float64 -> beta, se, p
    """
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    UT = np.ascontiguousarray(UT, dtype=np.float64)
    G = np.ascontiguousarray(G, dtype=np.int8)
    Uy = UT @ y
    UX = UT @ X
    if iUXUX is None:
        iUXUX = np.linalg.pinv(UX.T @ UX)
    iUXUX = np.ascontiguousarray(iUXUX, dtype=np.float64)
    UXUy = UX.T @ Uy
    Uy = np.ascontiguousarray(Uy, dtype=np.float64)
    UX = np.ascontiguousarray(UX, dtype=np.float64)
    UXUy = np.ascontiguousarray(UXUy, dtype=np.float64).ravel()
    return mlmi8(Uy, UX, iUXUX, UXUy, UT, G, float(vgs), step=int(step), threads=int(threads))

def poolMLM(y, X, UT, G_pool, vgs=1.0, ridge=1e-10):
    """
    # Multi-locus test: put all loci in G_pool into fixed effects simultaneously,
    
    :param y: (n,)
    :param X: (n,q0)
    :param UT: (n,n)
    :param G_pool: (k,n) int8 (rows are loci)
    :return: beta/se/p (k,3), for each locus in the pool.
    """
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    UT = np.ascontiguousarray(UT, dtype=np.float64)
    G_pool = np.ascontiguousarray(G_pool, dtype=np.int8)
    return mlmpf32(y, X, UT, G_pool, float(vgs), float(ridge))