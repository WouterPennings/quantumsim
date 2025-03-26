import scipy.sparse as sc
import numpy as np

def kron(A, B, format=None):
    coo_sparse = sc.coo_matrix

    A = coo_sparse(A)
    B = coo_sparse(B)

    if A.ndim != 2:
        raise ValueError(f"kron requires 2D input arrays. `A` is {A.ndim}D.")
    output_shape = (A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        # kronecker product is the zero matrix
        return coo_sparse(output_shape).asformat(format)

    # expand entries of a into blocks    
    row = A.row.repeat(B.nnz)
    col = A.col.repeat(B.nnz)
    data = A.data.repeat(B.nnz)

    if max(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1]) > np.iinfo('int32').max:
        row = row.astype(np.int64)
        col = col.astype(np.int64)

    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row = row.reshape(-1,B.nnz) + B.row
    col = col.reshape(-1,B.nnz) + B.col
    row = row.reshape(-1)
    col = col.reshape(-1)

    # compute block entries
    data = data.reshape(-1,B.nnz) * B.data
    data = data.reshape(-1)

    return coo_sparse((data,(row,col)), shape=output_shape).asformat(format)
    
a = np.eye(5)
b = np.eye(5)
b[3, 2] = 4
b[1, 2] = 2
b[3, 2] = 1

a = sc.coo_matrix(a)
b = sc.coo_matrix(b)

c = kron(a, b)
print(c)