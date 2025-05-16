import numpy as np
from numba import njit
import scipy.sparse as sparse

try:
    import cupy
    import cupyx.scipy.sparse as cupysparse
except:
    pass


@njit
def coo_spmv_column(rowIdx, colIdx, values, v):
    """
    Performs sparse matrix-vector (column based) multiplication using COO format.
    
    Parameters:
    - rowIdx (list[int]): Row indices of nonzero elements.
    - colIdx (list[int]): Column indices of nonzero elements.
    - values (list[float]): Nonzero values of the matrix.
    - v (numpy array): Dense vector for multiplication.
    
    Returns:
    - numpy array: Result vector y = A * v
    """
    out = np.zeros((len(v), 1), dtype=values.dtype)  # Initialize output vector
    nnz = len(values)  # Number of nonzero elements

    for i in range(nnz):  # Iterate over nonzero elements
        out[rowIdx[i], 0] += values[i] * v[colIdx[i], 0]

    return out

@njit
def coo_spmv_row(rowIdx, colIdx, values, v):
    """
    Performs sparse matrix-vector (row based) multiplication using COO format.
    
    Parameters:
    - rowIdx (list[int]): Row indices of nonzero elements.
    - colIdx (list[int]): Column indices of nonzero elements.
    - values (list[float]): Nonzero values of the matrix.
    - v (numpy array): Dense vector for multiplication.
    
    Returns:
    - numpy array: Result vector y = A * v
    """
    out = np.zeros(len(v), dtype=values.dtype)  # Initialize output vector
    nnz = len(values)  # Number of nonzero elements

    for i in range(nnz):  # Iterate over nonzero elements
        out[rowIdx[i]] += values[i] * v[colIdx[i]]

    return out


# Based on the scipy implementation
# Source: https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/_construct.py#L458
# Docs: https://docs.scipy.org/doc/scipy-1.15.1/reference/generated/scipy.sparse.kron.html
def coo_kron(A:sparse.coo_matrix, B:sparse.coo_matrix, format='coo'):
    output_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    if A.nnz == 0 or B.nnz == 0:
        # kronecker product is the zero matrix
        return sparse.coo_matrix(output_shape).asformat(format)

    # Expand entries of a into blocks
    # When using more then 32 qubits, increase to int64
    row = np.asarray(A.row, dtype=np.int32).repeat(B.nnz)
    col = np.asarray(A.col, dtype=np.int32).repeat(B.nnz)
    data = A.data.repeat(B.nnz)
    
    row *= B.shape[0]
    col *= B.shape[1]

    # increment block indices
    row = row.reshape(-1, B.nnz)
    row += B.row
    row = row.reshape(-1)

    col = col.reshape(-1, B.nnz)
    col += B.col
    col = col.reshape(-1)

    # compute block entries
    data = data.reshape(-1, B.nnz) * B.data
    data = data.reshape(-1)

    return sparse.coo_matrix((data, (row, col)), shape=output_shape).asformat(format)

try:
    # Based on the Cupy implementation
    # Source: https://github.com/cupy/cupy/blob/v13.4.1/cupyx/scipy/sparse/_construct.py#L496
    # Docs: https://docs.cupy.dev/en/v13.4.1/reference/generated/cupyx.scipy.sparse.kron.html
    def coo_kron_gpu(A:cupysparse.coo_matrix, B:cupysparse.coo_matrix, format='coo'):
        out_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

        if A.nnz == 0 or B.nnz == 0:
            # kronecker product is the zero matrix
            return cupysparse.coo_matrix(out_shape).asformat(format)

        # expand entries of A into blocks
        row = A.row.astype(cupy.int32, copy=True) * B.shape[0]
        row = row.repeat(B.nnz)
        col = A.col.astype(cupy.int32, copy=True) * B.shape[1]
        col = col.repeat(B.nnz)
        data = A.data.repeat(B.nnz) 

        # increment block indices
        row = row.reshape(-1, B.nnz)
        row += B.row
        row = row.ravel()

        col = col.reshape(-1, B.nnz)
        col += B.col
        col = col.ravel()

        # compute block entries
        data = data.reshape(-1, B.nnz) * B.data
        data = data.ravel()

        return cupysparse.coo_matrix(
            (data, (row, col)), shape=out_shape).asformat(format)
except NameError:
    pass
except Exception as e:
    # print(e)
    exit(1)