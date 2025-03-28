import ctypes
import numpy as np 

# Load shared library
coo_lib = ctypes.CDLL("./util.dll")  # Use "coo_spmv.dll" on Windows

# Define the function signature
coo_lib.coo_spmv.argtypes = [
    ctypes.POINTER(ctypes.c_int),   # rowIdx
    ctypes.POINTER(ctypes.c_int),   # colIdx
    ctypes.POINTER(ctypes.c_float), # values
    ctypes.POINTER(ctypes.c_float), # v
    ctypes.POINTER(ctypes.c_float), # out
    ctypes.c_int,  # nnz (number of non-zero elements)
    ctypes.c_int   # n (size of output vector)
]

def coo_spmv_c(rowIdx: np.ndarray, colIdx: np.ndarray, values: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert rowIdx.dtype == np.int32, "rowIdx must be of dtype np.int32"
    assert colIdx.dtype == np.int32, "colIdx must be of dtype np.int32"
    assert values.dtype == np.float32, "values must be of dtype np.float64"
    assert v.dtype == np.float32, "v must be of dtype np.float64"

    nnz = len(values)  # Number of non-zero elements
    n = len(v)         # Size of output vector

    # Allocate output array (zero-initialized)
    out = np.zeros(n, dtype=np.float64)

    # Call the C function
    coo_lib.coo_spmv(
        rowIdx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        colIdx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nnz,
        n
    )

    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    out_array = np.ctypeslib.as_array(out_ptr, shape=(n,1))
    print(out_array)

    return out_array