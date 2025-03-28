#include <stdlib.h>
#include <string.h>

void coo_spmv(int *rowIdx, int *colIdx, float *values, float *v, float *out, int nnz, int n) {
    // Initialize output vector to zero
    memset(out, 0, n * sizeof(double));

    // Perform SpMV
    for (int i = 0; i < nnz; i++) {
        out[rowIdx[i]] += values[i] * v[colIdx[i]];
    }
}
