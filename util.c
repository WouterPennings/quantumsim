// #include <stdlib.h>
// #include <string.h>

// void coo_spmv_row(int *rowIdx, int *colIdx, double *values, double *v, double *out, int nnz, int n) {
//     // Initialize output vector to zero
//     memset(out, 0, n * sizeof(double));

//     // Perform SpMV
//     for (int i = 0; i < nnz; i++) {
//         out[rowIdx[i]] += values[i] * v[colIdx[i]];
//     }
// }

#include <stdlib.h>
#include <string.h>
#include <complex.h>

void coo_spmv(
    int *rowIdx, int *colIdx, double _Complex *values, 
    double _Complex *v, double _Complex *out, int nnz, int n) {

    // Initialize output vector to zero
    memset(out, 0, n * sizeof(double _Complex));

    // Perform sparse matrix-vector multiplication
    for (int i = 0; i < nnz; i++) {
        out[rowIdx[i]] += values[i] * v[colIdx[i]];
    }
}
