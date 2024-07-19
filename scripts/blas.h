#pragma region type definition

typedef int blasint;

typedef
struct {
    float real;
    float imag;
} c32;

typedef
struct {
    double real;
    double imag;
} c64;

#define BLAS_INT blasint
#define BLAS_Complex8 c32
#define BLAS_Complex16 c64

#pragma endregion

#pragma region Level3 BLAS

void sgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const float *alpha, const float *a, const BLAS_INT *lda, const float *b, const BLAS_INT *ldb, const float *beta, float *c, const BLAS_INT *ldc);
void dgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const double *alpha, const double *a, const BLAS_INT *lda, const double *b, const BLAS_INT *ldb, const double *beta, double *c, const BLAS_INT *ldc);
void cgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *b, const BLAS_INT *ldb, const BLAS_Complex8 *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void zgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *b, const BLAS_INT *ldb, const BLAS_Complex16 *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);

#pragma endregion
