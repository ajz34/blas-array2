#pragma region type definition

typedef int blas_int;

typedef struct { float real; float imag; } c32;

typedef struct { double real; double imag; } c64;

#define BLAS_INT blas_int
#define BLAS_Complex8 c32
#define BLAS_Complex16 c64

#pragma endregion

#pragma region Scalar BLAS

double dcabs1_(const BLAS_Complex16 *z);
float scabs1_(const BLAS_Complex8 *c);

#pragma endregion

#pragma region Level1 BLAS

// asum: sum | real( x_i ) | + | imag( x_i ) |

double dasum_(const BLAS_INT *n, const double *x, const BLAS_INT *incx);
double dzasum_(const BLAS_INT *n, const BLAS_Complex16 *x, const BLAS_INT *incx);
float sasum_(const BLAS_INT *n, const float *x, const BLAS_INT *incx);
float scasum_(const BLAS_INT *n, const BLAS_Complex8 *x, const BLAS_INT *incx);

// axpy: y = ax + y

void caxpy_(const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, BLAS_Complex8 *y, const BLAS_INT *incy);
void daxpy_(const BLAS_INT *n, const double *alpha, const double *x, const BLAS_INT *incx, double *y, const BLAS_INT *incy);
void saxpy_(const BLAS_INT *n, const float *alpha, const float *x, const BLAS_INT *incx, float *y, const BLAS_INT *incy);
void zaxpy_(const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, BLAS_Complex16 *y, const BLAS_INT *incy);

// copy: y = x

void ccopy_(const BLAS_INT *n, const BLAS_Complex8 *x, const BLAS_INT *incx, BLAS_Complex8 *y, const BLAS_INT *incy);
void dcopy_(const BLAS_INT *n, const double *x, const BLAS_INT *incx, double *y, const BLAS_INT *incy);
void scopy_(const BLAS_INT *n, const float *x, const BLAS_INT *incx, float *y, const BLAS_INT *incy);
void zcopy_(const BLAS_INT *n, const BLAS_Complex16 *x, const BLAS_INT *incx, BLAS_Complex16 *y, const BLAS_INT *incy);

// dot: x^H x and x^T x

void cdotc_(BLAS_Complex8 *pres, const BLAS_INT *n, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *y, const BLAS_INT *incy);
void cdotu_(BLAS_Complex8 *pres, const BLAS_INT *n, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *y, const BLAS_INT *incy);
double ddot_(const BLAS_INT *n, const double *x, const BLAS_INT *incx, const double *y, const BLAS_INT *incy);
double dsdot_(const BLAS_INT *n, const float *x, const BLAS_INT *incx, const float *y, const BLAS_INT *incy);
float sdot_(const BLAS_INT *n, const float *x, const BLAS_INT *incx, const float *y, const BLAS_INT *incy);
float sdsdot_(const BLAS_INT *n, const float *sb, const float *x, const BLAS_INT *incx, const float *y, const BLAS_INT *incy);
void zdotc_(BLAS_Complex16 *pres, const BLAS_INT *n, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *y, const BLAS_INT *incy);
void zdotu_(BLAS_Complex16 *pres, const BLAS_INT *n, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *y, const BLAS_INT *incy);

// iamax: argmax_i | real( x_i ) | + | imag( x_i ) |

BLAS_INT icamax_(const BLAS_INT *n, const BLAS_Complex8 *x, const BLAS_INT *incx);
BLAS_INT idamax_(const BLAS_INT *n, const double *x, const BLAS_INT *incx);
BLAS_INT isamax_(const BLAS_INT *n, const float *x, const BLAS_INT *incx);
BLAS_INT izamax_(const BLAS_INT *n, const BLAS_Complex16 *x, const BLAS_INT *incx);

// nrm2: || x ||_2

double dnrm2_(const BLAS_INT *n, const double *x, const BLAS_INT *incx);
double dznrm2_(const BLAS_INT *n, const BLAS_Complex16 *x, const BLAS_INT *incx);
float snrm2_(const BLAS_INT *n, const float *x, const BLAS_INT *incx);
float scnrm2_(const BLAS_INT *n, const BLAS_Complex8 *x, const BLAS_INT *incx);

// scal: x = alpha x

void cscal_(const BLAS_INT *n, const BLAS_Complex8 *a, BLAS_Complex8 *x, const BLAS_INT *incx);
void csscal_(const BLAS_INT *n, const float *a, BLAS_Complex8 *x, const BLAS_INT *incx);
void dscal_(const BLAS_INT *n, const double *a, double *x, const BLAS_INT *incx);
void sscal_(const BLAS_INT *n, const float *a, float *x, const BLAS_INT *incx);
void zdscal_(const BLAS_INT *n, const double *a, BLAS_Complex16 *x, const BLAS_INT *incx);
void zscal_(const BLAS_INT *n, const BLAS_Complex16 *a, BLAS_Complex16 *x, const BLAS_INT *incx);

// swap: x <=> y

void cswap_(const BLAS_INT *n, BLAS_Complex8 *x, const BLAS_INT *incx, BLAS_Complex8 *y, const BLAS_INT *incy);
void dswap_(const BLAS_INT *n, double *x, const BLAS_INT *incx, double *y, const BLAS_INT *incy);
void sswap_(const BLAS_INT *n, float *x, const BLAS_INT *incx, float *y, const BLAS_INT *incy);
void zswap_(const BLAS_INT *n, BLAS_Complex16 *x, const BLAS_INT *incx, BLAS_Complex16 *y, const BLAS_INT *incy);

#pragma endregion

#pragma region Level1 BLAS plane rotations

// rot: apply plane rotation ([cz]rot in LAPACK)

void csrot_(const BLAS_INT *n, BLAS_Complex8 *x, const BLAS_INT *incx, BLAS_Complex8 *y, const BLAS_INT *incy, const float *c, const float *s);
void drot_(const BLAS_INT *n, double *x, const BLAS_INT *incx, double *y, const BLAS_INT *incy, const double *c, const double *s);
void srot_(const BLAS_INT *n, float *x, const BLAS_INT *incx, float *y, const BLAS_INT *incy, const float *c, const float *s);
void zdrot_(const BLAS_INT *n, BLAS_Complex16 *x, const BLAS_INT *incx, BLAS_Complex16 *y, const BLAS_INT *incy, const double *c, const double *s);

// rotg: generate plane rotation (cf. lartg)

void crotg_(BLAS_Complex8 *a, const BLAS_Complex8 *b, float *c, BLAS_Complex8 *s);
void drotg_(double *a, double *b, double *c, double *s);
void srotg_(float *a,float *b,float *c,float *s);
void zrotg_(BLAS_Complex16 *a, const BLAS_Complex16 *b, double *c, BLAS_Complex16 *s);

// rotm: apply modified (fast) plane rotation

void drotm_(const BLAS_INT *n, double *x, const BLAS_INT *incx, double *y, const BLAS_INT *incy, const double *param);
void srotm_(const BLAS_INT *n, float *x, const BLAS_INT *incx, float *y, const BLAS_INT *incy, const float *param);

// rotmg: generate modified (fast) plane rotation

void drotmg_(double *d1, double *d2, double *x1, const double *y1, double *param);
void srotmg_(float *d1, float *d2, float *x1, const float *y1, float *param);

#pragma endregion

#pragma region Level2 BLAS full

// gemv: general matrix-vector multiply

void cgemv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *beta, BLAS_Complex8 *y, const BLAS_INT *incy);
void dgemv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const double *alpha, const double *a, const BLAS_INT *lda, const double *x, const BLAS_INT *incx, const double *beta, double *y, const BLAS_INT *incy);
void sgemv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const float *alpha, const float *a, const BLAS_INT *lda, const float *x, const BLAS_INT *incx, const float *beta, float *y, const BLAS_INT *incy);
void zgemv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *beta, BLAS_Complex16 *y, const BLAS_INT *incy);

// ger: general matrix rank-1 update

void cgerc_(const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *y, const BLAS_INT *incy, BLAS_Complex8 *a, const BLAS_INT *lda);
void cgeru_(const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *y, const BLAS_INT *incy, BLAS_Complex8 *a, const BLAS_INT *lda);
void dger_(const BLAS_INT *m, const BLAS_INT *n, const double *alpha, const double *x, const BLAS_INT *incx, const double *y, const BLAS_INT *incy, double *a, const BLAS_INT *lda);
void sger_(const BLAS_INT *m, const BLAS_INT *n, const float *alpha, const float *x, const BLAS_INT *incx, const float *y, const BLAS_INT *incy, float *a, const BLAS_INT *lda);
void zgerc_(const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *y, const BLAS_INT *incy, BLAS_Complex16 *a, const BLAS_INT *lda);
void zgeru_(const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *y, const BLAS_INT *incy, BLAS_Complex16 *a, const BLAS_INT *lda);

// {he,sy}mv: Hermitian/symmetric matrix-vector multiply ([cz]symv in LAPACK)

void chemv_(const char *uplo, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *beta, BLAS_Complex8 *y, const BLAS_INT *incy);
void dsymv_(const char *uplo, const BLAS_INT *n, const double *alpha, const double *a, const BLAS_INT *lda, const double *x, const BLAS_INT *incx, const double *beta, double *y, const BLAS_INT *incy);
void ssymv_(const char *uplo, const BLAS_INT *n, const float *alpha, const float *a, const BLAS_INT *lda, const float *x, const BLAS_INT *incx, const float *beta, float *y, const BLAS_INT *incy);
void zhemv_(const char *uplo, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *beta, BLAS_Complex16 *y, const BLAS_INT *incy);

// {he,sy}r: Hermitian/symmetric rank-1 update

void cher_(const char *uplo, const BLAS_INT *n, const float *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, BLAS_Complex8 *a, const BLAS_INT *lda);
void dsyr_(const char *uplo, const BLAS_INT *n, const double *alpha, const double *x, const BLAS_INT *incx, double *a, const BLAS_INT *lda);
void ssyr_(const char *uplo, const BLAS_INT *n, const float *alpha, const float *x, const BLAS_INT *incx, float *a, const BLAS_INT *lda);
void zher_(const char *uplo, const BLAS_INT *n, const double *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, BLAS_Complex16 *a, const BLAS_INT *lda);

// {he,sy}r2: Hermitian/symmetric rank-2 update

void cher2_(const char *uplo, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *y, const BLAS_INT *incy, BLAS_Complex8 *a, const BLAS_INT *lda);
void dsyr2_(const char *uplo, const BLAS_INT *n, const double *alpha, const double *x, const BLAS_INT *incx, const double *y, const BLAS_INT *incy, double *a, const BLAS_INT *lda);
void ssyr2_(const char *uplo, const BLAS_INT *n, const float *alpha, const float *x, const BLAS_INT *incx, const float *y, const BLAS_INT *incy, float *a, const BLAS_INT *lda);
void zher2_(const char *uplo, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *y, const BLAS_INT *incy, BLAS_Complex16 *a, const BLAS_INT *lda);

// trmv: triangular matrix-vector multiply

void ctrmv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const BLAS_Complex8 *a, const BLAS_INT *lda, BLAS_Complex8 *b, const BLAS_INT *incx);
void dtrmv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const double *a, const BLAS_INT *lda, double *b, const BLAS_INT *incx);
void strmv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const float *a, const BLAS_INT *lda, float *b, const BLAS_INT *incx);
void ztrmv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const BLAS_Complex16 *a, const BLAS_INT *lda, BLAS_Complex16 *b, const BLAS_INT *incx);

// trsv: triangular matrix-vector solve

void ctrsv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const BLAS_Complex8 *a, const BLAS_INT *lda, BLAS_Complex8 *b, const BLAS_INT *incx);
void dtrsv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const double *a, const BLAS_INT *lda, double *b, const BLAS_INT *incx);
void strsv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const float *a, const BLAS_INT *lda, float *b, const BLAS_INT *incx);
void ztrsv_(const char *uplo, const char *transa, const char *diag, const BLAS_INT *n, const BLAS_Complex16 *a, const BLAS_INT *lda, BLAS_Complex16 *b, const BLAS_INT *incx);

#pragma endregion

#pragma region Level2 BLAS packed

// {hp,sp}mv: Hermitian/symmetric matrix-vector multiply

void chpmv_(const char *uplo, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *ap, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *beta, BLAS_Complex8 *y, const BLAS_INT *incy);
void dspmv_(const char *uplo, const BLAS_INT *n, const double *alpha, const double *ap, const double *x, const BLAS_INT *incx, const double *beta, double *y, const BLAS_INT *incy);
void sspmv_(const char *uplo, const BLAS_INT *n, const float *alpha, const float *ap, const float *x, const BLAS_INT *incx, const float *beta, float *y, const BLAS_INT *incy);
void zhpmv_(const char *uplo, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *ap, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *beta, BLAS_Complex16 *y, const BLAS_INT *incy);

// {hp,sp}r: Hermitian/symmetric rank-1 update

void chpr_(const char *uplo, const BLAS_INT *n, const float *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, BLAS_Complex8 *ap);
void dspr_(const char *uplo, const BLAS_INT *n, const double *alpha, const double *x, const BLAS_INT *incx, double *ap);
void sspr_(const char *uplo, const BLAS_INT *n, const float *alpha, const float *x, const BLAS_INT *incx, float *ap);
void zhpr_(const char *uplo, const BLAS_INT *n, const double *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, BLAS_Complex16 *ap);

// {hp,sp}r2: Hermitian/symmetric rank-2 update

void chpr2_(const char *uplo, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *y, const BLAS_INT *incy, BLAS_Complex8 *ap);
void dspr2_(const char *uplo, const BLAS_INT *n, const double *alpha, const double *x, const BLAS_INT *incx, const double *y, const BLAS_INT *incy, double *ap);
void sspr2_(const char *uplo, const BLAS_INT *n, const float *alpha, const float *x, const BLAS_INT *incx, const float *y, const BLAS_INT *incy, float *ap);
void zhpr2_(const char *uplo, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *y, const BLAS_INT *incy, BLAS_Complex16 *ap);

// tpmv: triangular matrix-vector multiply

void ctpmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_Complex8 *ap, BLAS_Complex8 *x, const BLAS_INT *incx);
void dtpmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const double *ap, double *x, const BLAS_INT *incx);
void stpmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const float *ap, float *x, const BLAS_INT *incx);
void ztpmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_Complex16 *ap, BLAS_Complex16 *x, const BLAS_INT *incx);

// tpsv: triangular matrix-vector solve

void ctpsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_Complex8 *ap, BLAS_Complex8 *x, const BLAS_INT *incx);
void dtpsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const double *ap, double *x, const BLAS_INT *incx);
void stpsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const float *ap, float *x, const BLAS_INT *incx);
void ztpsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_Complex16 *ap, BLAS_Complex16 *x, const BLAS_INT *incx);


#pragma endregion

#pragma region Level2 BLAS banded

// gbmv: general matrix-vector multiply

void cgbmv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *kl, const BLAS_INT *ku, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *beta, BLAS_Complex8 *y, const BLAS_INT *incy);
void dgbmv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *kl, const BLAS_INT *ku, const double *alpha, const double *a, const BLAS_INT *lda, const double *x, const BLAS_INT *incx, const double *beta, double *y, const BLAS_INT *incy);
void sgbmv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *kl, const BLAS_INT *ku, const float *alpha, const float *a, const BLAS_INT *lda, const float *x, const BLAS_INT *incx, const float *beta, float *y, const BLAS_INT *incy);
void zgbmv_(const char *trans, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *kl, const BLAS_INT *ku, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *beta, BLAS_Complex16 *y, const BLAS_INT *incy);

// {hb,sb}mv: Hermitian/symmetric matrix-vector multiply

void chbmv_(const char *uplo, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *x, const BLAS_INT *incx, const BLAS_Complex8 *beta, BLAS_Complex8 *y, const BLAS_INT *incy);
void dsbmv_(const char *uplo, const BLAS_INT *n, const BLAS_INT *k, const double *alpha, const double *a, const BLAS_INT *lda, const double *x, const BLAS_INT *incx, const double *beta, double *y, const BLAS_INT *incy);
void ssbmv_(const char *uplo, const BLAS_INT *n, const BLAS_INT *k, const float *alpha, const float *a, const BLAS_INT *lda, const float *x, const BLAS_INT *incx, const float *beta, float *y, const BLAS_INT *incy);
void zhbmv_(const char *uplo, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *x, const BLAS_INT *incx, const BLAS_Complex16 *beta, BLAS_Complex16 *y, const BLAS_INT *incy);

// tbmv: triangular matrix-vector multiply

void ctbmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *a, const BLAS_INT *lda, BLAS_Complex8 *x, const BLAS_INT *incx);
void dtbmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const double *a, const BLAS_INT *lda, double *x, const BLAS_INT *incx);
void stbmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const float *a, const BLAS_INT *lda, float *x, const BLAS_INT *incx);
void ztbmv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *a, const BLAS_INT *lda, BLAS_Complex16 *x, const BLAS_INT *incx);

// tbsv: triangular matrix-vector solve

void ctbsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *a, const BLAS_INT *lda, BLAS_Complex8 *x, const BLAS_INT *incx);
void dtbsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const double *a, const BLAS_INT *lda, double *x, const BLAS_INT *incx);
void stbsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const float *a, const BLAS_INT *lda, float *x, const BLAS_INT *incx);
void ztbsv_(const char *uplo, const char *trans, const char *diag, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *a, const BLAS_INT *lda, BLAS_Complex16 *x, const BLAS_INT *incx);

#pragma endregion

#pragma region Level3 BLAS

// gemm: general matrix-matrix multiplication

void cgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *b, const BLAS_INT *ldb, const BLAS_Complex8 *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void dgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const double *alpha, const double *a, const BLAS_INT *lda, const double *b, const BLAS_INT *ldb, const double *beta, double *c, const BLAS_INT *ldc);
void sgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const float *alpha, const float *a, const BLAS_INT *lda, const float *b, const BLAS_INT *ldb, const float *beta, float *c, const BLAS_INT *ldc);
void zgemm_(const char *transa, const char *transb, const BLAS_INT *m, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *b, const BLAS_INT *ldb, const BLAS_Complex16 *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);

// {he,sy}mm: Hermitian/symmetric matrix-matrix multiply

void chemm_(const char *side, const char *uplo, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *b, const BLAS_INT *ldb, const BLAS_Complex8 *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void csymm_(const char *side, const char *uplo, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *b, const BLAS_INT *ldb, const BLAS_Complex8 *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void dsymm_(const char *side, const char *uplo, const BLAS_INT *m, const BLAS_INT *n, const double *alpha, const double *a, const BLAS_INT *lda, const double *b, const BLAS_INT *ldb, const double *beta, double *c, const BLAS_INT *ldc);
void ssymm_(const char *side, const char *uplo, const BLAS_INT *m, const BLAS_INT *n, const float *alpha, const float *a, const BLAS_INT *lda, const float *b, const BLAS_INT *ldb, const float *beta, float *c, const BLAS_INT *ldc);
void zhemm_(const char *side, const char *uplo, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *b, const BLAS_INT *ldb, const BLAS_Complex16 *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);
void zsymm_(const char *side, const char *uplo, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *b, const BLAS_INT *ldb, const BLAS_Complex16 *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);

// {he,sy}rk: Hermitian/symmetric rank-k update

void cherk_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const float *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const float *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void csyrk_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void dsyrk_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const double *alpha, const double *a, const BLAS_INT *lda, const double *beta, double *c, const BLAS_INT *ldc);
void ssyrk_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const float *alpha, const float *a, const BLAS_INT *lda, const float *beta, float *c, const BLAS_INT *ldc);
void zherk_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const double *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const double *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);
void zsyrk_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);

// {he,sy}r2k: Hermitian/symmetric rank-2k update

void cher2k_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *b, const BLAS_INT *ldb, const float *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void csyr2k_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, const BLAS_Complex8 *b, const BLAS_INT *ldb, const BLAS_Complex8 *beta, BLAS_Complex8 *c, const BLAS_INT *ldc);
void dsyr2k_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const double *alpha, const double *a, const BLAS_INT *lda, const double *b, const BLAS_INT *ldb, const double *beta, double *c, const BLAS_INT *ldc);
void ssyr2k_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const float *alpha, const float *a, const BLAS_INT *lda, const float *b, const BLAS_INT *ldb, const float *beta, float *c, const BLAS_INT *ldc);
void zher2k_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *b, const BLAS_INT *ldb, const double *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);
void zsyr2k_(const char *uplo, const char *trans, const BLAS_INT *n, const BLAS_INT *k, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, const BLAS_Complex16 *b, const BLAS_INT *ldb, const BLAS_Complex16 *beta, BLAS_Complex16 *c, const BLAS_INT *ldc);

// trmm: triangular matrix-matrix multiplication

void ctrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, BLAS_Complex8 *b, const BLAS_INT *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const double *alpha, const double *a, const BLAS_INT *lda, double *b, const BLAS_INT *ldb);
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const float *alpha, const float *a, const BLAS_INT *lda, float *b, const BLAS_INT *ldb);
void ztrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, BLAS_Complex16 *b, const BLAS_INT *ldb);

// trsm: triangular matrix-matrix solve

void ctrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex8 *alpha, const BLAS_Complex8 *a, const BLAS_INT *lda, BLAS_Complex8 *b, const BLAS_INT *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const double *alpha, const double *a, const BLAS_INT *lda, double *b, const BLAS_INT *ldb);
void strsm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const float *alpha, const float *a, const BLAS_INT *lda, float *b, const BLAS_INT *ldb);
void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const BLAS_INT *m, const BLAS_INT *n, const BLAS_Complex16 *alpha, const BLAS_Complex16 *a, const BLAS_INT *lda, BLAS_Complex16 *b, const BLAS_INT *ldb);

#pragma endregion
