# BLAS Wrapper Structs

Document of BLAS wrappers is on-going work. If there is anything important important, we refer [LAPACK document](https://netlib.org/lapack/explore-html/index.html), or struct document in "Prototype" column.

## Level 3 BLAS

| BLAS | Prototype | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|
| gemm | [`GEMM_<F>`] | [`GEMM<F>`] | [`SGEMM`] | [`DGEMM`] | [`CGEMM`] | [`ZGEMM`] | general matrix-matrix multiply |
| symm | [`SYMM_<F, S>`] | [`SYMM<F>`] | [`SSYMM`] | [`DSYMM`] | [`CSYMM`] | [`ZSYMM`] | symmetric matrix-matrix multiply |
| hemm | [`SYMM_<F, S>`] | [`HEMM<F>`] |  |  | [`CHEMM`] | [`ZHEMM`] | hermitian matrix-matrix multiply |
| syrk | [`SYRK_<F, S>`] | [`SYRK<F>`] | [`SSYRK`] | [`DSYRK`] | [`CSYRK`] | [`ZSYRK`] | symmetric rank-k update |
| herk | [`SYRK_<F, S>`] | [`HERK<F>`] |  |  | [`CHERK`] | [`ZHERK`] | hermitian rank-k update |
| syr2k | [`SYR2K_<F, S>`] | [`SYR2K<F>`] | [`SSYR2K`] | [`DSYR2K`] | [`CSYR2K`] | [`ZSYR2K`] | symmetric rank-2k update |
| her2k | [`SYR2K_<F, S>`] | [`HER2K<F>`] |  |  | [`CHER2K`] | [`ZHER2K`] | hermitian rank-2k update |
| trmm | [`TRMM_<F>`] | [`TRMM<F>`] | [`STRMM`] | [`DTRMM`] | [`CTRMM`] | [`ZTRMM`] | triangular matrix-matrix multiply |
| trsm | [`TRSM_<F>`] | [`TRSM<F>`] | [`STRSM`] | [`DTRSM`] | [`CTRSM`] | [`ZTRSM`] | triangular matrix-matrix solve |

## Level 3 BLAS (extensions)

| BLAS | Prototype | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|
| gemmt | [`GEMMT_<F>`] | [`GEMMT<F>`] | [`SGEMMT`] | [`DGEMMT`] | [`CGEMMT`] | [`ZGEMMT`] | general matrix-matrix multiply, triangular update |

## Level 2 BLAS (full)

| BLAS | Prototype | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|
| gemv | [`GEMV_<F>`] | [`GEMV<F>`] | [`SGEMV`] | [`DGEMV`] | [`CGEMV`] | [`ZGEMV`] | general matrix-vector multiply |
| ger | [`GER_<F>`] | [`GER<F>`] | [`SGER`] | [`DGER`] | [`CGERU`] | [`ZGERU`] | general matrix rank-1 update |
| gerc | [`GERC_<F>`] | [`GERC<F>`] |  |  | [`CGERC`] | [`ZGERC`] | general matrix rank-1 update |
| symv | [`SYMV_<F, S>`] | [`SYMV<F>`] | [`SSYMV`] | [`DSYMV`] |  |  | symmetric matrix-vector multiply |
| hemv | [`SYMV_<F, S>`] | [`HEMV<F>`] |  |  | [`CHEMV`] | [`ZHEMV`] | hermitian matrix-vector multiply |
| syr | [`SYR_<F, S>`] | [`SYR<F>`] | [`SSYR`] | [`DSYR`] |  |  | symmetric rank-1 update |
| her | [`SYR_<F, S>`] | [`HER<F>`] |  |  | [`CHER`] | [`ZHER`] | hermitian rank-1 update |
| syr2 | [`SYR2_<F>`] | [`SYR2<F>`] | [`SSYR2`] | [`DSYR2`] |  |  | symmetric rank-2 update |
| her2 | [`SYR2_<F>`] | [`HER2<F>`] |  |  | [`CHER2`] | [`ZHER2`] | hermitian rank-2 update |
| trmv | [`TRMV_<F>`] | [`TRMV<F>`] | [`STRMV`] | [`DTRMV`] | [`CTRMV`] | [`ZTRMV`] | triangular matrix-vector multiply |
| trsv | [`TRSV_<F>`] | [`TRSV<F>`] | [`STRSV`] | [`DTRSV`] | [`CTRSV`] | [`ZTRSV`] | triangular matrix-vector solve |

## Level 2 BLAS (packed)

| BLAS | Prototype | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|
| spmv | [`SPMV_<F, S>`] | [`SPMV<F>`] | [`SSPMV`] | [`DSPMV`] |  |  | symmetric matrix-vector multiply |
| hpmv | [`SPMV_<F, S>`] | [`HPMV<F>`] |  |  | [`CHPMV`] | [`ZHPMV`] | hermitian matrix-vector multiply |
| spr | [`SPR_<F, S>`] | [`SPR<F>`] | [`SSPR`] | [`DSPR`] |  |  | symmetric rank-1 update |
| hpr | [`SPR_<F, S>`] | [`HPR<F>`] |  |  | [`CHPR`] | [`ZHPR`] | hermitian rank-1 update |
| spr2 | [`SPR2_<F>`] | [`SPR2<F>`] | [`SSPR2`] | [`DSPR2`] |  |  | symmetric rank-2 update |
| hpr2 | [`SPR2_<F>`] | [`HPR2<F>`] |  |  | [`CHPR2`] | [`ZHPR2`] | hermitian rank-2 update |
| tpmv | [`TPMV_<F>`] | [`TPMV<F>`] | [`STPMV`] | [`DTPMV`] | [`CTPMV`] | [`ZTPMV`] | triangular matrix-vector multiply |
| tpsv | [`TPSV_<F>`] | [`TPSV<F>`] | [`STPSV`] | [`DTPSV`] | [`CTPSV`] | [`ZTPSV`] | triangular matrix-vector solve |

## Level 2 BLAS (banded)

| BLAS | Prototype | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|
| gbmv | [`GBMV_<F>`] | [`GBMV<F>`] | [`SGBMV`] | [`DGBMV`] | [`CGBMV`] | [`ZGBMV`] | general matrix-vector multiply |
| sbmv | [`SBMV_<F, S>`] | [`SBMV<F>`] | [`SSBMV`] | [`DSBMV`] |  |  | symmetric matrix-vector multiply |
| hbmv | [`SBMV_<F, S>`] | [`HBMV<F>`] |  |  | [`CHBMV`] | [`ZHBMV`] | hermitian matrix-vector multiply |
| tbmv | [`TBMV_<F>`] | [`TBMV<F>`] | [`STBMV`] | [`DTBMV`] | [`CTBMV`] | [`ZTBMV`] | triangular matrix-vector multiply |
| tbsv | [`TBSV_<F>`] | [`TBSV<F>`] | [`STBSV`] | [`DTBSV`] | [`CTBSV`] | [`ZTBSV`] | triangular matrix-vector solve |

## Level 1 BLAS

| BLAS | Prototype | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|
| asum | [`ASUM_<F>`] | [`ASUM<F>`] | [`SASUM`] | [`DASUM`] | [`SCASUM`] | [`DZASUM`] | $\sum_i \big( \vert \mathrm{re} ( x_i ) \vert + \vert \mathrm{im} ( x_i ) \vert \big)$ |
| nrm2 | [`NRM2_<F>`] | [`NRM2<F>`] | [`SNRM2`] | [`DNRM2`] | [`SCNRM2`] | [`DZASUM`] | $\Vert \boldsymbol{x} \Vert_2$ |
| iamax | [`IAMAX_<F>`] | [`IAMAX<F>`] | [`ISAMAX`] | [`IDAMAX`] | [`ICAMAX`] | [`IZAMAX`] | $\arg \max_i \big( \vert \mathrm{re} ( x_i ) \vert + \vert \mathrm{im} ( x_i ) \vert \big)$ |
