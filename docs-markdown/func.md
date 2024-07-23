# BLAS Wrapper Structs

Document of BLAS wrappers is on-going work.

Terms of the following table is
- **Prototype**: Prototype of BLAS wrappers.
    - For users, prototype is **NOT** designed to be used by API caller. One may wish to call generic structs (such as `GEMM<F>`) or directly call specialization (such as `DGEMM`).
    - For documents and structure of BLAS wrapper, please refer to hyperlinks of prototype (on docs.rs). We also refer [LAPACK document](https://netlib.org/lapack/explore-html/index.html).
- **Num Trait**: Trait bound that could be applied on generics.
- **Generic**: Generic structs of BLAS wrapper. This is designed to be called by API user. For example of `GEMM<F>`, this is type alias of `GEMM_Builder<F>`.
- **Specialization**: Specialization structs of BLAS wrapper. This is designed to be called by API user, and have the same subroutine name to (legacy) BLAS.

Abbrivations
- symm: Symmetric
- hermi: Hermitian
- tri: Triangular

## Level 3 BLAS

| BLAS | Prototype | Num Trait | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|--|
| gemm  | [`GEMM_<F>`]  | [`GEMMNum`]  | [`GEMM<F>`]  | [`SGEMM`]  | [`DGEMM`]  | [`CGEMM`]  | [`ZGEMM`]  | general matrix-matrix multiply |
| symm  | [`SYMM_<F>`]  | [`SYMMNum`]  | [`SYMM<F>`]  | [`SSYMM`]  | [`DSYMM`]  | [`CSYMM`]  | [`ZSYMM`]  | symm matrix-matrix multiply |
| hemm  | [`HEMM_<F>`]  | [`HEMMNum`]  | [`HEMM<F>`]  |            |            | [`CHEMM`]  | [`ZHEMM`]  | hermi matrix-matrix multiply |
| syrk  | [`SYRK_<F>`]  | [`SYRKNum`]  | [`SYRK<F>`]  | [`SSYRK`]  | [`DSYRK`]  | [`CSYRK`]  | [`ZSYRK`]  | symm rank-k update |
| herk  | [`HERK_<F>`]  | [`HERKNum`]  | [`HERK<F>`]  |            |            | [`CHERK`]  | [`ZHERK`]  | hermi rank-k update |
| syr2k | [`SYR2K_<F>`] | [`SYR2KNum`] | [`SYR2K<F>`] | [`SSYR2K`] | [`DSYR2K`] | [`CSYR2K`] | [`ZSYR2K`] | symm rank-2k update |
| her2k | [`HER2K_<F>`] | [`HER2KNum`] | [`HER2K<F>`] |            |            | [`CHER2K`] | [`ZHER2K`] | hermi rank-2k update |
| trmm  | [`TRMM_<F>`]  | [`TRMMNum`]  | [`TRMM<F>`]  | [`STRMM`]  | [`DTRMM`]  | [`CTRMM`]  | [`ZTRMM`]  | tri matrix-matrix multiply |
| trsm  | [`TRSM_<F>`]  | [`TRSMNum`]  | [`TRSM<F>`]  | [`STRSM`]  | [`DTRSM`]  | [`CTRSM`]  | [`ZTRSM`]  | tri matrix-matrix solve |

## Level 3 BLAS (extensions)

| BLAS | Prototype | Num Trait | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|--|
| gemmt | [`GEMMT_<F>`] | [`GEMMTNum`] | [`GEMMT<F>`] | [`SGEMMT`] | [`DGEMMT`] | [`CGEMMT`] | [`ZGEMMT`] | general matrix-matrix multiply, tri update |

## Level 2 BLAS (full)

| BLAS | Prototype | Num Trait | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|--|
| gemv      | [`GEMV_<F>`] | [`GEMVNum`] | [`GEMV<F>`] | [`SGEMV`] | [`DGEMV`] | [`CGEMV`] | [`ZGEMV`] | general matrix-vector multiply |
| ger       | [`GER_<F>`]  | [`GERNum`] | [`GER<F>`]  | [`SGER`]  | [`DGER`]  | [`CGERU`] | [`ZGERU`] | general matrix rank-1 update |
| gerc      | [`GERC_<F>`] | [`GERCNum`] | [`GERC<F>`] |           |           | [`CGERC`] | [`ZGERC`] | general matrix rank-1 update |
| {sy,he}mv | [`HEMV_<F>`] | [`HEMVNum`] | [`HEMV<F>`] | [`SSYMV`] | [`DSYMV`] | [`CHEMV`] | [`ZHEMV`] | symm/hermi matrix-vector multiply |
| {sy,he}r  | [`HER_<F>`]  | [`HERNum`] | [`HER<F>`]  | [`SSYR`]  | [`DSYR`]  | [`CHER`]  | [`ZHER`]  | symm/hermi rank-1 update |
| {sy,he}r2 | [`HER2_<F>`] | [`HER2Num`] | [`HER2<F>`] | [`SSYR2`] | [`DSYR2`] | [`CHER2`] | [`ZHER2`] | symm/hermi rank-2 update |
| trmv      | [`TRMV_<F>`] | [`TRMVNum`] | [`TRMV<F>`] | [`STRMV`] | [`DTRMV`] | [`CTRMV`] | [`ZTRMV`] | tri matrix-vector multiply |
| trsv      | [`TRSV_<F>`] | [`TRSVNum`] | [`TRSV<F>`] | [`STRSV`] | [`DTRSV`] | [`CTRSV`] | [`ZTRSV`] | tri matrix-vector solve |

## Level 2 BLAS (packed)

| BLAS | Prototype | Num Trait | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|--|
| {sp,hp}mv | [`HPMV_<F>`] | [`HPMVNum`] | [`HPMV<F>`] | [`SSPMV`] | [`DSPMV`] | [`CHPMV`] | [`ZHPMV`] | symm/hermi matrix-vector multiply |
| {sp,hp}r  | [`HPR_<F>`]  | [`HPRNum`]   | [`HPR<F>`]  | [`SSPR`]  | [`DSPR`]  | [`CHPR`]  | [`ZHPR`]  | symm/hermi rank-1 update |
| {sp,hp}r2 | [`HPR2_<F>`] | [`HPR2Num`] | [`HPR2<F>`] | [`SSPR2`] | [`DSPR2`] | [`CHPR2`] | [`ZHPR2`] | symm/hermi rank-2 update |
| tpmv      | [`TPMV_<F>`] | [`TPMVNum`] | [`TPMV<F>`] | [`STPMV`] | [`DTPMV`] | [`CTPMV`] | [`ZTPMV`] | tri matrix-vector multiply |
| tpsv      | [`TPSV_<F>`] | [`TPSVNum`] | [`TPSV<F>`] | [`STPSV`] | [`DTPSV`] | [`CTPSV`] | [`ZTPSV`] | tri matrix-vector solve |

## Level 2 BLAS (banded)

| BLAS | Prototype | Num Trait | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|--|
| gbmv      | [`GBMV_<F>`] | [`GBMVNum`] | [`GBMV<F>`] | [`SGBMV`] | [`DGBMV`] | [`CGBMV`] | [`ZGBMV`] | general matrix-vector multiply |
| {sb,hb}mv | [`HBMV_<F>`] | [`HBMVNum`] | [`HBMV<F>`] | [`SSBMV`] | [`DSBMV`] | [`CHBMV`] | [`ZHBMV`] | symm/hermi matrix-vector multiply |
| tbmv      | [`TBMV_<F>`] | [`TBMVNum`] | [`TBMV<F>`] | [`STBMV`] | [`DTBMV`] | [`CTBMV`] | [`ZTBMV`] | tri matrix-vector multiply |
| tbsv      | [`TBSV_<F>`] | [`TBSVNum`] | [`TBSV<F>`] | [`STBSV`] | [`DTBSV`] | [`CTBSV`] | [`ZTBSV`] | tri matrix-vector solve |

## Level 1 BLAS

| BLAS | Prototype | Num Trait | Generic | f32 | f64 | c32 | c64 | Description |
|--|--|--|--|--|--|--|--|--|
| asum  | [`ASUM_<F>`]  | [`ASUMNum`]  | [`ASUM<F>`]  | [`SASUM`]  | [`DASUM`]  | [`SCASUM`] | [`DZASUM`] | $\sum_i \big( \vert \mathrm{re} ( x_i ) \vert + \vert \mathrm{im} ( x_i ) \vert \big)$ |
| nrm2  | [`NRM2_<F>`]  | [`NRM2Num`]  | [`NRM2<F>`]  | [`SNRM2`]  | [`DNRM2`]  | [`SCNRM2`] | [`DZASUM`] | $\Vert \boldsymbol{x} \Vert_2$ |
| iamax | [`IAMAX_<F>`] | [`IAMAXNum`] | [`IAMAX<F>`] | [`ISAMAX`] | [`IDAMAX`] | [`ICAMAX`] | [`IZAMAX`] | $\arg \max_i \big( \vert \mathrm{re} ( x_i ) \vert + \vert \mathrm{im} ( x_i ) \vert \big)$ |
