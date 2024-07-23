#![cfg(feature = "gemmt")]

use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

#[cfg(not(feature = "gemmt"))]
pub trait GEMMTNum {}

pub trait GEMMTNum: BLASFloat {
    unsafe fn gemmt(
        uplo: *const c_char,
        transa: *const c_char,
        transb: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
        alpha: *const Self,
        a: *const Self,
        lda: *const blas_int,
        b: *const Self,
        ldb: *const blas_int,
        beta: *const Self,
        c: *mut Self,
        ldc: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GEMMTNum for $type {
            unsafe fn gemmt(
                uplo: *const c_char,
                transa: *const c_char,
                transb: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                b: *const Self,
                ldb: *const blas_int,
                beta: *const Self,
                c: *mut Self,
                ldc: *const blas_int,
            ) {
                ffi::$func(uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
        }
    };
}

impl_func!(f32, sgemmt_);
impl_func!(f64, dgemmt_);
impl_func!(c32, cgemmt_);
impl_func!(c64, zgemmt_);

/* #endregion */

/* #region BLAS driver */

pub struct GEMMT_Driver<'a, 'b, 'c, F>
where
    F: GEMMTNum,
{
    uplo: c_char,
    transa: c_char,
    transb: c_char,
    n: blas_int,
    k: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    b: ArrayView2<'b, F>,
    ldb: blas_int,
    beta: F,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
}

impl<'a, 'b, 'c, F> BLASDriver<'c, F, Ix2> for GEMMT_Driver<'a, 'b, 'c, F>
where
    F: GEMMTNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { uplo, transa, transb, n, k, alpha, a, lda, b, ldb, beta, mut c, ldc } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(c.clone_to_view_mut());
        } else if k == 0 {
            if uplo == BLASLower.try_into()? {
                for i in 0..n {
                    c.view_mut().slice_mut(s![i.., i]).mapv_inplace(|v| v * beta);
                }
            } else if uplo == BLASUpper.try_into()? {
                for i in 0..n {
                    c.view_mut().slice_mut(s![..=i, i]).mapv_inplace(|v| v * beta);
                }
            } else {
                blas_invalid!(uplo)?
            }
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            F::gemmt(&uplo, &transa, &transb, &n, &k, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct GEMMT_<'a, 'b, 'c, F>
where
    F: GEMMTNum,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub transa: BLASTranspose,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub transb: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'b, 'c, F> BLASBuilder_<'c, F, Ix2> for GEMMT_<'a, 'b, 'c, F>
where
    F: GEMMTNum,
{
    fn driver(self) -> Result<GEMMT_Driver<'a, 'b, 'c, F>, BLASError> {
        let Self { a, b, c, alpha, beta, uplo, transa, transb, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref() && b.is_fpref());

        // initialize intent(hide)
        let (n, k) = match transa {
            BLASNoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASTrans | BLASConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(transa)?,
        };
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        match transb {
            BLASNoTrans => blas_assert_eq!(b.dim(), (k, n), InvalidDim)?,
            BLASTrans | BLASConjTrans => blas_assert_eq!(b.dim(), (n, k), InvalidDim)?,
            _ => blas_invalid!(transb)?,
        };

        // optional intent(out)
        let c = match c {
            Some(c) => {
                blas_assert_eq!(c.dim(), (n, n), InvalidDim)?;
                if c.view().is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.view().to_col_layout()?.into_owned();
                    ArrayOut2::ToBeCloned(c, c_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((n, n).f())),
        };
        let ldc = c.view().stride_of(Axis(1));

        // finalize
        let driver = GEMMT_Driver {
            uplo: uplo.try_into()?,
            transa: transa.try_into()?,
            transb: transb.try_into()?,
            n: n.try_into()?,
            k: k.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            b,
            ldb: ldb.try_into()?,
            beta,
            c,
            ldc: ldc.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type GEMMT<'a, 'b, 'c, F> = GEMMT_Builder<'a, 'b, 'c, F>;
pub type SGEMMT<'a, 'b, 'c> = GEMMT<'a, 'b, 'c, f32>;
pub type DGEMMT<'a, 'b, 'c> = GEMMT<'a, 'b, 'c, f64>;
pub type CGEMMT<'a, 'b, 'c> = GEMMT<'a, 'b, 'c, c32>;
pub type ZGEMMT<'a, 'b, 'c> = GEMMT<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F> BLASBuilder<'c, F, Ix2> for GEMMT_Builder<'a, 'b, 'c, F>
where
    F: GEMMTNum,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let GEMMT_ { a, b, c, alpha, beta, uplo, transa, transb, layout } = self.build()?;
        let at = a.t();
        let bt = b.t();

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            let (transa, a_cow) = flip_trans_fpref(transa, &a, &at, false)?;
            let (transb, b_cow) = flip_trans_fpref(transb, &b, &bt, false)?;
            let obj = GEMMT_ {
                a: a_cow.view(),
                b: b_cow.view(),
                c,
                alpha,
                beta,
                uplo,
                transa,
                transb,
                layout: Some(BLASColMajor),
            };
            return obj.driver()?.run_blas();
        } else if layout == BLASRowMajor {
            let (transa, a_cow) = flip_trans_cpref(transa, &a, &at, false)?;
            let (transb, b_cow) = flip_trans_cpref(transb, &b, &bt, false)?;
            let obj = GEMMT_ {
                a: b_cow.t(),
                b: a_cow.t(),
                c: c.map(|c| c.reversed_axes()),
                alpha,
                beta,
                uplo: uplo.flip()?,
                transa: transb,
                transb: transa,
                layout: Some(BLASColMajor),
            };
            return Ok(obj.driver()?.run_blas()?.reversed_axes());
        } else {
            return blas_raise!(RuntimeError, "This is designed not to execuate this line.");
        }
    }
}

/* #endregion */
