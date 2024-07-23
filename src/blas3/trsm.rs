use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TRSMNum: BLASFloat {
    unsafe fn trsm(
        side: *const c_char,
        uplo: *const c_char,
        transa: *const c_char,
        diag: *const c_char,
        m: *const blas_int,
        n: *const blas_int,
        alpha: *const Self,
        a: *const Self,
        lda: *const blas_int,
        b: *mut Self,
        ldb: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TRSMNum for $type {
            unsafe fn trsm(
                side: *const c_char,
                uplo: *const c_char,
                transa: *const c_char,
                diag: *const c_char,
                m: *const blas_int,
                n: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                b: *mut Self,
                ldb: *const blas_int,
            ) {
                ffi::$func(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
            }
        }
    };
}

impl_func!(f32, strsm_);
impl_func!(f64, dtrsm_);
impl_func!(c32, ctrsm_);
impl_func!(c64, ztrsm_);

/* #endregion */

/* #region BLAS driver */

pub struct TRSM_Driver<'a, 'b, F>
where
    F: BLASFloat,
{
    side: c_char,
    uplo: c_char,
    transa: c_char,
    diag: c_char,
    m: blas_int,
    n: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    b: ArrayOut2<'b, F>,
    ldb: blas_int,
}

impl<'a, 'b, F> BLASDriver<'b, F, Ix2> for TRSM_Driver<'a, 'b, F>
where
    F: TRSMNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'b, F>, BLASError> {
        let Self { side, uplo, transa, diag, m, n, alpha, a, lda, mut b, ldb } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(b.clone_to_view_mut());
        }

        unsafe {
            F::trsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a_ptr, &lda, b_ptr, &ldb);
        }
        return Ok(b.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct TRSM_<'a, 'b, F>
where
    F: TRSMNum,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayViewMut2<'b, F>,

    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "BLASLeft")]
    pub side: BLASSide,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub transa: BLASTranspose,
    #[builder(setter(into), default = "BLASNonUnit")]
    pub diag: BLASDiag,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'b, F> BLASBuilder_<'b, F, Ix2> for TRSM_<'a, 'b, F>
where
    F: TRSMNum,
{
    fn driver(self) -> Result<TRSM_Driver<'a, 'b, F>, BLASError> {
        let Self { a, b, alpha, side, uplo, transa, diag, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref());

        // initialize intent(hide)
        let (m, n) = b.dim();
        let lda = a.stride_of(Axis(1));

        // perform check
        match side {
            BLASLeft => blas_assert_eq!(a.dim(), (m, m), InvalidDim)?,
            BLASRight => blas_assert_eq!(a.dim(), (n, n), InvalidDim)?,
            _ => blas_invalid!(side)?,
        };

        // prepare output
        let b = if b.view().is_fpref() {
            ArrayOut2::ViewMut(b)
        } else {
            let b_buffer = b.view().to_col_layout()?.into_owned();
            ArrayOut2::ToBeCloned(b, b_buffer)
        };
        let ldb = b.view().stride_of(Axis(1));

        // finalize
        let driver = TRSM_Driver {
            side: side.try_into()?,
            uplo: uplo.try_into()?,
            transa: transa.try_into()?,
            diag: diag.try_into()?,
            m: m.try_into()?,
            n: n.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            b,
            ldb: ldb.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type TRSM<'a, 'b, F> = TRSM_Builder<'a, 'b, F>;
pub type STRSM<'a, 'b> = TRSM<'a, 'b, f32>;
pub type DTRSM<'a, 'b> = TRSM<'a, 'b, f64>;
pub type CTRSM<'a, 'b> = TRSM<'a, 'b, c32>;
pub type ZTRSM<'a, 'b> = TRSM<'a, 'b, c64>;

impl<'a, 'b, F> BLASBuilder<'b, F, Ix2> for TRSM_Builder<'a, 'b, F>
where
    F: TRSMNum,
{
    fn run(self) -> Result<ArrayOut2<'b, F>, BLASError> {
        // initialize
        let TRSM_ { a, b, alpha, side, uplo, transa, diag, layout } = self.build()?;
        let at = a.t();

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b.view());

        let layout = get_layout_row_preferred(&[layout, Some(layout_b)], &[layout_a]);
        if layout == BLASColMajor {
            // F-contiguous: B = op(A) B (if side = L)
            let (transa_new, a_cow) = flip_trans_fpref(transa, &a, &at, false)?;
            let uplo = if transa_new != transa { uplo.flip()? } else { uplo };
            let obj = TRSM_ {
                a: a_cow.view(),
                b,
                alpha,
                side,
                uplo,
                transa: transa_new,
                diag,
                layout: Some(BLASColMajor),
            };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: B' = B' op(A') (if side = L)
            let (transa_new, a_cow) = flip_trans_cpref(transa, &a, &at, false)?;
            let uplo = if transa_new != transa { uplo.flip()? } else { uplo };
            let obj = TRSM_ {
                a: a_cow.t(),
                b: b.reversed_axes(),
                alpha,
                side: side.flip()?,
                uplo: uplo.flip()?,
                transa: transa_new,
                diag,
                layout: Some(BLASColMajor),
            };
            return Ok(obj.driver()?.run_blas()?.reversed_axes());
        }
    }
}

/* #endregion */
