use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TPSVNum: BLASFloat {
    unsafe fn tpsv(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blas_int,
        ap: *const Self,
        x: *mut Self,
        incx: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TPSVNum for $type {
            unsafe fn tpsv(
                uplo: *const c_char,
                trans: *const c_char,
                diag: *const c_char,
                n: *const blas_int,
                ap: *const Self,
                x: *mut Self,
                incx: *const blas_int,
            ) {
                ffi::$func(uplo, trans, diag, n, ap, x, incx);
            }
        }
    };
}

impl_func!(f32, stpsv_);
impl_func!(f64, dtpsv_);
impl_func!(c32, ctpsv_);
impl_func!(c64, ztpsv_);

/* #endregion */

/* #region BLAS driver */

pub struct TPSV_Driver<'a, 'x, F>
where
    F: TPSVNum,
{
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: blas_int,
    ap: ArrayView1<'a, F>,
    x: ArrayOut1<'x, F>,
    incx: blas_int,
}

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TPSV_Driver<'a, 'x, F>
where
    F: TPSVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        let Self { uplo, trans, diag, n, ap, mut x, incx } = self;
        let ap_ptr = ap.as_ptr();
        let x_ptr = x.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(x);
        }

        unsafe {
            F::tpsv(&uplo, &trans, &diag, &n, ap_ptr, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct TPSV_<'a, 'x, F>
where
    F: TPSVNum,
{
    pub ap: ArrayView1<'a, F>,
    pub x: ArrayViewMut1<'x, F>,

    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into), default = "BLASNonUnit")]
    pub diag: BLASDiag,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TPSV_<'a, 'x, F>
where
    F: TPSVNum,
{
    fn driver(self) -> Result<TPSV_Driver<'a, 'x, F>, BLASError> {
        let Self { ap, x, uplo, trans, diag, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        let incap = ap.stride_of(Axis(0));
        assert!(incap <= 1);

        // initialize intent(hide)
        let np = ap.len_of(Axis(0));
        let n = x.len_of(Axis(0));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(np, n * (n + 1) / 2, InvalidDim)?;

        // prepare output
        let x = ArrayOut1::ViewMut(x);

        // finalize
        let driver = TPSV_Driver {
            uplo: uplo.try_into()?,
            trans: trans.try_into()?,
            diag: diag.try_into()?,
            n: n.try_into()?,
            ap,
            x,
            incx: incx.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type TPSV<'a, 'x, F> = TPSV_Builder<'a, 'x, F>;
pub type STPSV<'a, 'x> = TPSV<'a, 'x, f32>;
pub type DTPSV<'a, 'x> = TPSV<'a, 'x, f64>;
pub type CTPSV<'a, 'x> = TPSV<'a, 'x, c32>;
pub type ZTPSV<'a, 'x> = TPSV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TPSV_Builder<'a, 'x, F>
where
    F: TPSVNum,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout = obj.layout.unwrap_or(BLASRowMajor);

        if layout == BLASColMajor {
            // F-contiguous
            let ap_cow = obj.ap.to_seq_layout()?;
            let obj = TPSV_ { ap: ap_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let ap_cow = obj.ap.to_seq_layout()?;
            match obj.trans {
                BLASNoTrans => {
                    // N -> T
                    let obj = TPSV_ {
                        ap: ap_cow.view(),
                        trans: BLASTrans,
                        uplo: obj.uplo.flip()?,
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // T -> N
                    let obj = TPSV_ {
                        ap: ap_cow.view(),
                        trans: BLASNoTrans,
                        uplo: obj.uplo.flip()?,
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> N
                    let mut x = obj.x;
                    x.mapv_inplace(F::conj);
                    let obj = TPSV_ {
                        ap: ap_cow.view(),
                        x,
                        trans: BLASNoTrans,
                        uplo: obj.uplo.flip()?,
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    let mut x = obj.driver()?.run_blas()?;
                    x.view_mut().mapv_inplace(F::conj);
                    return Ok(x);
                },
                _ => return blas_invalid!(obj.trans)?,
            }
        }
    }
}

/* #endregion */
