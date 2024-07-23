use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait HPMVNum: BLASFloat {
    unsafe fn hpmv(
        uplo: *const c_char,
        n: *const blas_int,
        alpha: *const Self,
        ap: *const Self,
        x: *const Self,
        incx: *const blas_int,
        beta: *const Self,
        y: *mut Self,
        incy: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl HPMVNum for $type {
            unsafe fn hpmv(
                uplo: *const c_char,
                n: *const blas_int,
                alpha: *const Self,
                ap: *const Self,
                x: *const Self,
                incx: *const blas_int,
                beta: *const Self,
                y: *mut Self,
                incy: *const blas_int,
            ) {
                ffi::$func(uplo, n, alpha, ap, x, incx, beta, y, incy);
            }
        }
    };
}

impl_func!(f32, sspmv_);
impl_func!(f64, dspmv_);
impl_func!(c32, chpmv_);
impl_func!(c64, zhpmv_);

/* #endregion */

/* #region BLAS driver */

pub struct HPMV_Driver<'a, 'x, 'y, F>
where
    F: HPMVNum,
{
    uplo: c_char,
    n: blas_int,
    alpha: F,
    ap: ArrayView1<'a, F>,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: blas_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for HPMV_Driver<'a, 'x, 'y, F>
where
    F: HPMVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let Self { uplo, n, alpha, ap, x, incx, beta, mut y, incy, .. } = self;
        let ap_ptr = ap.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            F::hpmv(&uplo, &n, &alpha, ap_ptr, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HPMV_<'a, 'x, 'y, F>
where
    F: HPMVNum,
{
    pub ap: ArrayView1<'a, F>,
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub y: Option<ArrayViewMut1<'y, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for HPMV_<'a, 'x, 'y, F>
where
    F: HPMVNum,
{
    fn driver(self) -> Result<HPMV_Driver<'a, 'x, 'y, F>, BLASError> {
        let Self { ap, x, y, alpha, beta, uplo, layout, .. } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let incap = ap.stride_of(Axis(0));
        assert!(incap <= 1);
        assert_eq!(layout, Some(BLASColMajor));

        // initialize intent(hide)
        let np = ap.len_of(Axis(0));
        let n = x.len_of(Axis(0));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(np, n * (n + 1) / 2, InvalidDim)?;

        // prepare output
        let y = match y {
            Some(y) => {
                blas_assert_eq!(y.len_of(Axis(0)), n, InvalidDim)?;
                ArrayOut1::ViewMut(y)
            },
            None => ArrayOut1::Owned(Array1::zeros(n)),
        };
        let incy = y.view().stride_of(Axis(0));

        // finalize
        let driver = HPMV_Driver {
            uplo: uplo.into(),
            n: n.try_into()?,
            alpha,
            ap,
            x,
            incx: incx.try_into()?,
            beta,
            y,
            incy: incy.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type HPMV<'a, 'x, 'y, F> = HPMV_Builder<'a, 'x, 'y, F>;
pub type SSPMV<'a, 'x, 'y> = HPMV<'a, 'x, 'y, f32>;
pub type DSPMV<'a, 'x, 'y> = HPMV<'a, 'x, 'y, f64>;
pub type CHPMV<'a, 'x, 'y> = HPMV<'a, 'x, 'y, c32>;
pub type ZHPMV<'a, 'x, 'y> = HPMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for HPMV_Builder<'a, 'x, 'y, F>
where
    F: HPMVNum,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout = obj.layout.unwrap_or(BLASRowMajor);

        if layout == BLASColMajor {
            // F-contiguous
            let ap_cow = obj.ap.to_seq_layout()?;
            let obj = HPMV_ { ap: ap_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let ap_cow = obj.ap.to_seq_layout()?;
            if F::is_complex() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.map(|mut y| {
                    y.mapv_inplace(F::conj);
                    y
                });
                let obj = HPMV_ {
                    ap: ap_cow.view(),
                    x: x.view(),
                    y,
                    uplo: obj.uplo.flip(),
                    alpha: F::conj(obj.alpha),
                    beta: F::conj(obj.beta),
                    layout: Some(BLASColMajor),
                    ..obj
                };
                let mut y = obj.driver()?.run_blas()?;
                y.view_mut().mapv_inplace(F::conj);
                return Ok(y);
            } else {
                let obj =
                    HPMV_ { ap: ap_cow.view(), uplo: obj.uplo.flip(), layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            }
        }
    }
}

/* #endregion */
