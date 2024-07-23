use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::*;

/* #region BLAS func */

pub trait HPRNum: BLASFloat {
    unsafe fn hpr(
        uplo: *const c_char,
        n: *const blas_int,
        alpha: *const Self::RealFloat,
        x: *const Self,
        incx: *const blas_int,
        ap: *mut Self,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl HPRNum for $type {
            unsafe fn hpr(
                uplo: *const c_char,
                n: *const blas_int,
                alpha: *const Self::RealFloat,
                x: *const Self,
                incx: *const blas_int,
                ap: *mut Self,
            ) {
                ffi::$func(uplo, n, alpha, x, incx, ap);
            }
        }
    };
}

impl_func!(f32, sspr_);
impl_func!(f64, dspr_);
impl_func!(c32, chpr_);
impl_func!(c64, zhpr_);

/* #endregion */

/* #region BLAS driver */

pub struct HPR_Driver<'x, 'a, F>
where
    F: HPRNum,
{
    uplo: c_char,
    n: blas_int,
    alpha: F::RealFloat,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    ap: ArrayOut1<'a, F>,
}

impl<'x, 'a, F> BLASDriver<'a, F, Ix1> for HPR_Driver<'x, 'a, F>
where
    F: HPRNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'a, F>, BLASError> {
        let Self { uplo, n, alpha, x, incx, mut ap, .. } = self;
        let x_ptr = x.as_ptr();
        let ap_ptr = ap.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(ap.clone_to_view_mut());
        }

        unsafe {
            F::hpr(&uplo, &n, &alpha, x_ptr, &incx, ap_ptr);
        }
        return Ok(ap.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HPR_<'x, 'a, F>
where
    F: HPRNum,
{
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub ap: Option<ArrayViewMut1<'a, F>>,
    #[builder(setter(into), default = "F::RealFloat::one()")]
    pub alpha: F::RealFloat,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'x, 'a, F> BLASBuilder_<'a, F, Ix1> for HPR_<'x, 'a, F>
where
    F: HPRNum,
{
    fn driver(self) -> Result<HPR_Driver<'x, 'a, F>, BLASError> {
        let Self { x, ap, alpha, uplo, layout, .. } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));

        // prepare output
        let ap = match ap {
            Some(ap) => {
                blas_assert_eq!(ap.len_of(Axis(0)), n * (n + 1) / 2, InvalidDim)?;
                if ap.is_standard_layout() {
                    ArrayOut1::ViewMut(ap)
                } else {
                    let ap_buffer = ap.view().to_seq_layout()?.into_owned();
                    ArrayOut1::ToBeCloned(ap, ap_buffer)
                }
            },
            None => ArrayOut1::Owned(Array1::zeros(n * (n + 1) / 2)),
        };

        // finalize
        let driver = HPR_Driver { uplo: uplo.into(), n: n.try_into()?, alpha, x, incx: incx.try_into()?, ap };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type HPR<'x, 'a, F> = HPR_Builder<'x, 'a, F>;
pub type SSPR<'x, 'a> = HPR<'x, 'a, f32>;
pub type DSPR<'x, 'a> = HPR<'x, 'a, f64>;
pub type CHPR<'x, 'a> = HPR<'x, 'a, c32>;
pub type ZHPR<'x, 'a> = HPR<'x, 'a, c64>;

impl<'x, 'a, F> BLASBuilder<'a, F, Ix1> for HPR_Builder<'x, 'a, F>
where
    F: HPRNum,
{
    fn run(self) -> Result<ArrayOut1<'a, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        if obj.layout == Some(BLASColMajor) {
            // F-contiguous
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let uplo = obj.uplo.flip();
            if F::is_complex() {
                let x = obj.x.mapv(F::conj);
                let obj = HPR_ { x: x.view(), uplo, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            } else {
                let obj = HPR_ { uplo, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            };
        }
    }
}

/* #endregion */
