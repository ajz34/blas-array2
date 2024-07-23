use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::*;

/* #region BLAS func */

pub trait HERNum: BLASFloat {
    unsafe fn her(
        uplo: *const c_char,
        n: *const blas_int,
        alpha: *const Self::RealFloat,
        x: *const Self,
        incx: *const blas_int,
        a: *mut Self,
        lda: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl HERNum for $type {
            unsafe fn her(
                uplo: *const c_char,
                n: *const blas_int,
                alpha: *const Self::RealFloat,
                x: *const Self,
                incx: *const blas_int,
                a: *mut Self,
                lda: *const blas_int,
            ) {
                ffi::$func(uplo, n, alpha, x, incx, a, lda);
            }
        }
    };
}

impl_func!(f32, ssyr_);
impl_func!(f64, dsyr_);
impl_func!(c32, cher_);
impl_func!(c64, zher_);

/* #endregion */

/* #region BLAS driver */

pub struct HER_Driver<'x, 'a, F>
where
    F: HERNum,
{
    uplo: c_char,
    n: blas_int,
    alpha: F::RealFloat,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    a: ArrayOut2<'a, F>,
    lda: blas_int,
}

impl<'x, 'a, F> BLASDriver<'a, F, Ix2> for HER_Driver<'x, 'a, F>
where
    F: HERNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        let Self { uplo, n, alpha, x, incx, mut a, lda, .. } = self;
        let x_ptr = x.as_ptr();
        let a_ptr = a.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(a.clone_to_view_mut());
        }

        unsafe {
            F::her(&uplo, &n, &alpha, x_ptr, &incx, a_ptr, &lda);
        }
        return Ok(a.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HER_<'x, 'a, F>
where
    F: HERNum,
{
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub a: Option<ArrayViewMut2<'a, F>>,
    #[builder(setter(into), default = "F::RealFloat::one()")]
    pub alpha: F::RealFloat,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
}

impl<'x, 'a, F> BLASBuilder_<'a, F, Ix2> for HER_<'x, 'a, F>
where
    F: HERNum,
{
    fn driver(self) -> Result<HER_Driver<'x, 'a, F>, BLASError> {
        let Self { x, a, alpha, uplo, .. } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));

        // prepare output
        let a = match a {
            Some(a) => {
                blas_assert_eq!(a.dim(), (n, n), InvalidDim)?;
                if a.view().is_fpref() {
                    ArrayOut2::ViewMut(a)
                } else {
                    let a_buffer = a.view().to_col_layout()?.into_owned();
                    ArrayOut2::ToBeCloned(a, a_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((n, n).f())),
        };
        let lda = a.view().stride_of(Axis(1));

        // finalize
        let driver = HER_Driver {
            uplo: uplo.try_into()?,
            n: n.try_into()?,
            alpha,
            x,
            incx: incx.try_into()?,
            a,
            lda: lda.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type HER<'x, 'a, F> = HER_Builder<'x, 'a, F>;
pub type SSYR<'x, 'a> = HER<'x, 'a, f32>;
pub type DSYR<'x, 'a> = HER<'x, 'a, f64>;
pub type CHER<'x, 'a> = HER<'x, 'a, c32>;
pub type ZHER<'x, 'a> = HER<'x, 'a, c64>;

impl<'x, 'a, F> BLASBuilder<'a, F, Ix2> for HER_Builder<'x, 'a, F>
where
    F: HERNum,
{
    fn run(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        if obj.a.as_ref().map(|a| a.view().is_fpref()) == Some(true) {
            // F-contiguous
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let uplo = obj.uplo.flip()?;
            let a = obj.a.map(|a| a.reversed_axes());
            if F::is_complex() {
                let x = obj.x.mapv(F::conj);
                let obj = HER_ { a, x: x.view(), uplo, ..obj };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            } else {
                let obj = HER_ { a, uplo, ..obj };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            };
        }
    }
}

/* #endregion */
