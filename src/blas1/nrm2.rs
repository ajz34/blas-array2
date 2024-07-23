use crate::ffi::{self, blas_int};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::Zero;

/* #region BLAS func */

pub trait NRM2Num: BLASFloat {
    unsafe fn nrm2(n: *const blas_int, x: *const Self, incx: *const blas_int) -> Self::RealFloat;
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl NRM2Num for $type
        where
            $type: BLASFloat,
        {
            unsafe fn nrm2(
                n: *const blas_int,
                x: *const Self,
                incx: *const blas_int,
            ) -> <$type as BLASFloat>::RealFloat {
                ffi::$func(n, x, incx)
            }
        }
    };
}

impl_func!(f32, snrm2_);
impl_func!(f64, dnrm2_);
impl_func!(c32, scnrm2_);
impl_func!(c64, dznrm2_);

/* #endregion */

/* #region BLAS driver */

pub struct NRM2_Driver<'x, F>
where
    F: NRM2Num,
{
    n: blas_int,
    x: ArrayView1<'x, F>,
    incx: blas_int,
}

impl<'x, F> NRM2_Driver<'x, F>
where
    F: NRM2Num,
{
    pub fn run_blas(self) -> Result<F::RealFloat, BLASError> {
        let Self { n, x, incx } = self;
        let x_ptr = x.as_ptr();
        if n == 0 {
            return Ok(F::RealFloat::zero());
        } else {
            return unsafe { Ok(F::nrm2(&n, x_ptr, &incx)) };
        }
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct NRM2_<'x, F>
where
    F: NRM2Num,
{
    pub x: ArrayView1<'x, F>,
}

impl<'x, F> NRM2_<'x, F>
where
    F: NRM2Num,
{
    pub fn driver(self) -> Result<NRM2_Driver<'x, F>, BLASError> {
        let Self { x } = self;
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));
        let driver = NRM2_Driver { n: n.try_into()?, x, incx: incx.try_into()? };
        return Ok(driver);
    }
}

/* #region BLAS wrapper */

pub type NRM2<'x, F> = NRM2_Builder<'x, F>;
pub type SNRM2<'x> = NRM2<'x, f32>;
pub type DNRM2<'x> = NRM2<'x, f64>;
pub type SCNRM2<'x> = NRM2<'x, c32>;
pub type DZNRM2<'x> = NRM2<'x, c64>;

impl<'x, F> NRM2<'x, F>
where
    F: NRM2Num,
{
    pub fn run(self) -> Result<F::RealFloat, BLASError> {
        self.build()?.driver()?.run_blas()
    }
}

/* #endregion */
