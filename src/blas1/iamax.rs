use crate::ffi::{self, blas_int};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait IAMAXNum: BLASFloat {
    unsafe fn iamax(n: *const blas_int, x: *const Self, incx: *const blas_int) -> blas_int;
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl IAMAXNum for $type
        where
            $type: BLASFloat,
        {
            unsafe fn iamax(n: *const blas_int, x: *const Self, incx: *const blas_int) -> blas_int {
                ffi::$func(n, x, incx)
            }
        }
    };
}

impl_func!(f32, isamax_);
impl_func!(f64, idamax_);
impl_func!(c32, icamax_);
impl_func!(c64, izamax_);

/* #endregion */

/* #region BLAS driver */

pub struct IAMAX_Driver<'x, F>
where
    F: IAMAXNum,
{
    n: blas_int,
    x: ArrayView1<'x, F>,
    incx: blas_int,
}

impl<'x, F> IAMAX_Driver<'x, F>
where
    F: IAMAXNum,
{
    pub fn run_blas(self) -> Result<usize, BLASError> {
        let Self { n, x, incx } = self;
        let x_ptr = x.as_ptr();
        if n == 0 {
            return Ok(0);
        } else {
            // 0-index for C/Rust v.s. 1-index for Fortran
            return unsafe { Ok((F::iamax(&n, x_ptr, &incx) - 1).try_into()?) };
        }
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct IAMAX_<'x, F>
where
    F: IAMAXNum,
{
    pub x: ArrayView1<'x, F>,
}

impl<'x, F> IAMAX_<'x, F>
where
    F: IAMAXNum,
{
    pub fn driver(self) -> Result<IAMAX_Driver<'x, F>, BLASError> {
        let Self { x } = self;
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));
        let driver = IAMAX_Driver { n: n.try_into()?, x, incx: incx.try_into()? };
        return Ok(driver);
    }
}

/* #region BLAS wrapper */

pub type IAMAX<'x, F> = IAMAX_Builder<'x, F>;
pub type ISAMAX<'x> = IAMAX<'x, f32>;
pub type IDAMAX<'x> = IAMAX<'x, f64>;
pub type ICAMAX<'x> = IAMAX<'x, c32>;
pub type IZAMAX<'x> = IAMAX<'x, c64>;

impl<'x, F> IAMAX<'x, F>
where
    F: IAMAXNum,
{
    pub fn run(self) -> Result<usize, BLASError> {
        self.build()?.driver()?.run_blas()
    }
}

/* #endregion */
