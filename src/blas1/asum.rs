use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::c_int;
use ndarray::prelude::*;
use num_traits::Zero;

/* #region BLAS func */

pub trait ASUMFunc<F>
where
    F: BLASFloat,
{
    unsafe fn asum(n: *const c_int, x: *const F, incx: *const c_int) -> F::RealFloat;
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl ASUMFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn asum(
                n: *const c_int,
                x: *const $type,
                incx: *const c_int,
            ) -> <$type as BLASFloat>::RealFloat {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(n, x as *const FFIFloat, incx)
            }
        }
    };
}

impl_func!(f32, sasum_);
impl_func!(f64, dasum_);
impl_func!(c32, scasum_);
impl_func!(c64, dzasum_);

/* #endregion */

/* #region BLAS driver */

pub struct ASUM_Driver<'x, F>
where
    F: BLASFloat,
{
    n: c_int,
    x: ArrayView1<'x, F>,
    incx: c_int,
}

impl<'x, F> ASUM_Driver<'x, F>
where
    F: BLASFloat,
    F::RealFloat: Zero,
    BLASFunc: ASUMFunc<F>,
{
    pub fn run_blas(self) -> Result<F::RealFloat, AnyError> {
        let Self { n, x, incx } = self;
        let x_ptr = x.as_ptr();
        if n == 0 {
            return Ok(F::RealFloat::zero());
        } else {
            return unsafe { Ok(BLASFunc::asum(&n, x_ptr, &incx)) };
        }
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct ASUM_<'x, F>
where
    F: BLASFloat,
{
    pub x: ArrayView1<'x, F>,
}

impl<'x, F> ASUM_<'x, F>
where
    F: BLASFloat,
    BLASFunc: ASUMFunc<F>,
{
    pub fn driver(self) -> Result<ASUM_Driver<'x, F>, AnyError> {
        let Self { x } = self;
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));
        let driver = ASUM_Driver { n: n.try_into()?, x, incx: incx.try_into()? };
        return Ok(driver);
    }
}

/* #region BLAS wrapper */

pub type ASUM<'x, F> = ASUM_Builder<'x, F>;
pub type SASUM<'x> = ASUM<'x, f32>;
pub type DASUM<'x> = ASUM<'x, f64>;
pub type SCASUM<'x> = ASUM<'x, c32>;
pub type DZASUM<'x> = ASUM<'x, c64>;

impl<'x, F> ASUM<'x, F>
where
    F: BLASFloat,
    BLASFunc: ASUMFunc<F>,
{
    pub fn run(self) -> Result<F::RealFloat, AnyError> {
        self.build()?.driver()?.run_blas()
    }
}

/* #endregion */