use derive_builder::Builder;
use libc::c_int;
use ndarray::prelude::*;
use blas_sys;
use crate::util::*;

/* #region FFI binding */

pub trait ASUMFunc<F>
where
    F: FloatType
{
    fn asum(n: *const c_int, x: *const F, incx: *const c_int) -> F::RealFloat;
}

macro_rules! impl_subroutine {
    ($type:ty, $func:ident) => {

impl ASUMFunc<$type> for BLASFunc<$type>
where
    $type: FloatType
{
    fn asum(n: *const c_int, x: *const $type, incx: *const c_int) -> <$type as FloatType>::RealFloat {
        unsafe { blas_sys::$func(n, x as *const <$type as FloatType>::FFIFloat, incx) }
    }
}

    };
}

impl_subroutine!(f32, sasum_);
impl_subroutine!(f64, dasum_);
impl_subroutine!(c32, scasum_);
impl_subroutine!(c64, dzasum_);

/* #endregion */

/* #region Struct */

#[derive(Builder)]
pub struct ASUM<'a, F> {

    pub x: ArrayView1<'a, F>,

    #[builder(setter(skip))]
    n: c_int,

    #[builder(setter(skip))]
    incx: c_int,

    #[builder(private, default = "true")]
    flag_runnable: bool,
}

impl<'a, F> StructBLAS for ASUM<'a, F> {

    fn init_hidden(&mut self) {
        self.n = self.x.len() as c_int;
        self.incx = self.x.stride_of(Axis(0)) as c_int;
    }

    fn init_optional(&mut self) {}

    fn check(&self) -> Result<(), String> { Ok(()) }

    fn runnable(&self) -> bool { self.flag_runnable }
}

impl<'a, F> ASUM<'a, F>
where
    F: FloatType
{

    pub fn run(&mut self) -> Result<<F as FloatType>::RealFloat, String>
    where
        BLASFunc<F>: ASUMFunc<F>
    {
        self.initialize()?;
        self.flag_runnable = false;

        let n = &self.n as *const c_int;
        let x = self.x.as_ptr();
        let incx = &self.incx as *const c_int;
        Ok(BLASFunc::<F>::asum(n, x, incx))
    }
}

pub type SASUM<'a> = ASUM<'a, f32>;
pub type DASUM<'a> = ASUM<'a, f64>;
pub type SCASUM<'a> = ASUM<'a, c32>;
pub type DZASUM<'a> = ASUM<'a, c64>;

/* #endregion */