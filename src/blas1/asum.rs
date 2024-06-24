use derive_builder::Builder;
use libc::c_int;
use ndarray::prelude::*;
use blas_sys;
use crate::util::*;

pub trait ASUMFunc<F>
where
    F: BLASFloat
{
    fn asum(n: *const c_int, x: *const F, incx: *const c_int) -> F::RealFloat;
}

macro_rules! impl_subroutine {
    ($type:ty, $func:ident) => {

impl ASUMFunc<$type> for BLASFunc<$type>
where
    $type: BLASFloat
{
    fn asum(n: *const c_int, x: *const $type, incx: *const c_int) -> <$type as BLASFloat>::RealFloat {
        unsafe { blas_sys::$func(n, x as *const <$type as BLASFloat>::FFIFloat, incx) }
    }
}

    };
}

impl_subroutine!(f32, sasum_);
impl_subroutine!(f64, dasum_);
impl_subroutine!(c32, scasum_);
impl_subroutine!(c64, dzasum_);

#[derive(Builder)]
pub struct ASUM_<'a, F> {

    pub x: ArrayView1<'a, F>,

    #[builder(setter(skip))]
    n: c_int,

    #[builder(setter(skip))]
    incx: c_int,

    #[builder(private, default = "true")]
    flag_runnable: bool,
}

impl<'a, F> StructBLAS for ASUM_<'a, F> {

    fn init_hidden(&mut self) -> Result<(), AnyError> {
        self.n = self.x.len().try_into()?;
        self.incx = self.x.stride_of(Axis(0)).try_into()?;
        Ok(())
    }

    fn init_optional(&mut self) -> Result<(), AnyError> { Ok(()) }

    fn check(&self) -> Result<(), AnyError> { Ok(()) }

    fn runnable(&self) -> bool { self.flag_runnable }
}

impl<'a, F> ASUM_<'a, F>
where
    F: BLASFloat
{

    pub fn run(&mut self) -> Result<<F as BLASFloat>::RealFloat, AnyError>
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

pub type ASUM<'a, F> = ASUM_Builder<'a, F>;

pub type SASUM<'a> = ASUM_<'a, f32>;
pub type DASUM<'a> = ASUM_<'a, f64>;
pub type SCASUM<'a> = ASUM_<'a, c32>;
pub type DZASUM<'a> = ASUM_<'a, c64>;
