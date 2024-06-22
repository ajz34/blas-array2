/// BLAS functions for a given floating point type `F`.
/// 
/// This struct will be implemented in modules of each function.
pub struct BLASFunc<F> {
    _marker: std::marker::PhantomData<F>,
}

/// Trait for BLAS functions.
pub trait StructBLAS {
    // `optional, depend`
    fn init_optional(&mut self);
    // `intent(hide)`
    fn init_hidden(&mut self);
    // `check`
    fn check(&self) -> Result<(), String>;
    fn runnable(&self) -> bool;
    fn initialize(&mut self) -> Result<(), String> {
        if !self.runnable() { return Err(
            "Current BLAS is not runnable. This struct may have execuated once and shouldn't be execuated anymore.".into()); }
        self.init_optional();
        self.init_hidden();
        self.check()
    }
}

use blas_sys::{c_float_complex, c_double_complex};
use libc::{c_float, c_double};
use num_complex::Complex;

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Trait for defining real part float types
pub trait FloatType: Default + Copy + Clone + Send + Sync + std::fmt::Debug + 'static {
    type RealFloat;
    type FFIFloat;
}

impl FloatType for f32 {
    type RealFloat = f32;
    type FFIFloat = c_float;
}

impl FloatType for f64 {
    type RealFloat = f64;
    type FFIFloat = c_double;
}

impl FloatType for c32 {
    type RealFloat = f32;
    type FFIFloat = c_float_complex;
}

impl FloatType for c64 {
    type RealFloat = f64;
    type FFIFloat = c_double_complex;
}
