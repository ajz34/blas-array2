use crate::util::{AnyError, BLASError};

/// Marker struct of BLAS functions.
///
/// This struct will be implemented in modules of each function.
pub struct BLASFunc {}

/// Trait for BLAS functions.
pub trait StructBLAS {
    // `optional, depend`
    fn init_optional(&mut self) -> Result<(), AnyError>;
    // `intent(hide)`
    fn init_hidden(&mut self) -> Result<(), AnyError>;
    // `check`
    fn check(&self) -> Result<(), AnyError>;
    fn runnable(&self) -> bool;
    fn initialize(&mut self) -> Result<(), AnyError> {
        println!("runnable: {:?}", self.runnable());
        if !self.runnable() {
            return Err(BLASError(
                "Current BLAS is not runnable. This struct may have execuated once and shouldn't be execuated anymore."
                    .to_string(),
            )
            .into());
        }
        self.init_optional()?;
        self.init_hidden()?;
        self.check()?;
        Ok(())
    }
}

use blas_sys::{c_double_complex, c_float_complex};
use libc::{c_double, c_float};
use num_complex::Complex;
use num_traits::{Num, NumAssignOps};

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Trait for defining real part float types
pub trait BLASFloat:
    Num + NumAssignOps + Send + Sync + Copy + Clone + Default + std::fmt::Debug + std::fmt::Display + 'static
{
    type RealFloat: Num;
    type FFIFloat;
}

impl BLASFloat for f32 {
    type RealFloat = f32;
    type FFIFloat = c_float;
}

impl BLASFloat for f64 {
    type RealFloat = f64;
    type FFIFloat = c_double;
}

impl BLASFloat for c32 {
    type RealFloat = f32;
    type FFIFloat = c_float_complex;
}

impl BLASFloat for c64 {
    type RealFloat = f64;
    type FFIFloat = c_double_complex;
}

/// Trait marker for complex symmetry (whether it is symmetric or hermitian)
pub trait BLASSymm {}

/// Struct marker for symmetric matrix
pub struct BLASSymmetric {}
impl BLASSymm for BLASSymmetric {}

/// Struct marker for hermitian matrix
pub struct BLASHermitian {}
impl BLASSymm for BLASHermitian {}
