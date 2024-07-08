use crate::util::*;

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
use ndarray::Dimension;
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
    type RealFloat: BLASFloat;
    type FFIFloat;
    fn is_complex() -> bool;
    fn conj(x: Self) -> Self;
}

impl BLASFloat for f32 {
    type RealFloat = f32;
    type FFIFloat = c_float;
    #[inline]
    fn is_complex() -> bool {
        false
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x
    }
}

impl BLASFloat for f64 {
    type RealFloat = f64;
    type FFIFloat = c_double;
    #[inline]
    fn is_complex() -> bool {
        false
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x
    }
}

impl BLASFloat for c32 {
    type RealFloat = f32;
    type FFIFloat = c_float_complex;
    #[inline]
    fn is_complex() -> bool {
        true
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x.conj()
    }
}

impl BLASFloat for c64 {
    type RealFloat = f64;
    type FFIFloat = c_double_complex;
    #[inline]
    fn is_complex() -> bool {
        true
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x.conj()
    }
}

/// Trait marker for complex symmetry (whether it is symmetric or hermitian)
pub trait BLASSymm {
    type Float: BLASFloat;
    type HermitianFloat: BLASFloat;
    fn is_hermitian() -> bool;
}

/// Struct marker for symmetric matrix
pub struct BLASSymmetric<F>
where
    F: BLASFloat,
{
    _phantom: std::marker::PhantomData<F>,
}

impl<F> BLASSymm for BLASSymmetric<F>
where
    F: BLASFloat,
{
    type Float = F;
    type HermitianFloat = F;
    #[inline]
    fn is_hermitian() -> bool {
        false
    }
}

/// Struct marker for hermitian matrix
pub struct BLASHermitian<F>
where
    F: BLASFloat,
{
    _phantom: std::marker::PhantomData<F>,
}

impl<F> BLASSymm for BLASHermitian<F>
where
    F: BLASFloat,
{
    type Float = F;
    type HermitianFloat = <F as BLASFloat>::RealFloat;
    #[inline]
    fn is_hermitian() -> bool {
        true
    }
}

/// Marker struct of BLAS functions.
///
/// This struct will be implemented in modules of each function.
pub struct BLASFunc {}

/// Trait for BLAS drivers
pub trait BLASDriver<'c, F, D>
where
    D: Dimension,
{
    fn run(self) -> Result<ArrayOut<'c, F, D>, AnyError>;
}

/// Trait for BLAS builder prototypes
pub trait BLASBuilder_<'c, F, D>
where
    D: Dimension,
{
    fn driver(self) -> Result<impl BLASDriver<'c, F, D>, AnyError>;
}

pub trait BLASBuilder<'c, F, D>
where
    D: Dimension,
{
    fn run(self) -> Result<ArrayOut<'c, F, D>, AnyError>;
}
