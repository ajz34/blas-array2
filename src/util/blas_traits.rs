use crate::util::*;
use ndarray::Dimension;
use num_complex::*;
use num_traits::*;

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

// for legacy compatibility
// use blas_sys::{c_double_complex, c_float_complex};

#[allow(bad_style)]
pub type c_double_complex = [f64; 2];
#[allow(bad_style)]
pub type c_float_complex = [f32; 2];

/// Trait for defining real part float types
pub trait BLASFloat:
    Num + NumAssignOps + Send + Sync + Copy + Clone + Default + core::fmt::Debug + core::fmt::Display
{
    type RealFloat: BLASFloat;
    type FFIFloat;
    const EPSILON: Self::RealFloat;
    fn is_complex() -> bool;
    fn conj(x: Self) -> Self;
    fn abs(x: Self) -> Self::RealFloat;
}

impl BLASFloat for f32 {
    type RealFloat = f32;
    type FFIFloat = f32;
    const EPSILON: Self::RealFloat = f32::EPSILON;
    #[inline]
    fn is_complex() -> bool {
        false
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x
    }
    #[inline]
    fn abs(x: Self) -> Self::RealFloat {
        x.abs()
    }
}

impl BLASFloat for f64 {
    type RealFloat = f64;
    type FFIFloat = f64;
    const EPSILON: Self::RealFloat = f64::EPSILON;
    #[inline]
    fn is_complex() -> bool {
        false
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x
    }
    #[inline]
    fn abs(x: Self) -> Self::RealFloat {
        x.abs()
    }
}

impl BLASFloat for c32 {
    type RealFloat = f32;
    type FFIFloat = c_float_complex;
    const EPSILON: Self::RealFloat = f32::EPSILON;
    #[inline]
    fn is_complex() -> bool {
        true
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x.conj()
    }
    #[inline]
    fn abs(x: Self) -> Self::RealFloat {
        x.abs()
    }
}

impl BLASFloat for c64 {
    type RealFloat = f64;
    type FFIFloat = c_double_complex;
    const EPSILON: Self::RealFloat = f64::EPSILON;
    #[inline]
    fn is_complex() -> bool {
        true
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x.conj()
    }
    #[inline]
    fn abs(x: Self) -> Self::RealFloat {
        x.abs()
    }
}

/// Trait marker for complex symmetry (whether it is symmetric or hermitian)
pub trait BLASSymmetric {
    type Float: BLASFloat;
    type HermitianFloat: BLASFloat;
    fn is_hermitian() -> bool;
}

/// Struct marker for symmetric matrix
pub struct BLASSymm<F>
where
    F: BLASFloat,
{
    _phantom: core::marker::PhantomData<F>,
}

impl<F> BLASSymmetric for BLASSymm<F>
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
pub struct BLASHermi<F>
where
    F: BLASFloat,
{
    _phantom: core::marker::PhantomData<F>,
}

impl<F> BLASSymmetric for BLASHermi<F>
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
    fn run_blas(self) -> Result<ArrayOut<'c, F, D>, BLASError>;
}

/// Trait for BLAS builder prototypes
pub trait BLASBuilder_<'c, F, D>
where
    D: Dimension,
{
    fn driver(self) -> Result<impl BLASDriver<'c, F, D>, BLASError>;
}

pub trait BLASBuilder<'c, F, D>
where
    D: Dimension,
{
    fn run(self) -> Result<ArrayOut<'c, F, D>, BLASError>;
}
