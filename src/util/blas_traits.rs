use crate::util::*;
use ndarray::Dimension;
use num_complex::*;
use num_traits::*;

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

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
    fn from_real(x: Self::RealFloat) -> Self;
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
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        x
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
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        x
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
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        c32::new(x, 0.0)
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
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        c64::new(x, 0.0)
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

// Following test is assisted by DeepSeek
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_blasfloat() {
        let x = 3.0_f32;
        assert_eq!(<f32 as BLASFloat>::is_complex(), false);
        assert_eq!(<f32 as BLASFloat>::conj(x), x);
        assert_eq!(<f32 as BLASFloat>::abs(x), x);
        assert_eq!(<f32 as BLASFloat>::from_real(x), x);
        assert_eq!(<f32 as BLASFloat>::EPSILON, f32::EPSILON);
    }

    #[test]
    fn test_f64_blasfloat() {
        let x = 3.0_f64;
        assert_eq!(<f64 as BLASFloat>::is_complex(), false);
        assert_eq!(<f64 as BLASFloat>::conj(x), x);
        assert_eq!(<f64 as BLASFloat>::abs(x), x);
        assert_eq!(<f64 as BLASFloat>::from_real(x), x);
        assert_eq!(<f64 as BLASFloat>::EPSILON, f64::EPSILON);
    }

    #[test]
    fn test_c32_blasfloat() {
        let x = Complex::new(3.0_f32, 4.0_f32);
        assert_eq!(<c32 as BLASFloat>::is_complex(), true);
        assert_eq!(<c32 as BLASFloat>::conj(x), x.conj());
        assert_eq!(<c32 as BLASFloat>::abs(x), x.abs());
        assert_eq!(<c32 as BLASFloat>::from_real(3.0_f32), Complex::new(3.0_f32, 0.0_f32));
        assert_eq!(<c32 as BLASFloat>::EPSILON, f32::EPSILON);
    }

    #[test]
    fn test_c64_blasfloat() {
        let x = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(<c64 as BLASFloat>::is_complex(), true);
        assert_eq!(<c64 as BLASFloat>::conj(x), x.conj());
        assert_eq!(<c64 as BLASFloat>::abs(x), x.abs());
        assert_eq!(<c64 as BLASFloat>::from_real(3.0_f64), Complex::new(3.0_f64, 0.0_f64));
        assert_eq!(<c64 as BLASFloat>::EPSILON, f64::EPSILON);
    }
}
