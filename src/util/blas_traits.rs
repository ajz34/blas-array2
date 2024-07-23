use crate::util::*;
use ndarray::Dimension;
use num_complex::*;
use num_traits::*;

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Trait for defining real part float types
pub trait BLASFloat: Num + Copy {
    type RealFloat: BLASFloat;
    fn is_complex() -> bool;
    fn conj(x: Self) -> Self;
    fn from_real(x: Self::RealFloat) -> Self;
}

impl BLASFloat for f32 {
    type RealFloat = f32;
    #[inline]
    fn is_complex() -> bool {
        false
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x
    }
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        x
    }
}

impl BLASFloat for f64 {
    type RealFloat = f64;
    #[inline]
    fn is_complex() -> bool {
        false
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x
    }
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        x
    }
}

impl BLASFloat for c32 {
    type RealFloat = f32;
    #[inline]
    fn is_complex() -> bool {
        true
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x.conj()
    }
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        c32::new(x, 0.0)
    }
}

impl BLASFloat for c64 {
    type RealFloat = f64;
    #[inline]
    fn is_complex() -> bool {
        true
    }
    #[inline]
    fn conj(x: Self) -> Self {
        x.conj()
    }
    #[inline]
    fn from_real(x: Self::RealFloat) -> Self {
        c64::new(x, 0.0)
    }
}

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
        assert_eq!(<f32 as BLASFloat>::from_real(x), x);
    }

    #[test]
    fn test_f64_blasfloat() {
        let x = 3.0_f64;
        assert_eq!(<f64 as BLASFloat>::is_complex(), false);
        assert_eq!(<f64 as BLASFloat>::conj(x), x);
        assert_eq!(<f64 as BLASFloat>::from_real(x), x);
    }

    #[test]
    fn test_c32_blasfloat() {
        let x = Complex::new(3.0_f32, 4.0_f32);
        assert_eq!(<c32 as BLASFloat>::is_complex(), true);
        assert_eq!(<c32 as BLASFloat>::conj(x), x.conj());
        assert_eq!(<c32 as BLASFloat>::from_real(3.0_f32), Complex::new(3.0_f32, 0.0_f32));
    }

    #[test]
    fn test_c64_blasfloat() {
        let x = Complex::new(3.0_f64, 4.0_f64);
        assert_eq!(<c64 as BLASFloat>::is_complex(), true);
        assert_eq!(<c64 as BLASFloat>::conj(x), x.conj());
        assert_eq!(<c64 as BLASFloat>::from_real(3.0_f64), Complex::new(3.0_f64, 0.0_f64));
    }
}
