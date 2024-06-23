/// BLAS functions for a given floating point type `F`.
/// 
/// This struct will be implemented in modules of each function.
pub struct BLASFunc<F> {
    _marker: std::marker::PhantomData<F>,
}

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
            return Err(Box::new(BLASError(
                "Current BLAS is not runnable. This struct may have execuated once and shouldn't be execuated anymore.".to_string()
            )));
        }
        self.init_optional()?;
        self.init_hidden()?;
        self.check()?;
        Ok(())
    }
}

use blas_sys::{c_float_complex, c_double_complex};
use libc::{c_float, c_double};
use num_complex::Complex;
use num_traits::Num;

#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

/// Trait for defining real part float types
pub trait FloatType: Num + Send + Sync {
    type RealFloat: Num;
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

/* #region Enum for output array */

use ndarray::prelude::*;

pub enum ArrayOut<'a, F, D> {
    View(ArrayViewMut<'a, F, D>),
    Own(Array<F, D>),
}

pub type ArrayOut1<'a, F> = ArrayOut<'a, F, Ix1>;
pub type ArrayOut2<'a, F> = ArrayOut<'a, F, Ix2>;
pub type ArrayOut3<'a, F> = ArrayOut<'a, F, Ix3>;

/* #endregion */

/* #region Enum for BLAS flags */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASOrder {
    #[default]
    Undefined = -1,
    RowMajor = 101,
    ColMajor = 102,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASTrans {
    #[default]
    Undefined = -1,
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    ConjNoTrans = 114,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASUpLo {
    #[default]
    Undefined = -1,
    Upper = 121,
    Lower = 122,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASDiag {
    #[default]
    Undefined = -1,
    NonUnit = 131,
    Unit = 132,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASSide {
    #[default]
    Undefined = -1,
    Left = 141,
    Right = 142,
}

impl TryFrom<char> for BLASOrder {
    type Error = String;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'R' => Ok(BLASOrder::RowMajor),
            'C' => Ok(BLASOrder::ColMajor),
            _ => Err(format!("Invalid character for BLASOrder: {}", c)),
        }
    }
}

impl TryFrom<char> for BLASTrans {
    type Error = String;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'N' => Ok(BLASTrans::NoTrans),
            'T' => Ok(BLASTrans::Trans),
            'C' => Ok(BLASTrans::ConjTrans),
            _ => Err(format!("Invalid character for BLASTrans: {}", c)),
        }
    }
}

impl TryFrom<char> for BLASUpLo {
    type Error = String;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'U' => Ok(BLASUpLo::Upper),
            'L' => Ok(BLASUpLo::Lower),
            _ => Err(format!("Invalid character for BLASUpLo: {}", c)),
        }
    }
}

impl TryFrom<char> for BLASDiag {
    type Error = String;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'N' => Ok(BLASDiag::NonUnit),
            'U' => Ok(BLASDiag::Unit),
            _ => Err(format!("Invalid character for BLASDiag: {}", c)),
        }
    }
}

impl TryFrom<char> for BLASSide {
    type Error = String;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'L' => Ok(BLASSide::Left),
            'R' => Ok(BLASSide::Right),
            _ => Err(format!("Invalid character for BLASSide: {}", c)),
        }
    }
}

unsafe impl Send for BLASOrder {}
unsafe impl Send for BLASTrans {}
unsafe impl Send for BLASUpLo {}
unsafe impl Send for BLASDiag {}
unsafe impl Send for BLASSide {}

unsafe impl Sync for BLASOrder {}
unsafe impl Sync for BLASTrans {}
unsafe impl Sync for BLASUpLo {}
unsafe impl Sync for BLASDiag {}
unsafe impl Sync for BLASSide {}

/* #endregion */

/* #region Convenient */

pub type AnyError = Box<dyn std::error::Error>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BLASError(String);

impl std::error::Error for BLASError {}

impl std::fmt::Display for BLASError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/* #endregion */

