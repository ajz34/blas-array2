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
            return Err(BLASError(
                "Current BLAS is not runnable. This struct may have execuated once and shouldn't be execuated anymore.".to_string()
            ).into());
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

/* #region BLASFloat */

/// Trait for defining real part float types
pub trait BLASFloat: Num + Send + Sync + Copy + Clone + Default + std::fmt::Debug + std::fmt::Display + 'static {
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

/* #endregion BLASFloat */

/* #region Enum for output array */

use ndarray::prelude::*;

#[derive(Debug)]
pub enum ArrayOut<'a, F, D>
where
    D: Dimension,
    F: BLASFloat,
{
    ViewMut(ArrayViewMut<'a, F, D>),
    Owned(Array<F, D>),
}

impl<'a, F, D> ArrayOut<'a, F, D>
where
    D: Dimension,
    F: BLASFloat,
{
    pub fn view_mut(&'a mut self) -> ArrayViewMut<'a, F, D> {
        match self {
            Self::ViewMut(arr) => arr.view_mut(),
            Self::Owned(arr) => arr.view_mut(),
        }
    }

    pub fn into_owned(self) -> Array<F, D> {
        match self {
            Self::ViewMut(arr) => arr.to_owned(),
            Self::Owned(arr) => arr,
        }
    }

    pub fn is_view_mut(&mut self) -> bool {
        match self {
            Self::ViewMut(_) => true,
            Self::Owned(_) => false,
        }
    }

    pub fn is_owned(&mut self) -> bool {
        match self {
            Self::ViewMut(_) => false,
            Self::Owned(_) => true,
        }
    }
}

pub type ArrayOut1<'a, F> = ArrayOut<'a, F, Ix1>;
pub type ArrayOut2<'a, F> = ArrayOut<'a, F, Ix2>;
pub type ArrayOut3<'a, F> = ArrayOut<'a, F, Ix3>;

/* #endregion */

/* #region Enum for BLAS flags */

use libc::c_char;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASLayout {
    #[default]
    Undefined = -1,
    RowMajor = 101,
    ColMajor = 102,
    // extension of current crate
    Sequential = 103,
    NonContiguous = 104,
}

pub type BALSOrder = BLASLayout;

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

impl TryFrom<char> for BLASLayout {
    type Error = BLASError;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'R' => Ok(BLASLayout::RowMajor),
            'C' => Ok(BLASLayout::ColMajor),
            _ => Err(BLASError(format!("Invalid character for BLASOrder: {}", c))),
        }
    }
}

impl TryFrom<BLASLayout> for char {
    type Error = BLASError;
    fn try_from(o: BLASLayout) -> Result<Self, Self::Error> {
        match o {
            BLASLayout::RowMajor => Ok('R'),
            BLASLayout::ColMajor => Ok('C'),
            _ => Err(BLASError(format!("Invalid BLASOrder: {:?}", o))),
        }
    }
}

impl TryFrom<BLASLayout> for c_char {
    type Error = BLASError;
    fn try_from(o: BLASLayout) -> Result<Self, Self::Error> {
        match o {
            BLASLayout::RowMajor => Ok('R' as c_char),
            BLASLayout::ColMajor => Ok('C' as c_char),
            _ => Err(BLASError(format!("Invalid BLASOrder: {:?}", o))),
        }
    }
}

impl TryFrom<char> for BLASTrans {
    type Error = BLASError;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'N' => Ok(BLASTrans::NoTrans),
            'T' => Ok(BLASTrans::Trans),
            'C' => Ok(BLASTrans::ConjTrans),
            _ => Err(BLASError(format!("Invalid character for BLASTrans: {}", c))),
        }
    }
}

impl TryFrom<BLASTrans> for char {
    type Error = BLASError;
    fn try_from(t: BLASTrans) -> Result<Self, Self::Error> {
        match t {
            BLASTrans::NoTrans => Ok('N'),
            BLASTrans::Trans => Ok('T'),
            BLASTrans::ConjTrans => Ok('C'),
            _ => Err(BLASError(format!("Invalid BLASTrans: {:?}", t))),
        }
    }
}

impl TryFrom<BLASTrans> for c_char {
    type Error = BLASError;
    fn try_from(t: BLASTrans) -> Result<Self, Self::Error> {
        match t {
            BLASTrans::NoTrans => Ok('N' as c_char),
            BLASTrans::Trans => Ok('T' as c_char),
            BLASTrans::ConjTrans => Ok('C' as c_char),
            _ => Err(BLASError(format!("Invalid BLASTrans: {:?}", t))),
        }
    }
}

impl TryFrom<char> for BLASUpLo {
    type Error = BLASError;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'U' => Ok(BLASUpLo::Upper),
            'L' => Ok(BLASUpLo::Lower),
            _ => Err(BLASError(format!("Invalid character for BLASUpLo: {}", c))),
        }
    }
}

impl TryFrom<BLASUpLo> for char {
    type Error = BLASError;
    fn try_from(u: BLASUpLo) -> Result<Self, Self::Error> {
        match u {
            BLASUpLo::Upper => Ok('U'),
            BLASUpLo::Lower => Ok('L'),
            _ => Err(BLASError(format!("Invalid BLASUpLo: {:?}", u))),
        }
    }
}

impl TryFrom<BLASUpLo> for c_char {
    type Error = BLASError;
    fn try_from(u: BLASUpLo) -> Result<Self, Self::Error> {
        match u {
            BLASUpLo::Upper => Ok('U' as c_char),
            BLASUpLo::Lower => Ok('L' as c_char),
            _ => Err(BLASError(format!("Invalid BLASUpLo: {:?}", u))),
        }
    }
}

impl TryFrom<char> for BLASDiag {
    type Error = BLASError;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'N' => Ok(BLASDiag::NonUnit),
            'U' => Ok(BLASDiag::Unit),
            _ => Err(BLASError(format!("Invalid character for BLASDiag: {}", c))),
        }
    }
}

impl TryFrom<BLASDiag> for char {
    type Error = BLASError;
    fn try_from(d: BLASDiag) -> Result<Self, Self::Error> {
        match d {
            BLASDiag::NonUnit => Ok('N'),
            BLASDiag::Unit => Ok('U'),
            _ => Err(BLASError(format!("Invalid BLASDiag: {:?}", d))),
        }
    }
}

impl TryFrom<BLASDiag> for c_char {
    type Error = BLASError;
    fn try_from(d: BLASDiag) -> Result<Self, Self::Error> {
        match d {
            BLASDiag::NonUnit => Ok('N' as c_char),
            BLASDiag::Unit => Ok('U' as c_char),
            _ => Err(BLASError(format!("Invalid BLASDiag: {:?}", d))),
        }
    }
}

impl TryFrom<char> for BLASSide {
    type Error = BLASError;
    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'L' => Ok(BLASSide::Left),
            'R' => Ok(BLASSide::Right),
            _ => Err(BLASError(format!("Invalid character for BLASSide: {}", c))),
        }
    }
}

impl TryFrom<BLASSide> for char {
    type Error = BLASError;
    fn try_from(s: BLASSide) -> Result<Self, Self::Error> {
        match s {
            BLASSide::Left => Ok('L'),
            BLASSide::Right => Ok('R'),
            _ => Err(BLASError(format!("Invalid BLASSide: {:?}", s))),
        }
    }
}

impl TryFrom<BLASSide> for c_char {
    type Error = BLASError;
    fn try_from(s: BLASSide) -> Result<Self, Self::Error> {
        match s {
            BLASSide::Left => Ok('L' as c_char),
            BLASSide::Right => Ok('R' as c_char),
            _ => Err(BLASError(format!("Invalid BLASSide: {:?}", s))),
        }
    }
}

unsafe impl Send for BLASLayout {}
unsafe impl Send for BLASTrans {}
unsafe impl Send for BLASUpLo {}
unsafe impl Send for BLASDiag {}
unsafe impl Send for BLASSide {}

unsafe impl Sync for BLASLayout {}
unsafe impl Sync for BLASTrans {}
unsafe impl Sync for BLASUpLo {}
unsafe impl Sync for BLASDiag {}
unsafe impl Sync for BLASSide {}

impl BLASLayout {
    pub fn is_cpref(&self) -> bool {
        match self {
            BLASLayout::RowMajor => true,
            BLASLayout::Sequential => true,
            _ => false,
        }
    }

    pub fn is_fpref(&self) -> bool {
        match self {
            BLASLayout::ColMajor => true,
            BLASLayout::Sequential => true,
            _ => false,
        }
    }
}

/* #endregion */

/* #region Error handling */

pub type AnyError = Box<dyn std::error::Error>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BLASError(pub String);

impl std::error::Error for BLASError {}

impl BLASError {
    pub fn assert(cond: bool, s: String) -> Result<(), BLASError> {
        match cond {
            true => Ok(()),
            false => Err(BLASError(s)),
        }
    }
}

impl std::fmt::Display for BLASError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/* #endregion */

/* #region Strides */

pub fn get_layout_array2<F>(arr: &ArrayView2<F>) -> BLASLayout
{
    // Note that this only shows order of matrix (dimension information)
    // not c/f-contiguous (memory layout)
    // So some sequential (both c/f-contiguous) cases may be considered as only row or col major
    // Examples:
    // RowMajor     ==>   shape=[1, 4], strides=[0, 1], layout=CFcf (0xf)
    // ColMajor     ==>   shape=[4, 1], strides=[1, 0], layout=CFcf (0xf)
    // Sequential   ==>   shape=[1, 1], strides=[0, 0], layout=CFcf (0xf)
    // NonContig    ==>   shape=[4, 1], strides=[10, 0], layout=Custom (0x0)
    let d0 = arr.dim().0;
    let d1 = arr.dim().1;
    let s0 = arr.strides()[0];
    let s1 = arr.strides()[1];
    if d0 == 0 || d1 == 0 {
        // empty array
        return BLASLayout::Sequential;
    }
    else if d0 == 1 && d1 == 1 {
        // one element
        return BLASLayout::Sequential;
    } else if s1 == 1 {
        // row-major
        return BLASLayout::RowMajor;
    } else if s0 == 1 {
        // col-major
        return BLASLayout::ColMajor;
    } else {
        // non-contiguous
        return BLASLayout::NonContiguous;
    }
}

/* #endregion */