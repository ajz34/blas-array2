use crate::ffi::c_char;

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
pub use BLASLayout::{ColMajor as BLASColMajor, RowMajor as BLASRowMajor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASTranspose {
    #[default]
    Undefined = -1,
    NoTrans = 111,
    Trans = 112,
    ConjTrans = 113,
    ConjNoTrans = 114,
}

pub use BLASTranspose::{ConjTrans as BLASConjTrans, NoTrans as BLASNoTrans, Trans as BLASTrans};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASUpLo {
    #[default]
    Undefined = -1,
    Upper = 121,
    Lower = 122,
}

pub use BLASUpLo::{Lower as BLASLower, Upper as BLASUpper};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASDiag {
    #[default]
    Undefined = -1,
    NonUnit = 131,
    Unit = 132,
}

pub use BLASDiag::{NonUnit as BLASNonUnit, Unit as BLASUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BLASSide {
    #[default]
    Undefined = -1,
    Left = 141,
    Right = 142,
}

pub use BLASSide::{Left as BLASLeft, Right as BLASRight};

use super::{blas_invalid, BLASError};

impl From<char> for BLASLayout {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'R' => BLASRowMajor,
            'C' => BLASColMajor,
            _ => Self::Undefined,
        }
    }
}

impl TryFrom<BLASLayout> for char {
    type Error = BLASError;
    #[inline]
    fn try_from(layout: BLASLayout) -> Result<Self, BLASError> {
        match layout {
            BLASRowMajor => Ok('R'),
            BLASColMajor => Ok('C'),
            _ => blas_invalid!(layout),
        }
    }
}

impl TryFrom<BLASLayout> for c_char {
    type Error = BLASError;
    #[inline]
    fn try_from(layout: BLASLayout) -> Result<Self, BLASError> {
        match layout {
            BLASRowMajor => Ok('R' as c_char),
            BLASColMajor => Ok('C' as c_char),
            _ => blas_invalid!(layout),
        }
    }
}

impl From<char> for BLASTranspose {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'N' => BLASNoTrans,
            'T' => BLASTrans,
            'C' => BLASConjTrans,
            _ => Self::Undefined,
        }
    }
}

impl TryFrom<BLASTranspose> for char {
    type Error = BLASError;
    #[inline]
    fn try_from(trans: BLASTranspose) -> Result<Self, Self::Error> {
        match trans {
            BLASNoTrans => Ok('N'),
            BLASTrans => Ok('T'),
            BLASConjTrans => Ok('C'),
            _ => blas_invalid!(trans),
        }
    }
}

impl TryFrom<BLASTranspose> for c_char {
    type Error = BLASError;
    #[inline]
    fn try_from(trans: BLASTranspose) -> Result<Self, Self::Error> {
        match trans {
            BLASNoTrans => Ok('N' as c_char),
            BLASTrans => Ok('T' as c_char),
            BLASConjTrans => Ok('C' as c_char),
            _ => blas_invalid!(trans),
        }
    }
}

impl From<char> for BLASUpLo {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'U' => BLASUpper,
            'L' => BLASLower,
            _ => Self::Undefined,
        }
    }
}

impl TryFrom<BLASUpLo> for char {
    type Error = BLASError;
    #[inline]
    fn try_from(uplo: BLASUpLo) -> Result<Self, BLASError> {
        match uplo {
            BLASUpper => Ok('U'),
            BLASLower => Ok('L'),
            _ => blas_invalid!(uplo),
        }
    }
}

impl TryFrom<BLASUpLo> for c_char {
    type Error = BLASError;
    #[inline]
    fn try_from(uplo: BLASUpLo) -> Result<Self, Self::Error> {
        match uplo {
            BLASUpper => Ok('U' as c_char),
            BLASLower => Ok('L' as c_char),
            _ => blas_invalid!(uplo),
        }
    }
}

impl From<char> for BLASDiag {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'N' => BLASNonUnit,
            'U' => BLASUnit,
            _ => Self::Undefined,
        }
    }
}

impl TryFrom<BLASDiag> for char {
    type Error = BLASError;
    #[inline]
    fn try_from(diag: BLASDiag) -> Result<Self, Self::Error> {
        match diag {
            BLASNonUnit => Ok('N'),
            BLASUnit => Ok('U'),
            _ => blas_invalid!(diag),
        }
    }
}

impl TryFrom<BLASDiag> for c_char {
    type Error = BLASError;
    #[inline]
    fn try_from(diag: BLASDiag) -> Result<Self, Self::Error> {
        match diag {
            BLASNonUnit => Ok('N' as c_char),
            BLASUnit => Ok('U' as c_char),
            _ => blas_invalid!(diag),
        }
    }
}

impl From<char> for BLASSide {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'L' => BLASLeft,
            'R' => BLASRight,
            _ => Self::Undefined,
        }
    }
}

impl TryFrom<BLASSide> for char {
    type Error = BLASError;
    #[inline]
    fn try_from(side: BLASSide) -> Result<Self, Self::Error> {
        match side {
            BLASLeft => Ok('L'),
            BLASRight => Ok('R'),
            _ => blas_invalid!(side),
        }
    }
}

impl TryFrom<BLASSide> for c_char {
    type Error = BLASError;
    #[inline]
    fn try_from(side: BLASSide) -> Result<Self, Self::Error> {
        match side {
            BLASLeft => Ok('L' as c_char),
            BLASRight => Ok('R' as c_char),
            _ => blas_invalid!(side),
        }
    }
}

impl BLASLayout {
    #[inline]
    pub fn flip(&self) -> Result<Self, BLASError> {
        match self {
            BLASRowMajor => Ok(BLASColMajor),
            BLASColMajor => Ok(BLASRowMajor),
            _ => blas_invalid!(self),
        }
    }
}

impl BLASUpLo {
    #[inline]
    pub fn flip(&self) -> Result<Self, BLASError> {
        match self {
            BLASUpper => Ok(BLASLower),
            BLASLower => Ok(BLASUpper),
            _ => blas_invalid!(self),
        }
    }
}

impl BLASSide {
    #[inline]
    pub fn flip(&self) -> Result<Self, BLASError> {
        match self {
            BLASLeft => Ok(BLASRight),
            BLASRight => Ok(BLASLeft),
            _ => blas_invalid!(self),
        }
    }
}

impl BLASTranspose {
    #[inline]
    pub fn flip(&self, hermi: bool) -> Result<Self, BLASError> {
        match self {
            BLASNoTrans => match hermi {
                false => Ok(BLASTrans),
                true => Ok(BLASConjTrans),
            },
            BLASTrans => Ok(BLASNoTrans),
            BLASConjTrans => Ok(BLASNoTrans),
            _ => blas_invalid!(self),
        }
    }
}

unsafe impl Send for BLASLayout {}
unsafe impl Send for BLASTranspose {}
unsafe impl Send for BLASUpLo {}
unsafe impl Send for BLASDiag {}
unsafe impl Send for BLASSide {}

unsafe impl Sync for BLASLayout {}
unsafe impl Sync for BLASTranspose {}
unsafe impl Sync for BLASUpLo {}
unsafe impl Sync for BLASDiag {}
unsafe impl Sync for BLASSide {}

impl BLASLayout {
    #[inline]
    pub fn is_cpref(&self) -> bool {
        match self {
            BLASRowMajor => true,
            BLASLayout::Sequential => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_fpref(&self) -> bool {
        match self {
            BLASColMajor => true,
            BLASLayout::Sequential => true,
            _ => false,
        }
    }
}

pub(crate) fn get_layout_row_preferred(by_first: &[Option<BLASLayout>], by_all: &[BLASLayout]) -> BLASLayout {
    for x in by_first {
        if let Some(x) = x {
            if x.is_cpref() {
                return BLASRowMajor;
            } else if x.is_fpref() {
                return BLASColMajor;
            }
        }
    }

    if by_all.iter().all(|f| f.is_cpref()) {
        return BLASRowMajor;
    } else if by_all.iter().all(|f| f.is_fpref()) {
        return BLASColMajor;
    } else {
        return BLASRowMajor;
    }
}
