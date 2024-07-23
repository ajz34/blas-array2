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

// Following test is generated by DeepSeek
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blaslayout_from_char() {
        assert_eq!(BLASLayout::from('R'), BLASRowMajor);
        assert_eq!(BLASLayout::from('r'), BLASRowMajor);
        assert_eq!(BLASLayout::from('C'), BLASColMajor);
        assert_eq!(BLASLayout::from('c'), BLASColMajor);
        assert_eq!(BLASLayout::from('X'), BLASLayout::Undefined);
    }

    #[test]
    fn test_blaslayout_try_from_blaslayout_for_char() {
        assert_eq!(char::try_from(BLASRowMajor), Ok('R'));
        assert_eq!(char::try_from(BLASColMajor), Ok('C'));
        assert!(char::try_from(BLASLayout::Undefined).is_err());
    }

    #[test]
    fn test_blaslayout_try_from_blaslayout_for_c_char() {
        assert_eq!(c_char::try_from(BLASRowMajor), Ok('R' as c_char));
        assert_eq!(c_char::try_from(BLASColMajor), Ok('C' as c_char));
        assert!(c_char::try_from(BLASLayout::Undefined).is_err());
    }

    #[test]
    fn test_blastranspose_from_char() {
        assert_eq!(BLASTranspose::from('N'), BLASNoTrans);
        assert_eq!(BLASTranspose::from('n'), BLASNoTrans);
        assert_eq!(BLASTranspose::from('T'), BLASTrans);
        assert_eq!(BLASTranspose::from('t'), BLASTrans);
        assert_eq!(BLASTranspose::from('C'), BLASConjTrans);
        assert_eq!(BLASTranspose::from('c'), BLASConjTrans);
        assert_eq!(BLASTranspose::from('X'), BLASTranspose::Undefined);
    }

    #[test]
    fn test_blastranspose_try_from_blastranspose_for_char() {
        assert_eq!(char::try_from(BLASNoTrans), Ok('N'));
        assert_eq!(char::try_from(BLASTrans), Ok('T'));
        assert_eq!(char::try_from(BLASConjTrans), Ok('C'));
        assert!(char::try_from(BLASTranspose::Undefined).is_err());
    }

    #[test]
    fn test_blastranspose_try_from_blastranspose_for_c_char() {
        assert_eq!(c_char::try_from(BLASNoTrans), Ok('N' as c_char));
        assert_eq!(c_char::try_from(BLASTrans), Ok('T' as c_char));
        assert_eq!(c_char::try_from(BLASConjTrans), Ok('C' as c_char));
        assert!(c_char::try_from(BLASTranspose::Undefined).is_err());
    }

    #[test]
    fn test_blasuplo_from_char() {
        assert_eq!(BLASUpLo::from('U'), BLASUpper);
        assert_eq!(BLASUpLo::from('u'), BLASUpper);
        assert_eq!(BLASUpLo::from('L'), BLASLower);
        assert_eq!(BLASUpLo::from('l'), BLASLower);
        assert_eq!(BLASUpLo::from('X'), BLASUpLo::Undefined);
    }

    #[test]
    fn test_blasuplo_try_from_blasuplo_for_char() {
        assert_eq!(char::try_from(BLASUpper), Ok('U'));
        assert_eq!(char::try_from(BLASLower), Ok('L'));
        assert!(char::try_from(BLASUpLo::Undefined).is_err());
    }

    #[test]
    fn test_blasuplo_try_from_blasuplo_for_c_char() {
        assert_eq!(c_char::try_from(BLASUpper), Ok('U' as c_char));
        assert_eq!(c_char::try_from(BLASLower), Ok('L' as c_char));
        assert!(c_char::try_from(BLASUpLo::Undefined).is_err());
    }

    #[test]
    fn test_blasdiag_from_char() {
        assert_eq!(BLASDiag::from('N'), BLASNonUnit);
        assert_eq!(BLASDiag::from('n'), BLASNonUnit);
        assert_eq!(BLASDiag::from('U'), BLASUnit);
        assert_eq!(BLASDiag::from('u'), BLASUnit);
        assert_eq!(BLASDiag::from('X'), BLASDiag::Undefined);
    }

    #[test]
    fn test_blasdiag_try_from_blasdiag_for_char() {
        assert_eq!(char::try_from(BLASNonUnit), Ok('N'));
        assert_eq!(char::try_from(BLASUnit), Ok('U'));
        assert!(char::try_from(BLASDiag::Undefined).is_err());
    }

    #[test]
    fn test_blasdiag_try_from_blasdiag_for_c_char() {
        assert_eq!(c_char::try_from(BLASNonUnit), Ok('N' as c_char));
        assert_eq!(c_char::try_from(BLASUnit), Ok('U' as c_char));
        assert!(c_char::try_from(BLASDiag::Undefined).is_err());
    }

    #[test]
    fn test_blasside_from_char() {
        assert_eq!(BLASSide::from('L'), BLASLeft);
        assert_eq!(BLASSide::from('l'), BLASLeft);
        assert_eq!(BLASSide::from('R'), BLASRight);
        assert_eq!(BLASSide::from('r'), BLASRight);
        assert_eq!(BLASSide::from('X'), BLASSide::Undefined);
    }

    #[test]
    fn test_blasside_try_from_blasside_for_char() {
        assert_eq!(char::try_from(BLASLeft), Ok('L'));
        assert_eq!(char::try_from(BLASRight), Ok('R'));
        assert!(char::try_from(BLASSide::Undefined).is_err());
    }

    #[test]
    fn test_blasside_try_from_blasside_for_c_char() {
        assert_eq!(c_char::try_from(BLASLeft), Ok('L' as c_char));
        assert_eq!(c_char::try_from(BLASRight), Ok('R' as c_char));
        assert!(c_char::try_from(BLASSide::Undefined).is_err());
    }

    #[test]
    fn test_blaslayout_flip() {
        assert_eq!(BLASRowMajor.flip(), Ok(BLASColMajor));
        assert_eq!(BLASColMajor.flip(), Ok(BLASRowMajor));
        assert!(BLASLayout::Undefined.flip().is_err());
    }

    #[test]
    fn test_blasuplo_flip() {
        assert_eq!(BLASUpper.flip(), Ok(BLASLower));
        assert_eq!(BLASLower.flip(), Ok(BLASUpper));
        assert!(BLASUpLo::Undefined.flip().is_err());
    }

    #[test]
    fn test_blasside_flip() {
        assert_eq!(BLASLeft.flip(), Ok(BLASRight));
        assert_eq!(BLASRight.flip(), Ok(BLASLeft));
        assert!(BLASSide::Undefined.flip().is_err());
    }

    #[test]
    fn test_blastranspose_flip() {
        assert_eq!(BLASNoTrans.flip(false), Ok(BLASTrans));
        assert_eq!(BLASNoTrans.flip(true), Ok(BLASConjTrans));
        assert_eq!(BLASTrans.flip(false), Ok(BLASNoTrans));
        assert_eq!(BLASTrans.flip(true), Ok(BLASNoTrans));
        assert_eq!(BLASConjTrans.flip(false), Ok(BLASNoTrans));
        assert_eq!(BLASConjTrans.flip(true), Ok(BLASNoTrans));
        assert!(BLASTranspose::Undefined.flip(false).is_err());
    }

    #[test]
    fn test_blaslayout_is_cpref() {
        assert!(BLASRowMajor.is_cpref());
        assert!(!BLASColMajor.is_cpref());
        assert!(BLASLayout::Sequential.is_cpref());
        assert!(!BLASLayout::Undefined.is_cpref());
    }

    #[test]
    fn test_blaslayout_is_fpref() {
        assert!(!BLASRowMajor.is_fpref());
        assert!(BLASColMajor.is_fpref());
        assert!(BLASLayout::Sequential.is_fpref());
        assert!(!BLASLayout::Undefined.is_fpref());
    }

    #[test]
    fn test_get_layout_row_preferred() {
        let by_first = [Some(BLASRowMajor), Some(BLASColMajor), None];
        let by_all = [BLASRowMajor, BLASColMajor, BLASLayout::Sequential];

        assert_eq!(get_layout_row_preferred(&by_first, &by_all), BLASRowMajor);

        let by_first = [None, None, None];
        let by_all = [BLASRowMajor, BLASRowMajor, BLASRowMajor];

        assert_eq!(get_layout_row_preferred(&by_first, &by_all), BLASRowMajor);

        let by_first = [None, None, None];
        let by_all = [BLASColMajor, BLASColMajor, BLASColMajor];

        assert_eq!(get_layout_row_preferred(&by_first, &by_all), BLASColMajor);

        let by_first = [None, None, None];
        let by_all = [BLASRowMajor, BLASColMajor, BLASLayout::Sequential];

        assert_eq!(get_layout_row_preferred(&by_first, &by_all), BLASRowMajor);
    }
}
