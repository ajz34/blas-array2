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

impl From<char> for BLASLayout {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'R' => BLASRowMajor,
            'C' => BLASColMajor,
            _ => panic!("Invalid character for BLASOrder: {}", c),
        }
    }
}

impl From<BLASLayout> for char {
    #[inline]
    fn from(layout: BLASLayout) -> Self {
        match layout {
            BLASRowMajor => 'R',
            BLASColMajor => 'C',
            _ => panic!("Invalid BLASOrder: {:?}", layout),
        }
    }
}

impl From<BLASLayout> for c_char {
    #[inline]
    fn from(layout: BLASLayout) -> Self {
        match layout {
            BLASRowMajor => 'R' as c_char,
            BLASColMajor => 'C' as c_char,
            _ => panic!("Invalid BLASOrder: {:?}", layout),
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
            _ => panic!("Invalid character for BLASTrans: {}", c),
        }
    }
}

impl From<BLASTranspose> for char {
    #[inline]
    fn from(trans: BLASTranspose) -> Self {
        match trans {
            BLASNoTrans => 'N',
            BLASTrans => 'T',
            BLASConjTrans => 'C',
            _ => panic!("Invalid BLASTrans: {:?}", trans),
        }
    }
}

impl From<BLASTranspose> for c_char {
    #[inline]
    fn from(trans: BLASTranspose) -> Self {
        match trans {
            BLASNoTrans => 'N' as c_char,
            BLASTrans => 'T' as c_char,
            BLASConjTrans => 'C' as c_char,
            _ => panic!("Invalid BLASTrans: {:?}", trans),
        }
    }
}

impl From<char> for BLASUpLo {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'U' => BLASUpper,
            'L' => BLASLower,
            _ => panic!("Invalid character for BLASUpLo: {}", c),
        }
    }
}

impl From<BLASUpLo> for char {
    #[inline]
    fn from(uplo: BLASUpLo) -> Self {
        match uplo {
            BLASUpper => 'U',
            BLASLower => 'L',
            _ => panic!("Invalid BLASUpLo: {:?}", uplo),
        }
    }
}

impl From<BLASUpLo> for c_char {
    #[inline]
    fn from(uplo: BLASUpLo) -> Self {
        match uplo {
            BLASUpper => 'U' as c_char,
            BLASLower => 'L' as c_char,
            _ => panic!("Invalid BLASUpLo: {:?}", uplo),
        }
    }
}

impl From<char> for BLASDiag {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'N' => BLASNonUnit,
            'U' => BLASUnit,
            _ => panic!("Invalid character for BLASDiag: {}", c),
        }
    }
}

impl From<BLASDiag> for char {
    #[inline]
    fn from(diag: BLASDiag) -> Self {
        match diag {
            BLASNonUnit => 'N',
            BLASUnit => 'U',
            _ => panic!("Invalid BLASDiag: {:?}", diag),
        }
    }
}

impl From<BLASDiag> for c_char {
    #[inline]
    fn from(diag: BLASDiag) -> Self {
        match diag {
            BLASNonUnit => 'N' as c_char,
            BLASUnit => 'U' as c_char,
            _ => panic!("Invalid BLASDiag: {:?}", diag),
        }
    }
}

impl From<char> for BLASSide {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'L' => BLASLeft,
            'R' => BLASRight,
            _ => panic!("Invalid character for BLASSide: {}", c),
        }
    }
}

impl From<BLASSide> for char {
    #[inline]
    fn from(side: BLASSide) -> Self {
        match side {
            BLASLeft => 'L',
            BLASRight => 'R',
            _ => panic!("Invalid BLASSide: {:?}", side),
        }
    }
}

impl From<BLASSide> for c_char {
    #[inline]
    fn from(side: BLASSide) -> Self {
        match side {
            BLASLeft => 'L' as c_char,
            BLASRight => 'R' as c_char,
            _ => panic!("Invalid BLASSide: {:?}", side),
        }
    }
}

impl BLASLayout {
    #[inline]
    pub fn flip(&self) -> Self {
        match self {
            BLASRowMajor => BLASColMajor,
            BLASColMajor => BLASRowMajor,
            _ => panic!("Invalid BLASOrder: {:?}", self),
        }
    }
}

impl BLASUpLo {
    #[inline]
    pub fn flip(&self) -> Self {
        match self {
            BLASUpper => BLASLower,
            BLASLower => BLASUpper,
            _ => panic!("Invalid BLASUpLo: {:?}", self),
        }
    }
}

impl BLASSide {
    #[inline]
    pub fn flip(&self) -> Self {
        match self {
            BLASLeft => BLASRight,
            BLASRight => BLASLeft,
            _ => panic!("Invalid BLASSide: {:?}", self),
        }
    }
}

impl BLASTranspose {
    #[inline]
    pub fn flip(&self, hermi: bool) -> Self {
        match self {
            BLASNoTrans => match hermi {
                false => BLASTrans,
                true => BLASConjTrans,
            },
            BLASTrans => BLASNoTrans,
            BLASConjTrans => BLASNoTrans,
            _ => panic!("Invalid BLASTranspose: {:?}", self),
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
