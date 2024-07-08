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

impl From<char> for BLASLayout {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'R' => BLASLayout::RowMajor,
            'C' => BLASLayout::ColMajor,
            _ => panic!("Invalid character for BLASOrder: {}", c),
        }
    }
}

impl From<BLASLayout> for char {
    #[inline]
    fn from(layout: BLASLayout) -> Self {
        match layout {
            BLASLayout::RowMajor => 'R',
            BLASLayout::ColMajor => 'C',
            _ => panic!("Invalid BLASOrder: {:?}", layout),
        }
    }
}

impl From<BLASLayout> for c_char {
    #[inline]
    fn from(layout: BLASLayout) -> Self {
        match layout {
            BLASLayout::RowMajor => 'R' as c_char,
            BLASLayout::ColMajor => 'C' as c_char,
            _ => panic!("Invalid BLASOrder: {:?}", layout),
        }
    }
}

impl From<char> for BLASTrans {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'N' => BLASTrans::NoTrans,
            'T' => BLASTrans::Trans,
            'C' => BLASTrans::ConjTrans,
            _ => panic!("Invalid character for BLASTrans: {}", c),
        }
    }
}

impl From<BLASTrans> for char {
    #[inline]
    fn from(trans: BLASTrans) -> Self {
        match trans {
            BLASTrans::NoTrans => 'N',
            BLASTrans::Trans => 'T',
            BLASTrans::ConjTrans => 'C',
            _ => panic!("Invalid BLASTrans: {:?}", trans),
        }
    }
}

impl From<BLASTrans> for c_char {
    #[inline]
    fn from(trans: BLASTrans) -> Self {
        match trans {
            BLASTrans::NoTrans => 'N' as c_char,
            BLASTrans::Trans => 'T' as c_char,
            BLASTrans::ConjTrans => 'C' as c_char,
            _ => panic!("Invalid BLASTrans: {:?}", trans),
        }
    }
}

impl From<char> for BLASUpLo {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'U' => BLASUpLo::Upper,
            'L' => BLASUpLo::Lower,
            _ => panic!("Invalid character for BLASUpLo: {}", c),
        }
    }
}

impl From<BLASUpLo> for char {
    #[inline]
    fn from(uplo: BLASUpLo) -> Self {
        match uplo {
            BLASUpLo::Upper => 'U',
            BLASUpLo::Lower => 'L',
            _ => panic!("Invalid BLASUpLo: {:?}", uplo),
        }
    }
}

impl From<BLASUpLo> for c_char {
    #[inline]
    fn from(uplo: BLASUpLo) -> Self {
        match uplo {
            BLASUpLo::Upper => 'U' as c_char,
            BLASUpLo::Lower => 'L' as c_char,
            _ => panic!("Invalid BLASUpLo: {:?}", uplo),
        }
    }
}

impl From<char> for BLASDiag {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'N' => BLASDiag::NonUnit,
            'U' => BLASDiag::Unit,
            _ => panic!("Invalid character for BLASDiag: {}", c),
        }
    }
}

impl From<BLASDiag> for char {
    #[inline]
    fn from(diag: BLASDiag) -> Self {
        match diag {
            BLASDiag::NonUnit => 'N',
            BLASDiag::Unit => 'U',
            _ => panic!("Invalid BLASDiag: {:?}", diag),
        }
    }
}

impl From<BLASDiag> for c_char {
    #[inline]
    fn from(diag: BLASDiag) -> Self {
        match diag {
            BLASDiag::NonUnit => 'N' as c_char,
            BLASDiag::Unit => 'U' as c_char,
            _ => panic!("Invalid BLASDiag: {:?}", diag),
        }
    }
}

impl From<char> for BLASSide {
    #[inline]
    fn from(c: char) -> Self {
        match c.to_ascii_uppercase() {
            'L' => BLASSide::Left,
            'R' => BLASSide::Right,
            _ => panic!("Invalid character for BLASSide: {}", c),
        }
    }
}

impl From<BLASSide> for char {
    #[inline]
    fn from(side: BLASSide) -> Self {
        match side {
            BLASSide::Left => 'L',
            BLASSide::Right => 'R',
            _ => panic!("Invalid BLASSide: {:?}", side),
        }
    }
}

impl From<BLASSide> for c_char {
    #[inline]
    fn from(side: BLASSide) -> Self {
        match side {
            BLASSide::Left => 'L' as c_char,
            BLASSide::Right => 'R' as c_char,
            _ => panic!("Invalid BLASSide: {:?}", side),
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
    #[inline]
    pub fn is_cpref(&self) -> bool {
        match self {
            BLASLayout::RowMajor => true,
            BLASLayout::Sequential => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_fpref(&self) -> bool {
        match self {
            BLASLayout::ColMajor => true,
            BLASLayout::Sequential => true,
            _ => false,
        }
    }
}
