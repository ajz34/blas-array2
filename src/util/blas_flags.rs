use libc::c_char;
use crate::util::BLASError;

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
