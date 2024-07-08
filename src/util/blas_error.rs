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

#[macro_export]
macro_rules! blas_assert {
    ($cond:expr, $($arg:tt)*) => {
        BLASError::assert($cond,
            format!("{:}:{:}: ", file!(), line!()) + &format!($($arg)*)
        )
    };
}

#[macro_export]
macro_rules! blas_assert_eq {
    ($a:expr, $b:expr, $($arg:tt)*) => {
        BLASError::assert($a == $b,
            format!("{:}:{:}: ", file!(), line!()) + &format!($($arg)*) + &format!(": {:} = {:?}, {:} = {:?}", stringify!($a), $a, stringify!($b), $b)
        )
    };
    ($a:expr, $b:expr) => {
        BLASError::assert($a == $b,
            format!("{:}:{:}: ", file!(), line!()) + &format!(": {:} = {:?}, {:} = {:?}", stringify!($a), $a, stringify!($b), $b)
        )
    };
}

#[macro_export]
macro_rules! blas_raise {
    ($($arg:tt)*) => {
        Err(BLASError(format!("{:}:{:}: ", file!(), line!()) + &format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! blas_invalid {
    ($word:expr) => {
        Err(BLASError(
            format!("{:}:{:}: ", file!(), line!()) + &format!("Invalid keyowrd {:} = {:?}", stringify!($word), $word),
        ))
    };
}

impl std::fmt::Display for BLASError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
