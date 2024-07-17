use core::num::TryFromIntError;

use derive_builder::UninitializedFieldError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BLASError {
    OverflowDimension(String),
    InvalidDim(String),
    InvalidFlag(String),
    FailedCheck(String),
    UninitializedField(String),
    Miscellaneous(String),
}

/* #region impl BLASError */

impl std::error::Error for BLASError {}

impl From<UninitializedFieldError> for BLASError {
    fn from(e: UninitializedFieldError) -> BLASError {
        BLASError::UninitializedField(e.to_string())
    }
}

impl From<TryFromIntError> for BLASError {
    fn from(e: TryFromIntError) -> BLASError {
        BLASError::OverflowDimension(e.to_string())
    }
}

impl core::fmt::Display for BLASError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/* #endregion */

#[macro_export]
macro_rules! blas_assert {
    ($cond:expr, $errtype:ident, $($arg:tt)*) => {
        if $cond {
            Ok(())
        } else {
            Err(BLASError::$errtype(
                format!("{:}:{:}: ", file!(), line!())
                + &format!($($arg)*)
                + &format!(": {:?}", stringify!($cond))
            ))
        }
    };
    ($cond:expr, $errtype:ident) => {
        if $cond {
            Ok(())
        } else {
            Err(BLASError::$errtype(
                format!("{:}:{:}: ", file!(), line!())
                + &format!("{:?}", stringify!($cond))
            ))
        }
    };
}

#[macro_export]
macro_rules! blas_assert_eq {
    ($a:expr, $b:expr, $errtype:ident, $($arg:tt)*) => {
        if $a == $b {
            Ok(())
        } else {
            Err(BLASError::$errtype(
                format!("{:}:{:}: ", file!(), line!())
                + &format!($($arg)*)
                + &format!(": {:} = {:?}, {:} = {:?}", stringify!($a), $a, stringify!($b), $b)
            ))
        }
    };
    ($a:expr, $b:expr, $errtype:ident) => {
        if $a == $b {
            Ok(())
        } else {
            Err(BLASError::$errtype(
                format!("{:}:{:}: ", file!(), line!())
                + &format!("{:} = {:?}, {:} = {:?}", stringify!($a), $a, stringify!($b), $b)
            ))
        }
    };
}

#[macro_export]
macro_rules! blas_raise {
    ($errtype:ident) => {
        Err(BLASError::$errtype(format!("{:}:{:}: ", file!(), line!())))
    };
    ($errtype:ident, $($arg:tt)*) => {
        Err(BLASError::$errtype(format!("{:}:{:}: ", file!(), line!()) + &format!($($arg)*)))
    };
}

#[macro_export]
macro_rules! blas_invalid {
    ($word:expr) => {
        Err(BLASError::InvalidFlag(
            format!("{:}:{:}: ", file!(), line!())
                + &format!("Invalid keyowrd {:} = {:?}", stringify!($word), $word),
        ))
    };
}
