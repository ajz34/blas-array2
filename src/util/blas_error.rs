use core::num::TryFromIntError;

use derive_builder::UninitializedFieldError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BLASError {
    OverflowDimension(&'static str),
    InvalidDim(&'static str),
    InvalidFlag(&'static str),
    FailedCheck(&'static str),
    UninitializedField(&'static str),
    Miscellaneous(&'static str),
}

/* #region impl BLASError */

impl std::error::Error for BLASError {}

impl From<UninitializedFieldError> for BLASError {
    fn from(e: UninitializedFieldError) -> BLASError {
        BLASError::UninitializedField(e.field_name())
    }
}

impl From<TryFromIntError> for BLASError {
    fn from(_: TryFromIntError) -> BLASError {
        BLASError::OverflowDimension("TryFromIntError")
    }
}

impl core::fmt::Display for BLASError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/* #endregion */

/* #region macros */

#[macro_export]
macro_rules! blas_assert {
    ($cond:expr, $errtype:ident, $($arg:tt)*) => {
        if $cond {
            Ok(())
        } else {
            Err(BLASError::$errtype(concat!(
                file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
                $($arg),*, ": ", stringify!($cond)
            )))
        }
    };
    ($cond:expr, $errtype:ident) => {
        if $cond {
            Ok(())
        } else {
            Err(BLASError::$errtype(concat!(
                file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
                stringify!($cond)
            )))
        }
    };
}

#[macro_export]
macro_rules! blas_assert_eq {
    ($a:expr, $b:expr, $errtype:ident, $($arg:tt)*) => {
        if $a == $b {
            Ok(())
        } else {
            Err(BLASError::$errtype(concat!(
                file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
                $($arg),*, ": ", stringify!($a), " = ", stringify!($b)
            )))
        }
    };
    ($a:expr, $b:expr, $errtype:ident) => {
        if $a == $b {
            Ok(())
        } else {
            Err(BLASError::$errtype(concat!(
                file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
                stringify!($a), " = ", stringify!($b)
            )))
        }
    };
}

#[macro_export]
macro_rules! blas_raise {
    ($errtype:ident) => {
        Err(BLASError::$errtype(concat!(
            file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype)
        )))
    };
    ($errtype:ident, $($arg:tt)*) => {
        Err(BLASError::$errtype(concat!(
            file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
            $($arg),*
        )))
    };
}

#[macro_export]
macro_rules! blas_invalid {
    ($word:expr) => {
        Err(BLASError::InvalidFlag(concat!(
            file!(), ":", line!(), ": ", "BLASError::InvalidFlag", " :",
            stringify!($word)
        )))
    };
}

/* #endregion */
