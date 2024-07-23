#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::string::String;
use core::num::TryFromIntError;
use derive_builder::UninitializedFieldError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BLASError {
    OverflowDimension(String),
    InvalidDim(String),
    InvalidFlag(String),
    FailedCheck(String),
    UninitializedField(&'static str),
    ExplicitCopy(String),
    Miscellaneous(String),
    RuntimeError(String),
}

/* #region impl BLASError */

#[cfg(feature = "std")]
impl std::error::Error for BLASError {}

impl From<UninitializedFieldError> for BLASError {
    fn from(e: UninitializedFieldError) -> BLASError {
        BLASError::UninitializedField(e.field_name())
    }
}

impl From<TryFromIntError> for BLASError {
    fn from(_: TryFromIntError) -> BLASError {
        BLASError::OverflowDimension(String::from("TryFromIntError"))
    }
}

impl core::fmt::Display for BLASError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
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
            extern crate alloc;
            use alloc::string::String;
            Err(BLASError::$errtype(String::from(concat!(
                file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
                $($arg),*, ": ", stringify!($cond)
            ))))
        }
    };
    ($cond:expr, $errtype:ident) => {
        if $cond {
            Ok(())
        } else {
            extern crate alloc;
            use alloc::string::String;
            Err(BLASError::$errtype(String::from(concat!(
                file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
                stringify!($cond)
            ))))
        }
    };
}

#[macro_export]
macro_rules! blas_assert_eq {
    ($a:expr, $b:expr, $errtype:ident) => {
        if $a == $b {
            Ok(())
        } else {
            extern crate alloc;
            use alloc::string::String;
            use core::fmt::Write;
            let mut s = String::from(concat!(
                file!(),
                ":",
                line!(),
                ": ",
                "BLASError::",
                stringify!($errtype),
                " : "
            ));
            write!(s, "{:?} = {:?} not equal to {:?} = {:?}", stringify!($a), $a, stringify!($b), $b)
                .unwrap();
            Err(BLASError::$errtype(s))
        }
    };
}

#[macro_export]
macro_rules! blas_raise {
    ($errtype:ident) => {{
        extern crate alloc;
        use alloc::string::String;
        Err(BLASError::$errtype(String::from(concat!(
            file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype)
        ))))
    }};
    ($errtype:ident, $($arg:tt)*) => {{
        extern crate alloc;
        use alloc::string::String;
        Err(BLASError::$errtype(String::from(concat!(
            file!(), ":", line!(), ": ", "BLASError::", stringify!($errtype), " : ",
            $($arg),*
        ))))
    }};
}

#[macro_export]
macro_rules! blas_invalid {
    ($word:expr) => {{
        extern crate alloc;
        use alloc::string::String;
        use core::fmt::Write;
        let mut s = String::from(concat!(file!(), ":", line!(), ": ", "BLASError::InvalidFlag", " : "));
        write!(s, "{:?} = {:?}", stringify!($word), $word).unwrap();
        Err(BLASError::InvalidFlag(s))
    }};
}

/* #endregion */

/* #region macros (warning) */

#[macro_export]
macro_rules! blas_warn_layout_clone {
    ($array:expr) => {{
        #[cfg(feature = "std")]
        extern crate std;

        if cfg!(all(feature = "std", feature = "warn_on_copy")) {
            std::eprintln!(
                "Warning: Copying array due to non-standard layout, shape={:?}, strides={:?}",
                $array.shape(),
                $array.strides()
            );
            Result::<(), BLASError>::Ok(())
        } else if cfg!(feature = "error_on_copy") {
            blas_raise!(ExplicitCopy)
        } else {
            Result::<(), BLASError>::Ok(())
        }
    }};
    ($array:expr, $msg:tt) => {{
        #[cfg(feature = "std")]
        extern crate std;

        if cfg!(all(feature = "std", feature = "warn_on_copy")) {
            std::eprintln!("Warning: {:?}, shape={:?}, strides={:?}", $msg, $array.shape(), $array.strides());
            Result::<(), BLASError>::Ok(())
        } else if cfg!(feature = "error_on_copy") {
            blas_raise!(ExplicitCopy)
        } else {
            Result::<(), BLASError>::Ok(())
        }
    }};
}

/* #endregion */

// Following test is assisted by DeepSeek
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_uninitialized_field_error() {
        let error = UninitializedFieldError::new("field_name");
        let blas_error: BLASError = error.into();
        assert_eq!(blas_error, BLASError::UninitializedField("field_name"));
    }

    #[test]
    fn test_from_try_from_int_error() {
        let error: Result<i32, _> = (1000000000000 as usize).try_into();
        let blas_error: BLASError = error.unwrap_err().into();
        assert_eq!(blas_error, BLASError::OverflowDimension("TryFromIntError".to_string()));
    }

    #[test]
    fn test_blas_assert_macro_with_args() {
        let result = blas_assert!(false, InvalidFlag, "test_condition");
        if let Err(BLASError::InvalidFlag(msg)) = result {
            assert!(msg.contains("BLASError::InvalidFlag"));
            assert!(msg.contains("test_condition"));
        } else {
            panic!("Expected BLASError::InvalidFlag");
        }
    }

    #[test]
    fn test_blas_assert_macro_without_args() {
        let result = blas_assert!(false, InvalidFlag);
        if let Err(BLASError::InvalidFlag(msg)) = result {
            assert!(msg.contains("BLASError::InvalidFlag"));
        } else {
            panic!("Expected BLASError::InvalidFlag");
        }
    }

    #[test]
    fn test_blas_assert_eq_macro() {
        let result = blas_assert_eq!(1, 2, InvalidFlag);
        if let Err(BLASError::InvalidFlag(msg)) = result {
            assert!(msg.contains("BLASError::InvalidFlag"));
            assert!(msg.contains(r#""1" = 1 not equal to "2" = 2"#));
        } else {
            panic!("Expected BLASError::InvalidFlag");
        }
    }

    #[test]
    fn test_blas_raise_macro_without_args() {
        let result: Result<(), BLASError> = blas_raise!(InvalidFlag);
        if let Err(BLASError::InvalidFlag(msg)) = result {
            assert!(msg.contains("BLASError::InvalidFlag"));
        } else {
            panic!("Expected BLASError::InvalidFlag");
        }
    }

    #[test]
    fn test_blas_raise_macro_with_args() {
        let result: Result<(), BLASError> = blas_raise!(InvalidFlag, "test_message");
        if let Err(BLASError::InvalidFlag(msg)) = result {
            assert!(msg.contains("BLASError::InvalidFlag"));
            assert!(msg.contains("test_message"));
        } else {
            panic!("Expected BLASError::InvalidFlag");
        }
    }

    #[test]
    fn test_blas_invalid_macro() {
        let word = 1;
        let result: Result<(), BLASError> = blas_invalid!(word);
        if let Err(BLASError::InvalidFlag(msg)) = result {
            assert!(msg.contains("BLASError::InvalidFlag"));
            assert!(msg.contains(r#""word" = 1"#));
        } else {
            panic!("Expected BLASError::InvalidFlag");
        }
    }
}
