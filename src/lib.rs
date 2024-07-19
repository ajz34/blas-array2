#![doc = include_str!("../README.md")]
#![allow(non_camel_case_types)]
#![allow(refining_impl_trait_reachable)]
#![allow(non_upper_case_globals)]
#![cfg_attr(not(test), no_std)]

pub mod blas1;
pub mod blas2;
pub mod blas3;
pub mod ffi;
pub mod prelude;
pub mod util;
