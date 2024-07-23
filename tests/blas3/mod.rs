pub mod test_gemm;
pub mod test_symm;
pub mod test_syr2k;
pub mod test_syrk;
pub mod test_trmm;
pub mod test_trsm;

#[cfg(feature = "gemmt")]
#[cfg_attr(docsrs, doc(cfg(feature = "gemmt")))]
pub mod test_gemmt;
