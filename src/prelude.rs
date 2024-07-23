pub use crate::ffi::blas_int;
pub use crate::util::*;

pub use crate::blas1::asum::{ASUM, DASUM, DZASUM, SASUM, SCASUM};
pub use crate::blas1::iamax::{IAMAX, ICAMAX, IDAMAX, ISAMAX, IZAMAX};
pub use crate::blas1::nrm2::{DNRM2, DZNRM2, NRM2, SCNRM2, SNRM2};

pub use crate::blas2::gbmv::{GBMVNum, CGBMV, DGBMV, GBMV, SGBMV, ZGBMV};
pub use crate::blas2::gemv::{GEMVNum, CGEMV, DGEMV, GEMV, SGEMV, ZGEMV};
pub use crate::blas2::ger::{GERNum, CGERU, DGER, GER, SGER, ZGERU};
pub use crate::blas2::gerc::{GERCNum, CGERC, GERC, ZGERC};
pub use crate::blas2::hbmv::{HBMVNum, CHBMV, DSBMV, HBMV, SSBMV, ZHBMV};
pub use crate::blas2::hemv::{HEMVNum, CHEMV, DSYMV, HEMV, SSYMV, ZHEMV};
pub use crate::blas2::her::{HERNum, CHER, DSYR, HER, SSYR, ZHER};
pub use crate::blas2::her2::{HER2Num, CHER2, DSYR2, HER2, SSYR2, SYR2, ZHER2};
pub use crate::blas2::hpmv::{HPMVNum, CHPMV, DSPMV, HPMV, SSPMV, ZHPMV};
pub use crate::blas2::hpr::{HPRNum, CHPR, DSPR, HPR, SSPR, ZHPR};
pub use crate::blas2::hpr2::{HPR2Num, CHPR2, DSPR2, HPR2, SSPR2, ZHPR2};
pub use crate::blas2::tbmv::{TBMVNum, CTBMV, DTBMV, STBMV, TBMV, ZTBMV};
pub use crate::blas2::tbsv::{TBSVNum, CTBSV, DTBSV, STBSV, TBSV, ZTBSV};
pub use crate::blas2::tpmv::{TPMVNum, CTPMV, DTPMV, STPMV, TPMV, ZTPMV};
pub use crate::blas2::tpsv::{TPSVNum, CTPSV, DTPSV, STPSV, TPSV, ZTPSV};
pub use crate::blas2::trmv::{TRMVNum, CTRMV, DTRMV, STRMV, TRMV, ZTRMV};
pub use crate::blas2::trsv::{TRSVNum, CTRSV, DTRSV, STRSV, TRSV, ZTRSV};

pub use crate::blas3::gemm::{GEMMNum, CGEMM, DGEMM, GEMM, SGEMM, ZGEMM};
pub use crate::blas3::hemm::{HEMMNum, CHEMM, HEMM, ZHEMM};
pub use crate::blas3::her2k::{HER2KNum, CHER2K, HER2K, ZHER2K};
pub use crate::blas3::herk::{HERKNum, CHERK, HERK, ZHERK};
pub use crate::blas3::symm::{SYMMNum, CSYMM, DSYMM, SSYMM, SYMM, ZSYMM};
pub use crate::blas3::syr2k::{SYR2KNum, CSYR2K, DSYR2K, SSYR2K, SYR2K, ZSYR2K};
pub use crate::blas3::syrk::{SYRKNum, CSYRK, DSYRK, SSYRK, SYRK, ZSYRK};
pub use crate::blas3::trmm::{TRMMNum, CTRMM, DTRMM, STRMM, TRMM, ZTRMM};
pub use crate::blas3::trsm::{TRSMNum, CTRSM, DTRSM, STRSM, TRSM, ZTRSM};

#[cfg(feature = "gemmt")]
pub use crate::blas3::gemmt::{GEMMTNum, CGEMMT, DGEMMT, GEMMT, SGEMMT, ZGEMMT};

pub mod generic {
    pub use crate::blas1::asum::ASUM_;
    pub use crate::blas1::iamax::IAMAX_;
    pub use crate::blas1::nrm2::NRM2_;

    pub use crate::blas2::gbmv::GBMV_;
    pub use crate::blas2::gemv::GEMV_;
    pub use crate::blas2::ger::GER_;
    pub use crate::blas2::gerc::GERC_;
    pub use crate::blas2::hbmv::HBMV_;
    pub use crate::blas2::hemv::HEMV_;
    pub use crate::blas2::her::HER_;
    pub use crate::blas2::her2::HER2_;
    pub use crate::blas2::hpmv::HPMV_;
    pub use crate::blas2::hpr::HPR_;
    pub use crate::blas2::hpr2::HPR2_;
    pub use crate::blas2::tbmv::TBMV_;
    pub use crate::blas2::tbsv::TBSV_;
    pub use crate::blas2::tpmv::TPMV_;
    pub use crate::blas2::tpsv::TPSV_;
    pub use crate::blas2::trmv::TRMV_;
    pub use crate::blas2::trsv::TRSV_;

    pub use crate::blas3::gemm::GEMM_;
    pub use crate::blas3::symm::SYMM_;
    pub use crate::blas3::syr2k::SYR2K_;
    pub use crate::blas3::syrk::SYRK_;
    pub use crate::blas3::trmm::TRMM_;
    pub use crate::blas3::trsm::TRSM_;

    #[cfg(feature = "gemmt")]
    pub use crate::blas3::gemmt::GEMMT_;
}
