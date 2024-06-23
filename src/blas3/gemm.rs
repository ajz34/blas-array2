use libc::{c_int, c_char};
use ndarray::prelude::*;
use blas_sys;
use crate::util::*;

pub trait GEMMFunc<F>
where
    F: FloatType
{
    fn gemm(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const F,
        a: *const F,
        lda: *const c_int,
        b: *const F,
        ldb: *const c_int,
        beta: *const F,
        c: *mut F,
        ldc: *const c_int);
}

macro_rules! impl_subroutine {
    ($type:ty, $func:ident) => {

impl GEMMFunc<$type> for BLASFunc<$type>
where
    $type: FloatType
{
    fn gemm(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const $type,
        a: *const $type,
        lda: *const c_int,
        b: *const $type,
        ldb: *const c_int,
        beta: *const $type,
        c: *mut $type,
        ldc: *const c_int
    ){
        unsafe { blas_sys::$func(
            transa,
            transb,
            m,
            n,
            k,
            alpha as *const <$type as FloatType>::FFIFloat,
            a as *const <$type as FloatType>::FFIFloat,
            lda,
            b as *const <$type as FloatType>::FFIFloat,
            ldb,
            beta as *const <$type as FloatType>::FFIFloat,
            c as *mut <$type as FloatType>::FFIFloat,
            ldc)
        }
    }
}

    };
}

impl_subroutine!(f32, sgemm_);
impl_subroutine!(f64, dgemm_);
impl_subroutine!(c32, cgemm_);
impl_subroutine!(c64, zgemm_);
