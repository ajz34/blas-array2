use derive_builder::Builder;
use libc::{c_int, c_char};
use ndarray::prelude::*;
use blas_sys;
use crate::util::*;

pub trait GEMMFunc<F>
where
    F: BLASFloat
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
    $type: BLASFloat
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
            alpha as *const <$type as BLASFloat>::FFIFloat,
            a as *const <$type as BLASFloat>::FFIFloat,
            lda,
            b as *const <$type as BLASFloat>::FFIFloat,
            ldb,
            beta as *const <$type as BLASFloat>::FFIFloat,
            c as *mut <$type as BLASFloat>::FFIFloat,
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

pub struct GEMM_Driver<'a, F>
where
    F: BLASFloat
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'a, F>,
    pub c: ArrayOut2<'a, F>,
    pub alpha: F,
    pub beta: F,
    pub transa: c_char,
    pub transb: c_char,
    n: c_int,
    m: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
}

impl <'a, F> GEMM_Driver<'a, F>
where
    F: BLASFloat
{
    pub fn run(self) -> Result<ArrayOut2<'a, F>, AnyError>
    where
        BLASFunc<F>: GEMMFunc<F>
    {
        let transa = self.transa as c_char;
        let transb = self.transb as c_char;
        let m = self.m;
        let n = self.n;
        let k = self.k;
        let alpha = self.alpha;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
        let b_ptr = self.b.as_ptr();
        let ldb = self.ldb;
        let beta = self.beta;
        let mut c = self.c;
        let c_ptr = match &mut c {
            ArrayOut::ViewMut(c) => c.as_mut_ptr(),
            ArrayOut::Owned(c) => c.as_mut_ptr(),
        };
        let ldc = self.ldc;

        BLASFunc::<F>::gemm(
            &transa, &transb,
            &m, &n, &k,
            &alpha, a_ptr, &lda,
            b_ptr, &ldb,
            &beta, c_ptr, &ldc
        );
        Ok(c)
    }
}

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct GEMM_<'a, F>
where
    F: BLASFloat
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'a, F>,

    #[builder(default = "None")]
    pub c: Option<ArrayViewMut2<'a, F>>,
    #[builder(default = "F::one()")]
    pub alpha: F,
    #[builder(default = "F::zero()")]
    pub beta: F,
    #[builder(default = "BLASTrans::NoTrans")]
    pub transa: BLASTrans,
    #[builder(default = "BLASTrans::NoTrans")]
    pub transb: BLASTrans,
}

pub type GEMM<'a, F> = GEMM_Builder<'a, F>;

impl<'a, F> GEMM<'a, F>
where
    F: BLASFloat
{
    pub fn driver(self) -> Result<GEMM_Driver<'a, F>, AnyError>
    {
        // initialize
        let obj = self.build()?;

        let a = obj.a;
        let b = obj.b;
        let c = obj.c;
        let transa = obj.transa;
        let transb = obj.transb;
        let alpha = obj.alpha;
        let beta = obj.beta;
        
        // currently only fortran-preferred (col-major) is accepted
        let layout_a = get_layout_array2(&obj.a);
        let layout_b = get_layout_array2(&obj.b);
        if !layout_a.is_fpref() && !layout_b.is_fpref() {
            todo!("Only column major implemented.")
        }

        // initialize intent(hide)
        let (lda, ka) = a.dim();
        let (ldb, kb) = b.dim();
        let m = if transa != BLASTrans::NoTrans { ka } else { lda };
        let k = if transa != BLASTrans::NoTrans { lda } else { ka };
        let n = if transb != BLASTrans::NoTrans { ldb } else { kb };
        
        // perform check
        if transb != BLASTrans::NoTrans {
            BLASError::assert(
                k == kb,
                format!("Incompatible dimensions for matrix multiplication, k={k}, kb={kb}."))?;
        } else {
            BLASError::assert(
                k == ldb,
                format!("Incompatible dimensions for matrix multiplication, k={k}, ldb={ldb}."))?;
        }

        // optional intent(out)
        let c = match c {
            Some(c) => {
                let (ldc, kc) = c.dim();
                BLASError::assert(
                    m == ldc,
                    format!("Incompatible dimensions for matrix multiplication, m={m}, ldc={ldc}."))?;
                BLASError::assert(
                    n == kc,
                    format!("Incompatible dimensions for matrix multiplication, n={n}, kc={kc}."))?;
                ArrayOut2::ViewMut(c)
            },
            None => {
                ArrayOut2::Owned(Array2::zeros((m, n).f()))
            }
        };
        
        let driver = GEMM_Driver {
            a,
            b,
            c,
            alpha,
            beta,
            transa: transa.try_into()?,
            transb: transb.try_into()?,
            n: n.try_into()?,
            m: m.try_into()?,
            k: k.try_into()?,
            lda: lda.try_into()?,
            ldb: ldb.try_into()?,
            ldc: m.try_into()?,
        };

        return Ok(driver);
    }

    pub fn run(self) -> Result<ArrayOut2<'a, F>, AnyError>
    where
        BLASFunc<F>: GEMMFunc<F>
    {
        self.driver()?.run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm()
    {
        println!("test_gemm");
        let arr = Array1::<f64>::linspace(1.0, 35.0, 35);
        let a = Array2::from_shape_vec((5, 7).f(), arr.to_vec()).unwrap();
        let a = a.slice(s![1..4, 2..6]);
        let b = Array2::from_shape_vec((7, 5).f(), arr.to_vec()).unwrap();
        let b = b.slice(s![2..6, 1..4]);
        println!("a={:?}, b={:?}", a, b);
        let c = GEMM::default()
            .a(a.view())
            .b(b.view())
            .run().unwrap();
        println!("{:?}", c);
    }
}
