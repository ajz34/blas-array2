use derive_builder::Builder;
use libc::{c_int, c_char};
use ndarray::prelude::*;
use blas_sys;
use crate::util::*;

/* #region BLAS func */

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

macro_rules! impl_func {
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

impl_func!(f32, sgemm_);
impl_func!(f64, dgemm_);
impl_func!(c32, cgemm_);
impl_func!(c64, zgemm_);

/* #endregion */

/* #region BLAS driver */

pub struct GEMM_Driver<'a, 'b, 'c, F>
where
    F: BLASFloat
{
    a: ArrayView2<'a, F>,
    b: ArrayView2<'b, F>,
    c: ArrayOut2<'c, F>,
    alpha: F,
    beta: F,
    transa: c_char,
    transb: c_char,
    n: c_int,
    m: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
}

impl<'a, 'b, 'c, F> GEMM_Driver<'a, 'b, 'c, F>
where
    F: BLASFloat
{
    pub fn run(self) -> Result<ArrayOut2<'c, F>, AnyError>
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
            ArrayOut::ToBeCloned(_, c) => c.as_mut_ptr(),
        };
        let ldc = self.ldc;

        BLASFunc::<F>::gemm(
            &transa, &transb,
            &m, &n, &k,
            &alpha, a_ptr, &lda,
            b_ptr, &ldb,
            &beta, c_ptr, &ldc
        );
        Ok(c.clone_to_view_mut())
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct GEMM_<'a, 'b, 'c, F>
where
    F: BLASFloat
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(default = "F::one()")]
    pub alpha: F,
    #[builder(default = "F::zero()")]
    pub beta: F,
    #[builder(try_setter, default = "BLASTrans::NoTrans")]
    pub transa: BLASTrans,
    #[builder(try_setter, default = "BLASTrans::NoTrans")]
    pub transb: BLASTrans,
}

impl<'a, 'b, 'c, F> GEMM_<'a, 'b, 'c, F>
where 
    F: BLASFloat
{
    pub fn driver(self) -> Result<GEMM_Driver<'a, 'b, 'c, F>, AnyError>
    {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let transa = self.transa;
        let transb = self.transb;
        let alpha = self.alpha;
        let beta = self.beta;
        
        // currently only fortran-preferred (col-major) is accepted
        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        if !(layout_a.is_fpref() && layout_b.is_fpref()) {
            BLASError("Inner driver should be fortran-only. This is probably error of library author.".to_string());
        }

        // initialize intent(hide)
        let m = if transa != BLASTrans::NoTrans { a.dim().1 } else { a.dim().0 };
        let k = if transa != BLASTrans::NoTrans { a.dim().0 } else { a.dim().1 };
        let n = if transb != BLASTrans::NoTrans { b.dim().0 } else { b.dim().1 };
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));
        
        // perform check
        if transb != BLASTrans::NoTrans {
            BLASError::assert(
                k == b.dim().1,
                format!("Incompatible dimensions for matrix multiplication, k={:}, b.dim[1]={:}.", k, b.dim().1))?;
        } else {
            BLASError::assert(
                k == b.dim().0,
                format!("Incompatible dimensions for matrix multiplication, k={:}, b.dim[0]={:}.", k, b.dim().0))?;
        }

        // optional intent(out)
        let c = match c {
            Some(c) => {
                BLASError::assert(
                    m == c.dim().0,
                    format!("Incompatible dimensions for matrix multiplication, m={:}, c.dim[0]={:}.", m, c.dim().0))?;
                BLASError::assert(
                    n == c.dim().1,
                    format!("Incompatible dimensions for matrix multiplication, n={:}, c.dim[1]={:}.", n, c.dim().1))?;
                if get_layout_array2(&c.view()).is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    ArrayOut2::ToBeCloned(c, Array2::zeros((m, n).f()))
                }
            },
            None => {
                ArrayOut2::Owned(Array2::zeros((m, n).f()))
            }
        };

        let ldc = c.view().stride_of(Axis(1));
        
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
            ldc: ldc.try_into()?,
        };

        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type GEMM<'a, 'b, 'c, F> = GEMM_Builder<'a, 'b, 'c, F>;
pub type SGEMM<'a, 'b, 'c> = GEMM<'a, 'b, 'c, f32>;
pub type DGEMM<'a, 'b, 'c> = GEMM<'a, 'b, 'c, f64>;
pub type CGEMM<'a, 'b, 'c> = GEMM<'a, 'b, 'c, c32>;
pub type ZGEMM<'a, 'b, 'c> = GEMM<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F> GEMM<'a, 'b, 'c, F>
where
    F: BLASFloat,
    BLASFunc<F>: GEMMFunc<F>,
{
    pub fn run(self) -> Result<ArrayOut2<'c, F>, AnyError>
    {
        // initialize
        let obj = self.build()?;
        
        // currently only fortran-preferred (col-major) is accepted
        let layout_a = get_layout_array2(&obj.a);
        let layout_b = get_layout_array2(&obj.b);

        if layout_a.is_fpref() && layout_b.is_fpref() {
            return obj.driver()?.run()
        } else if layout_a.is_cpref() && layout_b.is_cpref() {
            let obj = GEMM_ {
                a: obj.b.reversed_axes(),
                b: obj.a.reversed_axes(),
                c: match obj.c {
                    Some(c) => Some(c.reversed_axes()),
                    None => None,
                },
                alpha: obj.alpha,
                beta: obj.beta,
                transa: obj.transb,
                transb: obj.transa,
            };
            let c = obj.driver()?.run()?.reversed_axes();
            return Ok(c);
        } else {
            let a_owned = match obj.a.is_standard_layout() {
                true => None,
                false => Some(obj.a.as_standard_layout().into_owned()),
            };
            let b_owned = match obj.b.is_standard_layout() {
                true => None,
                false => Some(obj.b.as_standard_layout().into_owned()),
            };
            let a_view = a_owned.as_ref().map_or(obj.a.view(), |a| a.view());
            let b_view = b_owned.as_ref().map_or(obj.b.view(), |b| b.view());
            let obj = GEMM_ {
                a: b_view.t(),
                b: a_view.t(),
                c: match obj.c {
                    Some(c) => Some(c.reversed_axes()),
                    None => None,
                },
                alpha: obj.alpha,
                beta: obj.beta,
                transa: obj.transb,
                transb: obj.transa,
            };
            let c = obj.driver()?.run()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */
