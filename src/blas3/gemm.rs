use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GEMMFunc<F>
where
    F: BLASFloat,
{
    unsafe fn gemm(
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
        ldc: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GEMMFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn gemm(
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
                ldc: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha as *const FFIFloat,
                    a as *const FFIFloat,
                    lda,
                    b as *const FFIFloat,
                    ldb,
                    beta as *const FFIFloat,
                    c as *mut FFIFloat,
                    ldc,
                );
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
    F: BLASFloat,
{
    transa: c_char,
    transb: c_char,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    b: ArrayView2<'b, F>,
    ldb: c_int,
    beta: F,
    c: ArrayOut2<'c, F>,
    ldc: c_int,
}

impl<'a, 'b, 'c, F> BLASDriver<'c, F, Ix2> for GEMM_Driver<'a, 'b, 'c, F>
where
    F: BLASFloat,
    BLASFunc: GEMMFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, AnyError> {
        let transa = self.transa;
        let transb = self.transb;
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

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(c.clone_to_view_mut());
        } else if k == 0 {
            if beta == F::zero() {
                c.view_mut().fill(F::zero());
            } else if beta != F::one() {
                c.view_mut().mapv_inplace(| v | v * beta);
            }
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::gemm(
                &transa, &transb, &m, &n, &k, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc,
            );
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct GEMM_<'a, 'b, 'c, F>
where
    F: BLASFloat,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub transa: BLASTranspose,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub transb: BLASTranspose,
}

impl<'a, 'b, 'c, F> BLASBuilder_<'c, F, Ix2> for GEMM_<'a, 'b, 'c, F>
where
    F: BLASFloat,
    BLASFunc: GEMMFunc<F>,
{
    fn driver(self) -> Result<GEMM_Driver<'a, 'b, 'c, F>, AnyError> {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let transa = self.transa;
        let transb = self.transb;
        let alpha = self.alpha;
        let beta = self.beta;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        assert!(layout_a.is_fpref() && layout_b.is_fpref());

        // initialize intent(hide)
        let (m, k) = match transa {
            BLASNoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASTrans | BLASConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(transa)?,
        };
        let n = match transb {
            BLASNoTrans => b.len_of(Axis(1)),
            BLASTrans | BLASConjTrans => b.len_of(Axis(0)),
            _ => blas_invalid!(transb)?,
        };
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        match transb {
            BLASNoTrans => blas_assert_eq!(b.len_of(Axis(0)), k, "Incompatible dimensions")?,
            BLASTrans | BLASConjTrans => blas_assert_eq!(b.len_of(Axis(1)), k, "Incompatible dimensions")?,
            _ => blas_invalid!(transb)?,
        }

        // optional intent(out)
        let c = match c {
            Some(c) => {
                blas_assert_eq!(c.dim(), (m, n), "Incompatible dimensions")?;
                if get_layout_array2(&c.view()).is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.t().as_standard_layout().into_owned().reversed_axes();
                    ArrayOut2::ToBeCloned(c, c_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((m, n).f())),
        };
        let ldc = c.view().stride_of(Axis(1));

        // finalize
        let driver = GEMM_Driver {
            transa: transa.into(),
            transb: transb.into(),
            m: m.try_into()?,
            n: n.try_into()?,
            k: k.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            b,
            ldb: ldb.try_into()?,
            beta,
            c,
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

impl<'a, 'b, 'c, F> BLASBuilder<'c, F, Ix2> for GEMM_Builder<'a, 'b, 'c, F>
where
    F: BLASFloat,
    BLASFunc: GEMMFunc<F>,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout_b = get_layout_array2(&obj.b);

        if layout_a.is_fpref() && layout_b.is_fpref() {
            // F-contiguous: C = op(A) op(B)
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: C' = op(B') op(A')
            let a_cow = obj.a.as_standard_layout();
            let b_cow = obj.b.as_standard_layout();
            let obj = GEMM_ {
                a: b_cow.t(),
                b: a_cow.t(),
                c: obj.c.map(|c| c.reversed_axes()),
                alpha: obj.alpha,
                beta: obj.beta,
                transa: obj.transb,
                transb: obj.transa,
            };
            let c = obj.driver()?.run_blas()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */
