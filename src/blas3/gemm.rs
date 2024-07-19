use crate::ffi::{self, blasint, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GEMMFunc<F>
where
    F: BLASFloat,
{
    unsafe fn gemm(
        transa: *const c_char,
        transb: *const c_char,
        m: *const blasint,
        n: *const blasint,
        k: *const blasint,
        alpha: *const F,
        a: *const F,
        lda: *const blasint,
        b: *const F,
        ldb: *const blasint,
        beta: *const F,
        c: *mut F,
        ldc: *const blasint,
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
                m: *const blasint,
                n: *const blasint,
                k: *const blasint,
                alpha: *const $type,
                a: *const $type,
                lda: *const blasint,
                b: *const $type,
                ldb: *const blasint,
                beta: *const $type,
                c: *mut $type,
                ldc: *const blasint,
            ) {
                ffi::$func(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blasint,
    b: ArrayView2<'b, F>,
    ldb: blasint,
    beta: F,
    c: ArrayOut2<'c, F>,
    ldc: blasint,
}

impl<'a, 'b, 'c, F> BLASDriver<'c, F, Ix2> for GEMM_Driver<'a, 'b, 'c, F>
where
    F: BLASFloat,
    BLASFunc: GEMMFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, mut c, ldc } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(c.clone_to_view_mut());
        } else if k == 0 {
            if beta == F::zero() {
                c.view_mut().fill(F::zero());
            } else if beta != F::one() {
                c.view_mut().mapv_inplace(|v| v * beta);
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
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
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
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'b, 'c, F> BLASBuilder_<'c, F, Ix2> for GEMM_<'a, 'b, 'c, F>
where
    F: BLASFloat,
    BLASFunc: GEMMFunc<F>,
{
    fn driver(self) -> Result<GEMM_Driver<'a, 'b, 'c, F>, BLASError> {
        let Self { a, b, c, alpha, beta, transa, transb, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref() && b.is_fpref());

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
            BLASNoTrans => blas_assert_eq!(b.len_of(Axis(0)), k, InvalidDim)?,
            BLASTrans | BLASConjTrans => blas_assert_eq!(b.len_of(Axis(1)), k, InvalidDim)?,
            _ => blas_invalid!(transb)?,
        }

        // optional intent(out)
        let c = match c {
            Some(c) => {
                blas_assert_eq!(c.dim(), (m, n), InvalidDim)?;
                if c.view().is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.view().to_col_layout()?.into_owned();
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
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let GEMM_ { a, b, c, alpha, beta, transa, transb, layout } = self.build()?;
        let at = a.t();
        let bt = b.t();

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            // F-contiguous: C = op(A) op(B)
            let (transa, a_cow) = flip_trans_fpref(transa, &a, &at, false)?;
            let (transb, b_cow) = flip_trans_fpref(transb, &b, &bt, false)?;
            let obj = GEMM_ {
                a: a_cow.view(),
                b: b_cow.view(),
                c,
                alpha,
                beta,
                transa,
                transb,
                layout: Some(BLASColMajor),
            };
            return obj.driver()?.run_blas();
        } else if layout == BLASRowMajor {
            // C-contiguous: C' = op(B') op(A')
            let (transa, a_cow) = flip_trans_cpref(transa, &a, &at, false)?;
            let (transb, b_cow) = flip_trans_cpref(transb, &b, &bt, false)?;
            let obj = GEMM_ {
                a: b_cow.t(),
                b: a_cow.t(),
                c: c.map(|c| c.reversed_axes()),
                alpha,
                beta,
                transa: transb,
                transb: transa,
                layout: Some(BLASColMajor),
            };
            return Ok(obj.driver()?.run_blas()?.reversed_axes());
        } else {
            panic!("This is designed not to execuate this line.");
        }
    }
}

/* #endregion */
