use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GEMVNum: BLASFloat {
    unsafe fn gemv(
        trans: *const c_char,
        m: *const blas_int,
        n: *const blas_int,
        alpha: *const Self,
        a: *const Self,
        lda: *const blas_int,
        x: *const Self,
        incx: *const blas_int,
        beta: *const Self,
        y: *mut Self,
        incy: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GEMVNum for $type {
            unsafe fn gemv(
                trans: *const c_char,
                m: *const blas_int,
                n: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                x: *const Self,
                incx: *const blas_int,
                beta: *const Self,
                y: *mut Self,
                incy: *const blas_int,
            ) {
                ffi::$func(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
            }
        }
    };
}

impl_func!(f32, sgemv_);
impl_func!(f64, dgemv_);
impl_func!(c32, cgemv_);
impl_func!(c64, zgemv_);

/* #endregion */

/* #region BLAS driver */

pub struct GEMV_Driver<'a, 'x, 'y, F>
where
    F: BLASFloat,
{
    trans: c_char,
    m: blas_int,
    n: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: blas_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for GEMV_Driver<'a, 'x, 'y, F>
where
    F: GEMVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let Self { trans, m, n, alpha, a, lda, x, incx, beta, mut y, incy } = self;
        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(y);
        }

        unsafe {
            F::gemv(&trans, &m, &n, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct GEMV_<'a, 'x, 'y, F>
where
    F: GEMVNum,
{
    pub a: ArrayView2<'a, F>,
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub y: Option<ArrayViewMut1<'y, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for GEMV_<'a, 'x, 'y, F>
where
    F: GEMVNum,
{
    fn driver(self) -> Result<GEMV_Driver<'a, 'x, 'y, F>, BLASError> {
        let Self { a, x, y, alpha, beta, trans } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());

        // initialize intent(hide)
        let (m, n) = a.dim();
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        match trans {
            BLASNoTrans => blas_assert_eq!(x.len_of(Axis(0)), n, InvalidDim)?,
            BLASTrans | BLASConjTrans => blas_assert_eq!(x.len_of(Axis(0)), m, InvalidDim)?,
            _ => blas_invalid!(trans)?,
        };

        // prepare output
        let y = match y {
            Some(y) => {
                match trans {
                    BLASNoTrans => blas_assert_eq!(y.len_of(Axis(0)), m, InvalidDim)?,
                    BLASTrans | BLASConjTrans => blas_assert_eq!(y.len_of(Axis(0)), n, InvalidDim)?,
                    _ => blas_invalid!(trans)?,
                };
                ArrayOut1::ViewMut(y)
            },
            None => ArrayOut1::Owned(Array1::zeros(match trans {
                BLASNoTrans => m,
                BLASTrans | BLASConjTrans => n,
                _ => blas_invalid!(trans)?,
            })),
        };
        let incy = y.view().stride_of(Axis(0));

        // finalize
        let driver = GEMV_Driver {
            trans: trans.try_into()?,
            m: m.try_into()?,
            n: n.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            x,
            incx: incx.try_into()?,
            beta,
            y,
            incy: incy.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type GEMV<'a, 'x, 'y, F> = GEMV_Builder<'a, 'x, 'y, F>;
pub type SGEMV<'a, 'x, 'y> = GEMV<'a, 'x, 'y, f32>;
pub type DGEMV<'a, 'x, 'y> = GEMV<'a, 'x, 'y, f64>;
pub type CGEMV<'a, 'x, 'y> = GEMV<'a, 'x, 'y, c32>;
pub type ZGEMV<'a, 'x, 'y> = GEMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for GEMV_Builder<'a, 'x, 'y, F>
where
    F: GEMVNum,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous: y = alpha op(A) x + beta y
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = obj.a.to_row_layout()?;
            match obj.trans {
                BLASNoTrans => {
                    // N -> T: y = alpha (A')' x + beta y
                    let obj = GEMV_ { a: a_cow.t(), trans: BLASTrans, ..obj };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // T -> N: y = alpha (A') x + beta y
                    let obj = GEMV_ { a: a_cow.t(), trans: BLASNoTrans, ..obj };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> N: y* = alpha* (A') x* + beta* y*; y = y*
                    let x = obj.x.mapv(F::conj);
                    let y = obj.y.map(|mut y| {
                        y.mapv_inplace(F::conj);
                        y
                    });
                    let obj = GEMV_ {
                        a: a_cow.t(),
                        trans: BLASNoTrans,
                        x: x.view(),
                        y,
                        alpha: F::conj(obj.alpha),
                        beta: F::conj(obj.beta),
                    };
                    let mut y = obj.driver()?.run_blas()?;
                    y.view_mut().mapv_inplace(F::conj);
                    return Ok(y);
                },
                _ => return blas_invalid!(&obj.trans)?,
            };
        }
    }
}

/* #endregion */
