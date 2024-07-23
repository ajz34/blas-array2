use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait HEMMNum: BLASFloat {
    unsafe fn hemm(
        side: *const c_char,
        uplo: *const c_char,
        m: *const blas_int,
        n: *const blas_int,
        alpha: *const Self,
        a: *const Self,
        lda: *const blas_int,
        b: *const Self,
        ldb: *const blas_int,
        beta: *const Self,
        c: *mut Self,
        ldc: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl HEMMNum for $type {
            unsafe fn hemm(
                side: *const c_char,
                uplo: *const c_char,
                m: *const blas_int,
                n: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                b: *const Self,
                ldb: *const blas_int,
                beta: *const Self,
                c: *mut Self,
                ldc: *const blas_int,
            ) {
                ffi::$func(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
            }
        }
    };
}

impl_func!(c32, chemm_);
impl_func!(c64, zhemm_);

/* #endregion */

/* #region BLAS driver */

pub struct HEMM_Driver<'a, 'b, 'c, F>
where
    F: HEMMNum,
{
    side: c_char,
    uplo: c_char,
    m: blas_int,
    n: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    b: ArrayView2<'b, F>,
    ldb: blas_int,
    beta: F,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
}

impl<'a, 'b, 'c, F> BLASDriver<'c, F, Ix2> for HEMM_Driver<'a, 'b, 'c, F>
where
    F: HEMMNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { side, uplo, m, n, alpha, a, lda, b, ldb, beta, mut c, ldc, .. } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            F::hemm(&side, &uplo, &m, &n, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HEMM_<'a, 'b, 'c, F>
where
    F: HEMMNum,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASLeft")]
    pub side: BLASSide,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'b, 'c, F> BLASBuilder_<'c, F, Ix2> for HEMM_<'a, 'b, 'c, F>
where
    F: HEMMNum,
{
    fn driver(self) -> Result<HEMM_Driver<'a, 'b, 'c, F>, BLASError> {
        let Self { a, b, c, alpha, beta, side, uplo, layout, .. } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref() && a.is_fpref());

        // initialize intent(hide)
        let m = b.len_of(Axis(0));
        let n = b.len_of(Axis(1));
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        match side {
            BLASLeft => blas_assert_eq!(a.dim(), (m, m), InvalidDim)?,
            BLASRight => blas_assert_eq!(a.dim(), (n, n), InvalidDim)?,
            _ => blas_invalid!(side)?,
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
        let driver = HEMM_Driver::<'a, 'b, 'c, F> {
            side: side.try_into()?,
            uplo: uplo.try_into()?,
            m: m.try_into()?,
            n: n.try_into()?,
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

pub type HEMM<'a, 'b, 'c, F> = HEMM_Builder<'a, 'b, 'c, F>;
pub type CHEMM<'a, 'b, 'c> = HEMM<'a, 'b, 'c, c32>;
pub type ZHEMM<'a, 'b, 'c> = HEMM<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F> BLASBuilder<'c, F, Ix2> for HEMM_Builder<'a, 'b, 'c, F>
where
    F: HEMMNum,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let HEMM_ { a, b, c, alpha, beta, side, uplo, layout, .. } = self.build()?;

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            // F-contiguous: C = op(A) op(B)
            let a_cow = a.to_col_layout()?;
            let b_cow = b.to_col_layout()?;
            let obj = HEMM_ {
                a: a_cow.view(),
                b: b_cow.view(),
                c,
                alpha,
                beta,
                side,
                uplo,
                layout: Some(BLASColMajor),
            };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: C' = op(B') op(A')
            let a_cow = a.to_row_layout()?;
            let b_cow = b.to_row_layout()?;
            let obj = HEMM_ {
                a: a_cow.t(),
                b: b_cow.t(),
                c: c.map(|c| c.reversed_axes()),
                alpha,
                beta,
                side: side.flip()?,
                uplo: uplo.flip()?,
                layout: Some(BLASColMajor),
            };
            let c = obj.driver()?.run_blas()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */
