use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GBMVNum: BLASFloat {
    unsafe fn gbmv(
        trans: *const c_char,
        m: *const blas_int,
        n: *const blas_int,
        kl: *const blas_int,
        ku: *const blas_int,
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
        impl GBMVNum for $type {
            unsafe fn gbmv(
                trans: *const c_char,
                m: *const blas_int,
                n: *const blas_int,
                kl: *const blas_int,
                ku: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                x: *const Self,
                incx: *const blas_int,
                beta: *const Self,
                y: *mut Self,
                incy: *const blas_int,
            ) {
                ffi::$func(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
            }
        }
    };
}

impl_func!(f32, sgbmv_);
impl_func!(f64, dgbmv_);
impl_func!(c32, cgbmv_);
impl_func!(c64, zgbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct GBMV_Driver<'a, 'x, 'y, F>
where
    F: GBMVNum,
{
    trans: c_char,
    m: blas_int,
    n: blas_int,
    kl: blas_int,
    ku: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: blas_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for GBMV_Driver<'a, 'x, 'y, F>
where
    F: GBMVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let Self { trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, mut y, incy } = self;
        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            F::gbmv(&trans, &m, &n, &kl, &ku, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct GBMV_<'a, 'x, 'y, F>
where
    F: GBMVNum,
{
    pub a: ArrayView2<'a, F>,
    pub x: ArrayView1<'x, F>,
    pub m: usize,
    pub kl: usize,

    #[builder(setter(into, strip_option), default = "None")]
    pub y: Option<ArrayViewMut1<'y, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for GBMV_<'a, 'x, 'y, F>
where
    F: GBMVNum,
{
    fn driver(self) -> Result<GBMV_Driver<'a, 'x, 'y, F>, BLASError> {
        let Self { a, x, m, kl, y, alpha, beta, trans, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());
        assert!(layout == Some(BLASLayout::ColMajor));

        // initialize intent(hide)
        let (k, n) = a.dim();
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert!(k > kl, InvalidDim)?;
        blas_assert!(m >= k, InvalidDim)?;
        let ku = k - 1 - kl;
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
        let driver = GBMV_Driver {
            trans: trans.try_into()?,
            m: m.try_into()?,
            n: n.try_into()?,
            kl: kl.try_into()?,
            ku: ku.try_into()?,
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

pub type GBMV<'a, 'x, 'y, F> = GBMV_Builder<'a, 'x, 'y, F>;
pub type SGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, f32>;
pub type DGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, f64>;
pub type CGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, c32>;
pub type ZGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for GBMV_Builder<'a, 'x, 'y, F>
where
    F: GBMVNum,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let GBMV_ { a, x, m, kl, y, alpha, beta, trans, layout } = self.build()?;

        let layout_a = get_layout_array2(&a);
        let layout = match layout {
            Some(layout) => layout,
            None => match layout_a {
                BLASLayout::Sequential => BLASColMajor,
                BLASRowMajor => BLASRowMajor,
                BLASColMajor => BLASColMajor,
                _ => blas_raise!(InvalidFlag, "Without defining layout, this function checks layout of input matrix `a` but it is not contiguous.")?,
            }
        };

        if layout == BLASColMajor {
            // F-contiguous
            let a_cow = a.to_col_layout()?;
            let obj = GBMV_ { a: a_cow.view(), x, m, kl, y, alpha, beta, trans, layout: Some(BLASColMajor) };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = a.to_row_layout()?;
            let k = a_cow.len_of(Axis(1));
            blas_assert!(k > kl, InvalidDim)?;
            let ku = k - kl - 1;
            match trans {
                BLASNoTrans => {
                    // N -> T
                    let obj = GBMV_ {
                        a: a_cow.t(),
                        x,
                        m,
                        kl: ku,
                        y,
                        alpha,
                        beta,
                        trans: BLASTrans,
                        layout: Some(BLASColMajor),
                    };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // N -> T
                    let obj = GBMV_ {
                        a: a_cow.t(),
                        x,
                        m,
                        kl: ku,
                        y,
                        alpha,
                        beta,
                        trans: BLASNoTrans,
                        layout: Some(BLASColMajor),
                    };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> N
                    let x = x.mapv(F::conj);
                    let y = y.map(|mut y| {
                        y.mapv_inplace(F::conj);
                        y
                    });
                    let obj = GBMV_ {
                        a: a_cow.t(),
                        x: x.view(),
                        m,
                        kl: ku,
                        y,
                        alpha: F::conj(alpha),
                        beta: F::conj(beta),
                        trans: BLASNoTrans,
                        layout: Some(BLASColMajor),
                    };
                    let mut y = obj.driver()?.run_blas()?;
                    y.view_mut().mapv_inplace(F::conj);
                    return Ok(y);
                },
                _ => return blas_invalid!(trans)?,
            }
        }
    }
}

/* #endregion */
