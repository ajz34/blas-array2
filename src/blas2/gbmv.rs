use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GBMVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn gbmv(
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        kl: *const c_int,
        ku: *const c_int,
        alpha: *const F,
        a: *const F,
        lda: *const c_int,
        x: *const F,
        incx: *const c_int,
        beta: *const F,
        y: *mut F,
        incy: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GBMVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn gbmv(
                trans: *const c_char,
                m: *const c_int,
                n: *const c_int,
                kl: *const c_int,
                ku: *const c_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const c_int,
                x: *const $type,
                incx: *const c_int,
                beta: *const $type,
                y: *mut $type,
                incy: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    trans,
                    m,
                    n,
                    kl,
                    ku,
                    alpha as *const FFIFloat,
                    a as *const FFIFloat,
                    lda,
                    x as *const FFIFloat,
                    incx,
                    beta as *const FFIFloat,
                    y as *mut FFIFloat,
                    incy,
                );
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
    F: BLASFloat,
{
    trans: c_char,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    x: ArrayView1<'x, F>,
    incx: c_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: c_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for GBMV_Driver<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: GBMVFunc<F>,
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
            BLASFunc::gbmv(&trans, &m, &n, &kl, &ku, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
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
    F: BLASFloat,
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
    F: BLASFloat,
    BLASFunc: GBMVFunc<F>,
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
            trans: trans.into(),
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
    F: BLASFloat,
    BLASFunc: GBMVFunc<F>,
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
