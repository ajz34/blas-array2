use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait HBMVNum: BLASFloat {
    unsafe fn hbmv(
        uplo: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
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
        impl HBMVNum for $type {
            unsafe fn hbmv(
                uplo: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                x: *const Self,
                incx: *const blas_int,
                beta: *const Self,
                y: *mut Self,
                incy: *const blas_int,
            ) {
                ffi::$func(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
            }
        }
    };
}

impl_func!(f32, ssbmv_);
impl_func!(f64, dsbmv_);
impl_func!(c32, chbmv_);
impl_func!(c64, zhbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct HBMV_Driver<'a, 'x, 'y, F>
where
    F: HBMVNum,
{
    uplo: c_char,
    n: blas_int,
    k: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: blas_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for HBMV_Driver<'a, 'x, 'y, F>
where
    F: HBMVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let Self { uplo, n, k, alpha, a, lda, x, incx, beta, mut y, incy, .. } = self;
        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            F::hbmv(&uplo, &n, &k, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HBMV_<'a, 'x, 'y, F>
where
    F: HBMVNum,
{
    pub a: ArrayView2<'a, F>,
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub y: Option<ArrayViewMut1<'y, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for HBMV_<'a, 'x, 'y, F>
where
    F: HBMVNum,
{
    fn driver(self) -> Result<HBMV_Driver<'a, 'x, 'y, F>, BLASError> {
        let Self { a, x, y, alpha, beta, uplo, layout, .. } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());
        assert!(layout == Some(BLASLayout::ColMajor));

        // initialize intent(hide)
        let (k_, n) = a.dim();
        blas_assert!(k_ > 0, InvalidDim, "Rows of input `a` must larger than zero.")?;
        let k = k_ - 1;
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(x.len_of(Axis(0)), n, InvalidDim)?;

        // prepare output
        let y = match y {
            Some(y) => {
                blas_assert_eq!(y.len_of(Axis(0)), n, InvalidDim)?;
                ArrayOut1::ViewMut(y)
            },
            None => ArrayOut1::Owned(Array1::zeros(n)),
        };
        let incy = y.view().stride_of(Axis(0));

        // finalize
        let driver = HBMV_Driver {
            uplo: uplo.try_into()?,
            n: n.try_into()?,
            k: k.try_into()?,
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

pub type HBMV<'a, 'x, 'y, F> = HBMV_Builder<'a, 'x, 'y, F>;
pub type SSBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, f32>;
pub type DSBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, f64>;
pub type CHBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, c32>;
pub type ZHBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for HBMV_Builder<'a, 'x, 'y, F>
where
    F: HBMVNum,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout = get_layout_row_preferred(&[obj.layout, Some(layout_a)], &[]);

        if layout == BLASColMajor {
            // F-contiguous
            let a_cow = obj.a.to_col_layout()?;
            let obj = HBMV_ { a: a_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = obj.a.to_row_layout()?;
            if F::is_complex() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.map(|mut y| {
                    y.mapv_inplace(F::conj);
                    y
                });
                let obj = HBMV_ {
                    a: a_cow.t(),
                    x: x.view(),
                    y,
                    uplo: obj.uplo.flip()?,
                    alpha: F::conj(obj.alpha),
                    beta: F::conj(obj.beta),
                    layout: Some(BLASColMajor),
                    ..obj
                };
                let mut y = obj.driver()?.run_blas()?;
                y.view_mut().mapv_inplace(F::conj);
                return Ok(y);
            } else {
                let obj = HBMV_ { a: a_cow.t(), uplo: obj.uplo.flip()?, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            }
        }
    }
}

/* #endregion */
