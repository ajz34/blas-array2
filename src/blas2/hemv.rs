use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait HEMVNum: BLASFloat {
    unsafe fn hemv(
        uplo: *const c_char,
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
        impl HEMVNum for $type {
            unsafe fn hemv(
                uplo: *const c_char,
                n: *const blas_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const blas_int,
                x: *const $type,
                incx: *const blas_int,
                beta: *const $type,
                y: *mut $type,
                incy: *const blas_int,
            ) {
                ffi::$func(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
            }
        }
    };
}

impl_func!(f32, ssymv_);
impl_func!(f64, dsymv_);
impl_func!(c32, chemv_);
impl_func!(c64, zhemv_);

/* #endregion */

/* #region BLAS driver */

pub struct HEMV_Driver<'a, 'x, 'y, F>
where
    F: HEMVNum,
{
    uplo: c_char,
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

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for HEMV_Driver<'a, 'x, 'y, F>
where
    F: HEMVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let Self { uplo, n, alpha, a, lda, x, incx, beta, mut y, incy, .. } = self;
        let a_ptr = a.as_ptr();
        let x_ptr = x.as_ptr();
        let y_ptr = y.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            F::hemv(&uplo, &n, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HEMV_<'a, 'x, 'y, F>
where
    F: BLASFloat,
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
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for HEMV_<'a, 'x, 'y, F>
where
    F: HEMVNum,
{
    fn driver(self) -> Result<HEMV_Driver<'a, 'x, 'y, F>, BLASError> {
        let Self { a, x, y, alpha, beta, uplo, .. } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());

        // initialize intent(hide)
        let (n_, n) = a.dim();
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(n, n_, InvalidDim)?;
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
        let driver = HEMV_Driver {
            uplo: uplo.into(),
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

pub type HEMV<'a, 'x, 'y, F> = HEMV_Builder<'a, 'x, 'y, F>;
pub type SSYMV<'a, 'x, 'y> = HEMV<'a, 'x, 'y, f32>;
pub type DSYMV<'a, 'x, 'y> = HEMV<'a, 'x, 'y, f64>;
pub type CHEMV<'a, 'x, 'y> = HEMV<'a, 'x, 'y, c32>;
pub type ZHEMV<'a, 'x, 'y> = HEMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for HEMV_Builder<'a, 'x, 'y, F>
where
    F: HEMVNum,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous
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
                let obj = HEMV_ {
                    a: a_cow.t(),
                    x: x.view(),
                    y,
                    uplo: obj.uplo.flip(),
                    alpha: F::conj(obj.alpha),
                    beta: F::conj(obj.beta),
                    ..obj
                };
                let mut y = obj.driver()?.run_blas()?;
                y.view_mut().mapv_inplace(F::conj);
                return Ok(y);
            } else {
                let obj = HEMV_ { a: a_cow.t(), uplo: obj.uplo.flip(), ..obj };
                return obj.driver()?.run_blas();
            }
        }
    }
}

/* #endregion */
