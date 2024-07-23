use crate::blas2::ger::{GERNum, GER_};
use crate::ffi::{self, blas_int};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GERCNum: BLASFloat + GERNum {
    unsafe fn gerc(
        m: *const blas_int,
        n: *const blas_int,
        alpha: *const Self,
        x: *const Self,
        incx: *const blas_int,
        y: *const Self,
        incy: *const blas_int,
        a: *mut Self,
        lda: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GERCNum for $type {
            unsafe fn gerc(
                m: *const blas_int,
                n: *const blas_int,
                alpha: *const Self,
                x: *const Self,
                incx: *const blas_int,
                y: *const Self,
                incy: *const blas_int,
                a: *mut Self,
                lda: *const blas_int,
            ) {
                ffi::$func(m, n, alpha, x, incx, y, incy, a, lda);
            }
        }
    };
}

impl_func!(c32, cgerc_);
impl_func!(c64, zgerc_);

/* #endregion */

/* #region BLAS driver */

pub struct GERC_Driver<'x, 'y, 'a, F>
where
    F: GERCNum,
{
    m: blas_int,
    n: blas_int,
    alpha: F,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    y: ArrayView1<'y, F>,
    incy: blas_int,
    a: ArrayOut2<'a, F>,
    lda: blas_int,
}

impl<'x, 'y, 'a, F> BLASDriver<'a, F, Ix2> for GERC_Driver<'x, 'y, 'a, F>
where
    F: GERCNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        let Self { m, n, alpha, x, incx, y, incy, mut a, lda } = self;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let a_ptr = a.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(a.clone_to_view_mut());
        }

        unsafe {
            F::gerc(&m, &n, &alpha, x_ptr, &incx, y_ptr, &incy, a_ptr, &lda);
        }
        return Ok(a.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct GERC_<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    pub x: ArrayView1<'x, F>,
    pub y: ArrayView1<'y, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub a: Option<ArrayViewMut2<'a, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
}

impl<'x, 'y, 'a, F> BLASBuilder_<'a, F, Ix2> for GERC_<'x, 'y, 'a, F>
where
    F: GERCNum,
{
    fn driver(self) -> Result<GERC_Driver<'x, 'y, 'a, F>, BLASError> {
        let Self { x, y, a, alpha } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let incy = y.stride_of(Axis(0));
        let m = x.len_of(Axis(0));
        let n = y.len_of(Axis(0));

        // prepare output
        let a = match a {
            Some(a) => {
                blas_assert_eq!(a.dim(), (m, n), InvalidDim)?;
                if a.view().is_fpref() {
                    ArrayOut2::ViewMut(a)
                } else {
                    let a_buffer = a.view().to_col_layout()?.into_owned();
                    ArrayOut2::ToBeCloned(a, a_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((m, n).f())),
        };
        let lda = a.view().stride_of(Axis(1));

        // finalize
        let driver = GERC_Driver {
            m: m.try_into()?,
            n: n.try_into()?,
            alpha,
            x,
            incx: incx.try_into()?,
            y,
            incy: incy.try_into()?,
            a,
            lda: lda.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type GERC<'x, 'y, 'a, F> = GERC_Builder<'x, 'y, 'a, F>;
pub type CGERC<'x, 'y, 'a> = GERC<'x, 'y, 'a, c32>;
pub type ZGERC<'x, 'y, 'a> = GERC<'x, 'y, 'a, c64>;

impl<'x, 'y, 'a, F> BLASBuilder<'a, F, Ix2> for GERC_Builder<'x, 'y, 'a, F>
where
    F: GERCNum,
{
    fn run(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        if obj.a.as_ref().map(|a| a.view().is_fpref()) == Some(true) {
            // F-contiguous
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a = obj.a.map(|a| a.reversed_axes());
            let y = obj.y.mapv(F::conj);
            let obj = GER_ { a, x: y.view(), y: obj.x, alpha: obj.alpha };
            let a = obj.driver()?.run_blas()?;
            return Ok(a.reversed_axes());
        }
    }
}

/* #endregion */
