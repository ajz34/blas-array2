use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SPR2Func<F>
where
    F: BLASFloat,
{
    unsafe fn spr2(
        uplo: *const c_char,
        n: *const blas_int,
        alpha: *const F,
        x: *const F,
        incx: *const blas_int,
        y: *const F,
        incy: *const blas_int,
        ap: *mut F,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl SPR2Func<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn spr2(
                uplo: *const c_char,
                n: *const blas_int,
                alpha: *const $type,
                x: *const $type,
                incx: *const blas_int,
                y: *const $type,
                incy: *const blas_int,
                ap: *mut $type,
            ) {
                ffi::$func(uplo, n, alpha, x, incx, y, incy, ap);
            }
        }
    };
}

impl_func!(f32, sspr2_);
impl_func!(f64, dspr2_);
impl_func!(c32, chpr2_);
impl_func!(c64, zhpr2_);

/* #endregion */

/* #region BLAS driver */

pub struct SPR2_Driver<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    n: blas_int,
    alpha: F,
    x: ArrayView1<'x, F>,
    incx: blas_int,
    y: ArrayView1<'y, F>,
    incy: blas_int,
    ap: ArrayOut1<'a, F>,
}

impl<'x, 'y, 'a, F> BLASDriver<'a, F, Ix1> for SPR2_Driver<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: SPR2Func<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'a, F>, BLASError> {
        let Self { uplo, n, alpha, x, incx, y, incy, mut ap } = self;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let ap_ptr = ap.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(ap.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::spr2(&uplo, &n, &alpha, x_ptr, &incx, y_ptr, &incy, ap_ptr);
        }
        return Ok(ap.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SPR2_<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    pub x: ArrayView1<'x, F>,
    pub y: ArrayView1<'y, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub ap: Option<ArrayViewMut1<'a, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'x, 'y, 'a, F> BLASBuilder_<'a, F, Ix1> for SPR2_<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: SPR2Func<F>,
{
    fn driver(self) -> Result<SPR2_Driver<'x, 'y, 'a, F>, BLASError> {
        let Self { x, y, ap, alpha, uplo, layout, .. } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let incy = y.stride_of(Axis(0));
        let n = x.len_of(Axis(0));

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));

        // check optional
        blas_assert_eq!(y.len_of(Axis(0)), n, InvalidDim)?;

        // prepare output
        let ap = match ap {
            Some(ap) => {
                blas_assert_eq!(ap.len_of(Axis(0)), n * (n + 1) / 2, InvalidDim)?;
                if ap.is_standard_layout() {
                    ArrayOut1::ViewMut(ap)
                } else {
                    let ap_buffer = ap.view().to_seq_layout()?.into_owned();
                    ArrayOut1::ToBeCloned(ap, ap_buffer)
                }
            },
            None => ArrayOut1::Owned(Array1::zeros(n * (n + 1) / 2)),
        };

        // finalize
        let driver = SPR2_Driver {
            uplo: uplo.into(),
            n: n.try_into()?,
            alpha,
            x,
            incx: incx.try_into()?,
            y,
            incy: incy.try_into()?,
            ap,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SPR2<'x, 'y, 'a, F> = SPR2_Builder<'x, 'y, 'a, F>;
pub type SSPR2<'x, 'y, 'a> = SPR2<'x, 'y, 'a, f32>;
pub type DSPR2<'x, 'y, 'a> = SPR2<'x, 'y, 'a, f64>;

pub type HPR2<'x, 'y, 'a, F> = SPR2_Builder<'x, 'y, 'a, F>;
pub type CHPR2<'x, 'y, 'a> = HPR2<'x, 'y, 'a, c32>;
pub type ZHPR2<'x, 'y, 'a> = HPR2<'x, 'y, 'a, c64>;

impl<'x, 'y, 'a, F> BLASBuilder<'a, F, Ix1> for SPR2_Builder<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: SPR2Func<F>,
{
    fn run(self) -> Result<ArrayOut1<'a, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        if obj.layout == Some(BLASColMajor) {
            // F-contiguous
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let uplo = obj.uplo.flip();
            if F::is_complex() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.mapv(F::conj);
                let obj = SPR2_ { y: x.view(), x: y.view(), uplo, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            } else {
                let obj = SPR2_ { uplo, x: obj.y, y: obj.x, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            };
        }
    }
}

/* #endregion */
