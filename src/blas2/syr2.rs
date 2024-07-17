use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SYR2Func<F>
where
    F: BLASFloat,
{
    unsafe fn syr2(
        uplo: *const c_char,
        n: *const c_int,
        alpha: *const F,
        x: *const F,
        incx: *const c_int,
        y: *const F,
        incy: *const c_int,
        a: *mut F,
        lda: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl SYR2Func<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn syr2(
                uplo: *const c_char,
                n: *const c_int,
                alpha: *const $type,
                x: *const $type,
                incx: *const c_int,
                y: *const $type,
                incy: *const c_int,
                a: *mut $type,
                lda: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    uplo,
                    n,
                    alpha as *const FFIFloat,
                    x as *const FFIFloat,
                    incx,
                    y as *const FFIFloat,
                    incy,
                    a as *mut FFIFloat,
                    lda,
                );
            }
        }
    };
}

impl_func!(f32, ssyr2_);
impl_func!(f64, dsyr2_);
impl_func!(c32, cher2_);
impl_func!(c64, zher2_);

/* #endregion */

/* #region BLAS driver */

pub struct SYR2_Driver<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    n: c_int,
    alpha: F,
    x: ArrayView1<'x, F>,
    incx: c_int,
    y: ArrayView1<'y, F>,
    incy: c_int,
    a: ArrayOut2<'a, F>,
    lda: c_int,
}

impl<'x, 'y, 'a, F> BLASDriver<'a, F, Ix2> for SYR2_Driver<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: SYR2Func<F>,
{
    fn run_blas(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        let Self { uplo, n, alpha, x, incx, y, incy, mut a, lda } = self;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let a_ptr = match &mut a {
            ArrayOut2::Owned(a) => a.as_mut_ptr(),
            ArrayOut2::ViewMut(a) => a.as_mut_ptr(),
            ArrayOut2::ToBeCloned(_, a) => a.as_mut_ptr(),
        };

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(a.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::syr2(&uplo, &n, &alpha, x_ptr, &incx, y_ptr, &incy, a_ptr, &lda);
        }
        return Ok(a.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SYR2_<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    pub x: ArrayView1<'x, F>,
    pub y: ArrayView1<'y, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub a: Option<ArrayViewMut2<'a, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
}

impl<'x, 'y, 'a, F> BLASBuilder_<'a, F, Ix2> for SYR2_<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: SYR2Func<F>,
{
    fn driver(self) -> Result<SYR2_Driver<'x, 'y, 'a, F>, BLASError> {
        let Self { x, y, a, alpha, uplo, .. } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let incy = y.stride_of(Axis(0));
        let n = x.len_of(Axis(0));

        // check optional
        blas_assert_eq!(y.len_of(Axis(0)), n, InvalidDim)?;

        // prepare output
        let a = match a {
            Some(a) => {
                blas_assert_eq!(a.dim(), (n, n), InvalidDim)?;
                if get_layout_array2(&a.view()).is_fpref() {
                    ArrayOut2::ViewMut(a)
                } else {
                    let a_buffer = a.t().as_standard_layout().into_owned().reversed_axes();
                    ArrayOut2::ToBeCloned(a, a_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((n, n).f())),
        };
        let lda = a.view().stride_of(Axis(1));

        // finalize
        let driver = SYR2_Driver {
            uplo: uplo.into(),
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

pub type SYR2<'x, 'y, 'a, F> = SYR2_Builder<'x, 'y, 'a, F>;
pub type SSYR2<'x, 'y, 'a> = SYR2<'x, 'y, 'a, f32>;
pub type DSYR2<'x, 'y, 'a> = SYR2<'x, 'y, 'a, f64>;

pub type HER2<'x, 'y, 'a, F> = SYR2_Builder<'x, 'y, 'a, F>;
pub type CHER2<'x, 'y, 'a> = HER2<'x, 'y, 'a, c32>;
pub type ZHER2<'x, 'y, 'a> = HER2<'x, 'y, 'a, c64>;

impl<'x, 'y, 'a, F> BLASBuilder<'a, F, Ix2> for SYR2_Builder<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: SYR2Func<F>,
{
    fn run(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        if obj.a.as_ref().map(|a| get_layout_array2(&a.view()).is_fpref()) == Some(true) {
            // F-contiguous
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let uplo = match obj.uplo {
                BLASUpper => BLASLower,
                BLASLower => BLASUpper,
                _ => blas_invalid!(obj.uplo)?,
            };
            let a = obj.a.map(|a| a.reversed_axes());
            if F::is_complex() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.mapv(F::conj);
                let obj = SYR2_ { a, y: x.view(), x: y.view(), uplo, ..obj };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            } else {
                let obj = SYR2_ { a, uplo, x: obj.y, y: obj.x, ..obj };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            };
        }
    }
}

/* #endregion */
