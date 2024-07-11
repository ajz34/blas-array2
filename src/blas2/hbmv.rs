use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait HBMVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn hbmv(
        uplo: *const c_char,
        n: *const c_int,
        k: *const c_int,
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
        impl HBMVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn hbmv(
                uplo: *const c_char,
                n: *const c_int,
                k: *const c_int,
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
                    uplo,
                    n,
                    k,
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

impl_func!(f32, ssbmv_);
impl_func!(f64, dsbmv_);
impl_func!(c32, chbmv_);
impl_func!(c64, zhbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct HBMV_Driver<'a, 'x, 'y, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    n: c_int,
    k: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    x: ArrayView1<'x, F>,
    incx: c_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: c_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for HBMV_Driver<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: HBMVFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, AnyError> {
        let uplo = self.uplo;
        let n = self.n;
        let k = self.k;
        let alpha = self.alpha;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
        let x_ptr = self.x.as_ptr();
        let incx = self.incx;
        let beta = self.beta;
        let mut y = self.y;
        let y_ptr = match &mut y {
            ArrayOut1::Owned(y) => y.as_mut_ptr(),
            ArrayOut1::ViewMut(y) => y.as_mut_ptr(),
            _ => panic!("Ix1 won't be ToBeCloned"),
        };
        let incy = self.incy;

        unsafe {
            BLASFunc::hbmv(&uplo, &n, &k, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct HBMV_<'a, 'x, 'y, F>
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
    #[builder(setter(into), default = "BLASUpLo::Upper")]
    pub uplo: BLASUpLo,
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for HBMV_<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: HBMVFunc<F>,
{
    fn driver(self) -> Result<HBMV_Driver<'a, 'x, 'y, F>, AnyError> {
        let a = self.a;
        let x = self.x;
        let y = self.y;
        let alpha = self.alpha;
        let beta = self.beta;
        let uplo = self.uplo;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());

        // initialize intent(hide)
        let (k_, n) = a.dim();
        blas_assert!(k_ > 0, "Rows of input `a` must larger than zero.")?;
        let k = k_ - 1;
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(x.len_of(Axis(0)), n, "Incompatible dimensions")?;

        // prepare output
        let y = match y {
            Some(y) => {
                blas_assert_eq!(y.len_of(Axis(0)), n, "Incompatible dimensions")?;
                ArrayOut1::ViewMut(y)
            },
            None => ArrayOut1::Owned(Array1::zeros(n)),
        };
        let incy = y.view().stride_of(Axis(0));

        // finalize
        let driver = HBMV_Driver {
            uplo: uplo.into(),
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
pub type SSYMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, f32>;
pub type DSYMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, f64>;
pub type CHBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, c32>;
pub type ZHBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for HBMV_Builder<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: HBMVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous: y = alpha op(A) x + beta y
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: transpose to F-contiguous
            eprintln!("{:}:{:}: Warning message from blas-array2", file!(), line!());
            eprintln!("Banded storage not suitable for C-contiguous without explicit transposition.");
            eprintln!("Also see https://github.com/Reference-LAPACK/lapack/issues/1032.");
            let a_fpref = obj.a.reversed_axes().as_standard_layout().reversed_axes().into_owned();
            let obj = HBMV_ { a: a_fpref.view(), ..obj };
            return obj.driver()?.run_blas();
        }
    }
}

/* #endregion */
