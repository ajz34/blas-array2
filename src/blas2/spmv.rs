use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SPMVFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn spmv(
        uplo: *const c_char,
        n: *const c_int,
        alpha: *const F,
        ap: *const F,
        x: *const F,
        incx: *const c_int,
        beta: *const F,
        y: *mut F,
        incy: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SPMVFunc<$type, $symm> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn spmv(
                uplo: *const c_char,
                n: *const c_int,
                alpha: *const $type,
                ap: *const $type,
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
                    alpha as *const FFIFloat,
                    ap as *const FFIFloat,
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

impl_func!(f32, BLASSymm<f32>, sspmv_);
impl_func!(f64, BLASSymm<f64>, dspmv_);
impl_func!(c32, BLASHermi<c32>, chpmv_);
impl_func!(c64, BLASHermi<c64>, zhpmv_);
// these two functions are actually in lapack, not blas
// impl_func!(c32, BLASSymm<c32>, cspmv_);
// impl_func!(c64, BLASSymm<c64>, zspmv_);

/* #endregion */

/* #region BLAS driver */

pub struct SPMV_Driver<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    uplo: c_char,
    n: c_int,
    alpha: F,
    ap: ArrayView1<'a, F>,
    x: ArrayView1<'x, F>,
    incx: c_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: c_int,
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'x, 'y, F, S> BLASDriver<'y, F, Ix1> for SPMV_Driver<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SPMVFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let uplo = self.uplo;
        let n = self.n;
        let alpha = self.alpha;
        let ap_ptr = self.ap.as_ptr();
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

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            BLASFunc::spmv(&uplo, &n, &alpha, ap_ptr, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"))]
pub struct SPMV_<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
{
    pub ap: ArrayView1<'a, F>,
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

    #[builder(private, default = "core::marker::PhantomData {}")]
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'x, 'y, F, S> BLASBuilder_<'y, F, Ix1> for SPMV_<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SPMVFunc<F, S>,
{
    fn driver(self) -> Result<SPMV_Driver<'a, 'x, 'y, F, S>, BLASError> {
        let ap = self.ap;
        let x = self.x;
        let y = self.y;
        let alpha = self.alpha;
        let beta = self.beta;
        let uplo = self.uplo;
        let layout = self.layout;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let incap = ap.stride_of(Axis(0));
        assert!(incap <= 1);
        assert!(layout.unwrap() == BLASColMajor);

        // initialize intent(hide)
        let np = ap.len_of(Axis(0));
        let n = x.len_of(Axis(0));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(np, n * (n + 1) / 2, InvalidDim)?;

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
        let driver = SPMV_Driver {
            uplo: uplo.into(),
            n: n.try_into()?,
            alpha,
            ap,
            x,
            incx: incx.try_into()?,
            beta,
            y,
            incy: incy.try_into()?,
            _phantom: core::marker::PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SPMV<'a, 'x, 'y, F> = SPMV_Builder<'a, 'x, 'y, F, BLASSymm<F>>;
pub type SSPMV<'a, 'x, 'y> = SPMV<'a, 'x, 'y, f32>;
pub type DSPMV<'a, 'x, 'y> = SPMV<'a, 'x, 'y, f64>;

pub type HPMV<'a, 'x, 'y, F> = SPMV_Builder<'a, 'x, 'y, F, BLASHermi<F>>;
pub type CHPMV<'a, 'x, 'y> = HPMV<'a, 'x, 'y, c32>;
pub type ZHPMV<'a, 'x, 'y> = HPMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F, S> BLASBuilder<'y, F, Ix1> for SPMV_Builder<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SPMVFunc<F, S>,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout = match obj.layout {
            Some(layout) => layout,
            None => BLASRowMajor,
        };

        if layout == BLASColMajor {
            // F-contiguous
            let ap_cow = obj.ap.as_standard_layout();
            let obj = SPMV_ { ap: ap_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let ap_cow = obj.ap.as_standard_layout();
            if S::is_hermitian() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.map(|mut y| {
                    y.mapv_inplace(F::conj);
                    y
                });
                let obj = SPMV_ {
                    ap: ap_cow.view(),
                    x: x.view(),
                    y,
                    uplo: match obj.uplo {
                        BLASUpper => BLASLower,
                        BLASLower => BLASUpper,
                        _ => blas_invalid!(obj.uplo)?,
                    },
                    alpha: F::conj(obj.alpha),
                    beta: F::conj(obj.beta),
                    layout: Some(BLASColMajor),
                    ..obj
                };
                let mut y = obj.driver()?.run_blas()?;
                y.view_mut().mapv_inplace(F::conj);
                return Ok(y);
            } else {
                let obj = SPMV_ {
                    ap: ap_cow.view(),
                    uplo: match obj.uplo {
                        BLASUpper => BLASLower,
                        BLASLower => BLASUpper,
                        _ => blas_invalid!(obj.uplo)?,
                    },
                    layout: Some(BLASColMajor),
                    ..obj
                };
                return obj.driver()?.run_blas();
            }
        }
    }
}

/* #endregion */
