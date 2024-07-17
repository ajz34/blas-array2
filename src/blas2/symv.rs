use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SYMVFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn hemv(
        uplo: *const c_char,
        n: *const c_int,
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
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYMVFunc<$type, $symm> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn hemv(
                uplo: *const c_char,
                n: *const c_int,
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

impl_func!(f32, BLASSymm<f32>, ssymv_);
impl_func!(f64, BLASSymm<f64>, dsymv_);
impl_func!(c32, BLASHermi<c32>, chemv_);
impl_func!(c64, BLASHermi<c64>, zhemv_);
// these two functions are actually in lapack, not blas
// impl_func!(c32, BLASSymm<c32>, csymv_);
// impl_func!(c64, BLASSymm<c64>, zsymv_);

/* #endregion */

/* #region BLAS driver */

pub struct SYMV_Driver<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    uplo: c_char,
    n: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    x: ArrayView1<'x, F>,
    incx: c_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: c_int,
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'x, 'y, F, S> BLASDriver<'y, F, Ix1> for SYMV_Driver<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMVFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        let uplo = self.uplo;
        let n = self.n;
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

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            BLASFunc::hemv(&uplo, &n, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"))]
pub struct SYMV_<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
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

    #[builder(private, default = "core::marker::PhantomData {}")]
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'x, 'y, F, S> BLASBuilder_<'y, F, Ix1> for SYMV_<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMVFunc<F, S>,
{
    fn driver(self) -> Result<SYMV_Driver<'a, 'x, 'y, F, S>, BLASError> {
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
        let driver = SYMV_Driver {
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
            _phantom: core::marker::PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SYMV<'a, 'x, 'y, F> = SYMV_Builder<'a, 'x, 'y, F, BLASSymm<F>>;
pub type SSYMV<'a, 'x, 'y> = SYMV<'a, 'x, 'y, f32>;
pub type DSYMV<'a, 'x, 'y> = SYMV<'a, 'x, 'y, f64>;

pub type HEMV<'a, 'x, 'y, F> = SYMV_Builder<'a, 'x, 'y, F, BLASHermi<F>>;
pub type CHEMV<'a, 'x, 'y> = HEMV<'a, 'x, 'y, c32>;
pub type ZHEMV<'a, 'x, 'y> = HEMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F, S> BLASBuilder<'y, F, Ix1> for SYMV_Builder<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMVFunc<F, S>,
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
            let a_cow = obj.a.as_standard_layout();
            if S::is_hermitian() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.map(|mut y| {
                    y.mapv_inplace(F::conj);
                    y
                });
                let obj = SYMV_ {
                    a: a_cow.t(),
                    x: x.view(),
                    y,
                    uplo: match obj.uplo {
                        BLASUpper => BLASLower,
                        BLASLower => BLASUpper,
                        _ => blas_invalid!(obj.uplo)?,
                    },
                    alpha: F::conj(obj.alpha),
                    beta: F::conj(obj.beta),
                    ..obj
                };
                let mut y = obj.driver()?.run_blas()?;
                y.view_mut().mapv_inplace(F::conj);
                return Ok(y);
            } else {
                let obj = SYMV_ {
                    a: a_cow.t(),
                    uplo: match obj.uplo {
                        BLASUpper => BLASLower,
                        BLASLower => BLASUpper,
                        _ => blas_invalid!(obj.uplo)?,
                    },
                    ..obj
                };
                return obj.driver()?.run_blas();
            }
        }
    }
}

/* #endregion */
