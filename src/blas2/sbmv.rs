use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SBMVFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
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
    ($type: ty, $symm: ty, $func: ident) => {
        impl SBMVFunc<$type, $symm> for BLASFunc
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

impl_func!(f32, BLASSymm<f32>, ssbmv_);
impl_func!(f64, BLASSymm<f64>, dsbmv_);
impl_func!(c32, BLASHermi<c32>, chbmv_);
impl_func!(c64, BLASHermi<c64>, zhbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct SBMV_Driver<'a, 'x, 'y, F, S>
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
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'x, 'y, F, S> BLASDriver<'y, F, Ix1> for SBMV_Driver<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SBMVFunc<F, S>,
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
            BLASFunc::hbmv(&uplo, &n, &k, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SBMV_<'a, 'x, 'y, F, S>
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
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,

    #[builder(private, default = "core::marker::PhantomData {}")]
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'x, 'y, F, S> BLASBuilder_<'y, F, Ix1> for SBMV_<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SBMVFunc<F, S>,
{
    fn driver(self) -> Result<SBMV_Driver<'a, 'x, 'y, F, S>, BLASError> {
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
        let driver = SBMV_Driver {
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
            _phantom: core::marker::PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SBMV<'a, 'x, 'y, F> = SBMV_Builder<'a, 'x, 'y, F, BLASSymm<F>>;
pub type SSBMV<'a, 'x, 'y> = SBMV<'a, 'x, 'y, f32>;
pub type DSBMV<'a, 'x, 'y> = SBMV<'a, 'x, 'y, f64>;

pub type HBMV<'a, 'x, 'y, F> = SBMV_Builder<'a, 'x, 'y, F, BLASHermi<F>>;
pub type CHBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, c32>;
pub type ZHBMV<'a, 'x, 'y> = HBMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F, S> BLASBuilder<'y, F, Ix1> for SBMV_Builder<'a, 'x, 'y, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SBMVFunc<F, S>,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout = get_layout_row_preferred(&[obj.layout, Some(layout_a)], &[]);

        if layout == BLASColMajor {
            // F-contiguous
            let a_cow = obj.a.to_col_layout()?;
            let obj = SBMV_ { a: a_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = obj.a.to_row_layout()?;
            if S::is_hermitian() {
                let x = obj.x.mapv(F::conj);
                let y = obj.y.map(|mut y| {
                    y.mapv_inplace(F::conj);
                    y
                });
                let obj = SBMV_ {
                    a: a_cow.t(),
                    x: x.view(),
                    y,
                    uplo: obj.uplo.flip(),
                    alpha: F::conj(obj.alpha),
                    beta: F::conj(obj.beta),
                    layout: Some(BLASColMajor),
                    ..obj
                };
                let mut y = obj.driver()?.run_blas()?;
                y.view_mut().mapv_inplace(F::conj);
                return Ok(y);
            } else {
                let obj = SBMV_ { a: a_cow.t(), uplo: obj.uplo.flip(), layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            }
        }
    }
}

/* #endregion */
