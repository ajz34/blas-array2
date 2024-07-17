use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SYRFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn syr(
        uplo: *const c_char,
        n: *const c_int,
        alpha: *const F,
        x: *const F,
        incx: *const c_int,
        a: *mut F,
        lda: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYRFunc<$type, $symm> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn syr(
                uplo: *const c_char,
                n: *const c_int,
                alpha: *const $type,
                x: *const $type,
                incx: *const c_int,
                a: *mut $type,
                lda: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                type FFIHermitialFloat = <<$symm as BLASSymmetric>::HermitianFloat as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    uplo,
                    n,
                    alpha as *const FFIHermitialFloat,
                    x as *const FFIFloat,
                    incx,
                    a as *mut FFIFloat,
                    lda,
                );
            }
        }
    };
}

impl_func!(f32, BLASSymm<f32>, ssyr_);
impl_func!(f64, BLASSymm<f64>, dsyr_);
impl_func!(c32, BLASHermi<c32>, cher_);
impl_func!(c64, BLASHermi<c64>, zher_);
// these two functions are actually in lapack, not blas
// impl_func!(c32, BLASSymm<c32>, csyr_);
// impl_func!(c64, BLASSymm<c64>, zsyr_);

/* #endregion */

/* #region BLAS driver */

pub struct SYR_Driver<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    uplo: c_char,
    n: c_int,
    alpha: F,
    x: ArrayView1<'x, F>,
    incx: c_int,
    a: ArrayOut2<'a, F>,
    lda: c_int,

    _phantom: core::marker::PhantomData<S>,
}

impl<'x, 'a, F, S> BLASDriver<'a, F, Ix2> for SYR_Driver<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYRFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut2<'a, F>, BLASError> {
        let Self { uplo, n, alpha, x, incx, mut a, lda, .. } = self;
        let x_ptr = x.as_ptr();
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
            BLASFunc::syr(&uplo, &n, &alpha, x_ptr, &incx, a_ptr, &lda);
        }
        return Ok(a.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SYR_<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub a: Option<ArrayViewMut2<'a, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,

    #[builder(private, default = "core::marker::PhantomData {}")]
    _phantom: core::marker::PhantomData<S>,
}

impl<'x, 'a, F, S> BLASBuilder_<'a, F, Ix2> for SYR_<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYRFunc<F, S>,
{
    fn driver(self) -> Result<SYR_Driver<'x, 'a, F, S>, BLASError> {
        let Self { x, a, alpha, uplo, .. } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));

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
        let driver = SYR_Driver {
            uplo: uplo.into(),
            n: n.try_into()?,
            alpha,
            x,
            incx: incx.try_into()?,
            a,
            lda: lda.try_into()?,
            _phantom: core::marker::PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SYR<'x, 'a, F> = SYR_Builder<'x, 'a, F, BLASSymm<F>>;
pub type SSYR<'x, 'a> = SYR<'x, 'a, f32>;
pub type DSYR<'x, 'a> = SYR<'x, 'a, f64>;

pub type HER<'x, 'a, F> = SYR_Builder<'x, 'a, F, BLASHermi<F>>;
pub type CHER<'x, 'a> = HER<'x, 'a, c32>;
pub type ZHER<'x, 'a> = HER<'x, 'a, c64>;

impl<'x, 'a, F, S> BLASBuilder<'a, F, Ix2> for SYR_Builder<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYRFunc<F, S>,
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
            if S::is_hermitian() {
                let x = obj.x.mapv(F::conj);
                let obj = SYR_ { a, x: x.view(), uplo, ..obj };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            } else {
                let obj = SYR_ { a, uplo, ..obj };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            };
        }
    }
}

/* #endregion */
