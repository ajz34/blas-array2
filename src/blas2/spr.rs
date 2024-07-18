use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SPRFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn spr(
        uplo: *const c_char,
        n: *const c_int,
        alpha: *const F,
        x: *const F,
        incx: *const c_int,
        ap: *mut F,
    );
}

macro_rules! impl_func {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SPRFunc<$type, $symm> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn spr(
                uplo: *const c_char,
                n: *const c_int,
                alpha: *const $type,
                x: *const $type,
                incx: *const c_int,
                ap: *mut $type,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                type FFIHermitialFloat = <<$symm as BLASSymmetric>::HermitianFloat as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    uplo,
                    n,
                    alpha as *const FFIHermitialFloat,
                    x as *const FFIFloat,
                    incx,
                    ap as *mut FFIFloat,
                );
            }
        }
    };
}

impl_func!(f32, BLASSymm<f32>, sspr_);
impl_func!(f64, BLASSymm<f64>, dspr_);
impl_func!(c32, BLASHermi<c32>, chpr_);
impl_func!(c64, BLASHermi<c64>, zhpr_);
// these two functions are actually in lapack, not blas
// impl_func!(c32, BLASSymm<c32>, cspr_);
// impl_func!(c64, BLASSymm<c64>, zspr_);

/* #endregion */

/* #region BLAS driver */

pub struct SPR_Driver<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    uplo: c_char,
    n: c_int,
    alpha: F,
    x: ArrayView1<'x, F>,
    incx: c_int,
    ap: ArrayOut1<'a, F>,

    _phantom: core::marker::PhantomData<S>,
}

impl<'x, 'a, F, S> BLASDriver<'a, F, Ix1> for SPR_Driver<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SPRFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut1<'a, F>, BLASError> {
        let Self { uplo, n, alpha, x, incx, mut ap, .. } = self;
        let x_ptr = x.as_ptr();
        let ap_ptr = ap.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(ap.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::spr(&uplo, &n, &alpha, x_ptr, &incx, ap_ptr);
        }
        return Ok(ap.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SPR_<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    pub x: ArrayView1<'x, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub ap: Option<ArrayViewMut1<'a, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,

    #[builder(private, default = "core::marker::PhantomData {}")]
    _phantom: core::marker::PhantomData<S>,
}

impl<'x, 'a, F, S> BLASBuilder_<'a, F, Ix1> for SPR_<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SPRFunc<F, S>,
{
    fn driver(self) -> Result<SPR_Driver<'x, 'a, F, S>, BLASError> {
        let Self { x, ap, alpha, uplo, layout, .. } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let n = x.len_of(Axis(0));

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));

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
        let driver = SPR_Driver {
            uplo: uplo.into(),
            n: n.try_into()?,
            alpha,
            x,
            incx: incx.try_into()?,
            ap,
            _phantom: core::marker::PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SPR<'x, 'a, F> = SPR_Builder<'x, 'a, F, BLASSymm<F>>;
pub type SSPR<'x, 'a> = SPR<'x, 'a, f32>;
pub type DSPR<'x, 'a> = SPR<'x, 'a, f64>;

pub type HPR<'x, 'a, F> = SPR_Builder<'x, 'a, F, BLASHermi<F>>;
pub type CHPR<'x, 'a> = HPR<'x, 'a, c32>;
pub type ZHPR<'x, 'a> = HPR<'x, 'a, c64>;

impl<'x, 'a, F, S> BLASBuilder<'a, F, Ix1> for SPR_Builder<'x, 'a, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SPRFunc<F, S>,
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
            if S::is_hermitian() {
                let x = obj.x.mapv(F::conj);
                let obj = SPR_ { x: x.view(), uplo, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            } else {
                let obj = SPR_ { uplo, layout: Some(BLASColMajor), ..obj };
                return obj.driver()?.run_blas();
            };
        }
    }
}

/* #endregion */
