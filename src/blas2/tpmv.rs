use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TPMVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn tpmv(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const c_int,
        ap: *const F,
        x: *mut F,
        incx: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TPMVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn tpmv(
                uplo: *const c_char,
                trans: *const c_char,
                diag: *const c_char,
                n: *const c_int,
                ap: *const $type,
                x: *mut $type,
                incx: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(uplo, trans, diag, n, ap as *const FFIFloat, x as *mut FFIFloat, incx);
            }
        }
    };
}

impl_func!(f32, stpmv_);
impl_func!(f64, dtpmv_);
impl_func!(c32, ctpmv_);
impl_func!(c64, ztpmv_);

/* #endregion */

/* #region BLAS driver */

pub struct TPMV_Driver<'a, 'x, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    ap: ArrayView1<'a, F>,
    x: ArrayOut1<'x, F>,
    incx: c_int,
}

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TPMV_Driver<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TPMVFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;
        let n = self.n;
        let ap_ptr = self.ap.as_ptr();
        let mut x = self.x;
        let x_ptr = match &mut x {
            ArrayOut1::ViewMut(y) => y.as_mut_ptr(),
            _ => panic!("Ix1 with triangular A, won't be ToBeCloned or Owned"),
        };
        let incx = self.incx;

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(x);
        }

        unsafe {
            BLASFunc::tpmv(&uplo, &trans, &diag, &n, ap_ptr, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"))]

pub struct TPMV_<'a, 'x, F>
where
    F: BLASFloat,
{
    pub ap: ArrayView1<'a, F>,
    pub x: ArrayViewMut1<'x, F>,

    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into), default = "BLASNonUnit")]
    pub diag: BLASDiag,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TPMV_<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TPMVFunc<F>,
{
    fn driver(self) -> Result<TPMV_Driver<'a, 'x, F>, BLASError> {
        let ap = self.ap;
        let x = self.x;
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;
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
        let x = ArrayOut1::ViewMut(x);

        // finalize
        let driver = TPMV_Driver {
            uplo: uplo.into(),
            trans: trans.into(),
            diag: diag.into(),
            n: n.try_into()?,
            ap,
            x,
            incx: incx.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type TPMV<'a, 'x, F> = TPMV_Builder<'a, 'x, F>;
pub type STPMV<'a, 'x> = TPMV<'a, 'x, f32>;
pub type DTPMV<'a, 'x> = TPMV<'a, 'x, f64>;
pub type CTPMV<'a, 'x> = TPMV<'a, 'x, c32>;
pub type ZTPMV<'a, 'x> = TPMV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TPMV_Builder<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TPMVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout = match obj.layout {
            Some(layout) => layout,
            None => BLASRowMajor,
        };

        if layout == BLASColMajor {
            // F-contiguous
            let ap_cow = obj.ap.as_standard_layout();
            let obj = TPMV_ { ap: ap_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let ap_cow = obj.ap.as_standard_layout();
            match obj.trans {
                BLASNoTrans => {
                    // N -> T
                    let obj = TPMV_ {
                        ap: ap_cow.view(),
                        trans: BLASTrans,
                        uplo: match obj.uplo {
                            BLASUpper => BLASLower,
                            BLASLower => BLASUpper,
                            _ => blas_invalid!(obj.uplo)?,
                        },
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // T -> N
                    let obj = TPMV_ {
                        ap: ap_cow.view(),
                        trans: BLASNoTrans,
                        uplo: match obj.uplo {
                            BLASUpper => BLASLower,
                            BLASLower => BLASUpper,
                            _ => blas_invalid!(obj.uplo)?,
                        },
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> N
                    let mut x = obj.x;
                    x.mapv_inplace(F::conj);
                    let obj = TPMV_ {
                        ap: ap_cow.view(),
                        x,
                        trans: BLASNoTrans,
                        uplo: match obj.uplo {
                            BLASUpper => BLASLower,
                            BLASLower => BLASUpper,
                            _ => blas_invalid!(obj.uplo)?,
                        },
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    let mut x = obj.driver()?.run_blas()?;
                    x.view_mut().mapv_inplace(F::conj);
                    return Ok(x);
                },
                _ => return blas_invalid!(obj.trans)?,
            }
        }
    }
}

/* #endregion */
