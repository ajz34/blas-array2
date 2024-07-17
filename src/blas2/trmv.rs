use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TRMVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn trmv(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const c_int,
        a: *const F,
        lda: *const c_int,
        x: *mut F,
        incx: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TRMVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn trmv(
                uplo: *const c_char,
                trans: *const c_char,
                diag: *const c_char,
                n: *const c_int,
                a: *const $type,
                lda: *const c_int,
                x: *mut $type,
                incx: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(uplo, trans, diag, n, a as *const FFIFloat, lda, x as *mut FFIFloat, incx);
            }
        }
    };
}

impl_func!(f32, strmv_);
impl_func!(f64, dtrmv_);
impl_func!(c32, ctrmv_);
impl_func!(c64, ztrmv_);

/* #endregion */

/* #region BLAS driver */

pub struct TRMV_Driver<'a, 'x, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    a: ArrayView2<'a, F>,
    lda: c_int,
    x: ArrayOut1<'x, F>,
    incx: c_int,
}

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TRMV_Driver<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TRMVFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;
        let n = self.n;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
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
            BLASFunc::trmv(&uplo, &trans, &diag, &n, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"))]
pub struct TRMV_<'a, 'x, F>
where
    F: BLASFloat,
{
    pub a: ArrayView2<'a, F>,
    pub x: ArrayViewMut1<'x, F>,

    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into), default = "BLASNonUnit")]
    pub diag: BLASDiag,
}

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TRMV_<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TRMVFunc<F>,
{
    fn driver(self) -> Result<TRMV_Driver<'a, 'x, F>, BLASError> {
        let a = self.a;
        let x = self.x;
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());

        // initialize intent(hide)
        let (n, n_) = a.dim();
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(n, n_, InvalidDim)?;
        blas_assert_eq!(x.len_of(Axis(0)), n, InvalidDim)?;

        // prepare output
        let x = ArrayOut1::ViewMut(x);

        // finalize
        let driver = TRMV_Driver {
            uplo: uplo.into(),
            trans: trans.into(),
            diag: diag.into(),
            n: n.try_into()?,
            a,
            lda: lda.try_into()?,
            x,
            incx: incx.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type TRMV<'a, 'x, F> = TRMV_Builder<'a, 'x, F>;
pub type STRMV<'a, 'x> = TRMV<'a, 'x, f32>;
pub type DTRMV<'a, 'x> = TRMV<'a, 'x, f64>;
pub type CTRMV<'a, 'x> = TRMV<'a, 'x, c32>;
pub type ZTRMV<'a, 'x> = TRMV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TRMV_Builder<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TRMVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous: x = op(A) x
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous:
            let a_cow = obj.a.as_standard_layout();
            match obj.trans {
                BLASNoTrans => {
                    // N -> T: x = op(A')' x
                    let obj = TRMV_ {
                        a: a_cow.t(),
                        trans: BLASTrans,
                        uplo: match obj.uplo {
                            BLASUpper => BLASLower,
                            BLASLower => BLASUpper,
                            _ => blas_invalid!(obj.uplo)?,
                        },
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // T -> N: x = op(A') x
                    let obj = TRMV_ {
                        a: a_cow.t(),
                        trans: BLASNoTrans,
                        uplo: match obj.uplo {
                            BLASUpper => BLASLower,
                            BLASLower => BLASUpper,
                            _ => blas_invalid!(obj.uplo)?,
                        },
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> T: x* = op(A') x*; x = x*
                    let mut x = obj.x;
                    x.mapv_inplace(F::conj);
                    let obj = TRMV_ {
                        a: a_cow.t(),
                        x,
                        trans: BLASNoTrans,
                        uplo: match obj.uplo {
                            BLASUpper => BLASLower,
                            BLASLower => BLASUpper,
                            _ => blas_invalid!(obj.uplo)?,
                        },
                        ..obj
                    };
                    let mut x = obj.driver()?.run_blas()?;
                    x.view_mut().mapv_inplace(F::conj);
                    return Ok(x);
                },
                _ => return blas_invalid!(&obj.trans)?,
            }
        }
    }
}

/* #endregion */
