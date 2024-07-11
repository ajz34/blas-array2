use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TRSVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn trsv(
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
        impl TRSVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn trsv(
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

impl_func!(f32, strsv_);
impl_func!(f64, dtrsv_);
impl_func!(c32, ctrsv_);
impl_func!(c64, ztrsv_);

/* #endregion */

/* #region BLAS driver */

pub struct TRSV_Driver<'a, 'x, F>
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

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TRSV_Driver<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TRSVFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, AnyError> {
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

        unsafe {
            BLASFunc::trsv(&uplo, &trans, &diag, &n, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct TRSV_<'a, 'x, F>
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

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TRSV_<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TRSVFunc<F>,
{
    fn driver(self) -> Result<TRSV_Driver<'a, 'x, F>, AnyError> {
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
        blas_assert_eq!(n, n_, "Incompatible dimensions")?;
        blas_assert_eq!(x.len_of(Axis(0)), n, "Incompatible dimensions")?;

        // prepare output
        let x = ArrayOut1::ViewMut(x);

        // finalize
        let driver = TRSV_Driver {
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

pub type TRSV<'a, 'x, F> = TRSV_Builder<'a, 'x, F>;
pub type STRSV<'a, 'x> = TRSV<'a, 'x, f32>;
pub type DTRSV<'a, 'x> = TRSV<'a, 'x, f64>;
pub type CTRSV<'a, 'x> = TRSV<'a, 'x, c32>;
pub type ZTRSV<'a, 'x> = TRSV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TRSV_Builder<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TRSVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, AnyError> {
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
                    let obj = TRSV_ {
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
                    let obj = TRSV_ {
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
                    let obj = TRSV_ {
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