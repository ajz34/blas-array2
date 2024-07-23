use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TRSVNum: BLASFloat {
    unsafe fn trsv(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blas_int,
        a: *const Self,
        lda: *const blas_int,
        x: *mut Self,
        incx: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TRSVNum for $type {
            unsafe fn trsv(
                uplo: *const c_char,
                trans: *const c_char,
                diag: *const c_char,
                n: *const blas_int,
                a: *const Self,
                lda: *const blas_int,
                x: *mut Self,
                incx: *const blas_int,
            ) {
                ffi::$func(uplo, trans, diag, n, a, lda, x, incx);
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
    F: TRSVNum,
{
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: blas_int,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    x: ArrayOut1<'x, F>,
    incx: blas_int,
}

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TRSV_Driver<'a, 'x, F>
where
    F: TRSVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        let Self { uplo, trans, diag, n, a, lda, mut x, incx } = self;
        let a_ptr = a.as_ptr();
        let x_ptr = x.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(x);
        }

        unsafe {
            F::trsv(&uplo, &trans, &diag, &n, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct TRSV_<'a, 'x, F>
where
    F: TRSVNum,
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
    F: TRSVNum,
{
    fn driver(self) -> Result<TRSV_Driver<'a, 'x, F>, BLASError> {
        let Self { a, x, uplo, trans, diag } = self;

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
        let driver = TRSV_Driver {
            uplo: uplo.try_into()?,
            trans: trans.try_into()?,
            diag: diag.try_into()?,
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
    F: TRSVNum,
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
            let a_cow = obj.a.to_row_layout()?;
            match obj.trans {
                BLASNoTrans => {
                    // N -> T: x = op(A')' x
                    let obj = TRSV_ { a: a_cow.t(), trans: BLASTrans, uplo: obj.uplo.flip()?, ..obj };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // T -> N: x = op(A') x
                    let obj = TRSV_ { a: a_cow.t(), trans: BLASNoTrans, uplo: obj.uplo.flip()?, ..obj };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> T: x* = op(A') x*; x = x*
                    let mut x = obj.x;
                    x.mapv_inplace(F::conj);
                    let obj = TRSV_ { a: a_cow.t(), x, trans: BLASNoTrans, uplo: obj.uplo.flip()?, ..obj };
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
