use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TRMVNum: BLASFloat {
    unsafe fn trmv(
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
        impl TRMVNum for $type {
            unsafe fn trmv(
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

impl_func!(f32, strmv_);
impl_func!(f64, dtrmv_);
impl_func!(c32, ctrmv_);
impl_func!(c64, ztrmv_);

/* #endregion */

/* #region BLAS driver */

pub struct TRMV_Driver<'a, 'x, F>
where
    F: TRMVNum,
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

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TRMV_Driver<'a, 'x, F>
where
    F: TRMVNum,
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
            F::trmv(&uplo, &trans, &diag, &n, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct TRMV_<'a, 'x, F>
where
    F: TRMVNum,
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
    F: TRMVNum,
{
    fn driver(self) -> Result<TRMV_Driver<'a, 'x, F>, BLASError> {
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
        let driver = TRMV_Driver {
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

pub type TRMV<'a, 'x, F> = TRMV_Builder<'a, 'x, F>;
pub type STRMV<'a, 'x> = TRMV<'a, 'x, f32>;
pub type DTRMV<'a, 'x> = TRMV<'a, 'x, f64>;
pub type CTRMV<'a, 'x> = TRMV<'a, 'x, c32>;
pub type ZTRMV<'a, 'x> = TRMV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TRMV_Builder<'a, 'x, F>
where
    F: TRMVNum,
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
                    let obj = TRMV_ { a: a_cow.t(), trans: BLASTrans, uplo: obj.uplo.flip()?, ..obj };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // T -> N: x = op(A') x
                    let obj = TRMV_ { a: a_cow.t(), trans: BLASNoTrans, uplo: obj.uplo.flip()?, ..obj };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> T: x* = op(A') x*; x = x*
                    let mut x = obj.x;
                    x.mapv_inplace(F::conj);
                    let obj = TRMV_ { a: a_cow.t(), x, trans: BLASNoTrans, uplo: obj.uplo.flip()?, ..obj };
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
