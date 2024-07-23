use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TBMVNum: BLASFloat {
    unsafe fn tbmv(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
        a: *const Self,
        lda: *const blas_int,
        x: *mut Self,
        incx: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TBMVNum for $type {
            unsafe fn tbmv(
                uplo: *const c_char,
                trans: *const c_char,
                diag: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                a: *const Self,
                lda: *const blas_int,
                x: *mut Self,
                incx: *const blas_int,
            ) {
                ffi::$func(uplo, trans, diag, n, k, a, lda, x, incx);
            }
        }
    };
}

impl_func!(f32, stbmv_);
impl_func!(f64, dtbmv_);
impl_func!(c32, ctbmv_);
impl_func!(c64, ztbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct TBMV_Driver<'a, 'x, F>
where
    F: TBMVNum,
{
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: blas_int,
    k: blas_int,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    x: ArrayOut1<'x, F>,
    incx: blas_int,
}

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TBMV_Driver<'a, 'x, F>
where
    F: TBMVNum,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        let Self { uplo, trans, diag, n, k, a, lda, mut x, incx } = self;
        let a_ptr = a.as_ptr();
        let x_ptr = x.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(x);
        }

        unsafe {
            F::tbmv(&uplo, &trans, &diag, &n, &k, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct TBMV_<'a, 'x, F>
where
    F: TBMVNum,
{
    pub a: ArrayView2<'a, F>,
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

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TBMV_<'a, 'x, F>
where
    F: TBMVNum,
{
    fn driver(self) -> Result<TBMV_Driver<'a, 'x, F>, BLASError> {
        let Self { a, x, uplo, trans, diag, layout } = self;

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
        let x = ArrayOut1::ViewMut(x);

        // finalize
        let driver = TBMV_Driver {
            uplo: uplo.into(),
            trans: trans.into(),
            diag: diag.into(),
            n: n.try_into()?,
            k: k.try_into()?,
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

pub type TBMV<'a, 'x, F> = TBMV_Builder<'a, 'x, F>;
pub type STBMV<'a, 'x> = TBMV<'a, 'x, f32>;
pub type DTBMV<'a, 'x> = TBMV<'a, 'x, f64>;
pub type CTBMV<'a, 'x> = TBMV<'a, 'x, c32>;
pub type ZTBMV<'a, 'x> = TBMV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TBMV_Builder<'a, 'x, F>
where
    F: TBMVNum,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, BLASError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout = get_layout_row_preferred(&[obj.layout, Some(layout_a)], &[]);

        if layout == BLASColMajor {
            // F-contiguous
            let a_cow = obj.a.to_col_layout()?;
            let obj = TBMV_ { a: a_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = obj.a.to_row_layout()?;
            match obj.trans {
                BLASNoTrans => {
                    // N -> T
                    let obj = TBMV_ {
                        a: a_cow.t(),
                        trans: BLASTrans,
                        uplo: obj.uplo.flip(),
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // N -> T
                    let obj = TBMV_ {
                        a: a_cow.t(),
                        trans: BLASNoTrans,
                        uplo: obj.uplo.flip(),
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> N
                    let mut x = obj.x;
                    x.mapv_inplace(F::conj);
                    let obj = TBMV_ {
                        a: a_cow.t(),
                        x,
                        trans: BLASNoTrans,
                        uplo: obj.uplo.flip(),
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
