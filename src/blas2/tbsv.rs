use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TBSVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn tbsv(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const c_int,
        k: *const c_int,
        a: *const F,
        lda: *const c_int,
        x: *mut F,
        incx: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TBSVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn tbsv(
                uplo: *const c_char,
                trans: *const c_char,
                diag: *const c_char,
                n: *const c_int,
                k: *const c_int,
                a: *const $type,
                lda: *const c_int,
                x: *mut $type,
                incx: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(uplo, trans, diag, n, k, a as *const FFIFloat, lda, x as *mut FFIFloat, incx);
            }
        }
    };
}

impl_func!(f32, stbsv_);
impl_func!(f64, dtbsv_);
impl_func!(c32, ctbsv_);
impl_func!(c64, ztbsv_);

/* #endregion */

/* #region BLAS driver */

pub struct TBSV_Driver<'a, 'x, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    a: ArrayView2<'a, F>,
    lda: c_int,
    x: ArrayOut1<'x, F>,
    incx: c_int,
}

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TBSV_Driver<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TBSVFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'x, F>, AnyError> {
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;
        let n = self.n;
        let k = self.k;
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
            BLASFunc::tbsv(&uplo, &trans, &diag, &n, &k, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct TBSV_<'a, 'x, F>
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
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TBSV_<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TBSVFunc<F>,
{
    fn driver(self) -> Result<TBSV_Driver<'a, 'x, F>, AnyError> {
        let a = self.a;
        let x = self.x;
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;
        let layout = self.layout;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());
        assert!(layout == Some(BLASLayout::ColMajor));

        // initialize intent(hide)
        let (k_, n) = a.dim();
        blas_assert!(k_ > 0, "Rows of input `a` must larger than zero.")?;
        let k = k_ - 1;
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert_eq!(x.len_of(Axis(0)), n, "Incompatible dimensions")?;

        // prepare output
        let x = ArrayOut1::ViewMut(x);

        // finalize
        let driver = TBSV_Driver {
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

pub type TBSV<'a, 'x, F> = TBSV_Builder<'a, 'x, F>;
pub type STBSV<'a, 'x> = TBSV<'a, 'x, f32>;
pub type DTBSV<'a, 'x> = TBSV<'a, 'x, f64>;
pub type CTBSV<'a, 'x> = TBSV<'a, 'x, c32>;
pub type ZTBSV<'a, 'x> = TBSV<'a, 'x, c64>;

impl<'a, 'x, F> BLASBuilder<'x, F, Ix1> for TBSV_Builder<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TBSVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout = match obj.layout {
            Some(layout) => layout,
            None => match layout_a {
                BLASLayout::Sequential => BLASColMajor,
                BLASRowMajor => BLASRowMajor,
                BLASColMajor => BLASColMajor,
                _ => blas_raise!("Without defining layout, this function checks layout of input matrix `a` but it is not contiguous.")?,
            }
        };

        if layout == BLASColMajor {
            // F-contiguous
            let a_cow = obj.a.reversed_axes();
            let a_cow = a_cow.as_standard_layout().reversed_axes();
            let obj = TBSV_ { a: a_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = obj.a.as_standard_layout();
            match obj.trans {
                BLASNoTrans => {
                    // N -> T
                    let obj = TBSV_ {
                        a: a_cow.t(),
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
                    // N -> T
                    let obj = TBSV_ {
                        a: a_cow.t(),
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
                    let obj = TBSV_ {
                        a: a_cow.t(),
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
