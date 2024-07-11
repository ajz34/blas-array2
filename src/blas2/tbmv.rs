use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TBMVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn tbmv(
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
        impl TBMVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn tbmv(
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

impl_func!(f32, stbmv_);
impl_func!(f64, dtbmv_);
impl_func!(c32, ctbmv_);
impl_func!(c64, ztbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct TBMV_Driver<'a, 'x, F>
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

impl<'a, 'x, F> BLASDriver<'x, F, Ix1> for TBMV_Driver<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TBMVFunc<F>,
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

        unsafe {
            BLASFunc::tbmv(&uplo, &trans, &diag, &n, &k, a_ptr, &lda, x_ptr, &incx);
        }
        return Ok(x);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct TBMV_<'a, 'x, F>
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

impl<'a, 'x, F> BLASBuilder_<'x, F, Ix1> for TBMV_<'a, 'x, F>
where
    F: BLASFloat,
    BLASFunc: TBMVFunc<F>,
{
    fn driver(self) -> Result<TBMV_Driver<'a, 'x, F>, AnyError> {
        let a = self.a;
        let x = self.x;
        let uplo = self.uplo;
        let trans = self.trans;
        let diag = self.diag;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());

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
    F: BLASFloat,
    BLASFunc: TBMVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'x, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous: y = alpha op(A) x + beta y
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: transpose to F-contiguous
            eprintln!("{:}:{:}: Warning message from blas-array2", file!(), line!());
            eprintln!("Banded storage not suitable for C-contiguous without explicit transposition.");
            eprintln!("Also see https://github.com/Reference-LAPACK/lapack/issues/1032.");
            let a_fpref = obj.a.reversed_axes().as_standard_layout().reversed_axes().into_owned();
            let obj = TBMV_ { a: a_fpref.view(), ..obj };
            return obj.driver()?.run_blas();
        }
    }
}

/* #endregion */