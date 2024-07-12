use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait TRSMFunc<F>
where
    F: BLASFloat,
{
    unsafe fn trsm(
        side: *const c_char,
        uplo: *const c_char,
        transa: *const c_char,
        diag: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const F,
        a: *const F,
        lda: *const c_int,
        b: *mut F,
        ldb: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl TRSMFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn trsm(
                side: *const c_char,
                uplo: *const c_char,
                transa: *const c_char,
                diag: *const c_char,
                m: *const c_int,
                n: *const c_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const c_int,
                b: *mut $type,
                ldb: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    side,
                    uplo,
                    transa,
                    diag,
                    m,
                    n,
                    alpha as *const FFIFloat,
                    a as *const FFIFloat,
                    lda,
                    b as *mut FFIFloat,
                    ldb,
                );
            }
        }
    };
}

impl_func!(f32, strsm_);
impl_func!(f64, dtrsm_);
impl_func!(c32, ctrsm_);
impl_func!(c64, ztrsm_);

/* #endregion */

/* #region BLAS driver */

pub struct TRSM_Driver<'a, 'b, F>
where
    F: BLASFloat,
{
    side: c_char,
    uplo: c_char,
    transa: c_char,
    diag: c_char,
    m: c_int,
    n: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    b: ArrayOut2<'b, F>,
    ldb: c_int,
}

impl<'a, 'b, F> BLASDriver<'b, F, Ix2> for TRSM_Driver<'a, 'b, F>
where
    F: BLASFloat,
    BLASFunc: TRSMFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut2<'b, F>, AnyError> {
        let side = self.side;
        let uplo = self.uplo;
        let transa = self.transa;
        let diag = self.diag;
        let m = self.m;
        let n = self.n;
        let alpha = self.alpha;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
        let mut b = self.b;
        let b_ptr = match &mut b {
            ArrayOut::ViewMut(b) => b.as_mut_ptr(),
            ArrayOut::Owned(b) => b.as_mut_ptr(),
            ArrayOut::ToBeCloned(_, b) => b.as_mut_ptr(),
        };
        let ldb = self.ldb;

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(b.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::trsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a_ptr, &lda, b_ptr, &ldb);
        }
        return Ok(b.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct TRSM_<'a, 'b, F>
where
    F: BLASFloat,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayViewMut2<'b, F>,

    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "BLASLeft")]
    pub side: BLASSide,
    #[builder(setter(into), default = "BLASUpper")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub transa: BLASTranspose,
    #[builder(setter(into), default = "BLASNonUnit")]
    pub diag: BLASDiag,
}

impl<'a, 'b, F> BLASBuilder_<'b, F, Ix2> for TRSM_<'a, 'b, F>
where
    F: BLASFloat,
    BLASFunc: TRSMFunc<F>,
{
    fn driver(self) -> Result<TRSM_Driver<'a, 'b, F>, AnyError> {
        let a = self.a;
        let b = self.b;
        let alpha = self.alpha;
        let side = self.side;
        let uplo = self.uplo;
        let transa = self.transa;
        let diag = self.diag;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());

        // initialize intent(hide)
        let (m, n) = b.dim();
        let lda = a.stride_of(Axis(1));

        // perform check
        match side {
            BLASLeft => blas_assert_eq!(a.dim(), (m, m), "Incompatible dimensions")?,
            BLASRight => blas_assert_eq!(a.dim(), (n, n), "Incompatible dimensions")?,
            _ => blas_invalid!(side)?,
        };

        // prepare output
        let b = if get_layout_array2(&b.view()).is_fpref() {
            ArrayOut2::ViewMut(b)
        } else {
            let b_buffer = b.t().as_standard_layout().into_owned().reversed_axes();
            ArrayOut2::ToBeCloned(b, b_buffer)
        };
        let ldb = b.view().stride_of(Axis(1));

        // finalize
        let driver = TRSM_Driver {
            side: side.into(),
            uplo: uplo.into(),
            transa: transa.into(),
            diag: diag.into(),
            m: m.try_into()?,
            n: n.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            b,
            ldb: ldb.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type TRSM<'a, 'b, F> = TRSM_Builder<'a, 'b, F>;
pub type STRSM<'a, 'b> = TRSM<'a, 'b, f32>;
pub type DTRSM<'a, 'b> = TRSM<'a, 'b, f64>;
pub type CTRSM<'a, 'b> = TRSM<'a, 'b, c32>;
pub type ZTRSM<'a, 'b> = TRSM<'a, 'b, c64>;

impl<'a, 'b, F> BLASBuilder<'b, F, Ix2> for TRSM_Builder<'a, 'b, F>
where
    F: BLASFloat,
    BLASFunc: TRSMFunc<F>,
{
    fn run(self) -> Result<ArrayOut2<'b, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous: B = op(A) B (if side = L)
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: B' = B' op(A') (if side = L)
            let a_cow = obj.a.as_standard_layout();
            let obj = TRSM_ {
                a: a_cow.t(),
                b: obj.b.reversed_axes(),
                alpha: obj.alpha,
                side: match obj.side {
                    BLASLeft => BLASRight,
                    BLASRight => BLASLeft,
                    _ => blas_invalid!(obj.side)?,
                },
                uplo: match obj.uplo {
                    BLASUpper => BLASLower,
                    BLASLower => BLASUpper,
                    _ => blas_invalid!(obj.uplo)?,
                },
                transa: obj.transa,
                diag: obj.diag,
            };
            let b = obj.driver()?.run_blas()?.reversed_axes();
            return Ok(b);
        }
    }
}

/* #endregion */
