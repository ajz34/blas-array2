use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SYMMFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn symm(
        side: *const c_char,
        uplo: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const F,
        a: *const F,
        lda: *const c_int,
        b: *const F,
        ldb: *const c_int,
        beta: *const F,
        c: *mut F,
        ldc: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYMMFunc<$type, $symm> for BLASFunc {
            unsafe fn symm(
                side: *const c_char,
                uplo: *const c_char,
                m: *const c_int,
                n: *const c_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const c_int,
                b: *const $type,
                ldb: *const c_int,
                beta: *const $type,
                c: *mut $type,
                ldc: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    side,
                    uplo,
                    m,
                    n,
                    alpha as *const FFIFloat,
                    a as *const FFIFloat,
                    lda,
                    b as *const FFIFloat,
                    ldb,
                    beta as *const FFIFloat,
                    c as *mut FFIFloat,
                    ldc,
                );
            }
        }
    };
}

impl_func!(f32, BLASSymm<f32>, ssymm_);
impl_func!(f64, BLASSymm<f64>, dsymm_);
impl_func!(c32, BLASSymm<c32>, csymm_);
impl_func!(c64, BLASSymm<c64>, zsymm_);
impl_func!(c32, BLASHermi<c32>, chemm_);
impl_func!(c64, BLASHermi<c64>, zhemm_);

/* #endregion */

/* #region BLAS driver */

pub struct SYMM_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    side: c_char,
    uplo: c_char,
    m: c_int,
    n: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    b: ArrayView2<'b, F>,
    ldb: c_int,
    beta: F,
    c: ArrayOut2<'c, F>,
    ldc: c_int,
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'b, 'c, F, S> BLASDriver<'c, F, Ix2> for SYMM_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMMFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let side = self.side;
        let uplo = self.uplo;
        let m = self.m;
        let n = self.n;
        let alpha = self.alpha;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
        let b_ptr = self.b.as_ptr();
        let ldb = self.ldb;
        let beta = self.beta;
        let mut c = self.c;
        let c_ptr = match &mut c {
            ArrayOut::ViewMut(c) => c.as_mut_ptr(),
            ArrayOut::Owned(c) => c.as_mut_ptr(),
            ArrayOut::ToBeCloned(_, c) => c.as_mut_ptr(),
        };
        let ldc = self.ldc;

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::symm(&side, &uplo, &m, &n, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SYMM_<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASLeft")]
    pub side: BLASSide,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,

    #[builder(private, default = "core::marker::PhantomData {}")]
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'b, 'c, F, S> BLASBuilder_<'c, F, Ix2> for SYMM_<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMMFunc<F, S>,
{
    fn driver(self) -> Result<SYMM_Driver<'a, 'b, 'c, F, S>, BLASError> {
        let Self { a, b, c, alpha, beta, side, uplo, layout, .. } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        assert!(layout_a.is_fpref() && layout_b.is_fpref());

        // initialize intent(hide)
        let m = b.len_of(Axis(0));
        let n = b.len_of(Axis(1));
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        match side {
            BLASLeft => blas_assert_eq!(a.dim(), (m, m), InvalidDim)?,
            BLASRight => blas_assert_eq!(a.dim(), (n, n), InvalidDim)?,
            _ => blas_invalid!(side)?,
        }

        // optional intent(out)
        let c = match c {
            Some(c) => {
                blas_assert_eq!(c.dim(), (m, n), InvalidDim)?;
                if get_layout_array2(&c.view()).is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.t().as_standard_layout().into_owned().reversed_axes();
                    ArrayOut2::ToBeCloned(c, c_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((m, n).f())),
        };
        let ldc = c.view().stride_of(Axis(1));

        // finalize
        let driver = SYMM_Driver::<'a, 'b, 'c, F, S> {
            side: side.into(),
            uplo: uplo.into(),
            m: m.try_into()?,
            n: n.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            b,
            ldb: ldb.try_into()?,
            beta,
            c,
            ldc: ldc.try_into()?,
            _phantom: core::marker::PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SYMM<'a, 'b, 'c, F> = SYMM_Builder<'a, 'b, 'c, F, BLASSymm<F>>;
pub type SSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, f32>;
pub type DSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, f64>;
pub type CSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, c32>;
pub type ZSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, c64>;

pub type HEMM<'a, 'b, 'c, F> = SYMM_Builder<'a, 'b, 'c, F, BLASHermi<F>>;
pub type CHEMM<'a, 'b, 'c> = HEMM<'a, 'b, 'c, c32>;
pub type ZHEMM<'a, 'b, 'c> = HEMM<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F, S> BLASBuilder<'c, F, Ix2> for SYMM_Builder<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMMFunc<F, S>,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let SYMM_ { a, b, c, alpha, beta, side, uplo, layout, .. } = self.build()?;
        let at = a.t();
        let bt = b.t();

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            // F-contiguous: C = op(A) op(B)
            let (uplo, a_cow) = match layout_a.is_fpref() || S::is_hermitian() {
                true => (uplo, at.as_standard_layout()),
                false => (uplo.flip(), a.as_standard_layout()),
            };
            let b_cow = bt.as_standard_layout();
            let obj = SYMM_ {
                a: a_cow.t(),
                b: b_cow.t(),
                c,
                alpha,
                beta,
                side,
                uplo,
                layout: Some(BLASColMajor),
                _phantom: core::marker::PhantomData {},
            };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: C' = op(B') op(A')
            let (uplo, a_cow) = match layout_a.is_cpref() || S::is_hermitian() {
                true => (uplo, a.as_standard_layout()),
                false => (uplo.flip(), at.as_standard_layout()),
            };
            let b_cow = b.as_standard_layout();
            let obj = SYMM_ {
                a: a_cow.t(),
                b: b_cow.t(),
                c: c.map(|c| c.reversed_axes()),
                alpha,
                beta,
                side: side.flip(),
                uplo: uplo.flip(),
                layout: Some(BLASColMajor),
                _phantom: core::marker::PhantomData {},
            };
            let c = obj.driver()?.run_blas()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */
