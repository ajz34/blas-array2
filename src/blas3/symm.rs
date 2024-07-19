use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
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
        m: *const blas_int,
        n: *const blas_int,
        alpha: *const F,
        a: *const F,
        lda: *const blas_int,
        b: *const F,
        ldb: *const blas_int,
        beta: *const F,
        c: *mut F,
        ldc: *const blas_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYMMFunc<$type, $symm> for BLASFunc {
            unsafe fn symm(
                side: *const c_char,
                uplo: *const c_char,
                m: *const blas_int,
                n: *const blas_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const blas_int,
                b: *const $type,
                ldb: *const blas_int,
                beta: *const $type,
                c: *mut $type,
                ldc: *const blas_int,
            ) {
                ffi::$func(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
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
    m: blas_int,
    n: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    b: ArrayView2<'b, F>,
    ldb: blas_int,
    beta: F,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
    _phantom: core::marker::PhantomData<S>,
}

impl<'a, 'b, 'c, F, S> BLASDriver<'c, F, Ix2> for SYMM_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYMMFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { side, uplo, m, n, alpha, a, lda, b, ldb, beta, mut c, ldc, .. } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

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
        assert!(a.is_fpref() && a.is_fpref());

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
                if c.view().is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.view().to_col_layout()?.into_owned();
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

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            // F-contiguous: C = op(A) op(B)
            let (uplo, a_cow) = match layout_a.is_fpref() || S::is_hermitian() {
                true => (uplo, a.to_col_layout()?),
                false => (uplo.flip(), at.to_col_layout()?),
            };
            let b_cow = b.to_col_layout()?;
            let obj = SYMM_ {
                a: a_cow.view(),
                b: b_cow.view(),
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
                true => (uplo, a.to_row_layout()?),
                false => (uplo.flip(), at.to_row_layout()?),
            };
            let b_cow = b.to_row_layout()?;
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
