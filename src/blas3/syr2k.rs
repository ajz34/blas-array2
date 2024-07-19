use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::{One, Zero};

/* #region BLAS func */

pub trait SYR2KFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn syr2k(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
        alpha: *const F,
        a: *const F,
        lda: *const blas_int,
        b: *const F,
        ldb: *const blas_int,
        beta: *const S::HermitianFloat,
        c: *mut F,
        ldc: *const blas_int,
    );
}

macro_rules! impl_syr2k {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYR2KFunc<$type, $symm> for BLASFunc {
            unsafe fn syr2k(
                uplo: *const c_char,
                trans: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const blas_int,
                b: *const $type,
                ldb: *const blas_int,
                beta: *const <$symm as BLASSymmetric>::HermitianFloat,
                c: *mut $type,
                ldc: *const blas_int,
            ) {
                type HermitialFloat = <$symm as BLASSymmetric>::HermitianFloat;
                ffi::$func(uplo, trans, n, k, alpha, a, lda, b, ldb, beta as *const HermitialFloat, c, ldc);
            }
        }
    };
}

impl_syr2k!(f32, BLASSymm<f32>, ssyr2k_);
impl_syr2k!(f64, BLASSymm<f64>, dsyr2k_);
impl_syr2k!(c32, BLASSymm<c32>, csyr2k_);
impl_syr2k!(c64, BLASSymm<c64>, zsyr2k_);
impl_syr2k!(c32, BLASHermi<c32>, cher2k_);
impl_syr2k!(c64, BLASHermi<c64>, zher2k_);

/* #endregion */

/* #region BLAS driver */

pub struct SYR2K_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    uplo: c_char,
    trans: c_char,
    n: blas_int,
    k: blas_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    b: ArrayView2<'b, F>,
    ldb: blas_int,
    beta: S::HermitianFloat,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
}

impl<'a, 'b, 'c, F, S> BLASDriver<'c, F, Ix2> for SYR2K_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYR2KFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { uplo, trans, n, k, alpha, a, lda, b, ldb, beta, mut c, ldc } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 || k == 0 {
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::syr2k(&uplo, &trans, &n, &k, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SYR2K_<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    S::HermitianFloat: Zero + One,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "S::HermitianFloat::zero()")]
    pub beta: S::HermitianFloat,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'b, 'c, F, S> BLASBuilder_<'c, F, Ix2> for SYR2K_<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYR2KFunc<F, S>,
{
    fn driver(self) -> Result<SYR2K_Driver<'a, 'b, 'c, F, S>, BLASError> {
        let Self { a, b, c, alpha, beta, uplo, trans, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref() && a.is_fpref());

        // initialize intent(hide)
        let (n, k) = match trans {
            BLASNoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASTrans | BLASConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(trans)?,
        };
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        // dimension of b
        match trans {
            BLASNoTrans => blas_assert_eq!(b.dim(), (n, k), InvalidDim)?,
            BLASTrans | BLASConjTrans => blas_assert_eq!(b.dim(), (k, n), InvalidDim)?,
            _ => blas_invalid!(trans)?,
        };
        // trans keyword
        match F::is_complex() {
            false => match trans {
                // ssyrk, dsyrk: NTC accepted
                BLASNoTrans | BLASTrans | BLASConjTrans => (),
                _ => blas_invalid!(trans)?,
            },
            true => match S::is_hermitian() {
                false => match trans {
                    // csyrk, zsyrk: NT accepted
                    BLASNoTrans | BLASTrans => (),
                    _ => blas_invalid!(trans)?,
                },
                true => match trans {
                    // cherk, zherk: NC accepted
                    BLASNoTrans | BLASConjTrans => (),
                    _ => blas_invalid!(trans)?,
                },
            },
        };

        // optional intent(out)
        let c = match c {
            Some(c) => {
                blas_assert_eq!(c.dim(), (n, n), InvalidDim)?;
                if c.view().is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.view().to_col_layout()?.into_owned();
                    ArrayOut2::ToBeCloned(c, c_buffer)
                }
            },
            None => ArrayOut2::Owned(Array2::zeros((n, n).f())),
        };
        let ldc = c.view().stride_of(Axis(1));

        // finalize
        let driver = SYR2K_Driver {
            uplo: uplo.into(),
            trans: trans.into(),
            n: n.try_into()?,
            k: k.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            b,
            ldb: ldb.try_into()?,
            beta,
            c,
            ldc: ldc.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SYR2K<'a, 'b, 'c, F> = SYR2K_Builder<'a, 'b, 'c, F, BLASSymm<F>>;
pub type SSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, f32>;
pub type DSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, f64>;
pub type CSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, c32>;
pub type ZSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, c64>;

pub type HER2K<'a, 'b, 'c, F> = SYR2K_Builder<'a, 'b, 'c, F, BLASHermi<F>>;
pub type CHER2K<'a, 'b, 'c> = HER2K<'a, 'b, 'c, c32>;
pub type ZHER2K<'a, 'b, 'c> = HER2K<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F, S> BLASBuilder<'c, F, Ix2> for SYR2K_Builder<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    BLASFunc: SYR2KFunc<F, S>,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let SYR2K_ { a, b, c, alpha, beta, uplo, trans, layout } = self.build()?;

        // Note that since we will change `trans` in outer wrapper to utilize mix-contiguous
        // additional check to this parameter is required
        match F::is_complex() {
            false => match trans {
                // ssyrk, dsyrk: NTC accepted
                BLASNoTrans | BLASTrans | BLASConjTrans => (),
                _ => blas_invalid!(trans)?,
            },
            true => match S::is_hermitian() {
                false => match trans {
                    // csyrk, zsyrk: NT accepted
                    BLASNoTrans | BLASTrans => (),
                    _ => blas_invalid!(trans)?,
                },
                true => match trans {
                    // cherk, zherk: NC accepted
                    BLASNoTrans | BLASConjTrans => (),
                    _ => blas_invalid!(trans)?,
                },
            },
        };

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        // syr2k is difficult to gain any improvement when input matrices layouts are mixed
        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            // F-contiguous: C = A op(B) + B op(A)
            let a_cow = a.to_col_layout()?;
            let b_cow = b.to_col_layout()?;
            let obj = SYR2K_ {
                a: a_cow.view(),
                b: b_cow.view(),
                c,
                alpha,
                beta,
                uplo,
                trans,
                layout: Some(BLASColMajor),
            };
            return obj.driver()?.run_blas();
        } else if layout == BLASRowMajor {
            // C-contiguous: C' = op(B') A' + op(A') B'
            let a_cow = a.to_row_layout()?;
            let b_cow = b.to_row_layout()?;
            let obj = SYR2K_ {
                a: b_cow.t(),
                b: a_cow.t(),
                c: c.map(|c| c.reversed_axes()),
                alpha,
                beta,
                uplo: uplo.flip(),
                trans: trans.flip(S::is_hermitian()),
                layout: Some(BLASColMajor),
            };
            return Ok(obj.driver()?.run_blas()?.reversed_axes());
        } else {
            panic!("This is designed not to execuate this line.");
        }
    }
}

/* #endregion */
