use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::{One, Zero};

/* #region BLAS func */

pub trait SYRKFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    unsafe fn syrk(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
        alpha: *const S::HermitianFloat,
        a: *const F,
        lda: *const blas_int,
        beta: *const S::HermitianFloat,
        c: *mut F,
        ldc: *const blas_int,
    );
}

macro_rules! impl_syrk {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYRKFunc<$type, $symm> for BLASFunc {
            unsafe fn syrk(
                uplo: *const c_char,
                trans: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                alpha: *const <$symm as BLASSymmetric>::HermitianFloat,
                a: *const $type,
                lda: *const blas_int,
                beta: *const <$symm as BLASSymmetric>::HermitianFloat,
                c: *mut $type,
                ldc: *const blas_int,
            ) {
                type HermitialFloat = <$symm as BLASSymmetric>::HermitianFloat;
                ffi::$func(
                    uplo,
                    trans,
                    n,
                    k,
                    alpha as *const HermitialFloat,
                    a,
                    lda,
                    beta as *const HermitialFloat,
                    c,
                    ldc,
                );
            }
        }
    };
}

impl_syrk!(f32, BLASSymm<f32>, ssyrk_);
impl_syrk!(f64, BLASSymm<f64>, dsyrk_);
impl_syrk!(c32, BLASSymm<c32>, csyrk_);
impl_syrk!(c64, BLASSymm<c64>, zsyrk_);
impl_syrk!(c32, BLASHermi<c32>, cherk_);
impl_syrk!(c64, BLASHermi<c64>, zherk_);

/* #endregion */

/* #region BLAS driver */

pub struct SYRK_Driver<'a, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
{
    uplo: c_char,
    trans: c_char,
    n: blas_int,
    k: blas_int,
    alpha: S::HermitianFloat,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    beta: S::HermitianFloat,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
}

impl<'a, 'c, F, S> BLASDriver<'c, F, Ix2> for SYRK_Driver<'a, 'c, F, S>
where
    F: BLASFloat,
    F: From<<S as BLASSymmetric>::HermitianFloat>,
    S: BLASSymmetric,
    BLASFunc: SYRKFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError>
    where
        BLASFunc: SYRKFunc<F, S>,
    {
        let Self { uplo, trans, n, k, alpha, a, lda, beta, mut c, ldc } = self;
        let a_ptr = a.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(c.clone_to_view_mut());
        } else if k == 0 {
            let beta_f = F::from(beta);
            if uplo == BLASLower.into() {
                for i in 0..n {
                    c.view_mut().slice_mut(s![i.., i]).mapv_inplace(|v| v * beta_f);
                }
            } else if uplo == BLASUpper.into() {
                for i in 0..n {
                    c.view_mut().slice_mut(s![..=i, i]).mapv_inplace(|v| v * beta_f);
                }
            } else {
                blas_invalid!(uplo)?
            }
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            BLASFunc::syrk(&uplo, &trans, &n, &k, &alpha, a_ptr, &lda, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct SYRK_<'a, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymmetric,
    S::HermitianFloat: Zero + One,
{
    pub a: ArrayView2<'a, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "S::HermitianFloat::one()")]
    pub alpha: S::HermitianFloat,
    #[builder(setter(into), default = "S::HermitianFloat::zero()")]
    pub beta: S::HermitianFloat,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'c, F, S> BLASBuilder_<'c, F, Ix2> for SYRK_<'a, 'c, F, S>
where
    F: BLASFloat,
    F: From<<S as BLASSymmetric>::HermitianFloat>,
    S: BLASSymmetric,
    BLASFunc: SYRKFunc<F, S>,
{
    fn driver(self) -> Result<SYRK_Driver<'a, 'c, F, S>, BLASError> {
        let Self { a, c, alpha, beta, uplo, trans, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref());

        // initialize intent(hide)
        let (n, k) = match trans {
            BLASNoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASTrans | BLASConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(trans)?,
        };
        let lda = a.stride_of(Axis(1));

        // perform check
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
        let driver = SYRK_Driver {
            uplo: uplo.into(),
            trans: trans.into(),
            n: n.try_into()?,
            k: k.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            beta,
            c,
            ldc: ldc.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SYRK<'a, 'c, F> = SYRK_Builder<'a, 'c, F, BLASSymm<F>>;
pub type SSYRK<'a, 'c> = SYRK<'a, 'c, f32>;
pub type DSYRK<'a, 'c> = SYRK<'a, 'c, f64>;
pub type CSYRK<'a, 'c> = SYRK<'a, 'c, c32>;
pub type ZSYRK<'a, 'c> = SYRK<'a, 'c, c64>;

pub type HERK<'a, 'c, F> = SYRK_Builder<'a, 'c, F, BLASHermi<F>>;
pub type CHERK<'a, 'c> = HERK<'a, 'c, c32>;
pub type ZHERK<'a, 'c> = HERK<'a, 'c, c64>;

impl<'a, 'c, F, S> BLASBuilder<'c, F, Ix2> for SYRK_Builder<'a, 'c, F, S>
where
    F: BLASFloat,
    F: From<<S as BLASSymmetric>::HermitianFloat>,
    S: BLASSymmetric,
    BLASFunc: SYRKFunc<F, S>,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let SYRK_ { a, c, alpha, beta, uplo, trans, layout } = self.build()?;
        let at = a.t();

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
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a]);
        if layout == BLASColMajor {
            // F-contiguous: C = A op(A) or C = op(A) A
            let (trans, a_cow) = flip_trans_fpref(trans, &a, &at, S::is_hermitian())?;
            let obj = SYRK_ { a: a_cow.view(), c, alpha, beta, uplo, trans, layout: Some(BLASColMajor) };
            return obj.driver()?.run_blas();
        } else if layout == BLASRowMajor {
            let (trans, a_cow) = flip_trans_cpref(trans, &a, &at, S::is_hermitian())?;
            let obj = SYRK_ {
                a: a_cow.t(),
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
