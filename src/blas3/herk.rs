use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::*;

/* #region BLAS func */

pub trait HERKNum: BLASFloat {
    unsafe fn herk(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
        alpha: *const Self::RealFloat,
        a: *const Self,
        lda: *const blas_int,
        beta: *const Self::RealFloat,
        c: *mut Self,
        ldc: *const blas_int,
    );
}

macro_rules! impl_herk {
    ($type: ty, $func: ident) => {
        impl HERKNum for $type {
            unsafe fn herk(
                uplo: *const c_char,
                trans: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                alpha: *const Self::RealFloat,
                a: *const Self,
                lda: *const blas_int,
                beta: *const Self::RealFloat,
                c: *mut Self,
                ldc: *const blas_int,
            ) {
                ffi::$func(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
            }
        }
    };
}

impl_herk!(c32, cherk_);
impl_herk!(c64, zherk_);

/* #endregion */

/* #region BLAS driver */

pub struct HERK_Driver<'a, 'c, F>
where
    F: BLASFloat,
{
    uplo: c_char,
    trans: c_char,
    n: blas_int,
    k: blas_int,
    alpha: F::RealFloat,
    a: ArrayView2<'a, F>,
    lda: blas_int,
    beta: F::RealFloat,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
}

impl<'a, 'c, F> BLASDriver<'c, F, Ix2> for HERK_Driver<'a, 'c, F>
where
    F: HERKNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { uplo, trans, n, k, alpha, a, lda, beta, mut c, ldc } = self;
        let a_ptr = a.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(c.clone_to_view_mut());
        } else if k == 0 {
            let beta_f = F::from_real(beta);
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
            F::herk(&uplo, &trans, &n, &k, &alpha, a_ptr, &lda, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HERK_<'a, 'c, F>
where
    F: HERKNum,
{
    pub a: ArrayView2<'a, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::RealFloat::one()")]
    pub alpha: F::RealFloat,
    #[builder(setter(into), default = "F::RealFloat::zero()")]
    pub beta: F::RealFloat,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'c, F> BLASBuilder_<'c, F, Ix2> for HERK_<'a, 'c, F>
where
    F: HERKNum,
{
    fn driver(self) -> Result<HERK_Driver<'a, 'c, F>, BLASError> {
        let Self { a, c, alpha, beta, uplo, trans, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref());

        // initialize intent(hide) (cherk, zherk: NC accepted)
        let (n, k) = match trans {
            BLASNoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(trans)?,
        };
        let lda = a.stride_of(Axis(1));

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
        let driver = HERK_Driver {
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

pub type HERK<'a, 'c, F> = HERK_Builder<'a, 'c, F>;
pub type CHERK<'a, 'c> = HERK<'a, 'c, c32>;
pub type ZHERK<'a, 'c> = HERK<'a, 'c, c64>;

impl<'a, 'c, F> BLASBuilder<'c, F, Ix2> for HERK_Builder<'a, 'c, F>
where
    F: HERKNum,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let HERK_ { a, c, alpha, beta, uplo, trans, layout } = self.build()?;
        let at = a.t();

        // Note that since we will change `trans` in outer wrapper to utilize mix-contiguous
        // additional check to this parameter is required
        match trans {
            // cherk, zherk: NC accepted
            BLASNoTrans | BLASConjTrans => (),
            _ => blas_invalid!(trans)?,
        };

        let layout_a = get_layout_array2(&a);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a]);
        if layout == BLASColMajor {
            // F-contiguous: C = A op(A) or C = op(A) A
            let (trans, a_cow) = flip_trans_fpref(trans, &a, &at, true)?;
            let obj = HERK_ { a: a_cow.view(), c, alpha, beta, uplo, trans, layout: Some(BLASColMajor) };
            return obj.driver()?.run_blas();
        } else if layout == BLASRowMajor {
            let (trans, a_cow) = flip_trans_cpref(trans, &a, &at, true)?;
            let obj = HERK_ {
                a: a_cow.t(),
                c: c.map(|c| c.reversed_axes()),
                alpha,
                beta,
                uplo: uplo.flip(),
                trans: trans.flip(true),
                layout: Some(BLASColMajor),
            };
            return Ok(obj.driver()?.run_blas()?.reversed_axes());
        } else {
            panic!("This is designed not to execuate this line.");
        }
    }
}

/* #endregion */
