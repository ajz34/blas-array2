use crate::ffi::{self, blas_int, c_char};
use crate::util::*;
use derive_builder::Builder;
use ndarray::prelude::*;
use num_traits::*;

/* #region BLAS func */

pub trait HER2KNum: BLASFloat {
    unsafe fn her2k(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const blas_int,
        k: *const blas_int,
        alpha: *const Self,
        a: *const Self,
        lda: *const blas_int,
        b: *const Self,
        ldb: *const blas_int,
        beta: *const Self::RealFloat,
        c: *mut Self,
        ldc: *const blas_int,
    );
}

macro_rules! impl_her2k {
    ($type: ty, $func: ident) => {
        impl HER2KNum for $type {
            unsafe fn her2k(
                uplo: *const c_char,
                trans: *const c_char,
                n: *const blas_int,
                k: *const blas_int,
                alpha: *const Self,
                a: *const Self,
                lda: *const blas_int,
                b: *const Self,
                ldb: *const blas_int,
                beta: *const Self::RealFloat,
                c: *mut Self,
                ldc: *const blas_int,
            ) {
                ffi::$func(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            }
        }
    };
}

impl_her2k!(c32, cher2k_);
impl_her2k!(c64, zher2k_);

/* #endregion */

/* #region BLAS driver */

pub struct HER2K_Driver<'a, 'b, 'c, F>
where
    F: HER2KNum,
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
    beta: F::RealFloat,
    c: ArrayOut2<'c, F>,
    ldc: blas_int,
}

impl<'a, 'b, 'c, F> BLASDriver<'c, F, Ix2> for HER2K_Driver<'a, 'b, 'c, F>
where
    F: HER2KNum,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        let Self { uplo, trans, n, k, alpha, a, lda, b, ldb, beta, mut c, ldc } = self;
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let c_ptr = c.get_data_mut_ptr();

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(c.clone_to_view_mut());
        } else if k == 0 {
            let beta_f = F::RealFloat::from(beta);
            if uplo == BLASLower.into() {
                for i in 0..n {
                    c.view_mut().slice_mut(s![i.., i]).mapv_inplace(|v| v * F::from_real(beta_f));
                }
            } else if uplo == BLASUpper.into() {
                for i in 0..n {
                    c.view_mut().slice_mut(s![..=i, i]).mapv_inplace(|v| v * F::from_real(beta_f));
                }
            } else {
                blas_invalid!(uplo)?
            }
            return Ok(c.clone_to_view_mut());
        }

        unsafe {
            F::her2k(&uplo, &trans, &n, &k, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned", build_fn(error = "BLASError"), no_std)]
pub struct HER2K_<'a, 'b, 'c, F>
where
    F: HER2KNum,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::RealFloat::zero()")]
    pub beta: F::RealFloat,
    #[builder(setter(into), default = "BLASLower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'b, 'c, F> BLASBuilder_<'c, F, Ix2> for HER2K_<'a, 'b, 'c, F>
where
    F: HER2KNum,
{
    fn driver(self) -> Result<HER2K_Driver<'a, 'b, 'c, F>, BLASError> {
        let Self { a, b, c, alpha, beta, uplo, trans, layout } = self;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        assert_eq!(layout, Some(BLASColMajor));
        assert!(a.is_fpref() && a.is_fpref());

        // initialize intent(hide)
        let (n, k) = match trans {
            BLASNoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(trans)?,
        };
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check (cher2k: NC accepted)
        // dimension of b
        match trans {
            BLASNoTrans => blas_assert_eq!(b.dim(), (n, k), InvalidDim)?,
            BLASConjTrans => blas_assert_eq!(b.dim(), (k, n), InvalidDim)?,
            _ => blas_invalid!(trans)?,
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
        let driver = HER2K_Driver {
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

pub type HER2K<'a, 'b, 'c, F> = HER2K_Builder<'a, 'b, 'c, F>;
pub type CHER2K<'a, 'b, 'c> = HER2K<'a, 'b, 'c, c32>;
pub type ZHER2K<'a, 'b, 'c> = HER2K<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F> BLASBuilder<'c, F, Ix2> for HER2K_Builder<'a, 'b, 'c, F>
where
    F: HER2KNum,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, BLASError> {
        // initialize
        let HER2K_ { a, b, c, alpha, beta, uplo, trans, layout } = self.build()?;

        // Note that since we will change `trans` in outer wrapper to utilize mix-contiguous
        // additional check to this parameter is required
        match trans {
            // cherk, zherk: NT accepted
            BLASNoTrans | BLASConjTrans => (),
            _ => blas_invalid!(trans)?,
        };

        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        let layout_c = c.as_ref().map(|c| get_layout_array2(&c.view()));

        // her2k is difficult to gain any improvement when input matrices layouts are mixed
        let layout = get_layout_row_preferred(&[layout, layout_c], &[layout_a, layout_b]);
        if layout == BLASColMajor {
            // F-contiguous: C = A op(B) + B op(A)
            let a_cow = a.to_col_layout()?;
            let b_cow = b.to_col_layout()?;
            let obj = HER2K_ {
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
            let obj = HER2K_ {
                a: b_cow.t(),
                b: a_cow.t(),
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
