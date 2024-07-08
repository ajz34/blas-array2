use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;
use num_traits::{One, Zero};

/* #region BLAS func */

pub trait SYR2KFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymm,
{
    unsafe fn syr2k(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const c_int,
        k: *const c_int,
        alpha: *const F,
        a: *const F,
        lda: *const c_int,
        b: *const F,
        ldb: *const c_int,
        beta: *const S::HermitianFloat,
        c: *mut F,
        ldc: *const c_int,
    );
}

macro_rules! impl_syr2k {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYR2KFunc<$type, $symm> for BLASFunc {
            unsafe fn syr2k(
                uplo: *const c_char,
                trans: *const c_char,
                n: *const c_int,
                k: *const c_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const c_int,
                b: *const $type,
                ldb: *const c_int,
                beta: *const <$symm as BLASSymm>::HermitianFloat,
                c: *mut $type,
                ldc: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                type FFIHermitialFloat = <<$symm as BLASSymm>::HermitianFloat as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    uplo,
                    trans,
                    n,
                    k,
                    alpha as *const FFIFloat,
                    a as *const FFIFloat,
                    lda,
                    b as *const FFIFloat,
                    ldb,
                    beta as *const FFIHermitialFloat,
                    c as *mut FFIFloat,
                    ldc,
                );
            }
        }
    };
}

impl_syr2k!(f32, BLASSymmetric<f32>, ssyr2k_);
impl_syr2k!(f64, BLASSymmetric<f64>, dsyr2k_);
impl_syr2k!(c32, BLASSymmetric<c32>, csyr2k_);
impl_syr2k!(c64, BLASSymmetric<c64>, zsyr2k_);
impl_syr2k!(c32, BLASHermitian<c32>, cher2k_);
impl_syr2k!(c64, BLASHermitian<c64>, zher2k_);

/* #endregion */

/* #region BLAS driver */

pub struct SYR2K_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
{
    uplo: c_char,
    trans: c_char,
    n: c_int,
    k: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    b: ArrayView2<'b, F>,
    ldb: c_int,
    beta: S::HermitianFloat,
    c: ArrayOut2<'c, F>,
    ldc: c_int,
}

impl<'a, 'b, 'c, F, S> BLASDriver<'c, F, Ix2> for SYR2K_Driver<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
    BLASFunc: SYR2KFunc<F, S>,
{
    fn run_blas(self) -> Result<ArrayOut2<'c, F>, AnyError> {
        let uplo = self.uplo;
        let trans = self.trans;
        let n = self.n;
        let k = self.k;
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

        unsafe {
            BLASFunc::syr2k(&uplo, &trans, &n, &k, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        return Ok(c.clone_to_view_mut());
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct SYR2K_<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
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
    #[builder(setter(into), default = "BLASUpLo::Lower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASTrans::NoTrans")]
    pub trans: BLASTrans,
}

impl<'a, 'b, 'c, F, S> BLASBuilder_<'c, F, Ix2> for SYR2K_<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
    BLASFunc: SYR2KFunc<F, S>,
{
    fn driver(self) -> Result<SYR2K_Driver<'a, 'b, 'c, F, S>, AnyError> {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let alpha = self.alpha;
        let beta = self.beta;
        let uplo = self.uplo;
        let trans = self.trans;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        assert!(layout_a.is_fpref() && layout_b.is_fpref());

        // initialize intent(hide)
        let (n, k) = match trans {
            BLASTrans::NoTrans => (a.len_of(Axis(0)), a.len_of(Axis(1))),
            BLASTrans::Trans | BLASTrans::ConjTrans => (a.len_of(Axis(1)), a.len_of(Axis(0))),
            _ => blas_invalid!(trans)?,
        };
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        // dimension of b
        match trans {
            BLASTrans::NoTrans => blas_assert_eq!(b.dim(), (n, k), "Incompatible dimensions")?,
            BLASTrans::Trans | BLASTrans::ConjTrans => blas_assert_eq!(b.dim(), (k, n), "Incompatible dimensions")?,
            _ => blas_invalid!(trans)?,
        };
        // trans keyword
        match F::is_complex() {
            false => match trans {
                // ssyrk, dsyrk: NTC accepted
                BLASTrans::NoTrans | BLASTrans::Trans | BLASTrans::ConjTrans => (),
                _ => blas_invalid!(trans)?,
            },
            true => match S::is_hermitian() {
                false => match trans {
                    // csyrk, zsyrk: NT accepted
                    BLASTrans::NoTrans | BLASTrans::Trans => (),
                    _ => blas_invalid!(trans)?,
                },
                true => match trans {
                    // cherk, zherk: NC accepted
                    BLASTrans::NoTrans | BLASTrans::ConjTrans => (),
                    _ => blas_invalid!(trans)?,
                },
            },
        };

        // optional intent(out)
        let c = match c {
            Some(c) => {
                blas_assert_eq!(c.dim(), (n, n), "Incompatible dimensions")?;
                if get_layout_array2(&c.view()).is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    let c_buffer = c.t().as_standard_layout().into_owned().reversed_axes();
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

pub type SYR2K<'a, 'b, 'c, F> = SYR2K_Builder<'a, 'b, 'c, F, BLASSymmetric<F>>;
pub type SSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, f32>;
pub type DSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, f64>;
pub type CSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, c32>;
pub type ZSYR2K<'a, 'b, 'c> = SYR2K<'a, 'b, 'c, c64>;

pub type HER2K<'a, 'b, 'c, F> = SYR2K_Builder<'a, 'b, 'c, F, BLASHermitian<F>>;
pub type CHER2K<'a, 'b, 'c> = HER2K<'a, 'b, 'c, c32>;
pub type ZHER2K<'a, 'b, 'c> = HER2K<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F, S> BLASBuilder<'c, F, Ix2> for SYR2K_Builder<'a, 'b, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
    BLASFunc: SYR2KFunc<F, S>,
{
    fn run(self) -> Result<ArrayOut2<'c, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout_b = get_layout_array2(&obj.b);

        if layout_a.is_fpref() && layout_b.is_fpref() {
            // F-contiguous: C = A op(B) + B op(A)
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous: C' = op(B') A' + op(A') B'
            let a_cow = obj.a.as_standard_layout();
            let b_cow = obj.b.as_standard_layout();
            let obj = SYR2K_::<'_, '_, '_, F, S> {
                a: b_cow.t(),
                b: a_cow.t(),
                c: obj.c.map(|c| c.reversed_axes()),
                alpha: obj.alpha,
                beta: obj.beta,
                uplo: match obj.uplo {
                    BLASUpLo::Lower => BLASUpLo::Upper,
                    BLASUpLo::Upper => BLASUpLo::Lower,
                    _ => blas_invalid!(obj.uplo)?,
                },
                trans: match F::is_complex() {
                    false => match obj.trans {
                        // ssyrk, dsyrk: NTC accepted
                        BLASTrans::NoTrans => BLASTrans::Trans,
                        BLASTrans::Trans => BLASTrans::NoTrans,
                        BLASTrans::ConjTrans => BLASTrans::NoTrans,
                        _ => blas_invalid!(obj.trans)?,
                    },
                    true => match S::is_hermitian() {
                        false => match obj.trans {
                            // csyrk, zsyrk: NT accepted
                            BLASTrans::NoTrans => BLASTrans::Trans,
                            BLASTrans::Trans => BLASTrans::NoTrans,
                            _ => blas_invalid!(obj.trans)?,
                        },
                        true => match obj.trans {
                            // cherk, zherk: NC accepted
                            BLASTrans::NoTrans => BLASTrans::ConjTrans,
                            BLASTrans::ConjTrans => BLASTrans::NoTrans,
                            _ => blas_invalid!(obj.trans)?,
                        },
                    },
                },
            };
            let c = obj.driver()?.run_blas()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */
