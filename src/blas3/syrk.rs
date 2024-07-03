use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;
use num_traits::{One, Zero};

/* #region BLAS func */

pub trait SYRKFunc<F, S>
where
    F: BLASFloat,
    S: BLASSymm,
{
    unsafe fn syrk(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const c_int,
        k: *const c_int,
        alpha: *const S::HermitianFloat,
        a: *const F,
        lda: *const c_int,
        beta: *const S::HermitianFloat,
        c: *mut F,
        ldc: *const c_int,
    );
}

macro_rules! impl_syrk {
    ($type: ty, $symm: ty, $func: ident) => {
        impl SYRKFunc<$type, $symm> for BLASFunc {
            unsafe fn syrk(
                uplo: *const c_char,
                trans: *const c_char,
                n: *const c_int,
                k: *const c_int,
                alpha: *const <$symm as BLASSymm>::HermitianFloat,
                a: *const $type,
                lda: *const c_int,
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
                    alpha as *const FFIHermitialFloat,
                    a as *const FFIFloat,
                    lda,
                    beta as *const FFIHermitialFloat,
                    c as *mut FFIFloat,
                    ldc,
                );
            }
        }
    };
}

impl_syrk!(f32, BLASSymmetric<f32>, ssyrk_);
impl_syrk!(f64, BLASSymmetric<f64>, dsyrk_);
impl_syrk!(c32, BLASSymmetric<c32>, csyrk_);
impl_syrk!(c64, BLASSymmetric<c64>, zsyrk_);
impl_syrk!(c32, BLASHermitian<c32>, cherk_);
impl_syrk!(c64, BLASHermitian<c64>, zherk_);

/* #endregion */

/* #region BLAS driver */

pub struct SYRK_Driver<'a, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
{
    uplo: c_char,
    trans: c_char,
    n: c_int,
    k: c_int,
    alpha: S::HermitianFloat,
    a: ArrayView2<'a, F>,
    lda: c_int,
    beta: S::HermitianFloat,
    c: ArrayOut2<'c, F>,
    ldc: c_int,
}

impl<'a, 'c, F, S> SYRK_Driver<'a, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
{
    pub fn run(self) -> Result<ArrayOut2<'c, F>, AnyError>
    where 
        BLASFunc: SYRKFunc<F, S>
    {
        let uplo = self.uplo;
        let trans = self.trans;
        let n = self.n;
        let k = self.k;
        let alpha = self.alpha;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
        let beta = self.beta;
        let mut c = self.c;
        let c_ptr = match &mut c {
            ArrayOut::ViewMut(c) => c.as_mut_ptr(),
            ArrayOut::Owned(c) => c.as_mut_ptr(),
            ArrayOut::ToBeCloned(_, c) => c.as_mut_ptr(),
        };
        let ldc = self.ldc;

        unsafe {
            BLASFunc::syrk(&uplo, &trans, &n, &k, &alpha, a_ptr, &lda, &beta, c_ptr, &ldc);
        }
        Ok(c.clone_to_view_mut())
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct SYRK_<'a, 'c, F, S>
where
    F: BLASFloat,
    S: BLASSymm,
    S::HermitianFloat: Zero + One,
{
    pub a: ArrayView2<'a, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(default = "S::HermitianFloat::one()")]
    pub alpha: S::HermitianFloat,
    #[builder(default = "S::HermitianFloat::zero()")]
    pub beta: S::HermitianFloat,
    #[builder(setter(into), default = "BLASUpLo::Lower")]
    pub uplo: BLASUpLo,
    #[builder(setter(into), default = "BLASTrans::NoTrans")]
    pub trans: BLASTrans,
}

impl<'a, 'c, F, S> SYRK_<'a, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
{
    pub fn driver(self) -> Result<SYRK_Driver<'a, 'c, F, S>, AnyError> {
        let a = self.a;
        let c = self.c;
        let alpha = self.alpha;
        let beta = self.beta;
        let uplo = self.uplo;
        let trans = self.trans;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        if !layout_a.is_fpref() {
            Err(BLASError("Inner driver should be fortran-only. This is probably error of library author.".to_string()))?;
        }

        // initialize intent(hide)
        let (n, k) = match trans {
            BLASTrans::NoTrans => (a.dim().0, a.dim().1),
            BLASTrans::Trans | BLASTrans::ConjTrans => (a.dim().1, a.dim().0),
            _ => Err(BLASError(format!("Unknown trans {trans:?}")))?,
        };
        let lda = a.stride_of(Axis(1));

        // optional intent(out)
        let c = match c {
            Some(c) => {
                BLASError::assert(
                    c.dim() == (n, n),
                    format!("Incompatible dimensions, c.dim={:?}, (n,n)={:?}.", c.dim(), (n, n)),
                )?;
                if get_layout_array2(&c.view()).is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    ArrayOut2::ToBeCloned(c, Array2::zeros((n, n).f()))
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

pub type SYRK<'a, 'c, F> = SYRK_Builder<'a, 'c, F, BLASSymmetric<F>>;
pub type SSYRK<'a, 'c> = SYRK<'a, 'c, f32>;
pub type DSYRK<'a, 'c> = SYRK<'a, 'c, f64>;
pub type CSYRK<'a, 'c> = SYRK<'a, 'c, c32>;
pub type ZSYRK<'a, 'c> = SYRK<'a, 'c, c64>;

pub type HERK<'a, 'c, F> = SYRK_Builder<'a, 'c, F, BLASHermitian<F>>;
pub type CHERK<'a, 'c> = HERK<'a, 'c, c32>;
pub type ZHERK<'a, 'c> = HERK<'a, 'c, c64>;

impl<'a, 'c, F, S> SYRK_Builder<'a, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
    BLASFunc: SYRKFunc<F, S>,
{
    pub fn run(self) -> Result<ArrayOut2<'c, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);

        if layout_a.is_fpref() {
            // F-contiguous: C = A op(A) or C = op(A) A
            return obj.driver()?.run();
        } else {
            // C-contiguous: C' = op(A') A' or C' = A' op(A')
            let a_cow = obj.a.as_standard_layout();
            let a_view = a_cow.view();
            let obj = SYRK_::<'_, '_, F, S> {
                a: a_view.t(),
                c: obj.c.map(|c| c.reversed_axes()),
                alpha: obj.alpha,
                beta: obj.beta,
                uplo: match obj.uplo {
                    BLASUpLo::Lower => BLASUpLo::Upper,
                    BLASUpLo::Upper => BLASUpLo::Lower,
                    _ => Err(BLASError(format!("Unsupported BLASUpLo {:?}", obj.uplo)))?,
                },
                trans: match F::is_complex() {
                    false => match obj.trans {
                        // ssyrk, dsyrk: NTC accepted
                        BLASTrans::NoTrans => BLASTrans::Trans,
                        BLASTrans::Trans => BLASTrans::NoTrans,
                        BLASTrans::ConjTrans => BLASTrans::NoTrans,
                        _ => Err(BLASError(format!("Unsupported BLASTrans {:?}", obj.trans)))?,
                    },
                    true => match S::is_hermitian() {
                        false => match obj.trans {
                            // csyrk, zsyrk: NT accepted
                            BLASTrans::NoTrans => BLASTrans::Trans,
                            BLASTrans::Trans => BLASTrans::NoTrans,
                            _ => Err(BLASError(format!("Unsupported BLASTrans {:?}", obj.trans)))?,
                        },
                        true => match obj.trans {
                            // cherk, zherk: NC accepted
                            BLASTrans::NoTrans => BLASTrans::ConjTrans,
                            BLASTrans::ConjTrans => BLASTrans::NoTrans,
                            _ => Err(BLASError(format!("Unsupported BLASTrans {:?}", obj.trans)))?,
                        }
                    },
                },
            };
            let c = obj.driver()?.run()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */
