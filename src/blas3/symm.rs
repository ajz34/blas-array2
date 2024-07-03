use std::marker::PhantomData;

use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait SYMMFunc<F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
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
        impl SYMMFunc<$type, $symm> for BLASFunc
        {
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
                blas_sys::$func(
                    side,
                    uplo,
                    m,
                    n,
                    alpha as *const <$type as BLASFloat>::FFIFloat,
                    a as *const <$type as BLASFloat>::FFIFloat,
                    lda,
                    b as *const <$type as BLASFloat>::FFIFloat,
                    ldb,
                    beta as *const <$type as BLASFloat>::FFIFloat,
                    c as *mut <$type as BLASFloat>::FFIFloat,
                    ldc,
                );
            }
        }
    };
}

impl_func!(f32, BLASSymmetric, ssymm_);
impl_func!(f64, BLASSymmetric, dsymm_);
impl_func!(c32, BLASSymmetric, csymm_);
impl_func!(c64, BLASSymmetric, zsymm_);
impl_func!(c32, BLASHermitian, chemm_);
impl_func!(c64, BLASHermitian, zhemm_);

/* #endregion */

/* #region BLAS driver */

pub struct SYMM_Driver<'a, 'b, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
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
    _phantom: std::marker::PhantomData<S>,
}

impl<'a, 'b, 'c, F, S> SYMM_Driver<'a, 'b, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
{
    pub fn run(self) -> Result<ArrayOut2<'c, F>, AnyError>
    where 
        BLASFunc: SYMMFunc<F, S>,
    {
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

        unsafe {
            BLASFunc::symm(&side, &uplo, &m, &n, &alpha, a_ptr, &lda, b_ptr, &ldb, &beta, c_ptr, &ldc);
        }
        Ok(c.clone_to_view_mut())
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct SYMM_<'a, 'b, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
{
    pub a: ArrayView2<'a, F>,
    pub b: ArrayView2<'b, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub c: Option<ArrayViewMut2<'c, F>>,
    #[builder(default = "F::one()")]
    pub alpha: F,
    #[builder(default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASSide::Left")]
    pub side: BLASSide,
    #[builder(setter(into), default = "BLASUpLo::Upper")]
    pub uplo: BLASUpLo,
    
    #[builder(private, default = "PhantomData {}")]
    _phantom: std::marker::PhantomData<S>,
}

impl<'a, 'b, 'c, F, S> SYMM_<'a, 'b, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
{
    pub fn driver(self) -> Result<SYMM_Driver<'a, 'b, 'c, F, S>, AnyError> {
        let a = self.a;
        let b = self.b;
        let c = self.c;
        let alpha = self.alpha;
        let beta = self.beta;
        let side = self.side;
        let uplo = self.uplo;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        let layout_b = get_layout_array2(&b);
        if !(layout_a.is_fpref() && layout_b.is_fpref()) {
            BLASError("Inner driver should be fortran-only. This is probably error of library author.".to_string());
        }

        // initialize intent(hide)
        let m = b.dim().0;
        let n = b.dim().1;
        let lda = a.stride_of(Axis(1));
        let ldb = b.stride_of(Axis(1));

        // perform check
        match side {
            BLASSide::Left => {
                BLASError::assert(
                    a.dim() == (m, m),
                    format!("Incompatible dimensions, a.dim={:?}, (m,m)={:?}.", a.dim(), (m, m))
                )?
            },
            BLASSide::Right => {
                BLASError::assert(
                    a.dim() == (n, n),
                    format!("Incompatible dimensions, a.dim={:?}, (n,n)={:?}.", a.dim(), (n, n))
                )?
            },
            _ => Err(BLASError(format!("Unknown side {side:?}")))?,
        }

        // optional intent(out)
        let c = match c {
            Some(c) => {
                BLASError::assert(
                    c.dim() == (m, n),
                    format!("Incompatible dimensions, c.dim={:?}, (m,n)={:?}.", c.dim(), (m, n)),
                )?;
                if get_layout_array2(&c.view()).is_fpref() {
                    ArrayOut2::ViewMut(c)
                } else {
                    ArrayOut2::ToBeCloned(c, Array2::zeros((m, n).f()))
                }
            }
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
            _phantom: PhantomData {},
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type SYMM<'a, 'b, 'c, F> = SYMM_Builder<'a, 'b, 'c, F, BLASSymmetric>;
pub type SSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, f32>;
pub type DSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, f64>;
pub type CSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, c32>;
pub type ZSYMM<'a, 'b, 'c> = SYMM<'a, 'b, 'c, c64>;

pub type HEMM<'a, 'b, 'c, F> = SYMM_Builder<'a, 'b, 'c, F, BLASHermitian>;
pub type CHEMM<'a, 'b, 'c> = HEMM<'a, 'b, 'c, c32>;
pub type ZHEMM<'a, 'b, 'c> = HEMM<'a, 'b, 'c, c64>;

impl<'a, 'b, 'c, F, S> SYMM_Builder<'a, 'b, 'c, F, S>
where 
    F: BLASFloat,
    S: BLASSymm,
    BLASFunc: SYMMFunc<F, S>,
{
    pub fn run(self) -> Result<ArrayOut2<'c, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout_b = get_layout_array2(&obj.b);

        if layout_a.is_fpref() && layout_b.is_fpref() {
            // F-contiguous: C = A B
            return obj.driver()?.run();
        } else if layout_a.is_cpref() && layout_b.is_cpref() {
            // C-contiguous: C' = B' A'
            let obj = SYMM_::<'_, '_, '_, F, S> {
                a: obj.a.reversed_axes(),
                b: obj.b.reversed_axes(),
                c: match obj.c {
                    Some(c) => Some(c.reversed_axes()),
                    None => None,
                },
                alpha: obj.alpha,
                beta: obj.beta,
                side: match obj.side {
                    BLASSide::Left => BLASSide::Right,
                    BLASSide::Right => BLASSide::Left,
                    _ => Err(BLASError(format!("Unsupported BLASSide {:?}", obj.side)))?,
                },
                uplo: match obj.uplo {
                    BLASUpLo::Lower => BLASUpLo::Upper,
                    BLASUpLo::Upper => BLASUpLo::Lower,
                    _ => Err(BLASError(format!("Unsupported BLASUpLo {:?}", obj.uplo)))?,
                },
                _phantom: PhantomData {},
            };
            let c = obj.driver()?.run()?.reversed_axes();
            return Ok(c);
        } else {
            // neither F-contiguous nor C-contiguous
            // copy to C contiguous if necessary, then C' = B' A'
            let a_cow = obj.a.as_standard_layout();
            let b_cow = obj.b.as_standard_layout();
            let a_view = a_cow.view();
            let b_view = b_cow.view();
            let obj = SYMM_::<'_, '_, '_, F, S> {
                a: a_view.t(),
                b: b_view.t(),
                c: match obj.c {
                    Some(c) => Some(c.reversed_axes()),
                    None => None,
                },
                alpha: obj.alpha,
                beta: obj.beta,
                side: match obj.side {
                    BLASSide::Left => BLASSide::Right,
                    BLASSide::Right => BLASSide::Left,
                    _ => Err(BLASError(format!("Unsupported BLASSide {:?}", obj.side)))?,
                },
                uplo: match obj.uplo {
                    BLASUpLo::Lower => BLASUpLo::Upper,
                    BLASUpLo::Upper => BLASUpLo::Lower,
                    _ => Err(BLASError(format!("Unsupported BLASUpLo {:?}", obj.uplo)))?,
                },
                _phantom: PhantomData {},
            };
            let c = obj.driver()?.run()?.reversed_axes();
            return Ok(c);
        }
    }
}

/* #endregion */

