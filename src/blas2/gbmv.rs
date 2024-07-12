use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::{c_char, c_int};
use ndarray::prelude::*;

/* #region BLAS func */

pub trait GBMVFunc<F>
where
    F: BLASFloat,
{
    unsafe fn gbmv(
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        kl: *const c_int,
        ku: *const c_int,
        alpha: *const F,
        a: *const F,
        lda: *const c_int,
        x: *const F,
        incx: *const c_int,
        beta: *const F,
        y: *mut F,
        incy: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GBMVFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn gbmv(
                trans: *const c_char,
                m: *const c_int,
                n: *const c_int,
                kl: *const c_int,
                ku: *const c_int,
                alpha: *const $type,
                a: *const $type,
                lda: *const c_int,
                x: *const $type,
                incx: *const c_int,
                beta: *const $type,
                y: *mut $type,
                incy: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    trans,
                    m,
                    n,
                    kl,
                    ku,
                    alpha as *const FFIFloat,
                    a as *const FFIFloat,
                    lda,
                    x as *const FFIFloat,
                    incx,
                    beta as *const FFIFloat,
                    y as *mut FFIFloat,
                    incy,
                );
            }
        }
    };
}

impl_func!(f32, sgbmv_);
impl_func!(f64, dgbmv_);
impl_func!(c32, cgbmv_);
impl_func!(c64, zgbmv_);

/* #endregion */

/* #region BLAS driver */

pub struct GBMV_Driver<'a, 'x, 'y, F>
where
    F: BLASFloat,
{
    trans: c_char,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    alpha: F,
    a: ArrayView2<'a, F>,
    lda: c_int,
    x: ArrayView1<'x, F>,
    incx: c_int,
    beta: F,
    y: ArrayOut1<'y, F>,
    incy: c_int,
}

impl<'a, 'x, 'y, F> BLASDriver<'y, F, Ix1> for GBMV_Driver<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: GBMVFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut1<'y, F>, AnyError> {
        let trans = self.trans;
        let m = self.m;
        let n = self.n;
        let kl = self.kl;
        let ku = self.ku;
        let alpha = self.alpha;
        let a_ptr = self.a.as_ptr();
        let lda = self.lda;
        let x_ptr = self.x.as_ptr();
        let incx = self.incx;
        let beta = self.beta;
        let mut y = self.y;
        let y_ptr = match &mut y {
            ArrayOut1::Owned(y) => y.as_mut_ptr(),
            ArrayOut1::ViewMut(y) => y.as_mut_ptr(),
            _ => panic!("Ix1 won't be ToBeCloned"),
        };
        let incy = self.incy;

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if n == 0 {
            return Ok(y);
        }

        unsafe {
            BLASFunc::gbmv(&trans, &m, &n, &kl, &ku, &alpha, a_ptr, &lda, x_ptr, &incx, &beta, y_ptr, &incy);
        }
        return Ok(y);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]

pub struct GBMV_<'a, 'x, 'y, F>
where
    F: BLASFloat,
{
    pub a: ArrayView2<'a, F>,
    pub x: ArrayView1<'x, F>,
    pub m: usize,
    pub kl: usize,
    pub ku: usize,

    #[builder(setter(into, strip_option), default = "None")]
    pub y: Option<ArrayViewMut1<'y, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
    #[builder(setter(into), default = "F::zero()")]
    pub beta: F,
    #[builder(setter(into), default = "BLASNoTrans")]
    pub trans: BLASTranspose,
    #[builder(setter(into, strip_option), default = "None")]
    pub layout: Option<BLASLayout>,
}

impl<'a, 'x, 'y, F> BLASBuilder_<'y, F, Ix1> for GBMV_<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: GBMVFunc<F>,
{
    fn driver(self) -> Result<GBMV_Driver<'a, 'x, 'y, F>, AnyError> {
        let a = self.a;
        let x = self.x;
        let y = self.y;
        let m = self.m;
        let kl = self.kl;
        let ku = self.ku;
        let alpha = self.alpha;
        let beta = self.beta;
        let trans = self.trans;
        let layout = self.layout;

        // only fortran-preferred (col-major) is accepted in inner wrapper
        let layout_a = get_layout_array2(&a);
        assert!(layout_a.is_fpref());
        assert!(layout == Some(BLASLayout::ColMajor));

        // initialize intent(hide)
        let (k, n) = a.dim();
        let lda = a.stride_of(Axis(1));
        let incx = x.stride_of(Axis(0));

        // perform check
        blas_assert!(m >= kl + ku + 1, "Incompatible dimensions")?;
        blas_assert!(k == kl + ku + 1, "Incompatible dimensions")?;
        match trans {
            BLASNoTrans => {
                blas_assert_eq!(x.len_of(Axis(0)), n, "Incompatible dimensions")?;
            },
            BLASTrans | BLASConjTrans => blas_assert_eq!(x.len_of(Axis(0)), m, "Incompatible dimensions")?,
            _ => blas_invalid!(trans)?,
        };

        // prepare output
        let y = match y {
            Some(y) => {
                match trans {
                    BLASNoTrans => blas_assert_eq!(y.len_of(Axis(0)), m, "Incompatible dimensions")?,
                    BLASTrans | BLASConjTrans => {
                        blas_assert_eq!(y.len_of(Axis(0)), n, "Incompatible dimensions")?
                    },
                    _ => blas_invalid!(trans)?,
                };
                ArrayOut1::ViewMut(y)
            },
            None => ArrayOut1::Owned(Array1::zeros(match trans {
                BLASNoTrans => m,
                BLASTrans | BLASConjTrans => n,
                _ => blas_invalid!(trans)?,
            })),
        };
        let incy = y.view().stride_of(Axis(0));

        // finalize
        let driver = GBMV_Driver {
            trans: trans.into(),
            m: m.try_into()?,
            n: n.try_into()?,
            kl: kl.try_into()?,
            ku: ku.try_into()?,
            alpha,
            a,
            lda: lda.try_into()?,
            x,
            incx: incx.try_into()?,
            beta,
            y,
            incy: incy.try_into()?,
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type GBMV<'a, 'x, 'y, F> = GBMV_Builder<'a, 'x, 'y, F>;
pub type SGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, f32>;
pub type DGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, f64>;
pub type CGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, c32>;
pub type ZGBMV<'a, 'x, 'y> = GBMV<'a, 'x, 'y, c64>;

impl<'a, 'x, 'y, F> BLASBuilder<'y, F, Ix1> for GBMV_Builder<'a, 'x, 'y, F>
where
    F: BLASFloat,
    BLASFunc: GBMVFunc<F>,
{
    fn run(self) -> Result<ArrayOut1<'y, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        let layout_a = get_layout_array2(&obj.a);
        let layout = match obj.layout {
            Some(layout) => layout,
            None => match layout_a {
                BLASLayout::Sequential => BLASColMajor,
                BLASRowMajor => BLASRowMajor,
                BLASColMajor => BLASColMajor,
                _ => blas_raise!("Without defining layout, this function checks layout of input matrix `a` but it is not contiguous.")?,
            }
        };

        if layout == BLASColMajor {
            // F-contiguous
            let a_cow = obj.a.reversed_axes();
            let a_cow = a_cow.as_standard_layout().reversed_axes();
            let obj = GBMV_ { a: a_cow.view(), layout: Some(BLASColMajor), ..obj };
            return obj.driver()?.run_blas();
        } else {
            // C-contiguous
            let a_cow = obj.a.as_standard_layout();
            match obj.trans {
                BLASNoTrans => {
                    // N -> T
                    let obj = GBMV_ {
                        a: a_cow.t(),
                        trans: BLASTrans,
                        ku: obj.kl,
                        kl: obj.ku,
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASTrans => {
                    // N -> T
                    let obj = GBMV_ {
                        a: a_cow.t(),
                        trans: BLASNoTrans,
                        ku: obj.kl,
                        kl: obj.ku,
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    return obj.driver()?.run_blas();
                },
                BLASConjTrans => {
                    // C -> N
                    let x = obj.x.mapv(F::conj);
                    let y = obj.y.map(|mut y| {
                        y.mapv_inplace(F::conj);
                        y
                    });
                    let obj = GBMV_ {
                        a: a_cow.t(),
                        trans: BLASNoTrans,
                        x: x.view(),
                        y,
                        ku: obj.kl,
                        kl: obj.ku,
                        alpha: F::conj(obj.alpha),
                        beta: F::conj(obj.beta),
                        layout: Some(BLASColMajor),
                        ..obj
                    };
                    let mut y = obj.driver()?.run_blas()?;
                    y.view_mut().mapv_inplace(F::conj);
                    return Ok(y);
                },
                _ => return blas_invalid!(obj.trans)?,
            }
        }
    }
}

/* #endregion */
