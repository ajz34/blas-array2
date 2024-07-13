use crate::blas2::ger::GER_;
use crate::util::*;
use blas_sys;
use derive_builder::Builder;
use libc::c_int;
use ndarray::prelude::*;

use super::ger::GERFunc;

/* #region BLAS func */

pub trait GERCFunc<F>
where
    F: BLASFloat,
{
    unsafe fn gerc(
        m: *const c_int,
        n: *const c_int,
        alpha: *const F,
        x: *const F,
        incx: *const c_int,
        y: *const F,
        incy: *const c_int,
        a: *mut F,
        lda: *const c_int,
    );
}

macro_rules! impl_func {
    ($type: ty, $func: ident) => {
        impl GERCFunc<$type> for BLASFunc
        where
            $type: BLASFloat,
        {
            unsafe fn gerc(
                m: *const c_int,
                n: *const c_int,
                alpha: *const $type,
                x: *const $type,
                incx: *const c_int,
                y: *const $type,
                incy: *const c_int,
                a: *mut $type,
                lda: *const c_int,
            ) {
                type FFIFloat = <$type as BLASFloat>::FFIFloat;
                blas_sys::$func(
                    m,
                    n,
                    alpha as *const FFIFloat,
                    x as *const FFIFloat,
                    incx,
                    y as *const FFIFloat,
                    incy,
                    a as *mut FFIFloat,
                    lda,
                );
            }
        }
    };
}

impl_func!(c32, cgerc_);
impl_func!(c64, zgerc_);

/* #endregion */

/* #region BLAS driver */

pub struct GERC_Driver<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    m: c_int,
    n: c_int,
    alpha: F,
    x: ArrayView1<'x, F>,
    incx: c_int,
    y: ArrayView1<'y, F>,
    incy: c_int,
    a: ArrayOut2<'a, F>,
    lda: c_int,
}

impl<'x, 'y, 'a, F> BLASDriver<'a, F, Ix2> for GERC_Driver<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: GERCFunc<F>,
{
    fn run_blas(self) -> Result<ArrayOut2<'a, F>, AnyError> {
        let Self { m, n, alpha, x, incx, y, incy, mut a, lda } = self;
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_ptr();
        let a_ptr = match &mut a {
            ArrayOut2::Owned(a) => a.as_mut_ptr(),
            ArrayOut2::ViewMut(a) => a.as_mut_ptr(),
            ArrayOut2::ToBeCloned(_, a) => a.as_mut_ptr(),
        };

        // assuming dimension checks has been performed
        // unconditionally return Ok if output does not contain anything
        if m == 0 || n == 0 {
            return Ok(a);
        }

        unsafe {
            BLASFunc::gerc(&m, &n, &alpha, x_ptr, &incx, y_ptr, &incy, a_ptr, &lda);
        }
        return Ok(a);
    }
}

/* #endregion */

/* #region BLAS builder */

#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct GERC_<'x, 'y, 'a, F>
where
    F: BLASFloat,
{
    pub x: ArrayView1<'x, F>,
    pub y: ArrayView1<'y, F>,

    #[builder(setter(into, strip_option), default = "None")]
    pub a: Option<ArrayViewMut2<'a, F>>,
    #[builder(setter(into), default = "F::one()")]
    pub alpha: F,
}

impl<'x, 'y, 'a, F> BLASBuilder_<'a, F, Ix2> for GERC_<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: GERCFunc<F>,
{
    fn driver(self) -> Result<GERC_Driver<'x, 'y, 'a, F>, AnyError> {
        let Self { x, y, a, alpha } = self;

        // initialize intent(hide)
        let incx = x.stride_of(Axis(0));
        let incy = y.stride_of(Axis(0));
        let m = x.len_of(Axis(0));
        let n = y.len_of(Axis(0));

        // prepare output
        assert!(a.is_some(), "Currently rank-update does not allow inner driver accept empty output");
        let a = a.unwrap();
        // dimension check and assign
        blas_assert_eq!(a.dim(), (m, n), "Incompatible dimensions")?;
        let layout_a = get_layout_array2(&a.view());
        assert!(layout_a.is_fpref(), "This function is designed not to handle C-contiguous array");
        let a = ArrayOut2::ViewMut(a);
        let lda = a.view().stride_of(Axis(1));

        // finalize
        let driver = GERC_Driver {
            m: m.try_into().unwrap(),
            n: n.try_into().unwrap(),
            alpha,
            x,
            incx: incx.try_into().unwrap(),
            y,
            incy: incy.try_into().unwrap(),
            a,
            lda: lda.try_into().unwrap(),
        };
        return Ok(driver);
    }
}

/* #endregion */

/* #region BLAS wrapper */

pub type GERC<'x, 'y, 'a, F> = GERC_Builder<'x, 'y, 'a, F>;
pub type CGERCU<'x, 'y, 'a> = GERC<'x, 'y, 'a, c32>;
pub type ZGERCU<'x, 'y, 'a> = GERC<'x, 'y, 'a, c64>;

impl<'x, 'y, 'a, F> BLASBuilder<'a, F, Ix2> for GERC_Builder<'x, 'y, 'a, F>
where
    F: BLASFloat,
    BLASFunc: GERCFunc<F> + GERFunc<F>,
{
    fn run(self) -> Result<ArrayOut2<'a, F>, AnyError> {
        // initialize
        let obj = self.build()?;

        if let Some(mut a) = obj.a {
            let layout = get_layout_array2(&a.view());
            if layout.is_fpref() {
                // F-contiguous
                let obj = GERC_ { a: Some(a), ..obj };
                return obj.driver()?.run_blas();
            } else if layout.is_cpref() {
                // C-contiguous
                let y = obj.y.mapv(F::conj);
                let obj = GER_ { a: Some(a.reversed_axes()), x: y.view(), y: obj.x, alpha: obj.alpha };
                let a = obj.driver()?.run_blas()?;
                return Ok(a.reversed_axes());
            } else {
                // other kinds of contiguous: use F-contiguous then assign
                let mut a_new = a.t().as_standard_layout().into_owned().reversed_axes();
                let obj = GERC_ { a: Some(a_new.view_mut()), ..obj };
                obj.driver()?.run_blas()?;
                a.assign(&a_new.view());
                return Ok(ArrayOut::ViewMut(a));
            }
        } else {
            // empty output: C-contiguous
            let m = obj.x.len_of(Axis(0));
            let n = obj.y.len_of(Axis(0));
            let y = obj.y.mapv(F::conj);
            let mut a = Array2::zeros((n, m).f());
            let obj = GER_ { a: Some(a.view_mut()), x: y.view(), y: obj.x, alpha: obj.alpha };
            obj.driver()?.run_blas()?;
            return Ok(ArrayOut::Owned(a.reversed_axes()));
        }
    }
}

/* #endregion */