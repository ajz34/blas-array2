use crate::util::*;
use blas_array2::prelude::*;
use cblas_sys::*;
use itertools::*;
use ndarray::prelude::*;

#[cfg(test)]
mod valid_cblas {
    use super::*;

    #[test]
    fn test_cblas_c32_view() {
        type F = c32;
        type FFI = <F as TestFloat>::FFIFloat;

        for (layout, uplo, incap, incx, incy) in iproduct!(['C', 'R'], ['U', 'L'], [1, 2], [1, 2], [1, 2]) {
            let n = 8;
            let np = n * (n + 1) / 2;

            let ap_slc = slice_1d(np, incap);
            let x_slc = slice_1d(n, incx);
            let y_slc = slice_1d(n, incy);

            let mut ap_raw = random_array::<F>(1000);
            let x_raw = random_array::<F>(100);
            let y_raw = random_array::<F>(100);
            let alpha = F::rand();

            let mut a_origin = ap_raw.clone();
            let mut ap_naive = ap_raw.slice(ap_slc).into_owned();
            let x_naive = x_raw.slice(x_slc).into_owned();
            let y_naive = y_raw.slice(y_slc).into_owned();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_chpr2(
                    to_cblas_layout(layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    [alpha].as_ptr() as *const FFI,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    y_naive.as_ptr() as *const FFI,
                    incy.try_into().unwrap(),
                    ap_naive.as_mut_ptr() as *mut FFI,
                );
            }

            HPR2::<F>::default()
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice(y_slc))
                .ap(ap_raw.slice_mut(ap_slc))
                .alpha(alpha)
                .uplo(uplo)
                .layout(layout)
                .run()
                .unwrap();

            check_same(&ap_raw.slice(ap_slc), &ap_naive.view(), 4.0 * F::EPSILON);
            ap_raw.slice_mut(ap_slc).fill(F::from(0.0));
            a_origin.slice_mut(ap_slc).fill(F::from(0.0));
            check_same(&ap_raw.view(), &a_origin.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_c32_owned() {
        type F = c32;
        type FFI = <F as TestFloat>::FFIFloat;

        for (layout, uplo, incx, incy) in iproduct!(['C', 'R'], ['U', 'L'], [1, 2], [1, 2]) {
            let n = 8;
            let np = n * (n + 1) / 2;

            let x_slc = slice_1d(n, incx);
            let y_slc = slice_1d(n, incy);

            let x_raw = random_array::<F>(100);
            let y_raw = random_array::<F>(100);
            let alpha = F::rand();

            let mut ap_naive = Array1::<F>::zeros(np);
            let x_naive = x_raw.slice(x_slc).into_owned();
            let y_naive = y_raw.slice(y_slc).into_owned();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_chpr2(
                    to_cblas_layout(layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    [alpha].as_ptr() as *const FFI,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    y_naive.as_ptr() as *const FFI,
                    incy.try_into().unwrap(),
                    ap_naive.as_mut_ptr() as *mut FFI,
                );
            }

            let a_out = HPR2::<F>::default()
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice(y_slc))
                .alpha(alpha)
                .uplo(uplo)
                .layout(layout)
                .run()
                .unwrap()
                .into_owned();

            check_same(&a_out.view(), &ap_naive.view(), 4.0 * F::EPSILON);
        }
    }
}
