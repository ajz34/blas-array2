use crate::util::*;
use blas_array2::blas2::spr::{HPR, SPR};
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
        type FRE = f32;
        type FFI = <F as BLASFloat>::FFIFloat;

        for (layout, uplo, incap, incx) in iproduct!(['C', 'R'], ['U', 'L'], [1, 2], [1, 2]) {
            println!("test info: {:?}", (layout, uplo, incap, incx));
            let n = 8;
            let np = n * (n + 1) / 2;

            let ap_slc = slice_1d(np, incap);
            let x_slc = slice_1d(n, incx);

            let mut ap_raw = random_array::<F>(1000);
            let x_raw = random_array::<F>(100);
            let alpha = FRE::rand();

            let mut a_origin = ap_raw.clone();
            let mut ap_naive = ap_raw.slice(ap_slc).into_owned();
            let x_naive = x_raw.slice(x_slc).into_owned();
            let incx = 1;
            unsafe {
                cblas_chpr(
                    to_cblas_layout(layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    ap_naive.as_mut_ptr() as *mut FFI,
                );
            }

            HPR::<F>::default()
                .x(x_raw.slice(x_slc))
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
        type FRE = f32;
        type FFI = <F as BLASFloat>::FFIFloat;

        for (layout, uplo, incx) in iproduct!(['C', 'R'], ['U', 'L'], [1, 2]) {
            println!("test info: {:?}", (layout, uplo, incx));
            let n = 8;
            let np = n * (n + 1) / 2;

            let x_slc = slice_1d(n, incx);

            let x_raw = random_array::<F>(100);
            let alpha = FRE::rand();

            let mut ap_naive = Array1::<F>::zeros(np);
            let x_naive = x_raw.slice(x_slc).into_owned();
            let incx = 1;
            unsafe {
                cblas_chpr(
                    to_cblas_layout(layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    ap_naive.as_mut_ptr() as *mut FFI,
                );
            }

            let a_out = HPR::<F>::default()
                .x(x_raw.slice(x_slc))
                .alpha(alpha)
                .uplo(uplo)
                .layout(layout)
                .run()
                .unwrap()
                .into_owned();

            check_same(&a_out.view(), &ap_naive.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_f32_owned() {
        type F = f32;
        type FRE = f32;
        type FFI = <F as BLASFloat>::FFIFloat;

        for (layout, uplo, incx) in iproduct!(['C', 'R'], ['U', 'L'], [1, 2]) {
            println!("test info: {:?}", (layout, uplo, incx));
            let n = 8;
            let np = n * (n + 1) / 2;

            let x_slc = slice_1d(n, incx);

            let x_raw = random_array::<F>(100);
            let alpha = FRE::rand();

            let mut ap_naive = Array1::<F>::zeros(np);
            let x_naive = x_raw.slice(x_slc).into_owned();
            let incx = 1;
            unsafe {
                cblas_sspr(
                    to_cblas_layout(layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    ap_naive.as_mut_ptr() as *mut FFI,
                );
            }

            let a_out = SPR::<F>::default()
                .x(x_raw.slice(x_slc))
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
