use crate::util::*;
use blas_array2::blas2::spmv::{HPMV, SPMV};
use blas_array2::prelude::*;
use cblas_sys::*;
use itertools::*;

#[cfg(test)]
mod valid_row_major {
    use super::*;

    #[test]
    fn test_cblas_row_major_c32() {
        type F = c32;
        for (cblas_layout, uplo, stride_x, stride_y) in iproduct!(['R', 'C'], ['U', 'L'], [1, 3], [1, 3]) {
            let n = 8;
            let np = n * (n + 1) / 2;

            // slice definition
            let a_slc = slice_1d(np, 3);
            let x_slc = slice_1d(n, stride_x);
            let y_slc = slice_1d(n, stride_y);

            // type definition
            type FFI = <F as BLASFloat>::FFIFloat;

            // data assignment
            let alpha = F::rand();
            let beta = F::rand();
            let a_raw = random_array(1000);
            let x_raw = random_array(1000);
            let mut y_raw = random_array(1000);
            let mut y_origin = y_raw.clone();

            // cblas computation
            let a_naive = a_raw.slice(a_slc).into_owned();
            let x_naive = x_raw.slice(x_slc).into_owned();
            let mut y_naive = y_raw.slice_mut(y_slc).into_owned();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_chpmv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    [alpha].as_ptr() as *const FFI,
                    a_naive.as_ptr() as *const FFI,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    [beta].as_ptr() as *const FFI,
                    y_naive.as_mut_ptr() as *mut FFI,
                    incy.try_into().unwrap(),
                );
            }

            HPMV::<F>::default()
                .ap(a_raw.slice(a_slc))
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice_mut(y_slc))
                .alpha(alpha)
                .beta(beta)
                .uplo(uplo)
                .layout(cblas_layout)
                .run()
                .unwrap();

            check_same(&y_raw.slice(y_slc), &y_naive.view(), 4.0 * F::EPSILON);
            y_raw.slice_mut(y_slc).fill(F::from(0.0));
            y_origin.slice_mut(y_slc).fill(F::from(0.0));
            check_same(&y_raw.view(), &y_origin.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_row_major_f32() {
        type F = f32;
        for (cblas_layout, uplo, stride_x, stride_y) in iproduct!(['R', 'C'], ['U', 'L'], [1, 3], [1, 3]) {
            let n = 8;
            let np = n * (n + 1) / 2;

            // slice definition
            let a_slc = slice_1d(np, 3);
            let x_slc = slice_1d(n, stride_x);
            let y_slc = slice_1d(n, stride_y);

            // type definition
            type FFI = <F as BLASFloat>::FFIFloat;

            // data assignment
            let alpha = F::rand();
            let beta = F::rand();
            let a_raw = random_array(1000);
            let x_raw = random_array(1000);
            let mut y_raw = random_array(1000);
            let mut y_origin = y_raw.clone();

            // cblas computation
            let a_naive = a_raw.slice(a_slc).into_owned();
            let x_naive = x_raw.slice(x_slc).into_owned();
            let mut y_naive = y_raw.slice_mut(y_slc).into_owned();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_sspmv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    a_naive.as_ptr() as *const FFI,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    beta,
                    y_naive.as_mut_ptr() as *mut FFI,
                    incy.try_into().unwrap(),
                );
            }

            SPMV::<F>::default()
                .ap(a_raw.slice(a_slc))
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice_mut(y_slc))
                .alpha(alpha)
                .beta(beta)
                .uplo(uplo)
                .layout(cblas_layout)
                .run()
                .unwrap();

            check_same(&y_raw.slice(y_slc), &y_naive.view(), 4.0 * F::EPSILON);
            y_raw.slice_mut(y_slc).fill(F::from(0.0));
            y_origin.slice_mut(y_slc).fill(F::from(0.0));
            check_same(&y_raw.view(), &y_origin.view(), 4.0 * F::EPSILON);
        }
    }
}
