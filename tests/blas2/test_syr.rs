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
        type FRE = f32;
        type FFI = <F as TestFloat>::FFIFloat;

        for (layout, uplo, as0, as1, incx) in iproduct!(['C', 'R'], ['U', 'L'], [1, 2], [1, 2], [1, 2]) {
            let n = 8;

            let a_slc = slice(n, n, as0, as1);
            let x_slc = slice_1d(n, incx);

            let mut a_raw = random_matrix::<F>(100, 100, layout.into());
            let x_raw = random_array::<F>(100);
            let alpha = FRE::rand();

            let mut a_origin = a_raw.clone();
            let mut a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), 'C');
            let x_naive = x_raw.slice(x_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            unsafe {
                cblas_cher(
                    to_cblas_layout('C'),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    a_naive.as_mut_ptr() as *mut FFI,
                    lda.try_into().unwrap(),
                );
            }

            HER::<F>::default()
                .x(x_raw.slice(x_slc))
                .a(a_raw.slice_mut(a_slc))
                .alpha(alpha)
                .uplo(uplo)
                .run()
                .unwrap();

            check_same(&a_raw.slice(a_slc), &a_naive.view(), 4.0 * F::EPSILON);
            a_raw.slice_mut(a_slc).fill(F::from(0.0));
            a_origin.slice_mut(a_slc).fill(F::from(0.0));
            check_same(&a_raw.view(), &a_origin.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_c32_owned() {
        type F = c32;
        type FRE = f32;
        type FFI = <F as TestFloat>::FFIFloat;

        for (uplo, incx) in iproduct!(['U', 'L'], [1, 2]) {
            let n = 8;

            let x_slc = slice_1d(n, incx);

            let x_raw = random_array::<F>(100);
            let alpha = FRE::rand();

            let mut a_naive = Array2::<F>::zeros((n, n));
            let x_naive = x_raw.slice(x_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            unsafe {
                cblas_cher(
                    to_cblas_layout('R'),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    a_naive.as_mut_ptr() as *mut FFI,
                    lda.try_into().unwrap(),
                );
            }

            let a_out =
                HER::<F>::default().x(x_raw.slice(x_slc)).alpha(alpha).uplo(uplo).run().unwrap().into_owned();

            check_same(&a_out.view(), &a_naive.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_f32_owned() {
        type F = f32;
        type FRE = f32;
        type FFI = <F as TestFloat>::FFIFloat;

        for (uplo, incx) in iproduct!(['U', 'L'], [1, 2]) {
            let n = 8;

            let x_slc = slice_1d(n, incx);

            let x_raw = random_array::<F>(100);
            let alpha = FRE::rand();

            let mut a_naive = Array2::<F>::zeros((n, n));
            let x_naive = x_raw.slice(x_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            unsafe {
                cblas_ssyr(
                    to_cblas_layout('R'),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    alpha,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    a_naive.as_mut_ptr() as *mut FFI,
                    lda.try_into().unwrap(),
                );
            }

            let a_out =
                HER::<F>::default().x(x_raw.slice(x_slc)).alpha(alpha).uplo(uplo).run().unwrap().into_owned();

            check_same(&a_out.view(), &a_naive.view(), 4.0 * F::EPSILON);
        }
    }
}
