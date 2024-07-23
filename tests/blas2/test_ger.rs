use crate::util::*;
use blas_array2::blas2::ger::GER;
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

        for (layout, as0, as1, incx, incy) in iproduct!(['C', 'R'], [1, 2], [1, 2], [1, 2], [1, 2]) {
            let m = 8;
            let n = 9;

            let a_slc = slice(m, n, as0, as1);
            let x_slc = slice_1d(m, incx);
            let y_slc = slice_1d(n, incy);

            let mut a_raw = random_matrix::<F>(100, 100, layout.into());
            let x_raw = random_array::<F>(100);
            let y_raw = random_array::<F>(100);
            let alpha = F::rand();

            let mut a_origin = a_raw.clone();
            let mut a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), 'C');
            let x_naive = x_raw.slice(x_slc).into_owned();
            let y_naive = y_raw.slice(y_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_cgeru(
                    to_cblas_layout('C'),
                    m.try_into().unwrap(),
                    n.try_into().unwrap(),
                    [alpha].as_ptr() as *const FFI,
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    y_naive.as_ptr() as *const FFI,
                    incy.try_into().unwrap(),
                    a_naive.as_mut_ptr() as *mut FFI,
                    lda.try_into().unwrap(),
                );
            }

            GER::<F>::default()
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice(y_slc))
                .a(a_raw.slice_mut(a_slc))
                .alpha(alpha)
                .run()
                .unwrap();

            check_same(&a_raw.slice(a_slc), &a_naive.view(), 4.0 * F::EPSILON);
            a_raw.slice_mut(a_slc).fill(F::from(0.0));
            a_origin.slice_mut(a_slc).fill(F::from(0.0));
            check_same(&a_raw.view(), &a_origin.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_f32_owned() {
        type F = f32;
        type FFI = <F as TestFloat>::FFIFloat;

        for (incx, incy) in iproduct!([1, 2], [1, 2]) {
            let m = 8;
            let n = 9;

            let x_slc = slice_1d(m, incx);
            let y_slc = slice_1d(n, incy);

            let x_raw = random_array::<F>(100);
            let y_raw = random_array::<F>(100);
            let alpha = F::rand();

            let mut a_naive = Array2::<F>::zeros((m, n));
            let x_naive = x_raw.slice(x_slc).into_owned();
            let y_naive = y_raw.slice(y_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_sger(
                    to_cblas_layout('R'),
                    m.try_into().unwrap(),
                    n.try_into().unwrap(),
                    alpha.clone(),
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    y_naive.as_ptr() as *const FFI,
                    incy.try_into().unwrap(),
                    a_naive.as_mut_ptr() as *mut FFI,
                    lda.try_into().unwrap(),
                );
            }

            let a_out = GER::<F>::default()
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice(y_slc))
                .alpha(alpha)
                .run()
                .unwrap()
                .into_owned();

            check_same(&a_out.view(), &a_naive.view(), 4.0 * F::EPSILON);
        }
    }
}
