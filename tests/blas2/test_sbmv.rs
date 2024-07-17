use crate::util::*;
use approx::*;
use blas_array2::blas2::sbmv::{HBMV, SBMV};
use blas_array2::prelude::*;
use cblas_sys::*;
use itertools::*;
use ndarray::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid_col_major {
    use super::*;

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F:ty,
            ($($a_slc: expr),+), ($($x_slc: expr),+), ($($y_slc: expr),+),
            $a_layout: expr,
            $uplo: expr,
            $blas: ident, $symm: ident
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$F>::rand();
                let n = 8;
                let k = 3;
                let uplo = $uplo;
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let x_raw = random_array(100);
                let mut y_raw = random_array(100);

                let a_slc = slice($($a_slc),+);
                let x_slc = slice_1d($($x_slc),+);
                let y_slc = slice_1d($($y_slc),+);

                let mut a_naive = Array2::<$F>::zeros((n, n));
                if uplo == 'U' {
                    for j in 0..n {
                        let m = k as isize - j as isize;
                        for i in (if j > k { j - k } else { 0 })..(j + 1) {
                            let mi = (m + i as isize) as usize;
                            let i = i as usize;
                            a_naive[[i, j]] = a_raw.slice(a_slc)[[mi, j]];
                        }
                    }
                } else {
                    for j in 0..n {
                        let m = - (j as isize);
                        for i in j..core::cmp::min(n, j + k + 1) {
                            let mi = (m + i as isize) as usize;
                            let i = i as usize;
                            a_naive[[i, j]] = a_raw.slice(a_slc)[[mi, j]];
                        }
                    }
                }
                let a_naive = $symm(&a_naive.view(), uplo.into());
                let x_naive = x_raw.slice(x_slc).into_owned();
                let mut y_naive = y_raw.clone();
                let y_bare = alpha * gemv(&a_naive.view(), &x_naive.view());
                let y_assign = &y_bare + beta * &y_naive.slice(&y_slc);
                y_naive.slice_mut(y_slc).assign(&y_assign);

                // mut_view
                let y_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .x(x_raw.slice(x_slc))
                    .y(y_raw.slice_mut(y_slc))
                    .uplo(uplo)
                    .alpha(alpha)
                    .beta(beta)
                    .layout('C')
                    .run()
                    .unwrap();
                if let ArrayOut1::ViewMut(_) = y_out {
                    let err = (&y_naive - &y_raw).mapv(|x| x.abs()).sum();
                    let acc = y_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }

                // owned
                let y_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .x(x_raw.slice(x_slc))
                    .uplo(uplo)
                    .alpha(alpha)
                    .beta(beta)
                    .layout('C')
                    .run()
                    .unwrap();
                if let ArrayOut1::Owned(y_out) = y_out {
                    let err = (&y_bare - &y_out).mapv(|x| x.abs()).sum();
                    let acc = y_bare.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    test_macro!(test_000: inline, f32, (4, 8, 1, 1), (8, 1), (8, 1), 'R', 'U', SBMV, symmetrize);
    test_macro!(test_001: inline, f32, (4, 8, 1, 1), (8, 1), (8, 3), 'C', 'L', SBMV, symmetrize);
    test_macro!(test_002: inline, f32, (4, 8, 3, 3), (8, 3), (8, 1), 'R', 'U', SBMV, symmetrize);
    test_macro!(test_003: inline, f32, (4, 8, 3, 3), (8, 3), (8, 3), 'C', 'L', SBMV, symmetrize);
    test_macro!(test_004: inline, f64, (4, 8, 1, 1), (8, 3), (8, 1), 'R', 'L', SBMV, symmetrize);
    test_macro!(test_005: inline, f64, (4, 8, 1, 1), (8, 3), (8, 3), 'C', 'U', SBMV, symmetrize);
    test_macro!(test_006: inline, f64, (4, 8, 3, 3), (8, 1), (8, 1), 'C', 'U', SBMV, symmetrize);
    test_macro!(test_007: inline, f64, (4, 8, 3, 3), (8, 1), (8, 3), 'R', 'L', SBMV, symmetrize);
    test_macro!(test_008: inline, c32, (4, 8, 1, 3), (8, 1), (8, 1), 'R', 'L', HBMV, hermitianize);
    test_macro!(test_009: inline, c32, (4, 8, 1, 3), (8, 1), (8, 3), 'C', 'U', HBMV, hermitianize);
    test_macro!(test_010: inline, c32, (4, 8, 3, 1), (8, 3), (8, 1), 'C', 'L', HBMV, hermitianize);
    test_macro!(test_011: inline, c32, (4, 8, 3, 1), (8, 3), (8, 3), 'R', 'U', HBMV, hermitianize);
    test_macro!(test_012: inline, c64, (4, 8, 1, 3), (8, 3), (8, 1), 'C', 'U', HBMV, hermitianize);
    test_macro!(test_013: inline, c64, (4, 8, 1, 3), (8, 3), (8, 3), 'R', 'L', HBMV, hermitianize);
    test_macro!(test_014: inline, c64, (4, 8, 3, 1), (8, 1), (8, 1), 'C', 'L', HBMV, hermitianize);
    test_macro!(test_015: inline, c64, (4, 8, 3, 1), (8, 1), (8, 3), 'R', 'U', HBMV, hermitianize);
}

#[cfg(test)]
mod valid_row_major {
    use super::*;

    #[test]
    fn test_cblas_row_major_c32() {
        let cblas_layout = 'R';
        type F = c32;
        for (a_layout, uplo) in iproduct!(['R', 'C'], ['U', 'L']) {
            let n = 8;
            let k = 3;

            // slice definition
            let a_slc = slice(n, k + 1, 3, 3);
            let x_slc = slice_1d(n, 3);
            let y_slc = slice_1d(n, 3);

            // type definition
            type FFI = <F as BLASFloat>::FFIFloat;

            // data assignment
            let alpha = F::rand();
            let beta = F::rand();
            let a_raw = random_matrix::<F>(100, 100, a_layout.into());
            let x_raw = random_array::<F>(1000);
            let mut y_raw = random_array::<F>(1000);
            let mut y_origin = y_raw.clone();

            // cblas computation - mut
            let a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), cblas_layout);
            let x_naive = x_raw.slice(x_slc).into_owned();
            let mut y_naive = y_raw.slice_mut(y_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_chbmv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    k.try_into().unwrap(),
                    [alpha].as_ptr() as *const FFI,
                    a_naive.as_ptr() as *const FFI,
                    lda.try_into().unwrap(),
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    [beta].as_ptr() as *const FFI,
                    y_naive.as_mut_ptr() as *mut FFI,
                    incy.try_into().unwrap(),
                );
            }

            let y_out = HBMV::<F>::default()
                .a(a_raw.slice(a_slc))
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice_mut(y_slc))
                .alpha(alpha)
                .beta(beta)
                .uplo(uplo)
                .layout(cblas_layout)
                .run()
                .unwrap()
                .into_owned();

            check_same(&y_out.view(), &y_naive.view(), 4.0 * F::EPSILON);
            check_same(&y_raw.slice(y_slc), &y_naive.view(), 4.0 * F::EPSILON);
            y_raw.slice_mut(y_slc).fill(F::from(0.0));
            y_origin.slice_mut(y_slc).fill(F::from(0.0));
            check_same(&y_raw.view(), &y_origin.view(), 4.0 * F::EPSILON);
        }
    }

    #[test]
    fn test_cblas_row_major_f32() {
        let cblas_layout = 'R';
        type F = f32;
        for (a_layout, uplo) in iproduct!(['R', 'C'], ['U', 'L']) {
            let n = 8;
            let k = 3;

            // slice definition
            let a_slc = slice(n, k + 1, 3, 3);
            let x_slc = slice_1d(n, 3);
            let y_slc = slice_1d(n, 3);

            // type definition
            type FFI = <F as BLASFloat>::FFIFloat;

            // data assignment
            let alpha = F::rand();
            let beta = F::rand();
            let a_raw = random_matrix::<F>(100, 100, a_layout.into());
            let x_raw = random_array::<F>(1000);
            let mut y_raw = random_array::<F>(1000);
            let mut y_origin = y_raw.clone();

            // cblas computation - mut
            let a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), cblas_layout);
            let x_naive = x_raw.slice(x_slc).into_owned();
            let mut y_naive = y_raw.slice_mut(y_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_ssbmv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_uplo(uplo),
                    n.try_into().unwrap(),
                    k.try_into().unwrap(),
                    alpha,
                    a_naive.as_ptr() as *const FFI,
                    lda.try_into().unwrap(),
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    beta,
                    y_naive.as_mut_ptr() as *mut FFI,
                    incy.try_into().unwrap(),
                );
            }

            let y_out = SBMV::<F>::default()
                .a(a_raw.slice(a_slc))
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice_mut(y_slc))
                .alpha(alpha)
                .beta(beta)
                .uplo(uplo)
                .layout(cblas_layout)
                .run()
                .unwrap()
                .into_owned();

            check_same(&y_out.view(), &y_naive.view(), 4.0 * F::EPSILON);
            check_same(&y_raw.slice(y_slc), &y_naive.view(), 4.0 * F::EPSILON);
            y_raw.slice_mut(y_slc).fill(F::from(0.0));
            y_origin.slice_mut(y_slc).fill(F::from(0.0));
            check_same(&y_raw.view(), &y_origin.view(), 4.0 * F::EPSILON);
        }
    }
}
