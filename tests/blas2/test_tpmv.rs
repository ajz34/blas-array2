use crate::util::*;
use approx::*;
use blas_array2::blas2::tpmv::TPMV;
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
            ($($a_slc: expr),+), ($($x_slc: expr),+),
            $a_layout: expr,
            $uplo: expr, $trans: expr, $diag: expr
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let uplo = $uplo;
                let trans = $trans;
                let diag = $diag;
                let a_raw = random_array(1000);
                let mut x_raw = random_array(100);

                let a_slc = slice_1d($($a_slc),+);
                let x_slc = slice_1d($($x_slc),+);
                let n = x_raw.slice(x_slc).len_of(Axis(0));

                let mut a_naive = Array2::<$F>::zeros((n, n));
                unpack_tril(&a_raw.slice(a_slc), &mut a_naive.view_mut(), 'C', uplo);
                if diag == 'U' {
                    for i in 0..n {
                        a_naive[[i, i]] = <$F>::from(1.0);
                    }
                }
                let a_naive = transpose(&a_naive.view(), trans.try_into().unwrap());
                let x_assign = gemv(&a_naive.view(), &x_raw.slice(x_slc));
                let mut x_naive = x_raw.clone();
                x_naive.slice_mut(x_slc).assign(&x_assign);

                // mut_view
                let x_out = TPMV::default()
                    .ap(a_raw.slice(a_slc))
                    .x(x_raw.slice_mut(x_slc))
                    .uplo(uplo)
                    .trans(trans)
                    .diag(diag)
                    .layout('C')
                    .run()
                    .unwrap();
                if let ArrayOut1::ViewMut(_) = x_out {
                    let err = (&x_naive - &x_raw).mapv(|x| x.abs()).sum();
                    let acc = x_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    test_macro!(test_000: inline, f32, (36, 1), (8, 1), 'C', 'U', 'N', 'N');
    test_macro!(test_001: inline, f32, (36, 1), (8, 1), 'R', 'L', 'N', 'N');
    test_macro!(test_002: inline, f32, (36, 1), (8, 3), 'C', 'U', 'T', 'U');
    test_macro!(test_003: inline, f32, (36, 3), (8, 1), 'C', 'L', 'T', 'U');
    test_macro!(test_004: inline, f32, (36, 3), (8, 3), 'R', 'U', 'C', 'U');
    test_macro!(test_005: inline, f32, (36, 3), (8, 3), 'C', 'L', 'C', 'N');
    test_macro!(test_006: inline, f64, (36, 1), (8, 1), 'C', 'L', 'C', 'U');
    test_macro!(test_007: inline, f64, (36, 3), (8, 1), 'R', 'L', 'T', 'U');
    test_macro!(test_008: inline, f64, (36, 3), (8, 3), 'R', 'U', 'T', 'N');
    test_macro!(test_009: inline, f64, (36, 1), (8, 1), 'R', 'U', 'C', 'N');
    test_macro!(test_010: inline, f64, (36, 1), (8, 3), 'C', 'U', 'N', 'U');
    test_macro!(test_011: inline, f64, (36, 3), (8, 3), 'C', 'L', 'N', 'N');
    test_macro!(test_012: inline, c32, (36, 1), (8, 3), 'R', 'L', 'C', 'U');
    test_macro!(test_013: inline, c32, (36, 3), (8, 1), 'C', 'L', 'C', 'N');
    test_macro!(test_014: inline, c32, (36, 3), (8, 3), 'C', 'U', 'N', 'N');
    test_macro!(test_015: inline, c32, (36, 1), (8, 1), 'C', 'U', 'T', 'U');
    test_macro!(test_016: inline, c32, (36, 1), (8, 3), 'R', 'L', 'N', 'U');
    test_macro!(test_017: inline, c32, (36, 3), (8, 1), 'R', 'U', 'T', 'N');
    test_macro!(test_018: inline, c64, (36, 1), (8, 3), 'C', 'L', 'T', 'N');
    test_macro!(test_019: inline, c64, (36, 3), (8, 1), 'C', 'U', 'N', 'U');
    test_macro!(test_020: inline, c64, (36, 3), (8, 3), 'R', 'U', 'C', 'U');
    test_macro!(test_021: inline, c64, (36, 1), (8, 1), 'C', 'U', 'C', 'N');
    test_macro!(test_022: inline, c64, (36, 1), (8, 3), 'R', 'L', 'T', 'N');
    test_macro!(test_023: inline, c64, (36, 3), (8, 1), 'R', 'L', 'N', 'U');
}

#[cfg(test)]
mod valid_row_major {
    use super::*;

    #[test]
    fn test_cblas_row_major() {
        type F = c32;
        for (cblas_layout, uplo, trans, diag) in
            iproduct!(['R', 'C'], ['U', 'L'], ['N', 'T', 'C'], ['U', 'N'])
        {
            let n = 8;
            let np = n * (n + 1) / 2;

            // slice definition
            let a_slc = slice_1d(np, 3);
            let x_slc = slice_1d(n, 3);

            // type definition
            type FFI = <F as TestFloat>::FFIFloat;

            // data assignment
            let a_raw = random_array(1000);
            let mut x_raw = random_array(1000);
            let mut x_origin = x_raw.clone();

            // cblas computation
            let a_naive = a_raw.slice(a_slc).into_owned();
            let mut x_naive = x_raw.slice_mut(x_slc).into_owned();
            let incx = 1;
            unsafe {
                cblas_ctpmv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_uplo(uplo),
                    to_cblas_trans(trans),
                    to_cblas_diag(diag),
                    n.try_into().unwrap(),
                    a_naive.as_ptr() as *const FFI,
                    x_naive.as_mut_ptr() as *mut FFI,
                    incx.try_into().unwrap(),
                );
            }

            TPMV::<F>::default()
                .ap(a_raw.slice(a_slc))
                .x(x_raw.slice_mut(x_slc))
                .uplo(uplo)
                .trans(trans)
                .diag(diag)
                .layout(cblas_layout)
                .run()
                .unwrap();

            check_same(&x_raw.slice(x_slc), &x_naive.view(), 4.0 * F::EPSILON);
            x_raw.slice_mut(x_slc).fill(F::from(0.0));
            x_origin.slice_mut(x_slc).fill(F::from(0.0));
            check_same(&x_raw.view(), &x_origin.view(), 4.0 * F::EPSILON);
        }
    }
}
