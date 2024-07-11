use crate::util::*;
use approx::*;
use blas_array2::blas2::tbsv::TBSV;
use blas_array2::prelude::*;
use ndarray::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid {
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
                let n = 8;
                let k = 3;
                let uplo = $uplo;
                let trans = $trans;
                let diag = $diag;
                let a_raw = random_matrix(100, 100, 'R'.into());
                let mut x_raw = random_array(100);

                let a_slc = slice($($a_slc),+);
                let x_slc = slice_1d($($x_slc),+);

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
                        for i in j..std::cmp::min(n, j + k + 1) {
                            let mi = (m + i as isize) as usize;
                            let i = i as usize;
                            a_naive[[i, j]] = a_raw.slice(a_slc)[[mi, j]];
                        }
                    }
                }
                if diag == 'U' {
                    for i in 0..n {
                        a_naive[[i, i]] = <$F>::from(1.0);
                    }
                }
                let a_naive = transpose(&a_naive.view(), trans.into());
                let mut x_naive = x_raw.clone();
                let x_origin = x_raw.clone();

                // mut_view
                let x_out = TBSV::default()
                    .a(a_raw.slice(a_slc))
                    .x(x_raw.slice_mut(x_slc))
                    .uplo(uplo)
                    .trans(trans)
                    .diag(diag)
                    .run()
                    .unwrap();
                println!("{:?}", x_out.view());
                if let ArrayOut1::ViewMut(_) = x_out {
                    let x_assign = gemv(&a_naive.view(), &x_raw.slice(x_slc));
                    x_naive.slice_mut(x_slc).assign(&x_assign);
                    let err = (&x_naive - &x_origin).mapv(|x| x.abs()).sum();
                    let acc = x_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::sqrt(RT::EPSILON));
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    test_macro!(test_000: inline, f32, (4, 8, 1, 1), (8, 1), 'R', 'U', 'N', 'N');
    test_macro!(test_001: inline, f32, (4, 8, 1, 1), (8, 1), 'R', 'L', 'N', 'N');
    test_macro!(test_002: inline, f32, (4, 8, 1, 1), (8, 3), 'C', 'U', 'T', 'U');
    test_macro!(test_003: inline, f32, (4, 8, 3, 3), (8, 1), 'C', 'L', 'T', 'U');
    test_macro!(test_004: inline, f32, (4, 8, 3, 3), (8, 3), 'R', 'U', 'C', 'U');
    test_macro!(test_005: inline, f32, (4, 8, 3, 3), (8, 3), 'C', 'L', 'C', 'N');
    test_macro!(test_006: inline, f64, (4, 8, 1, 1), (8, 1), 'C', 'L', 'C', 'U');
    test_macro!(test_007: inline, f64, (4, 8, 1, 3), (8, 1), 'R', 'L', 'T', 'U');
    test_macro!(test_008: inline, f64, (4, 8, 1, 3), (8, 3), 'R', 'U', 'T', 'N');
    test_macro!(test_009: inline, f64, (4, 8, 3, 1), (8, 1), 'R', 'U', 'C', 'N');
    test_macro!(test_010: inline, f64, (4, 8, 3, 1), (8, 3), 'C', 'U', 'N', 'U');
    test_macro!(test_011: inline, f64, (4, 8, 3, 3), (8, 3), 'C', 'L', 'N', 'N');
    test_macro!(test_012: inline, c32, (4, 8, 1, 1), (8, 3), 'R', 'L', 'C', 'U');
    test_macro!(test_013: inline, c32, (4, 8, 1, 3), (8, 1), 'C', 'L', 'C', 'N');
    test_macro!(test_014: inline, c32, (4, 8, 1, 3), (8, 3), 'C', 'U', 'N', 'N');
    test_macro!(test_015: inline, c32, (4, 8, 3, 1), (8, 1), 'C', 'U', 'T', 'U');
    test_macro!(test_016: inline, c32, (4, 8, 3, 1), (8, 3), 'R', 'L', 'N', 'U');
    test_macro!(test_017: inline, c32, (4, 8, 3, 3), (8, 1), 'R', 'U', 'T', 'N');
    test_macro!(test_018: inline, c64, (4, 8, 1, 1), (8, 3), 'C', 'L', 'T', 'N');
    test_macro!(test_019: inline, c64, (4, 8, 1, 3), (8, 1), 'C', 'U', 'N', 'U');
    test_macro!(test_020: inline, c64, (4, 8, 1, 3), (8, 3), 'R', 'U', 'C', 'U');
    test_macro!(test_021: inline, c64, (4, 8, 3, 1), (8, 1), 'C', 'U', 'C', 'N');
    test_macro!(test_022: inline, c64, (4, 8, 3, 1), (8, 3), 'R', 'L', 'T', 'N');
    test_macro!(test_023: inline, c64, (4, 8, 3, 3), (8, 1), 'R', 'L', 'N', 'U');
}
