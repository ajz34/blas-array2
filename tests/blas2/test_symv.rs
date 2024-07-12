use crate::util::*;
use approx::*;
use blas_array2::blas2::symv::{HEMV, SYMV};
use blas_array2::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid {
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
                let a_raw = random_matrix::<$F>(100, 100, $a_layout.into());
                let x_raw = random_array::<$F>(100);
                let mut y_raw = random_array::<$F>(100);

                let a_slc = slice($($a_slc),+);
                let x_slc = slice_1d($($x_slc),+);
                let y_slc = slice_1d($($y_slc),+);

                let a_naive = $symm(&a_raw.slice(a_slc), $uplo.into());
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
                    .uplo($uplo)
                    .alpha(alpha)
                    .beta(beta)
                    .run()
                    .unwrap();
                if let ArrayOut1::ViewMut(_) = y_out {
                    let err = (&y_naive - &y_raw).mapv(|x| x.abs()).sum();
                    let acc = y_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }

                // owned
                let y_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .x(x_raw.slice(x_slc))
                    .uplo($uplo)
                    .alpha(alpha)
                    .beta(beta)
                    .run()
                    .unwrap();
                if let ArrayOut1::Owned(y_out) = y_out {
                    let err = (&y_bare - &y_out).mapv(|x| x.abs()).sum();
                    let acc = y_bare.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    test_macro!(test_000: inline, f32, (8, 8, 1, 1), (8, 1), (8, 1), 'R', 'U', SYMV, symmetrize);
    test_macro!(test_001: inline, f32, (8, 8, 1, 1), (8, 1), (8, 3), 'C', 'L', SYMV, symmetrize);
    test_macro!(test_002: inline, f32, (8, 8, 3, 3), (8, 3), (8, 1), 'R', 'U', SYMV, symmetrize);
    test_macro!(test_003: inline, f32, (8, 8, 3, 3), (8, 3), (8, 3), 'C', 'L', SYMV, symmetrize);
    test_macro!(test_004: inline, f64, (8, 8, 1, 1), (8, 3), (8, 1), 'R', 'L', SYMV, symmetrize);
    test_macro!(test_005: inline, f64, (8, 8, 1, 1), (8, 3), (8, 3), 'C', 'U', SYMV, symmetrize);
    test_macro!(test_006: inline, f64, (8, 8, 3, 3), (8, 1), (8, 1), 'C', 'U', SYMV, symmetrize);
    test_macro!(test_007: inline, f64, (8, 8, 3, 3), (8, 1), (8, 3), 'R', 'L', SYMV, symmetrize);
    test_macro!(test_008: inline, c32, (8, 8, 1, 3), (8, 1), (8, 1), 'R', 'L', HEMV, hermitianize);
    test_macro!(test_009: inline, c32, (8, 8, 1, 3), (8, 1), (8, 3), 'C', 'U', HEMV, hermitianize);
    test_macro!(test_010: inline, c32, (8, 8, 3, 1), (8, 3), (8, 1), 'C', 'L', HEMV, hermitianize);
    test_macro!(test_011: inline, c32, (8, 8, 3, 1), (8, 3), (8, 3), 'R', 'U', HEMV, hermitianize);
    test_macro!(test_012: inline, c64, (8, 8, 1, 3), (8, 3), (8, 1), 'C', 'U', HEMV, hermitianize);
    test_macro!(test_013: inline, c64, (8, 8, 1, 3), (8, 3), (8, 3), 'R', 'L', HEMV, hermitianize);
    test_macro!(test_014: inline, c64, (8, 8, 3, 1), (8, 1), (8, 1), 'C', 'L', HEMV, hermitianize);
    test_macro!(test_015: inline, c64, (8, 8, 3, 1), (8, 1), (8, 3), 'R', 'U', HEMV, hermitianize);
}
