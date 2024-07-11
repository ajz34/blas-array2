use crate::util::*;
use approx::*;
use blas_array2::blas3::trsm::TRSM;
use blas_array2::util::*;
use ndarray::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <f32 as BLASFloat>::RealFloat;
        let alpha = f32::rand();
        let a_raw = random_matrix(100, 100, 'R'.into());
        let mut b_raw = random_matrix(100, 100, 'R'.into());
        let a_slc = slice(8, 8, 1, 1);
        let b_slc = slice(8, 9, 1, 1);
        let side = 'L';
        let uplo = 'U';
        let transa = 'N';
        let diag = 'N';

        let mut a_naive = a_raw.slice(a_slc).into_owned();
        let b_naive = b_raw.slice(b_slc).into_owned();

        if BLASDiag::from(diag) == BLASUnit {
            for i in 0..a_naive.len_of(Axis(0)) {
                a_naive[[i, i]] = 1.0;
            }
        }
        match uplo.into() {
            BLASLower => {
                for i in 0..a_naive.len_of(Axis(0)) {
                    for j in i + 1..a_naive.len_of(Axis(1)) {
                        a_naive[[i, j]] = 0.0;
                    }
                }
            },
            BLASUpper => {
                for i in 0..a_naive.len_of(Axis(0)) {
                    for j in 0..i {
                        a_naive[[i, j]] = 0.0;
                    }
                }
            },
            _ => panic!(),
        }
        let a_naive = transpose(&a_naive.view(), transa.into());

        let b_out = TRSM::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice_mut(b_slc))
            .alpha(alpha)
            .side(side)
            .uplo(uplo)
            .transa(transa)
            .diag(diag)
            .run()
            .unwrap();
        let b_out = b_out.into_owned();
        let b_out = match side.into() {
            BLASLeft => gemm(&a_naive.view(), &b_out.view()),
            BLASRight => gemm(&b_out.view(), &a_naive.view()),
            _ => panic!(),
        };

        let err = (alpha * &b_naive - &b_out).mapv(|x| x.abs()).sum();
        let acc = b_naive.view().mapv(|x| x.abs()).sum() as RT;
        let err_div = err / acc;
        // This involves two matrix multiplications, so the error is expected to be larger
        assert_relative_eq!(err_div, 0.0, epsilon = 2.0 * f32::sqrt(RT::EPSILON));
    }

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F: ty,
            ($($a_slc: expr),+), ($($b_slc: expr),+),
            $a_layout: expr, $b_layout: expr,
            $side: expr, $uplo: expr, $transa: expr, $diag: expr
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let a_raw = random_matrix::<$F>(100, 100, $a_layout.into());
                let mut b_raw = random_matrix::<$F>(100, 100, $b_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);

                let mut a_naive = a_raw.slice(a_slc).into_owned();
                let b_naive = b_raw.slice(b_slc).into_owned();

                if BLASDiag::from($diag) == BLASUnit {
                    for i in 0..a_naive.len_of(Axis(0)) {
                        a_naive[[i, i]] = <$F>::from(1.0);
                    }
                }
                match $uplo.into() {
                    BLASLower => {
                        for i in 0..a_naive.len_of(Axis(0)) {
                            for j in i+1..a_naive.len_of(Axis(1)) {
                                a_naive[[i, j]] = <$F>::from(0.0);
                            }
                        }
                    },
                    BLASUpper => {
                        for i in 0..a_naive.len_of(Axis(0)) {
                            for j in 0..i {
                                a_naive[[i, j]] = <$F>::from(0.0);
                            }
                        }
                    },
                    _ => panic!(),
                }
                let a_naive = transpose(&a_naive.view(), $transa.into());

                let b_out = TRSM::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice_mut(b_slc))
                    .alpha(alpha)
                    .side($side)
                    .uplo($uplo)
                    .transa($transa)
                    .diag($diag)
                    .run()
                    .unwrap();
                let b_out = b_out.into_owned();
                let b_out = match $side.into() {
                    BLASLeft => gemm(&a_naive.view(), &b_out.view()),
                    BLASRight => gemm(&b_out.view(), &a_naive.view()),
                    _ => panic!(),
                };

                let err = (alpha * &b_naive - &b_out).mapv(|x| x.abs()).sum();
                let acc = b_naive.view().mapv(|x| x.abs()).sum() as RT;
                let err_div = err / acc;
                // This involves two matrix multiplications, so the error is expected to be larger
                assert_relative_eq!(err_div, 0.0, epsilon = 2.0 * RT::sqrt(RT::EPSILON));
            }
        };
    }

    // successful tests
    test_macro!(test_000: inline, f32, (8, 8, 1, 1), (8, 9, 1, 1), 'R', 'R', 'L', 'L', 'N', 'N');
    test_macro!(test_001: inline, f32, (9, 9, 1, 1), (8, 9, 1, 1), 'C', 'C', 'R', 'U', 'T', 'U');
    test_macro!(test_002: inline, f32, (9, 9, 1, 1), (8, 9, 3, 3), 'R', 'R', 'R', 'U', 'C', 'U');
    test_macro!(test_003: inline, f32, (8, 8, 3, 3), (8, 9, 1, 3), 'C', 'C', 'L', 'L', 'N', 'N');
    test_macro!(test_004: inline, f32, (8, 8, 3, 3), (8, 9, 3, 1), 'R', 'C', 'L', 'U', 'C', 'U');
    test_macro!(test_005: inline, f32, (9, 9, 3, 3), (8, 9, 3, 3), 'C', 'R', 'R', 'L', 'T', 'N');
    test_macro!(test_006: inline, f64, (8, 8, 1, 1), (8, 9, 1, 3), 'C', 'C', 'L', 'L', 'C', 'U');
    test_macro!(test_007: inline, f64, (9, 9, 1, 3), (8, 9, 1, 1), 'C', 'R', 'R', 'L', 'C', 'N');
    test_macro!(test_008: inline, f64, (9, 9, 1, 3), (8, 9, 3, 1), 'R', 'R', 'R', 'U', 'N', 'N');
    test_macro!(test_009: inline, f64, (8, 8, 3, 1), (8, 9, 1, 1), 'R', 'C', 'L', 'U', 'T', 'N');
    test_macro!(test_010: inline, f64, (8, 8, 3, 1), (8, 9, 3, 3), 'R', 'R', 'L', 'L', 'T', 'U');
    test_macro!(test_011: inline, f64, (9, 9, 3, 3), (8, 9, 3, 3), 'C', 'C', 'R', 'U', 'N', 'U');
    test_macro!(test_012: inline, c32, (9, 9, 1, 1), (8, 9, 3, 1), 'C', 'C', 'R', 'L', 'N', 'U');
    test_macro!(test_013: inline, c32, (8, 8, 1, 3), (8, 9, 1, 3), 'C', 'R', 'L', 'U', 'T', 'U');
    test_macro!(test_014: inline, c32, (8, 8, 1, 3), (8, 9, 3, 3), 'R', 'C', 'L', 'L', 'C', 'N');
    test_macro!(test_015: inline, c32, (9, 9, 3, 1), (8, 9, 1, 3), 'R', 'C', 'R', 'U', 'N', 'N');
    test_macro!(test_016: inline, c32, (8, 8, 3, 1), (8, 9, 3, 1), 'C', 'R', 'L', 'U', 'C', 'N');
    test_macro!(test_017: inline, c32, (9, 9, 3, 3), (8, 9, 1, 1), 'R', 'R', 'R', 'L', 'T', 'U');
    test_macro!(test_018: inline, c64, (9, 9, 1, 1), (8, 9, 3, 3), 'R', 'C', 'R', 'L', 'T', 'N');
    test_macro!(test_019: inline, c64, (8, 8, 1, 3), (8, 9, 1, 3), 'R', 'R', 'L', 'U', 'N', 'U');
    test_macro!(test_020: inline, c64, (8, 8, 1, 3), (8, 9, 3, 1), 'C', 'C', 'L', 'U', 'T', 'N');
    test_macro!(test_021: inline, c64, (9, 9, 3, 1), (8, 9, 1, 3), 'C', 'R', 'R', 'U', 'C', 'N');
    test_macro!(test_022: inline, c64, (8, 8, 3, 1), (8, 9, 3, 1), 'C', 'R', 'L', 'L', 'N', 'U');
    test_macro!(test_023: inline, c64, (9, 9, 3, 3), (8, 9, 1, 1), 'R', 'C', 'R', 'L', 'C', 'U');
}
