use crate::util::*;
use approx::*;
use blas_array2::blas3::syr2k::{HER2K, SYR2K};
use blas_array2::util::*;
use ndarray::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid_owned {
    use super::*;

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F: ty,
            ($($a_slc: expr),+), ($($b_slc: expr),+),
            $a_layout: expr, $b_layout: expr,
            $uplo: expr,
            $trans: expr,
            $blas: ident, $blas_trans: expr, $blas_ty: ty
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$blas_ty>::rand();
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let b_raw = random_matrix(100, 100, $b_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);
                let uplo = $uplo;
                let trans = $trans;

                let c_out = $blas::<$F>::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .uplo(uplo)
                    .trans(trans)
                    .run()
                    .unwrap();

                let a_naive = a_raw.slice(a_slc).to_owned();
                let b_naive = b_raw.slice(b_slc).to_owned();
                let c_assign = match trans.into() {
                    BLASNoTrans => (
                        <$F>::from(alpha) * gemm(&a_naive.view(), &transpose(&b_naive.view(), $blas_trans.into()).view())
                    ),
                    BLASTrans | BLASConjTrans => (
                        <$F>::from(alpha) * gemm(&transpose(&a_naive.view(), $blas_trans.into()).view(), &b_naive.view())
                    ),
                    _ => panic!("Invalid"),
                };
                let c_assign = &c_assign + &transpose(&c_assign.view(), $blas_trans.into());
                let mut c_naive = Array2::zeros(c_assign.dim());
                tril_assign(&mut c_naive.view_mut(), &c_assign.view(), uplo);

                if let ArrayOut2::Owned(c_out) = c_out {
                    let err = (&c_naive - &c_out).mapv(|x| x.abs()).sum();
                    let acc = c_naive.mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    // successful tests
    test_macro!(test_000: inline, f32, (7, 5, 1, 1), (7, 5, 1, 1), 'R', 'R', 'L', 'T', SYR2K, 'T', f32);
    test_macro!(test_001: inline, f32, (7, 5, 1, 1), (7, 5, 1, 1), 'R', 'R', 'U', 'N', SYR2K, 'T', f32);
    test_macro!(test_002: inline, f32, (7, 5, 3, 3), (7, 5, 3, 3), 'C', 'C', 'L', 'N', SYR2K, 'T', f32);
    test_macro!(test_003: inline, f32, (7, 5, 3, 3), (7, 5, 3, 3), 'C', 'C', 'U', 'T', SYR2K, 'T', f32);
    test_macro!(test_004: inline, f64, (7, 5, 1, 1), (7, 5, 1, 3), 'C', 'C', 'L', 'N', SYR2K, 'T', f64);
    test_macro!(test_005: inline, f64, (7, 5, 1, 3), (7, 5, 3, 1), 'R', 'C', 'L', 'T', SYR2K, 'T', f64);
    test_macro!(test_006: inline, f64, (7, 5, 3, 1), (7, 5, 3, 1), 'C', 'R', 'U', 'T', SYR2K, 'T', f64);
    test_macro!(test_007: inline, f64, (7, 5, 3, 3), (7, 5, 1, 3), 'R', 'R', 'U', 'N', SYR2K, 'T', f64);
    test_macro!(test_008: inline, c32, (7, 5, 1, 1), (7, 5, 1, 3), 'C', 'C', 'U', 'T', SYR2K, 'T', c32);
    test_macro!(test_009: inline, c32, (7, 5, 1, 3), (7, 5, 3, 1), 'C', 'R', 'L', 'T', SYR2K, 'T', c32);
    test_macro!(test_010: inline, c32, (7, 5, 3, 1), (7, 5, 3, 3), 'R', 'R', 'L', 'N', SYR2K, 'T', c32);
    test_macro!(test_011: inline, c32, (7, 5, 3, 3), (7, 5, 1, 1), 'R', 'C', 'U', 'N', SYR2K, 'T', c32);
    test_macro!(test_012: inline, c64, (7, 5, 1, 1), (7, 5, 3, 3), 'R', 'C', 'U', 'T', SYR2K, 'T', c64);
    test_macro!(test_013: inline, c64, (7, 5, 1, 3), (7, 5, 1, 3), 'C', 'R', 'U', 'N', SYR2K, 'T', c64);
    test_macro!(test_014: inline, c64, (7, 5, 3, 1), (7, 5, 3, 1), 'C', 'R', 'L', 'N', SYR2K, 'T', c64);
    test_macro!(test_015: inline, c64, (7, 5, 3, 3), (7, 5, 1, 1), 'R', 'C', 'L', 'T', SYR2K, 'T', c64);
    test_macro!(test_016: inline, c32, (7, 5, 1, 1), (7, 5, 3, 1), 'C', 'C', 'U', 'N', HER2K, 'C', f32);
    test_macro!(test_017: inline, c32, (7, 5, 1, 3), (7, 5, 3, 3), 'R', 'R', 'L', 'N', HER2K, 'C', f32);
    test_macro!(test_018: inline, c32, (7, 5, 3, 1), (7, 5, 1, 3), 'R', 'C', 'L', 'C', HER2K, 'C', f32);
    test_macro!(test_019: inline, c32, (7, 5, 3, 3), (7, 5, 1, 1), 'C', 'R', 'U', 'C', HER2K, 'C', f32);
    test_macro!(test_020: inline, c64, (7, 5, 1, 3), (7, 5, 1, 1), 'C', 'C', 'L', 'N', HER2K, 'C', f64);
    test_macro!(test_021: inline, c64, (7, 5, 1, 3), (7, 5, 3, 3), 'R', 'R', 'U', 'C', HER2K, 'C', f64);
    test_macro!(test_022: inline, c64, (7, 5, 3, 1), (7, 5, 1, 3), 'C', 'R', 'L', 'C', HER2K, 'C', f64);
    test_macro!(test_023: inline, c64, (7, 5, 3, 1), (7, 5, 3, 1), 'R', 'C', 'U', 'N', HER2K, 'C', f64);

    // valid and invalid transpositions
    test_macro!(test_101: should_panic, c64, (7, 5, 1, 1), (7, 5, 1, 1), 'R', 'R', 'L', 'C', SYR2K, 'T', c64);
    test_macro!(test_102: should_panic, c64, (7, 5, 1, 1), (7, 5, 1, 1), 'R', 'R', 'L', 'T', HER2K, 'C', f64);
    test_macro!(test_103: should_panic, c64, (7, 5, 1, 1), (7, 5, 1, 1), 'C', 'C', 'L', 'C', SYR2K, 'T', c64);
    test_macro!(test_104: should_panic, c64, (7, 5, 1, 1), (7, 5, 1, 1), 'C', 'C', 'L', 'T', HER2K, 'C', f64);
}

#[cfg(test)]
mod valid_view {
    use super::*;

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F: ty,
            ($($a_slc: expr),+), ($($b_slc: expr),+), ($($c_slc: expr),+),
            $a_layout: expr, $b_layout: expr, $c_layout: expr,
            $uplo: expr,
            $trans: expr,
            $blas: ident, $blas_trans: expr, $blas_ty: ty
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$blas_ty>::rand();
                let beta = <$blas_ty>::rand();
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let b_raw = random_matrix(100, 100, $b_layout.into());
                let mut c_raw = random_matrix(100, 100, $c_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);
                let c_slc = slice($($c_slc),+);
                let uplo = $uplo;
                let trans = $trans;

                let mut c_naive = c_raw.clone();

                let c_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .c(c_raw.slice_mut(c_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .uplo(uplo)
                    .trans(trans)
                    .run()
                    .unwrap();

                let a_naive = a_raw.slice(a_slc).to_owned();
                let b_naive = b_raw.slice(b_slc).to_owned();
                let mut c_assign_0 = <$F>::from(beta) * &c_naive.slice(c_slc);
                if $blas_trans == 'C' {
                    for i in 0..c_assign_0.len_of(Axis(0)) {
                        c_assign_0[[i, i]] = <$F>::from(0.5) * (c_assign_0[[i, i]] + c_assign_0[[i, i]].conj());
                    }
                }
                let c_assign_1 = match trans.into() {
                    BLASNoTrans => {
                        <$F>::from(alpha) * gemm(&a_naive.view(), &transpose(&b_naive.view(), $blas_trans.into()).view())
                    },
                    BLASTrans | BLASConjTrans => {
                        <$F>::from(alpha) * gemm(&transpose(&a_naive.view(), $blas_trans.into()).view(), &b_naive.view())
                    },
                    _ => panic!("Invalid"),
                };
                let c_assign = &c_assign_0 + &c_assign_1 + &transpose(&c_assign_1.view(), $blas_trans.into()).view();
                tril_assign(&mut c_naive.slice_mut(c_slc), &c_assign.view(), uplo);

                if let ArrayOut2::ViewMut(_) = c_out {
                    let err = (&c_naive - &c_raw).mapv(|x| x.abs()).sum();
                    let acc = c_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    // successful tests
    test_macro!(test_000: inline, f32, (7, 5, 1, 1), (7, 5, 1, 1), (5, 5, 1, 1), 'R', 'R', 'R', 'L', 'T', SYR2K, 'T', f32);
    test_macro!(test_001: inline, f32, (7, 5, 1, 1), (7, 5, 3, 3), (7, 7, 3, 3), 'C', 'C', 'C', 'U', 'N', SYR2K, 'T', f32);
    test_macro!(test_002: inline, f32, (7, 5, 3, 3), (7, 5, 1, 1), (7, 7, 1, 1), 'R', 'C', 'C', 'U', 'N', SYR2K, 'T', f32);
    test_macro!(test_003: inline, f32, (7, 5, 3, 3), (7, 5, 3, 3), (5, 5, 3, 3), 'C', 'R', 'R', 'L', 'T', SYR2K, 'T', f32);
    test_macro!(test_004: inline, f64, (7, 5, 1, 1), (7, 5, 1, 1), (7, 7, 1, 3), 'C', 'R', 'R', 'U', 'N', SYR2K, 'T', f64);
    test_macro!(test_005: inline, f64, (7, 5, 1, 1), (7, 5, 3, 3), (5, 5, 3, 1), 'R', 'C', 'C', 'L', 'T', SYR2K, 'T', f64);
    test_macro!(test_006: inline, f64, (7, 5, 3, 3), (7, 5, 1, 1), (7, 7, 3, 1), 'C', 'R', 'C', 'L', 'N', SYR2K, 'T', f64);
    test_macro!(test_007: inline, f64, (7, 5, 3, 3), (7, 5, 3, 3), (5, 5, 1, 3), 'R', 'C', 'R', 'U', 'T', SYR2K, 'T', f64);
    test_macro!(test_008: inline, c32, (7, 5, 1, 1), (7, 5, 1, 1), (5, 5, 3, 3), 'C', 'C', 'R', 'U', 'T', SYR2K, 'T', c32);
    test_macro!(test_009: inline, c32, (7, 5, 1, 1), (7, 5, 3, 3), (7, 7, 1, 1), 'R', 'R', 'C', 'L', 'N', SYR2K, 'T', c32);
    test_macro!(test_010: inline, c32, (7, 5, 3, 3), (7, 5, 1, 3), (7, 7, 1, 3), 'C', 'C', 'C', 'L', 'N', SYR2K, 'T', c32);
    test_macro!(test_011: inline, c32, (7, 5, 3, 3), (7, 5, 3, 1), (5, 5, 3, 1), 'R', 'R', 'R', 'U', 'T', SYR2K, 'T', c32);
    test_macro!(test_012: inline, c64, (7, 5, 1, 3), (7, 5, 1, 1), (5, 5, 3, 3), 'R', 'C', 'C', 'L', 'T', SYR2K, 'T', c64);
    test_macro!(test_013: inline, c64, (7, 5, 1, 3), (7, 5, 3, 3), (7, 7, 1, 1), 'C', 'R', 'R', 'U', 'N', SYR2K, 'T', c64);
    test_macro!(test_014: inline, c64, (7, 5, 3, 1), (7, 5, 1, 3), (5, 5, 3, 1), 'C', 'R', 'C', 'U', 'T', SYR2K, 'T', c64);
    test_macro!(test_015: inline, c64, (7, 5, 3, 1), (7, 5, 3, 1), (7, 7, 1, 3), 'R', 'C', 'R', 'L', 'N', SYR2K, 'T', c64);
    test_macro!(test_016: inline, c32, (7, 5, 1, 3), (7, 5, 1, 3), (5, 5, 1, 1), 'C', 'C', 'R', 'L', 'C', HER2K, 'C', f32);
    test_macro!(test_017: inline, c32, (7, 5, 1, 3), (7, 5, 3, 1), (7, 7, 3, 3), 'R', 'R', 'C', 'U', 'N', HER2K, 'C', f32);
    test_macro!(test_018: inline, c32, (7, 5, 3, 1), (7, 5, 1, 3), (5, 5, 1, 3), 'R', 'R', 'C', 'U', 'C', HER2K, 'C', f32);
    test_macro!(test_019: inline, c32, (7, 5, 3, 1), (7, 5, 3, 1), (7, 7, 3, 1), 'C', 'C', 'R', 'L', 'N', HER2K, 'C', f32);
    test_macro!(test_020: inline, c64, (7, 5, 1, 3), (7, 5, 1, 3), (7, 7, 3, 1), 'R', 'C', 'R', 'U', 'N', HER2K, 'C', f64);
    test_macro!(test_021: inline, c64, (7, 5, 1, 3), (7, 5, 3, 1), (5, 5, 1, 3), 'C', 'R', 'C', 'L', 'C', HER2K, 'C', f64);
    test_macro!(test_022: inline, c64, (7, 5, 3, 1), (7, 5, 1, 3), (7, 7, 3, 3), 'R', 'R', 'R', 'L', 'N', HER2K, 'C', f64);
    test_macro!(test_023: inline, c64, (7, 5, 3, 1), (7, 5, 3, 1), (5, 5, 1, 1), 'C', 'C', 'C', 'U', 'C', HER2K, 'C', f64);
}
