use crate::util::*;
use approx::*;
use blas_array2::blas3::syrk::{HERK, SYRK};
use blas_array2::util::*;
use ndarray::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid_owned {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <c32 as BLASFloat>::RealFloat;
        let alpha = c32::rand();
        let beta = c32::rand();
        let a_raw = random_matrix(100, 100, 'R'.into());
        let a_slc = slice(7, 5, 1, 3);
        let uplo = 'U';
        let trans = 'T';

        let c_out = SYRK::default()
            .a(a_raw.slice(a_slc))
            .alpha(alpha)
            .beta(beta)
            .uplo(uplo)
            .trans(trans)
            .run()
            .unwrap();

        let a_naive = a_raw.slice(a_slc).to_owned();
        let c_assign = match trans.into() {
            BLASNoTrans => alpha * gemm(&a_naive.view(), &transpose(&a_naive.view(), trans.into()).view()),
            BLASTrans | BLASConjTrans => {
                alpha * gemm(&transpose(&a_naive.view(), trans.into()).view(), &a_naive.view())
            },
            _ => panic!("Invalid"),
        };
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

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F: ty,
            ($($a_slc: expr),+),
            $a_layout: expr,
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
                let a_slc = slice($($a_slc),+);
                let uplo = $uplo;
                let trans = $trans;

                let c_out = $blas::<$F>::default()
                    .a(a_raw.slice(a_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .uplo(uplo)
                    .trans(trans)
                    .run()
                    .unwrap();

                let a_naive = a_raw.slice(a_slc).to_owned();
                let c_assign = match trans.into() {
                    BLASNoTrans => <$F>::from(alpha) * gemm(&a_naive.view(), &transpose(&a_naive.view(), $blas_trans.into()).view()),
                    BLASTrans | BLASConjTrans => <$F>::from(alpha) * gemm(&transpose(&a_naive.view(), $blas_trans.into()).view(), &a_naive.view()),
                    _ => panic!("Invalid"),
                };
                let mut c_naive = Array2::zeros(c_assign.dim());
                tril_assign(&mut c_naive.view_mut(), &c_assign.view(), uplo);

                if let ArrayOut2::Owned(c_out) = c_out {
                    println!("{:7.3?}", &c_naive);
                    println!("{:7.3?}", &c_out);
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
    test_macro!(test_000: inline, f32, (7, 5, 1, 1), 'R', 'L', 'T', SYRK, 'T', f32);
    test_macro!(test_001: inline, f32, (7, 5, 1, 1), 'R', 'U', 'N', SYRK, 'T', f32);
    test_macro!(test_002: inline, f32, (7, 5, 3, 3), 'C', 'L', 'T', SYRK, 'T', f32);
    test_macro!(test_003: inline, f32, (7, 5, 3, 3), 'C', 'U', 'N', SYRK, 'T', f32);
    test_macro!(test_004: inline, f64, (7, 5, 1, 1), 'R', 'L', 'T', SYRK, 'T', f64);
    test_macro!(test_005: inline, f64, (7, 5, 1, 3), 'C', 'L', 'N', SYRK, 'T', f64);
    test_macro!(test_006: inline, f64, (7, 5, 3, 1), 'C', 'U', 'T', SYRK, 'T', f64);
    test_macro!(test_007: inline, f64, (7, 5, 3, 3), 'R', 'U', 'N', SYRK, 'T', f64);
    test_macro!(test_008: inline, c32, (7, 5, 1, 1), 'C', 'L', 'T', SYRK, 'T', c32);
    test_macro!(test_009: inline, c32, (7, 5, 1, 3), 'R', 'U', 'N', SYRK, 'T', c32);
    test_macro!(test_010: inline, c32, (7, 5, 3, 1), 'R', 'L', 'N', SYRK, 'T', c32);
    test_macro!(test_011: inline, c32, (7, 5, 3, 3), 'C', 'U', 'T', SYRK, 'T', c32);
    test_macro!(test_012: inline, c64, (7, 5, 1, 1), 'C', 'U', 'N', SYRK, 'T', c64);
    test_macro!(test_013: inline, c64, (7, 5, 1, 3), 'R', 'U', 'T', SYRK, 'T', c64);
    test_macro!(test_014: inline, c64, (7, 5, 3, 1), 'C', 'L', 'N', SYRK, 'T', c64);
    test_macro!(test_015: inline, c64, (7, 5, 3, 3), 'R', 'L', 'T', SYRK, 'T', c64);
    test_macro!(test_016: inline, c32, (7, 5, 1, 1), 'C', 'U', 'C', HERK, 'C', f32);
    test_macro!(test_017: inline, c32, (7, 5, 1, 3), 'C', 'L', 'N', HERK, 'C', f32);
    test_macro!(test_018: inline, c32, (7, 5, 3, 1), 'R', 'U', 'N', HERK, 'C', f32);
    test_macro!(test_019: inline, c32, (7, 5, 3, 3), 'R', 'L', 'C', HERK, 'C', f32);
    test_macro!(test_020: inline, c64, (7, 5, 1, 3), 'R', 'L', 'N', HERK, 'C', f64);
    test_macro!(test_021: inline, c64, (7, 5, 1, 3), 'C', 'U', 'C', HERK, 'C', f64);
    test_macro!(test_022: inline, c64, (7, 5, 3, 1), 'R', 'U', 'C', HERK, 'C', f64);
    test_macro!(test_023: inline, c64, (7, 5, 3, 1), 'C', 'L', 'N', HERK, 'C', f64);

    // valid and invalid transpositions
    test_macro!(test_100: inline, f32, (7, 5, 1, 1), 'R', 'L', 'C', SYRK, 'T', f32);
    test_macro!(test_101: should_panic, c64, (7, 5, 1, 1), 'R', 'L', 'C', SYRK, 'T', c64);
    test_macro!(test_102: should_panic, c64, (7, 5, 1, 1), 'R', 'L', 'T', HERK, 'C', f64);
    test_macro!(test_103: should_panic, c64, (7, 5, 1, 1), 'C', 'L', 'C', SYRK, 'T', c64);
    test_macro!(test_104: should_panic, c64, (7, 5, 1, 1), 'C', 'L', 'T', HERK, 'C', f64);
}

#[cfg(test)]
mod valid_view {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <c32 as BLASFloat>::RealFloat;
        let alpha = c32::rand();
        let beta = c32::rand();
        let a_raw = random_matrix(100, 100, 'R'.into());
        let mut c_raw = random_matrix(100, 100, 'R'.into());
        let a_slc = slice(7, 5, 1, 1);
        let c_slc = slice(7, 7, 3, 3);
        let uplo = 'U';
        let trans = 'N';
        let blas_trans = BLASTrans;

        let mut c_naive = c_raw.clone();

        let c_out = SYRK::default()
            .a(a_raw.slice(a_slc))
            .c(c_raw.slice_mut(c_slc))
            .alpha(alpha)
            .beta(beta)
            .uplo(uplo)
            .trans(trans)
            .run()
            .unwrap();

        let a_naive = a_raw.slice(a_slc).to_owned();
        let c_assign = match trans.into() {
            BLASNoTrans => {
                <c32>::from(alpha)
                    * gemm(&a_naive.view(), &transpose(&a_naive.view(), blas_trans.into()).view())
                    + beta * &c_naive.slice(c_slc)
            },
            BLASTrans | BLASConjTrans => {
                <c32>::from(alpha)
                    * gemm(&transpose(&a_naive.view(), blas_trans.into()).view(), &a_naive.view())
                    + beta * &c_naive.slice(c_slc)
            },
            _ => panic!("Invalid"),
        };
        tril_assign(&mut c_naive.slice_mut(c_slc), &c_assign.view(), uplo);

        if let ArrayOut2::ViewMut(_) = c_out {
            let err = (&c_naive - &c_raw).mapv(|x| x.abs()).sum();
            let acc = c_naive.view().mapv(|x| x.abs()).sum() as RT;
            let err_div = err / acc;
            assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
        } else {
            panic!("Failed");
        }
    }

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F: ty,
            ($($a_slc: expr),+), ($($c_slc: expr),+),
            $a_layout: expr, $c_layout: expr,
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
                let mut c_raw = random_matrix(100, 100, $c_layout.into());
                let a_slc = slice($($a_slc),+);
                let c_slc = slice($($c_slc),+);
                let uplo = $uplo;
                let trans = $trans;

                let mut c_naive = c_raw.clone();

                let c_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .c(c_raw.slice_mut(c_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .uplo(uplo)
                    .trans(trans)
                    .run()
                    .unwrap();

                let a_naive = a_raw.slice(a_slc).to_owned();
                let mut c_assign = <$F>::from(beta) * &c_naive.slice(c_slc);
                if $blas_trans == 'C' {
                    for i in 0..c_assign.len_of(Axis(0)) {
                        c_assign[[i, i]] = <$F>::from(0.5) * (c_assign[[i, i]] + c_assign[[i, i]].conj());
                    }
                }
                c_assign += &(match trans.into() {
                    BLASNoTrans => {
                        <$F>::from(alpha) * gemm(&a_naive.view(), &transpose(&a_naive.view(), $blas_trans.into()).view())
                    },
                    BLASTrans | BLASConjTrans => {
                        <$F>::from(alpha) * gemm(&transpose(&a_naive.view(), $blas_trans.into()).view(), &a_naive.view())
                    },
                    _ => panic!("Invalid"),
                });
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
    test_macro!(test_000: inline, f32, (7, 5, 1, 1), (5, 5, 1, 1), 'R', 'R', 'L', 'T', SYRK, 'T', f32);
    test_macro!(test_001: inline, f32, (7, 5, 1, 1), (7, 7, 1, 1), 'R', 'R', 'U', 'N', SYRK, 'T', f32);
    test_macro!(test_002: inline, f32, (7, 5, 3, 3), (7, 7, 3, 3), 'C', 'C', 'L', 'N', SYRK, 'T', f32);
    test_macro!(test_003: inline, f32, (7, 5, 3, 3), (5, 5, 3, 3), 'C', 'C', 'U', 'T', SYRK, 'T', f32);
    test_macro!(test_004: inline, f64, (7, 5, 1, 1), (7, 7, 1, 3), 'C', 'C', 'L', 'N', SYRK, 'T', f64);
    test_macro!(test_005: inline, f64, (7, 5, 1, 3), (5, 5, 3, 1), 'R', 'C', 'L', 'T', SYRK, 'T', f64);
    test_macro!(test_006: inline, f64, (7, 5, 3, 1), (5, 5, 3, 1), 'C', 'R', 'U', 'T', SYRK, 'T', f64);
    test_macro!(test_007: inline, f64, (7, 5, 3, 3), (7, 7, 1, 3), 'R', 'R', 'U', 'N', SYRK, 'T', f64);
    test_macro!(test_008: inline, c32, (7, 5, 1, 1), (5, 5, 1, 3), 'C', 'C', 'U', 'T', SYRK, 'T', c32);
    test_macro!(test_009: inline, c32, (7, 5, 1, 3), (5, 5, 3, 1), 'C', 'R', 'L', 'T', SYRK, 'T', c32);
    test_macro!(test_010: inline, c32, (7, 5, 3, 1), (7, 7, 3, 3), 'R', 'R', 'L', 'N', SYRK, 'T', c32);
    test_macro!(test_011: inline, c32, (7, 5, 3, 3), (7, 7, 1, 1), 'R', 'C', 'U', 'N', SYRK, 'T', c32);
    test_macro!(test_012: inline, c64, (7, 5, 1, 1), (5, 5, 3, 3), 'R', 'C', 'U', 'T', SYRK, 'T', c64);
    test_macro!(test_013: inline, c64, (7, 5, 1, 3), (7, 7, 1, 3), 'C', 'R', 'U', 'N', SYRK, 'T', c64);
    test_macro!(test_014: inline, c64, (7, 5, 3, 1), (7, 7, 3, 1), 'C', 'R', 'L', 'N', SYRK, 'T', c64);
    test_macro!(test_015: inline, c64, (7, 5, 3, 3), (5, 5, 1, 1), 'R', 'C', 'L', 'T', SYRK, 'T', c64);
    test_macro!(test_016: inline, c32, (7, 5, 1, 1), (7, 7, 3, 1), 'C', 'C', 'U', 'N', HERK, 'C', f32);
    test_macro!(test_017: inline, c32, (7, 5, 1, 3), (7, 7, 3, 3), 'R', 'R', 'L', 'N', HERK, 'C', f32);
    test_macro!(test_018: inline, c32, (7, 5, 3, 1), (5, 5, 1, 3), 'R', 'C', 'L', 'C', HERK, 'C', f32);
    test_macro!(test_019: inline, c32, (7, 5, 3, 3), (5, 5, 1, 1), 'C', 'R', 'U', 'C', HERK, 'C', f32);
    test_macro!(test_020: inline, c64, (7, 5, 1, 3), (7, 7, 1, 1), 'C', 'C', 'L', 'N', HERK, 'C', f64);
    test_macro!(test_021: inline, c64, (7, 5, 1, 3), (5, 5, 3, 3), 'R', 'R', 'U', 'C', HERK, 'C', f64);
    test_macro!(test_022: inline, c64, (7, 5, 3, 1), (5, 5, 1, 3), 'C', 'R', 'L', 'C', HERK, 'C', f64);
    test_macro!(test_023: inline, c64, (7, 5, 3, 1), (7, 7, 3, 1), 'R', 'C', 'U', 'N', HERK, 'C', f64);
}
