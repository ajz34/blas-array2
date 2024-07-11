use crate::util::*;
use approx::*;
use blas_array2::blas3::symm::{HEMM, SYMM};
use blas_array2::util::*;
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
        let b_raw = random_matrix(100, 100, 'R'.into());
        let a_slc = slice(7, 7, 1, 1);
        let b_slc = slice(7, 9, 1, 1);
        let side = 'L';
        let uplo = 'U';

        let c_out = SYMM::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .alpha(alpha)
            .beta(beta)
            .side(side)
            .uplo(uplo)
            .run()
            .unwrap();

        let a_naive = symmetrize(&a_raw.slice(a_slc), uplo.into());
        let b_naive = &b_raw.slice(b_slc).into_owned();
        let c_naive = match side.into() {
            BLASLeft => alpha * gemm(&a_naive.view(), &b_naive.view()),
            BLASRight => alpha * gemm(&b_naive.view(), &a_naive.view()),
            _ => panic!("Invalid"),
        };

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
            ($($a_slc: expr),+), ($($b_slc: expr),+),
            $a_layout: expr, $b_layout: expr,
            $side: expr, $uplo: expr,
            $blas: ident, $symm: ident
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$F>::rand();
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let b_raw = random_matrix(100, 100, $b_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);
                let side = $side;
                let uplo = $uplo;

                let c_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .side(side)
                    .uplo(uplo)
                    .run()
                    .unwrap();

                let a_naive = $symm(&a_raw.slice(a_slc), uplo.into());
                let b_naive = &b_raw.slice(b_slc).into_owned();
                let c_naive = match side.into() {
                    BLASLeft => alpha * gemm(&a_naive.view(), &b_naive.view()),
                    BLASRight => alpha * gemm(&b_naive.view(), &a_naive.view()),
                    _ => panic!("Invalid"),
                };

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
    test_macro!(test_000: inline, f32, (7, 7, 1, 1), (7, 9, 1, 1), 'R', 'R', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_001: inline, f32, (7, 7, 1, 1), (7, 9, 1, 1), 'R', 'C', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_002: inline, f32, (9, 9, 3, 3), (7, 9, 3, 3), 'C', 'R', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_003: inline, f32, (9, 9, 3, 3), (7, 9, 3, 3), 'C', 'C', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_004: inline, f64, (7, 7, 1, 1), (7, 9, 3, 3), 'C', 'R', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_005: inline, f64, (7, 7, 3, 3), (7, 9, 1, 1), 'C', 'R', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_006: inline, f64, (9, 9, 1, 3), (7, 9, 1, 3), 'R', 'C', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_007: inline, f64, (9, 9, 3, 1), (7, 9, 3, 1), 'R', 'C', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_008: inline, c32, (7, 7, 1, 1), (7, 9, 3, 3), 'C', 'C', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_009: inline, c32, (7, 7, 3, 3), (7, 9, 1, 3), 'R', 'R', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_010: inline, c32, (9, 9, 1, 3), (7, 9, 3, 1), 'R', 'R', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_011: inline, c32, (9, 9, 3, 1), (7, 9, 1, 1), 'C', 'C', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_012: inline, c64, (7, 7, 1, 3), (7, 9, 3, 1), 'C', 'C', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_013: inline, c64, (7, 7, 3, 1), (7, 9, 3, 3), 'R', 'C', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_014: inline, c64, (9, 9, 1, 3), (7, 9, 1, 3), 'R', 'R', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_015: inline, c64, (9, 9, 3, 1), (7, 9, 1, 1), 'C', 'R', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_016: inline, c32, (7, 7, 1, 3), (7, 9, 1, 3), 'C', 'C', 'L', 'U', HEMM, hermitianize);
    test_macro!(test_017: inline, c32, (7, 7, 3, 3), (7, 9, 3, 1), 'R', 'R', 'L', 'U', HEMM, hermitianize);
    test_macro!(test_018: inline, c32, (9, 9, 1, 1), (7, 9, 3, 1), 'C', 'R', 'R', 'L', HEMM, hermitianize);
    test_macro!(test_019: inline, c32, (9, 9, 3, 1), (7, 9, 1, 3), 'R', 'C', 'R', 'L', HEMM, hermitianize);
    test_macro!(test_020: inline, c64, (7, 7, 3, 1), (7, 9, 1, 3), 'C', 'R', 'L', 'U', HEMM, hermitianize);
    test_macro!(test_021: inline, c64, (7, 7, 3, 3), (7, 9, 3, 1), 'R', 'C', 'L', 'L', HEMM, hermitianize);
    test_macro!(test_022: inline, c64, (9, 9, 1, 1), (7, 9, 3, 3), 'R', 'R', 'R', 'L', HEMM, hermitianize);
    test_macro!(test_023: inline, c64, (9, 9, 1, 3), (7, 9, 1, 1), 'C', 'C', 'R', 'U', HEMM, hermitianize);
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
        let b_raw = random_matrix(100, 100, 'R'.into());
        let mut c_raw = random_matrix(100, 100, 'R'.into());
        let a_slc = slice(7, 7, 1, 1);
        let b_slc = slice(7, 9, 1, 1);
        let c_slc = slice(7, 9, 1, 1);
        let side = 'L';
        let uplo = 'U';

        let mut c_naive = c_raw.clone();

        let c_out = SYMM::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .c(c_raw.slice_mut(c_slc))
            .alpha(alpha)
            .beta(beta)
            .side(side)
            .uplo(uplo)
            .run()
            .unwrap();

        let a_naive = symmetrize(&a_raw.slice(a_slc), uplo.into());
        let b_naive = &b_raw.slice(b_slc).into_owned();
        let c_assign = match side.into() {
            BLASLeft => alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_naive.slice(c_slc),
            BLASRight => alpha * gemm(&b_naive.view(), &a_naive.view()) + beta * &c_naive.slice(c_slc),
            _ => panic!("Invalid"),
        };
        c_naive.slice_mut(c_slc).assign(&c_assign);

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
            ($($a_slc: expr),+), ($($b_slc: expr),+), ($($c_slc: expr),+),
            $a_layout: expr, $b_layout: expr, $c_layout: expr,
            $side: expr, $uplo: expr,
            $blas: ident, $symm: ident
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$F>::rand();
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let b_raw = random_matrix(100, 100, $b_layout.into());
                let mut c_raw = random_matrix(100, 100, $c_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);
                let c_slc = slice($($c_slc),+);
                let side = $side;
                let uplo = $uplo;

                let mut c_naive = c_raw.clone();

                let c_out = $blas::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .c(c_raw.slice_mut(c_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .side(side)
                    .uplo(uplo)
                    .run()
                    .unwrap();

                let a_naive = $symm(&a_raw.slice(a_slc), uplo.into());
                let b_naive = &b_raw.slice(b_slc).into_owned();
                let c_assign = match side.into() {
                    BLASLeft => alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_naive.slice(c_slc),
                    BLASRight => alpha * gemm(&b_naive.view(), &a_naive.view()) + beta * &c_naive.slice(c_slc),
                    _ => panic!("Invalid"),
                };
                c_naive.slice_mut(c_slc).assign(&c_assign);

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
    test_macro!(test_000: inline, f32, (7, 7, 1, 1), (7, 9, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_001: inline, f32, (7, 7, 1, 3), (7, 9, 3, 3), (7, 9, 3, 3), 'C', 'C', 'C', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_002: inline, f32, (9, 9, 3, 1), (7, 9, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_003: inline, f32, (9, 9, 3, 3), (7, 9, 3, 3), (7, 9, 3, 3), 'R', 'R', 'R', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_004: inline, f64, (7, 7, 1, 1), (7, 9, 1, 1), (7, 9, 3, 3), 'R', 'R', 'C', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_005: inline, f64, (7, 7, 1, 3), (7, 9, 3, 3), (7, 9, 1, 1), 'C', 'C', 'R', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_006: inline, f64, (9, 9, 3, 1), (7, 9, 1, 3), (7, 9, 1, 3), 'R', 'C', 'R', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_007: inline, f64, (9, 9, 3, 3), (7, 9, 3, 1), (7, 9, 3, 1), 'C', 'R', 'C', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_008: inline, c32, (7, 7, 1, 1), (7, 9, 1, 3), (7, 9, 3, 3), 'C', 'R', 'C', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_009: inline, c32, (7, 7, 1, 3), (7, 9, 3, 1), (7, 9, 1, 1), 'R', 'C', 'R', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_010: inline, c32, (9, 9, 3, 1), (7, 9, 3, 1), (7, 9, 3, 3), 'C', 'C', 'R', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_011: inline, c32, (9, 9, 3, 3), (7, 9, 1, 3), (7, 9, 1, 1), 'R', 'R', 'C', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_012: inline, c64, (7, 7, 3, 1), (7, 9, 1, 3), (7, 9, 3, 1), 'C', 'C', 'R', 'L', 'L', SYMM, symmetrize);
    test_macro!(test_013: inline, c64, (7, 7, 3, 3), (7, 9, 3, 1), (7, 9, 1, 3), 'R', 'R', 'C', 'L', 'U', SYMM, symmetrize);
    test_macro!(test_014: inline, c64, (9, 9, 1, 1), (7, 9, 3, 3), (7, 9, 1, 3), 'R', 'C', 'C', 'R', 'L', SYMM, symmetrize);
    test_macro!(test_015: inline, c64, (9, 9, 1, 3), (7, 9, 1, 1), (7, 9, 3, 1), 'C', 'R', 'R', 'R', 'U', SYMM, symmetrize);
    test_macro!(test_016: inline, c32, (7, 7, 3, 1), (7, 9, 3, 1), (7, 9, 1, 3), 'C', 'R', 'R', 'L', 'L', HEMM, hermitianize);
    test_macro!(test_017: inline, c32, (7, 7, 3, 3), (7, 9, 1, 3), (7, 9, 3, 1), 'R', 'C', 'C', 'L', 'U', HEMM, hermitianize);
    test_macro!(test_018: inline, c32, (9, 9, 1, 1), (7, 9, 3, 1), (7, 9, 3, 1), 'R', 'C', 'C', 'R', 'L', HEMM, hermitianize);
    test_macro!(test_019: inline, c32, (9, 9, 1, 3), (7, 9, 1, 3), (7, 9, 1, 3), 'C', 'R', 'R', 'R', 'U', HEMM, hermitianize);
    test_macro!(test_020: inline, c64, (7, 7, 3, 1), (7, 9, 3, 3), (7, 9, 1, 1), 'C', 'R', 'C', 'L', 'U', HEMM, hermitianize);
    test_macro!(test_021: inline, c64, (7, 7, 3, 3), (7, 9, 1, 1), (7, 9, 3, 3), 'R', 'C', 'R', 'L', 'L', HEMM, hermitianize);
    test_macro!(test_022: inline, c64, (9, 9, 1, 1), (7, 9, 3, 3), (7, 9, 3, 1), 'R', 'R', 'R', 'R', 'U', HEMM, hermitianize);
    test_macro!(test_023: inline, c64, (9, 9, 1, 3), (7, 9, 1, 1), (7, 9, 1, 3), 'C', 'C', 'C', 'R', 'L', HEMM, hermitianize);
}
