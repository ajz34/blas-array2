use crate::util::*;
use approx::*;
use blas_array2::blas3::gemm::GEMM;
use blas_array2::util::*;
use num_complex::*;

#[cfg(test)]
mod valid_owned {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <f32 as BLASFloat>::RealFloat;
        let alpha = f32::rand();
        let beta = f32::rand();
        let a_raw = random_matrix(100, 100, 'R'.into());
        let b_raw = random_matrix(100, 100, 'R'.into());
        let a_slc = slice(7, 8, 1, 1); // s![5..12, 10..18]
        let b_slc = slice(8, 9, 1, 1); // s![5..13, 10..19]

        let c_out = GEMM::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .transa('N')
            .transb('N')
            .alpha(alpha)
            .beta(beta)
            .run()
            .unwrap();

        let a_naive = transpose(&a_raw.slice(a_slc), 'N'.into());
        let b_naive = transpose(&b_raw.slice(b_slc), 'N'.into());
        let c_naive = alpha * gemm(&a_naive.view(), &b_naive.view());

        if let ArrayOut2::Owned(c_out) = c_out {
            let err = (&c_naive - &c_out).mapv(|x| x.abs()).sum();
            let acc = c_naive.mapv(|x| x.abs()).sum() as RT;
            let err_div = err / acc;
            assert_abs_diff_eq!(err_div, 0.0, epsilon = 2.0 * RT::EPSILON);
        } else {
            panic!("GEMM failed");
        }
    }

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F:ty,
            ($($a_slc: expr),+), ($($b_slc: expr),+),
            $a_layout: expr, $b_layout: expr,
            $a_trans: expr, $b_trans: expr
        ) => {
            #[test]
            #[$attr]
            pub fn $test_name()
            {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$F>::rand();
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let b_raw = random_matrix(100, 100, $b_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);

                let c_out = GEMM::<$F>::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .transa($a_trans)
                    .transb($b_trans)
                    .alpha(alpha)
                    .beta(beta)
                    .run().unwrap();

                let a_naive = transpose(&a_raw.slice(a_slc), $a_trans.into());
                let b_naive = transpose(&b_raw.slice(b_slc), $b_trans.into());
                let c_naive = alpha * gemm(&a_naive.view(), &b_naive.view());

                if let ArrayOut2::Owned(c_out) = c_out {
                    let err = (&c_naive - &c_out).mapv(|x| x.abs()).sum();
                    let acc = c_naive.mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=2.0*RT::EPSILON);
                } else {
                    panic!("GEMM failed");
                }
            }
        };
    }

    // successful tests
    test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    test_macro!(test_001: inline, f32, (7, 8, 1, 1), (8, 9, 3, 3), 'C', 'C', 'N', 'N');
    test_macro!(test_002: inline, f32, (8, 7, 3, 3), (9, 8, 1, 1), 'C', 'C', 'T', 'T');
    test_macro!(test_003: inline, f32, (8, 7, 3, 3), (9, 8, 3, 3), 'R', 'R', 'T', 'T');
    test_macro!(test_004: inline, f64, (7, 8, 3, 3), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    test_macro!(test_005: inline, f64, (7, 8, 3, 3), (8, 9, 3, 3), 'C', 'C', 'N', 'N');
    test_macro!(test_006: inline, f64, (8, 7, 1, 1), (9, 8, 1, 3), 'R', 'C', 'T', 'T');
    test_macro!(test_007: inline, f64, (8, 7, 1, 1), (9, 8, 3, 1), 'C', 'R', 'T', 'T');
    test_macro!(test_008: inline, c32, (7, 8, 1, 3), (9, 8, 1, 1), 'C', 'C', 'N', 'T');
    test_macro!(test_009: inline, c32, (7, 8, 1, 3), (9, 8, 3, 3), 'R', 'R', 'N', 'T');
    test_macro!(test_010: inline, c32, (8, 7, 3, 1), (8, 9, 1, 3), 'C', 'R', 'T', 'N');
    test_macro!(test_011: inline, c32, (8, 7, 3, 1), (8, 9, 3, 1), 'R', 'C', 'T', 'N');
    test_macro!(test_012: inline, c64, (7, 8, 3, 1), (9, 8, 1, 3), 'C', 'R', 'N', 'T');
    test_macro!(test_013: inline, c64, (7, 8, 3, 1), (9, 8, 3, 1), 'R', 'C', 'N', 'T');
    test_macro!(test_014: inline, c64, (8, 7, 1, 3), (8, 9, 1, 3), 'R', 'C', 'T', 'N');
    test_macro!(test_015: inline, c64, (8, 7, 1, 3), (8, 9, 3, 1), 'C', 'R', 'T', 'N');

    // unrecognized transpose
    test_macro!(test_100: should_panic, f32, (7, 8, 1, 1), (9, 8, 1, 3), 'R', 'R', BLASTrans::ConjNoTrans, 'T');
    test_macro!(test_101: should_panic, f32, (7, 8, 1, 1), (9, 8, 1, 3), 'R', 'R', 'N', BLASTrans::ConjNoTrans);

    // dimension mismatch (k)
    test_macro!(test_102: should_panic, f32, (7, 5, 1, 1), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    test_macro!(test_103: should_panic, f32, (7, 5, 1, 1), (9, 8, 1, 3), 'R', 'R', 'N', 'T');
}

mod valid_view {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <f32 as BLASFloat>::RealFloat;
        let alpha = f32::rand();
        let beta = f32::rand();
        let a_raw = random_matrix(100, 100, 'R'.into());
        let b_raw = random_matrix(100, 100, 'R'.into());
        let mut c_raw = random_matrix(100, 100, 'C'.into());
        let a_slc = slice(7, 8, 1, 1);
        let b_slc = slice(9, 8, 3, 3);
        let c_slc = slice(7, 9, 1, 3);

        let mut c_naive = c_raw.clone();

        let c_out = GEMM::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .c(c_raw.slice_mut(c_slc))
            .transa('N')
            .transb('T')
            .alpha(alpha)
            .beta(beta)
            .run()
            .unwrap();

        let a_naive = transpose(&a_raw.slice(a_slc), 'N'.into());
        let b_naive = transpose(&b_raw.slice(b_slc), 'T'.into());
        let c_assign = &(alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_naive.slice(c_slc));
        c_naive.slice_mut(c_slc).assign(c_assign);

        if let ArrayOut2::ViewMut(_) = c_out {
            let err = (&c_naive - &c_raw).mapv(|x| x.abs()).sum();
            let acc = c_naive.view().mapv(|x| x.abs()).sum() as RT;
            let err_div = err / acc;
            assert_abs_diff_eq!(err_div, 0.0, epsilon = 2.0 * RT::EPSILON);
        } else {
            panic!("GEMM failed");
        }
    }

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F:ty,
            ($($a_slc: expr),+), ($($b_slc: expr),+), ($($c_slc: expr),+),
            $a_layout: expr, $b_layout: expr, $c_layout: expr,
            $a_trans: expr, $b_trans: expr
        ) => {
            #[test]
            #[$attr]
            pub fn $test_name()
            {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$F>::rand();
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let b_raw = random_matrix(100, 100, $b_layout.into());
                let mut c_raw = random_matrix(100, 100, $c_layout.into());
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);
                let c_slc = slice($($c_slc),+);

                let mut c_naive = c_raw.clone();

                let c_out = GEMM::<$F>::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .c(c_raw.slice_mut(c_slc))
                    .transa($a_trans)
                    .transb($b_trans)
                    .alpha(alpha)
                    .beta(beta)
                    .run().unwrap();

                let a_naive = transpose(&a_raw.slice(a_slc), $a_trans.into());
                let b_naive = transpose(&b_raw.slice(b_slc), $b_trans.into());
                let c_assign = &(alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_naive.slice(c_slc));
                c_naive.slice_mut(c_slc).assign(c_assign);

                if let ArrayOut2::ViewMut(_) = c_out {
                    let err = (&c_naive - &c_raw).mapv(|x| x.abs()).sum();
                    let acc = c_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=2.0*RT::EPSILON);
                } else {
                    panic!("GEMM failed");
                }
            }
        };
    }

    // successful tests
    test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 9, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'N', 'N');
    test_macro!(test_001: inline, f32, (7, 8, 1, 1), (8, 9, 3, 3), (7, 9, 3, 3), 'C', 'C', 'C', 'N', 'N');
    test_macro!(test_002: inline, f32, (8, 7, 3, 3), (9, 8, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'T', 'T');
    test_macro!(test_003: inline, f32, (8, 7, 3, 3), (9, 8, 3, 3), (7, 9, 3, 3), 'R', 'R', 'R', 'T', 'T');
    test_macro!(test_004: inline, f64, (7, 8, 3, 3), (8, 9, 1, 1), (7, 9, 3, 3), 'R', 'R', 'C', 'N', 'N');
    test_macro!(test_005: inline, f64, (7, 8, 3, 3), (8, 9, 3, 3), (7, 9, 1, 1), 'C', 'C', 'R', 'N', 'N');
    test_macro!(test_006: inline, f64, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 3, 3), 'C', 'C', 'R', 'T', 'T');
    test_macro!(test_007: inline, f64, (8, 7, 1, 1), (9, 8, 3, 3), (7, 9, 1, 1), 'R', 'R', 'C', 'T', 'T');
    test_macro!(test_008: inline, c32, (7, 8, 1, 3), (9, 8, 1, 3), (7, 9, 1, 3), 'R', 'C', 'R', 'N', 'T');
    test_macro!(test_009: inline, c32, (7, 8, 1, 3), (9, 8, 3, 1), (7, 9, 3, 1), 'C', 'R', 'C', 'N', 'T');
    test_macro!(test_010: inline, c32, (8, 7, 3, 1), (8, 9, 1, 3), (7, 9, 3, 1), 'C', 'R', 'R', 'T', 'N');
    test_macro!(test_011: inline, c32, (8, 7, 3, 1), (8, 9, 3, 1), (7, 9, 1, 3), 'R', 'C', 'C', 'T', 'N');
    test_macro!(test_012: inline, c64, (7, 8, 3, 1), (9, 8, 1, 3), (7, 9, 1, 3), 'C', 'R', 'C', 'N', 'T');
    test_macro!(test_013: inline, c64, (7, 8, 3, 1), (9, 8, 3, 1), (7, 9, 3, 1), 'R', 'C', 'R', 'N', 'T');
    test_macro!(test_014: inline, c64, (8, 7, 1, 3), (8, 9, 1, 3), (7, 9, 3, 1), 'R', 'C', 'C', 'T', 'N');
    test_macro!(test_015: inline, c64, (8, 7, 1, 3), (8, 9, 3, 1), (7, 9, 1, 3), 'C', 'R', 'R', 'T', 'N');

    // dimension mismatch (m, n)
    test_macro!(test_100: should_panic, f32, (7, 8, 1, 1), (8, 9, 1, 1), (7, 8, 1, 1), 'R', 'R', 'R', 'N', 'N');
}
