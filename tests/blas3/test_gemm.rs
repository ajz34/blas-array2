use crate::util::*;
use approx::*;
use blas_array2::blas3::gemm::GEMM;
use blas_array2::util::*;
use num_complex::*;

#[cfg(test)]
mod demonstration {
    /// Following code performs matrix multiplication out-place
    /// ```
    /// c = a * b
    /// ```
    /// Output
    /// ```
    /// [[-22.000, -28.000],
    ///  [-40.000, -52.000]]
    /// ```
    #[test]
    fn demonstration_dgemm_simple() {
        use blas_array2::blas3::gemm::DGEMM;
        use blas_array2::util::*;
        use ndarray::prelude::*;
        let a = array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
        let b = array![[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]];
        let c_out = DGEMM::default().a(a.view()).b(b.view()).run().unwrap().into_owned();
        println!("{:7.3?}", c_out);
    }

    /// Following code performs matrix multiplication in-place
    /// ```
    /// c = alpha * a * transpose(b) + beta * c
    /// where
    ///     alpha = 1.0 (by default)
    ///     beta = 1.5
    ///     a = [[1.0, 2.0, ___],
    ///          [3.0, 4.0, ___]]
    ///         (sliced by `s![.., ..2]`)
    ///     b = [[-1.0, -2.0],
    ///          [-3.0, -4.0],
    ///          [-5.0, -6.0]]
    ///     c = [[1.0, 1.0, 1.0],
    ///          [___, ___, ___],
    ///          [1.0, 1.0, 1.0]]
    ///         (Column-major, sliced by `s![0..3;2, ..]`)
    /// ```
    /// Output of `c` is
    /// ```
    /// [[-3.500,  -9.500, -15.500],
    ///  [ 1.000,   1.000,   1.000],
    ///  [-9.500, -23.500, -37.500]]
    /// ```
    #[test]
    fn demonstration_dgemm_complicated() {
        use blas_array2::blas3::gemm::DGEMM;
        use blas_array2::util::*;
        use ndarray::prelude::*;

        let a = array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
        let b = array![[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]];
        let mut c = Array::ones((3, 3).f());

        let c_out = DGEMM::default()
            .a(a.slice(s![.., ..2]))
            .b(b.view())
            .c(c.slice_mut(s![0..3;2, ..]))
            .transb('T')
            .beta(1.5)
            .run()
            .unwrap();
        // one can get the result as an owned array
        // but the result may not refer to the same memory location as `c`
        println!("{:4.3?}", c_out.into_owned());
        // this modification on `c` is actually performed in-place
        // so if `c` is pre-defined, not calling `into_owned` could be more efficient
        println!("{:4.3?}", c);
    }
}

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
            panic!("Failed");
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
                    panic!("Failed");
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
            panic!("Failed");
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
                let c_assign = alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_naive.slice(c_slc);
                c_naive.slice_mut(c_slc).assign(&c_assign);

                if let ArrayOut2::ViewMut(_) = c_out {
                    let err = (&c_naive - &c_raw).mapv(|x| x.abs()).sum();
                    let acc = c_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=2.0*RT::EPSILON);
                } else {
                    panic!("Failed");
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
