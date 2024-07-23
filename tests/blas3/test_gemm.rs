use crate::util::*;
use approx::*;
use blas_array2::blas3::gemm::GEMM;
use blas_array2::util::*;
use cblas_sys::*;
use ndarray::prelude::*;
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
        use blas_array2::prelude::*;
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
        use blas_array2::prelude::*;
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
        type RT = <f32 as TestFloat>::RealFloat;
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
            assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
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
                type RT = <$F as TestFloat>::RealFloat;
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

                let a_naive = transpose(&a_raw.slice(a_slc), $a_trans.try_into().unwrap());
                let b_naive = transpose(&b_raw.slice(b_slc), $b_trans.try_into().unwrap());
                let c_naive = alpha * gemm(&a_naive.view(), &b_naive.view());

                if let ArrayOut2::Owned(c_out) = c_out {
                    let err = (&c_naive - &c_out).mapv(|x| x.abs()).sum();
                    let acc = c_naive.mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon=4.0 * RT::EPSILON);
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
    test_macro!(test_100: should_panic, f32, (7, 8, 1, 1), (9, 8, 1, 3), 'R', 'R', BLASTranspose::ConjNoTrans, 'T');
    test_macro!(test_101: should_panic, f32, (7, 8, 1, 1), (9, 8, 1, 3), 'R', 'R', 'N', BLASTranspose::ConjNoTrans);

    // dimension mismatch (k)
    test_macro!(test_102: should_panic, f32, (7, 5, 1, 1), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    test_macro!(test_103: should_panic, f32, (7, 5, 1, 1), (9, 8, 1, 3), 'R', 'R', 'N', 'T');
}

#[cfg(test)]
mod valid_view {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <f32 as TestFloat>::RealFloat;
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
            assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
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
                type RT = <$F as TestFloat>::RealFloat;
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

                let a_naive = transpose(&a_raw.slice(a_slc), $a_trans.try_into().unwrap());
                let b_naive = transpose(&b_raw.slice(b_slc), $b_trans.try_into().unwrap());
                let c_assign = alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_naive.slice(c_slc);
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

    // zero dimension
    test_macro!(test_099: inline, f32, (0, 8, 1, 1), (8, 9, 1, 1), (0, 9, 1, 1), 'R', 'R', 'R', 'N', 'N');
    test_macro!(test_098: inline, f32, (7, 0, 1, 1), (0, 9, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'N', 'N');
}

#[cfg(test)]
mod valid_cblas {
    use super::*;

    #[test]
    fn test_cblas() {
        // set parameters of test configuration
        type F = c32;
        let a_slc = slice(7, 8, 1, 1);
        let b_slc = slice(8, 9, 1, 1);
        let c_slc = slice(7, 9, 1, 1);
        let a_layout = 'R';
        let b_layout = 'R';
        let c_layout = 'R';
        let transa = 'N';
        let transb = 'N';
        let cblas_layout = 'R';

        // type definition
        type FFI = <F as TestFloat>::FFIFloat;

        // data assignment
        let alpha = F::rand();
        let beta = F::rand();
        let a_raw = random_matrix::<F>(100, 100, a_layout.into());
        let b_raw = random_matrix::<F>(100, 100, b_layout.into());
        let mut c_raw = random_matrix::<F>(100, 100, c_layout.into());
        let mut c_origin = c_raw.clone();

        // cblas computation - mut
        let a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), cblas_layout);
        let b_naive = ndarray_to_layout(b_raw.slice(b_slc).into_owned(), cblas_layout);
        let mut c_naive = ndarray_to_layout(c_raw.slice(c_slc).into_owned(), cblas_layout);
        let (m, n) = c_naive.dim();
        let k = if transa == 'N' { a_naive.len_of(Axis(1)) } else { a_naive.len_of(Axis(0)) };
        let lda = *a_naive.strides().iter().max().unwrap();
        let ldb = *b_naive.strides().iter().max().unwrap();
        let ldc = *c_naive.strides().iter().max().unwrap();
        unsafe {
            cblas_cgemm(
                to_cblas_layout(cblas_layout),
                to_cblas_trans(transa),
                to_cblas_trans(transb),
                m.try_into().unwrap(),
                n.try_into().unwrap(),
                k.try_into().unwrap(),
                [alpha].as_ptr() as *const FFI,
                a_naive.as_ptr() as *const FFI,
                lda.try_into().unwrap(),
                b_naive.as_ptr() as *const FFI,
                ldb.try_into().unwrap(),
                [beta].as_ptr() as *const FFI,
                c_naive.as_mut_ptr() as *mut FFI,
                ldc.try_into().unwrap(),
            );
        }

        let c_out = GEMM::<F>::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .c(c_raw.slice_mut(c_slc))
            .alpha(alpha)
            .beta(beta)
            .transa(transa)
            .transb(transb)
            .run()
            .unwrap()
            .into_owned();

        check_same(&c_out.view(), &c_naive.view(), 4.0 * F::EPSILON);
        check_same(&c_raw.slice(c_slc), &c_naive.view(), 4.0 * F::EPSILON);
        c_raw.slice_mut(c_slc).fill(F::from(0.0));
        c_origin.slice_mut(c_slc).fill(F::from(0.0));
        check_same(&c_raw.view(), &c_origin.view(), 4.0 * F::EPSILON);

        // cblas computation - own
        c_naive.fill(F::from(0.0));
        unsafe {
            cblas_cgemm(
                to_cblas_layout(cblas_layout),
                to_cblas_trans(transa),
                to_cblas_trans(transb),
                m.try_into().unwrap(),
                n.try_into().unwrap(),
                k.try_into().unwrap(),
                [alpha].as_ptr() as *const FFI,
                a_naive.as_ptr() as *const FFI,
                lda.try_into().unwrap(),
                b_naive.as_ptr() as *const FFI,
                ldb.try_into().unwrap(),
                [beta].as_ptr() as *const FFI,
                c_naive.as_mut_ptr() as *mut FFI,
                ldc.try_into().unwrap(),
            );
        }

        let c_out = GEMM::<F>::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .alpha(alpha)
            .beta(beta)
            .transa(transa)
            .transb(transb)
            .run()
            .unwrap()
            .into_owned();
        check_same(&c_out.view(), &c_naive.view(), 4.0 * F::EPSILON);
    }

    macro_rules! test_macro {
        (
            $test_name: ident: $attr: ident,
            $F:ty, $cblas_func: ident,
            ($($a_slc: expr),+), ($($b_slc: expr),+), ($($c_slc: expr),+),
            $a_layout: expr, $b_layout: expr, $c_layout: expr,
            $a_trans: expr, $b_trans: expr, $cblas_layout: expr
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                // set parameters of test configuration
                type F = $F;
                let a_slc = slice($($a_slc),+);
                let b_slc = slice($($b_slc),+);
                let c_slc = slice($($c_slc),+);
                let a_layout = $a_layout;
                let b_layout = $b_layout;
                let c_layout = $c_layout;
                let transa = $a_trans;
                let transb = $b_trans;
                let cblas_layout = $cblas_layout;

                // type definition
                type FFI = <F as TestFloat>::FFIFloat;

                // data assignment
                let alpha = F::rand();
                let beta = F::rand();
                let a_raw = random_matrix::<F>(100, 100, a_layout.into());
                let b_raw = random_matrix::<F>(100, 100, b_layout.into());
                let mut c_raw = random_matrix::<F>(100, 100, c_layout.into());
                let mut c_origin = c_raw.clone();

                // cblas computation - mut
                let a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), cblas_layout);
                let b_naive = ndarray_to_layout(b_raw.slice(b_slc).into_owned(), cblas_layout);
                let mut c_naive = ndarray_to_layout(c_raw.slice(c_slc).into_owned(), cblas_layout);
                let (m, n) = c_naive.dim();
                let k = if transa == 'N' { a_naive.len_of(Axis(1)) } else { a_naive.len_of(Axis(0)) };
                let lda = *a_naive.strides().iter().max().unwrap();
                let ldb = *b_naive.strides().iter().max().unwrap();
                let ldc = *c_naive.strides().iter().max().unwrap();
                unsafe {
                    $cblas_func(
                        to_cblas_layout(cblas_layout),
                        to_cblas_trans(transa),
                        to_cblas_trans(transb),
                        m.try_into().unwrap(),
                        n.try_into().unwrap(),
                        k.try_into().unwrap(),
                        [alpha].as_ptr() as *const FFI,
                        a_naive.as_ptr() as *const FFI,
                        lda.try_into().unwrap(),
                        b_naive.as_ptr() as *const FFI,
                        ldb.try_into().unwrap(),
                        [beta].as_ptr() as *const FFI,
                        c_naive.as_mut_ptr() as *mut FFI,
                        ldc.try_into().unwrap(),
                    );
                }

                let c_out = GEMM::<F>::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .c(c_raw.slice_mut(c_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .transa(transa)
                    .transb(transb)
                    .run()
                    .unwrap()
                    .into_owned();

                check_same(&c_out.view(), &c_naive.view(), 4.0 * F::EPSILON);
                check_same(&c_raw.slice(c_slc), &c_naive.view(), 4.0 * F::EPSILON);
                c_raw.slice_mut(c_slc).fill(F::from(0.0));
                c_origin.slice_mut(c_slc).fill(F::from(0.0));
                check_same(&c_raw.view(), &c_origin.view(), 4.0 * F::EPSILON);

                // cblas computation - own
                c_naive.fill(F::from(0.0));
                unsafe {
                    $cblas_func(
                        to_cblas_layout(cblas_layout),
                        to_cblas_trans(transa),
                        to_cblas_trans(transb),
                        m.try_into().unwrap(),
                        n.try_into().unwrap(),
                        k.try_into().unwrap(),
                        [alpha].as_ptr() as *const FFI,
                        a_naive.as_ptr() as *const FFI,
                        lda.try_into().unwrap(),
                        b_naive.as_ptr() as *const FFI,
                        ldb.try_into().unwrap(),
                        [beta].as_ptr() as *const FFI,
                        c_naive.as_mut_ptr() as *mut FFI,
                        ldc.try_into().unwrap(),
                    );
                }

                let c_out = GEMM::<F>::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .transa(transa)
                    .transb(transb)
                    .run()
                    .unwrap()
                    .into_owned();
                check_same(&c_out.view(), &c_naive.view(), 4.0 * F::EPSILON);
            }
        };
    }

    test_macro!(test_000: inline, c32, cblas_cgemm, (7, 8, 1, 1), (8, 9, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'N', 'N', 'R');
    test_macro!(test_001: inline, c32, cblas_cgemm, (7, 8, 1, 1), (8, 9, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'N', 'N', 'R');
    test_macro!(test_002: inline, c32, cblas_cgemm, (7, 8, 1, 1), (8, 9, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'N', 'N', 'C');
    test_macro!(test_003: inline, c32, cblas_cgemm, (7, 8, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'N', 'T', 'R');
    test_macro!(test_004: inline, c32, cblas_cgemm, (8, 7, 1, 1), (8, 9, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'T', 'N', 'C');
    test_macro!(test_005: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'T', 'T', 'R');
    test_macro!(test_006: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'T', 'C', 'R');
    test_macro!(test_007: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'T', 'C', 'R');
    test_macro!(test_008: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'C', 'T', 'C');
    test_macro!(test_009: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'C', 'T', 'C');
    test_macro!(test_010: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'C', 'C', 'C');
    test_macro!(test_011: inline, c32, cblas_cgemm, (8, 7, 1, 1), (9, 8, 1, 1), (7, 9, 1, 1), 'C', 'C', 'C', 'C', 'C', 'C');
}
