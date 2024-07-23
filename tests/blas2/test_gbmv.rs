use crate::util::*;
use approx::*;
use blas_array2::blas2::gbmv::GBMV;
use blas_array2::util::*;
use cblas_sys::*;
use itertools::iproduct;
use ndarray::prelude::*;
use num_complex::*;

#[cfg(test)]
mod valid_col_major {
    use super::*;

    #[test]
    fn test_example() {
        type RT = <f32 as BLASFloat>::RealFloat;
        let alpha = <f32>::rand();
        let beta = <f32>::rand();
        let m = 10;
        let n = 8;
        let kl = 4;
        let ku = 2;
        let trans = 'T';
        let a_raw = random_matrix(100, 100, 'R'.into());
        let x_raw = random_array(100);
        let mut y_raw = random_array(100);

        let a_slc = slice(kl + ku + 1, n, 1, 1);
        let x_slc = slice_1d(if trans == 'N' { n } else { m }, 3);
        let y_slc = slice_1d(if trans == 'N' { m } else { n }, 3);

        let mut a_naive = Array2::<f32>::zeros((m, n));
        for j in 0..n {
            let k = ku as isize - j as isize;
            for i in (if j > ku { j - ku } else { 0 })..core::cmp::min(m, j + kl + 1) {
                a_naive[[i as usize, j]] = a_raw.slice(a_slc)[[(k + i as isize) as usize, j]];
            }
        }

        let a_naive = transpose(&a_naive.view(), trans.try_into().unwrap());
        let x_naive = x_raw.slice(x_slc).into_owned();
        let mut y_naive = y_raw.clone();
        let y_bare = alpha * gemv(&a_naive.view(), &x_naive.view());
        let y_assign = &y_bare + beta * &y_naive.slice(&y_slc);
        y_naive.slice_mut(y_slc).assign(&y_assign);

        // mut_view
        let y_out = GBMV::default()
            .a(a_raw.slice(a_slc))
            .x(x_raw.slice(x_slc))
            .y(y_raw.slice_mut(y_slc))
            .m(m)
            .kl(kl)
            .layout('C')
            .trans(trans)
            .alpha(alpha)
            .beta(beta)
            .run()
            .unwrap();
        if let ArrayOut1::ViewMut(_) = y_out {
            let err = (&y_naive - &y_raw).mapv(|x| x.abs()).sum();
            let acc = y_naive.view().mapv(|x| x.abs()).sum() as RT;
            let err_div = err / acc;
            assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
        } else {
            panic!("Failed");
        }

        // owned
        let y_out = GBMV::default()
            .a(a_raw.slice(a_slc))
            .x(x_raw.slice(x_slc))
            .m(m)
            .kl(kl)
            .layout('C')
            .trans(trans)
            .alpha(alpha)
            .beta(beta)
            .run()
            .unwrap();
        if let ArrayOut1::Owned(y_out) = y_out {
            let err = (&y_bare - &y_out).mapv(|x| x.abs()).sum();
            let acc = y_bare.view().mapv(|x| x.abs()).sum() as RT;
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
            ($($a_slc: expr),+), ($($x_slc: expr),+), ($($y_slc: expr),+),
            $a_layout: expr,
            $trans: expr
        ) => {
            #[test]
            #[$attr]
            fn $test_name() {
                type RT = <$F as BLASFloat>::RealFloat;
                let alpha = <$F>::rand();
                let beta = <$F>::rand();
                let m = 10;
                let n = 8;
                let kl = 4;
                let ku = 2;
                let trans = $trans;
                let a_raw = random_matrix(100, 100, $a_layout.into());
                let x_raw = random_array(100);
                let mut y_raw = random_array(100);

                let a_slc = slice($($a_slc),+);
                let x_slc = slice_1d($($x_slc),+);
                let y_slc = slice_1d($($y_slc),+);

                let mut a_naive = Array2::zeros((m, n));
                for j in 0..n {
                    let k = ku as isize - j as isize;
                    for i in (if j > ku { j - ku } else { 0 })..core::cmp::min(m, j + kl + 1) {
                        a_naive[[i as usize, j]] = a_raw.slice(a_slc)[[(k + i as isize) as usize, j]];
                    }
                }

                let a_naive = transpose(&a_naive.view(), trans.try_into().unwrap());
                let x_naive = x_raw.slice(x_slc).into_owned();
                let mut y_naive = y_raw.clone();
                let y_bare = alpha * gemv(&a_naive.view(), &x_naive.view());
                let y_assign = &y_bare + beta * &y_naive.slice(&y_slc);
                y_naive.slice_mut(y_slc).assign(&y_assign);

                // mut_view
                let y_out = GBMV::default()
                    .a(a_raw.slice(a_slc))
                    .x(x_raw.slice(x_slc))
                    .y(y_raw.slice_mut(y_slc))
                    .m(m)
                    .kl(kl)
                    .layout('C')
                    .trans(trans)
                    .alpha(alpha)
                    .beta(beta)
                    .run()
                    .unwrap();
                if let ArrayOut1::ViewMut(_) = y_out {
                    let err = (&y_naive - &y_raw).mapv(|x| x.abs()).sum();
                    let acc = y_naive.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }

                // owned
                let y_out = GBMV::default()
                    .a(a_raw.slice(a_slc))
                    .x(x_raw.slice(x_slc))
                    .m(m)
                    .kl(kl)
                    .layout('C')
                    .trans(trans)
                    .alpha(alpha)
                    .beta(beta)
                    .run()
                    .unwrap();
                if let ArrayOut1::Owned(y_out) = y_out {
                    let err = (&y_bare - &y_out).mapv(|x| x.abs()).sum();
                    let acc = y_bare.view().mapv(|x| x.abs()).sum() as RT;
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        };
    }

    test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 1), (10, 1), 'R', 'N');
    test_macro!(test_001: inline, f32, (7, 8, 1, 1), (8, 1), (10, 1), 'C', 'N');
    test_macro!(test_002: inline, f32, (7, 8, 1, 1), (10, 3), (8, 3), 'R', 'T');
    test_macro!(test_003: inline, f32, (7, 8, 3, 3), (10, 1), (8, 3), 'C', 'T');
    test_macro!(test_004: inline, f32, (7, 8, 3, 3), (10, 3), (8, 1), 'R', 'C');
    test_macro!(test_005: inline, f32, (7, 8, 3, 3), (10, 3), (8, 3), 'C', 'C');
    test_macro!(test_006: inline, f64, (7, 8, 1, 1), (10, 1), (8, 3), 'C', 'C');
    test_macro!(test_007: inline, f64, (7, 8, 1, 3), (10, 1), (8, 1), 'C', 'T');
    test_macro!(test_008: inline, f64, (7, 8, 1, 3), (10, 3), (8, 1), 'R', 'T');
    test_macro!(test_009: inline, f64, (7, 8, 3, 1), (10, 1), (8, 1), 'R', 'C');
    test_macro!(test_010: inline, f64, (7, 8, 3, 1), (8, 3), (10, 3), 'R', 'N');
    test_macro!(test_011: inline, f64, (7, 8, 3, 3), (8, 3), (10, 3), 'C', 'N');
    test_macro!(test_012: inline, c32, (7, 8, 1, 1), (10, 3), (8, 1), 'C', 'C');
    test_macro!(test_013: inline, c32, (7, 8, 1, 3), (10, 1), (8, 3), 'C', 'C');
    test_macro!(test_014: inline, c32, (7, 8, 1, 3), (8, 3), (10, 3), 'R', 'N');
    test_macro!(test_015: inline, c32, (7, 8, 3, 1), (10, 1), (8, 3), 'R', 'T');
    test_macro!(test_016: inline, c32, (7, 8, 3, 1), (8, 3), (10, 1), 'C', 'N');
    test_macro!(test_017: inline, c32, (7, 8, 3, 3), (10, 1), (8, 1), 'R', 'T');
    test_macro!(test_018: inline, c64, (7, 8, 1, 1), (10, 3), (8, 3), 'C', 'T');
    test_macro!(test_019: inline, c64, (7, 8, 1, 3), (8, 1), (10, 3), 'R', 'N');
    test_macro!(test_020: inline, c64, (7, 8, 1, 3), (10, 3), (8, 1), 'R', 'C');
    test_macro!(test_021: inline, c64, (7, 8, 3, 1), (10, 1), (8, 3), 'R', 'C');
    test_macro!(test_022: inline, c64, (7, 8, 3, 1), (10, 3), (8, 1), 'C', 'T');
    test_macro!(test_023: inline, c64, (7, 8, 3, 3), (8, 1), (10, 1), 'C', 'N');
}

#[cfg(test)]
mod valid_row_major {

    use super::*;

    #[test]
    fn test_cblas_row_major() {
        let cblas_layout = 'R';

        // set parameters of test configuration
        type F = c32;
        for (a_layout, trans) in iproduct!(['R', 'C'], ['N', 'T', 'C']) {
            let m = 10;
            let n = 8;
            let ku = 3;
            let kl = 2;
            let k = ku + kl + 1;

            // slice definition
            let a_slc = slice(n, k, 3, 3);
            let x_slc = slice_1d(if trans == 'N' { m } else { n }, 3);
            let y_slc = slice_1d(if trans == 'N' { n } else { m }, 3);

            // type definition
            type FFI = <F as BLASFloat>::FFIFloat;

            // data assignment
            let alpha = F::rand();
            let beta = F::rand();
            let a_raw = random_matrix::<F>(100, 100, a_layout.into());
            let x_raw = random_array::<F>(1000);
            let mut y_raw = random_array::<F>(1000);
            let mut y_origin = y_raw.clone();

            // cblas computation - mut
            let a_naive = ndarray_to_layout(a_raw.slice(a_slc).into_owned(), cblas_layout);
            let x_naive = x_raw.slice(x_slc).into_owned();
            let mut y_naive = y_raw.slice_mut(y_slc).into_owned();
            let lda = *a_naive.strides().iter().max().unwrap();
            let incx = 1;
            let incy = 1;
            unsafe {
                cblas_cgbmv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_trans(trans),
                    n.try_into().unwrap(),
                    m.try_into().unwrap(),
                    kl.try_into().unwrap(),
                    ku.try_into().unwrap(),
                    [alpha].as_ptr() as *const FFI,
                    a_naive.as_ptr() as *const FFI,
                    lda.try_into().unwrap(),
                    x_naive.as_ptr() as *const FFI,
                    incx.try_into().unwrap(),
                    [beta].as_ptr() as *const FFI,
                    y_naive.as_mut_ptr() as *mut FFI,
                    incy.try_into().unwrap(),
                );
            }

            let y_out = GBMV::<F>::default()
                .a(a_raw.slice(a_slc))
                .x(x_raw.slice(x_slc))
                .y(y_raw.slice_mut(y_slc))
                .m(m)
                .kl(kl)
                .alpha(alpha)
                .beta(beta)
                .trans(trans)
                .layout(cblas_layout)
                .run()
                .unwrap()
                .into_owned();

            check_same(&y_out.view(), &y_naive.view(), 4.0 * F::EPSILON);
            check_same(&y_raw.slice(y_slc), &y_naive.view(), 4.0 * F::EPSILON);
            y_raw.slice_mut(y_slc).fill(F::from(0.0));
            y_origin.slice_mut(y_slc).fill(F::from(0.0));
            check_same(&y_raw.view(), &y_origin.view(), 4.0 * F::EPSILON);
        }
    }
}
