#![cfg(feature = "gemmt")]

use crate::util::*;
use approx::*;
use blas_array2::blas3::gemmt::GEMMT;
use blas_array2::util::*;
use itertools::iproduct;
use ndarray::prelude::*;

#[cfg(test)]
mod valid_own {
    use super::*;

    #[test]
    fn test_example() {
        let uplo = 'U';
        let transa = 'N';
        let transb = 'N';
        let layout = 'C';
        let layout_a = 'C';
        let layout_b = 'C';
        let as0 = 1;
        let as1 = 1;
        let bs0 = 1;
        let bs1 = 1;

        type RT = <f32 as BLASFloat>::RealFloat;
        let alpha = f32::rand();
        let beta = f32::rand();
        let a_raw = random_matrix(100, 100, layout_a.into());
        let b_raw = random_matrix(100, 100, layout_b.into());
        let a_slc = slice(7, 8, as0, as1);
        let b_slc = slice(8, 7, bs0, bs1);

        let c_out = GEMMT::<f32>::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .alpha(alpha)
            .beta(beta)
            .transa(transa)
            .transb(transb)
            .uplo(uplo)
            .layout(layout)
            .run()
            .unwrap();

        let a_naive = transpose(&a_raw.slice(a_slc), transa.try_into().unwrap());
        let b_naive = transpose(&b_raw.slice(b_slc), transb.try_into().unwrap());
        let c_assign = alpha * gemm(&a_naive.view(), &b_naive.view());
        let mut c_naive = Array::zeros(c_assign.dim());
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

    macro_rules! test_macro_own {
        ($type:ty) => {{
            let list_uplo = ['U', 'L'];
            let list_transa = ['N', 'T', 'C'];
            let list_transb = ['N', 'T', 'C'];
            let list_layout = ['C', 'R'];
            let list_layout_a = ['C', 'R'];
            let list_layout_b = ['C', 'R'];
            let list_as0 = [1, 2];
            let list_as1 = [1, 2];
            let list_bs0 = [1, 2];
            let list_bs1 = [1, 2];

            type RT = <$type as BLASFloat>::RealFloat;
            let alpha = <$type>::rand();
            let beta = <$type>::rand();
            let a_buffer = random_array::<$type>(400).to_vec();
            let b_buffer = random_array::<$type>(400).to_vec();

            for cfg in iproduct!(
                list_uplo,
                list_transa,
                list_transb,
                list_layout,
                list_layout_a,
                list_layout_b,
                list_as0,
                list_as1,
                list_bs0,
                list_bs1,
            ) {
                println!("test config {:?}", cfg);
                let (uplo, transa, transb, layout, layout_a, layout_b, as0, as1, bs0, bs1) = cfg;

                let (ad0, ad1) = if transa == 'N' { (3, 5) } else { (5, 3) };
                let (bd0, bd1) = if transb == 'N' { (5, 3) } else { (3, 5) };

                let a_shape =
                    if layout_a == 'C' { (20, 20).strides((1, 20)) } else { (20, 20).strides((20, 1)) };
                let b_shape =
                    if layout_b == 'C' { (20, 20).strides((1, 20)) } else { (20, 20).strides((20, 1)) };
                let a_raw = ArrayView2::from_shape(a_shape, &a_buffer).unwrap();
                let b_raw = ArrayView2::from_shape(b_shape, &b_buffer).unwrap();
                let a_slc = slice(ad0, ad1, as0, as1);
                let b_slc = slice(bd0, bd1, bs0, bs1);

                let a_naive = transpose(&a_raw.slice(a_slc), transa.try_into().unwrap());
                let b_naive = transpose(&b_raw.slice(b_slc), transb.try_into().unwrap());
                let c_assign = gemm(&a_naive.view(), &b_naive.view()) * alpha;
                let mut c_naive = Array::zeros(c_assign.dim());
                tril_assign(&mut c_naive.view_mut(), &c_assign.view(), uplo);

                let c_out = GEMMT::<$type>::default()
                    .a(a_raw.slice(a_slc))
                    .b(b_raw.slice(b_slc))
                    .alpha(alpha)
                    .beta(beta)
                    .transa(transa)
                    .transb(transb)
                    .uplo(uplo)
                    .layout(layout)
                    .run()
                    .unwrap();

                if let ArrayOut2::Owned(c_out) = c_out {
                    let err = (&c_naive - &c_out).mapv(|x| <$type>::abs(x)).sum();
                    let acc = c_naive.mapv(|x| <$type>::abs(x)).sum();
                    let err_div = err / acc;
                    assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
                } else {
                    panic!("Failed");
                }
            }
        }};
    }

    #[test]
    fn test_own() {
        test_macro_own!(f32);
        test_macro_own!(f64);
        test_macro_own!(c32);
        test_macro_own!(c64);
    }
}

#[cfg(test)]
mod valid_view {
    use super::*;

    #[test]
    fn test_example() {
        let uplo = 'U';
        let transa = 'N';
        let transb = 'N';
        let layout = 'C';
        let layout_a = 'C';
        let layout_b = 'C';
        let layout_c = 'C';
        let as0 = 1;
        let as1 = 1;
        let bs0 = 1;
        let bs1 = 1;
        let cs0 = 1;
        let cs1 = 1;

        type RT = <f32 as BLASFloat>::RealFloat;
        let alpha = f32::rand();
        let beta = f32::rand();
        let a_buffer = random_array::<f32>(400).to_vec();
        let b_buffer = random_array::<f32>(400).to_vec();
        let mut c_buffer = random_array::<f32>(400).to_vec();

        let (ad0, ad1) = if transa == 'N' { (3, 5) } else { (5, 3) };
        let (bd0, bd1) = if transb == 'N' { (5, 3) } else { (3, 5) };
        let (cd0, cd1) = if transa == 'N' { (3, 3) } else { (5, 5) };

        let a_shape = if layout_a == 'C' { (20, 20).strides((1, 20)) } else { (20, 20).strides((20, 1)) };
        let b_shape = if layout_b == 'C' { (20, 20).strides((1, 20)) } else { (20, 20).strides((20, 1)) };
        let c_shape = if layout_c == 'C' { (20, 20).strides((1, 20)) } else { (20, 20).strides((20, 1)) };

        let a_raw = ArrayView2::from_shape(a_shape, &a_buffer).unwrap();
        let b_raw = ArrayView2::from_shape(b_shape, &b_buffer).unwrap();
        let mut c_raw = ArrayViewMut2::from_shape(c_shape, &mut c_buffer).unwrap();

        let a_slc = slice(ad0, ad1, as0, as1);
        let b_slc = slice(bd0, bd1, bs0, bs1);
        let c_slc = slice(cd0, cd1, cs0, cs1);

        let a_naive = transpose(&a_raw.slice(a_slc), transa.try_into().unwrap());
        let b_naive = transpose(&b_raw.slice(b_slc), transb.try_into().unwrap());
        let c_assign = alpha * gemm(&a_naive.view(), &b_naive.view()) + beta * &c_raw.slice(c_slc);
        let mut c_naive = c_raw.slice(c_slc).to_owned();
        tril_assign(&mut c_naive.view_mut(), &c_assign.view(), uplo);

        let c_out = GEMMT::<f32>::default()
            .a(a_raw.slice(a_slc))
            .b(b_raw.slice(b_slc))
            .c(c_raw.slice_mut(c_slc))
            .alpha(alpha)
            .beta(beta)
            .transa(transa)
            .transb(transb)
            .uplo(uplo)
            .layout(layout)
            .run()
            .unwrap()
            .into_owned();

        let err = (&c_naive - &c_out).mapv(|x| x.abs()).sum();
        let acc = c_naive.mapv(|x| x.abs()).sum() as RT;
        let err_div = err / acc;
        assert_abs_diff_eq!(err_div, 0.0, epsilon = 4.0 * RT::EPSILON);
    }
}
