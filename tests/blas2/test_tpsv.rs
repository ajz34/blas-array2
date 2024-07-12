use crate::util::*;
use blas_array2::blas2::tpsv::TPSV;
use blas_array2::prelude::*;
use cblas_sys::*;
use itertools::*;

#[cfg(test)]
mod valid_row_major {
    use super::*;

    #[test]
    fn test_cblas_row_major() {
        type F = c32;
        for (cblas_layout, uplo, trans, diag, stride_a, stride_x) in
            iproduct!(['R', 'C'], ['U', 'L'], ['N', 'T', 'C'], ['U', 'N'], [1, 3], [1, 3])
        {
            println!("test info: {:?}", (cblas_layout, uplo, trans, diag, stride_a, stride_x));
            let n = 8;
            let np = n * (n + 1) / 2;

            // slice definition
            let a_slc = slice_1d(np, stride_a);
            let x_slc = slice_1d(n, stride_x);

            // type definition
            type FFI = <F as BLASFloat>::FFIFloat;

            // data assignment
            let a_raw = random_array(1000);
            let mut x_raw = random_array(1000);
            let mut x_origin = x_raw.clone();

            // cblas computation
            let a_naive = a_raw.slice(a_slc).into_owned();
            let mut x_naive = x_raw.slice_mut(x_slc).into_owned();
            let incx = 1;
            unsafe {
                cblas_ctpsv(
                    to_cblas_layout(cblas_layout),
                    to_cblas_uplo(uplo),
                    to_cblas_trans(trans),
                    to_cblas_diag(diag),
                    n.try_into().unwrap(),
                    a_naive.as_ptr() as *const FFI,
                    x_naive.as_mut_ptr() as *mut FFI,
                    incx.try_into().unwrap(),
                );
            }

            TPSV::<F>::default()
                .ap(a_raw.slice(a_slc))
                .x(x_raw.slice_mut(x_slc))
                .uplo(uplo)
                .trans(trans)
                .diag(diag)
                .layout(cblas_layout)
                .run()
                .unwrap();

            check_same(&x_raw.slice(x_slc), &x_naive.view(), 4.0 * F::EPSILON);
            x_raw.slice_mut(x_slc).fill(F::from(0.0));
            x_origin.slice_mut(x_slc).fill(F::from(0.0));
            check_same(&x_raw.view(), &x_origin.view(), 4.0 * F::EPSILON);
        }
    }
}
