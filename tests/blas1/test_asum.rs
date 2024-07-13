use crate::util::*;
use approx::*;
use blas_array2::blas1::asum::ASUM;
use blas_array2::util::*;

#[cfg(test)]
mod valid {
    use super::*;

    #[test]
    fn test_example() {
        for incx in [1, 2] {
            let n = 100;
            let x = random_array(1000);
            let x_slc = slice_1d(n, incx);
            let out = ASUM::default().x(x.slice(x_slc)).run().unwrap();
            let expected = x.slice(x_slc).view().mapv(f64::abs).sum();
            assert_relative_eq!(out, expected, epsilon = 1.0e-6);
        }

        for incx in [1, 2] {
            let n = 100;
            let x = random_array::<c64>(1000);
            let x_slc = slice_1d(n, incx);
            let out = ASUM::default().x(x.slice(x_slc)).run().unwrap();
            let expected = x.slice(x_slc).view().mapv(|v| v.re.abs() + v.im.abs()).sum();
            assert_relative_eq!(out, expected, epsilon = 1.0e-6);
        }
    }
}
