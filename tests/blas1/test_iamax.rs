use crate::util::*;
use blas_array2::blas1::iamax::IAMAX;
use blas_array2::util::*;

#[cfg(test)]
mod valid {
    use super::*;

    #[test]
    fn test_example() {
        for incx in [1, 2] {
            let n = 100;
            let x = random_array::<f64>(1000);
            let x_slc = slice_1d(n, incx);
            let out = IAMAX::default().x(x.slice(x_slc)).run().unwrap();
            let expected = x
                .slice(x_slc)
                .mapv(f64::abs)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(out, expected);
        }

        for incx in [1, 2] {
            let n = 100;
            let x = random_array::<c64>(1000);
            let x_slc = slice_1d(n, incx);
            let out = IAMAX::default().x(x.slice(x_slc)).run().unwrap();
            let expected = x
                .slice(x_slc)
                .mapv(|v| v.re.abs() + v.im.abs())
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(out, expected);
        }
    }
}
