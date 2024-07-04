use blas_array2::util::BLASFloat;
use ndarray::prelude::*;
use num_complex::ComplexFloat;

pub fn gemm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>)
where 
    F: BLASFloat
{
    let (m, k) = a.dim();
    let n = b.dim().1;
    assert_eq!(b.dim().0, k);
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = F::zero();
            for l in 0..k {
                sum += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = sum;
        }
    }
}

pub fn symmetrize<F>(a: &ArrayView2<F>, uplo: char)
where 
    F: BLASFloat
{
    let mut a = a.into_owned();
    if uplo == 'L' {
        for i in 0..a.dim().0 {
            for j in 0..i {
                a[[j, i]] = a[[i, j]];
            }
        }
    } else if uplo == 'U' {
        for i in 0..a.dim().0 {
            for j in i+1..a.dim().1 {
                a[[j, i]] = a[[i, j]];
            }
        }
    }
}

pub fn hermitianize<F>(a: &ArrayView2<F>, uplo: char)
where 
    F: BLASFloat + ComplexFloat + From<<F as ComplexFloat>::Real>
{
    let mut a = a.into_owned();
    if uplo == 'L' {
        for i in 0..a.dim().0 {
            a[[i, i]] = a[[i, i]].re().into();
            for j in 0..i {
                a[[j, i]] = a[[i, j]].conj();
            }
        }
    } else if uplo == 'U' {
        for i in 0..a.dim().0 {
            a[[i, i]] = a[[i, i]].re().into();
            for j in i+1..a.dim().1 {
                a[[j, i]] = a[[i, j]].conj();
            }
        }
    }
}
