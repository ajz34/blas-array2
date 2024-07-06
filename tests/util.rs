use blas_array2::util::*;
use ndarray::{prelude::*, SliceInfo, SliceInfoElem};
use num_complex::ComplexFloat;
use rand::{thread_rng, Rng};

/* #region Random matrix */

pub trait RandomNumber<F> {
    fn rand() -> F;
}

impl RandomNumber<f32> for f32 {
    fn rand() -> f32 {
        thread_rng().gen()
    }
}

impl RandomNumber<f64> for f64 {
    fn rand() -> f64 {
        thread_rng().gen()
    }
}

impl RandomNumber<c32> for c32 {
    fn rand() -> c32 {
        let re = thread_rng().gen();
        let im = thread_rng().gen();
        c32::new(re, im)
    }
}

impl RandomNumber<c64> for c64 {
    fn rand() -> c64 {
        let re = thread_rng().gen();
        let im = thread_rng().gen();
        c64::new(re, im)
    }
}

pub fn random_matrix<F>(row: usize, col: usize, layout: BLASLayout) -> Array2<F>
where
    F: RandomNumber<F> + BLASFloat
{
    let mut matrix = match layout {
        BLASLayout::RowMajor => Array2::zeros((row, col)),
        BLASLayout::ColMajor => Array2::zeros((row, col).f()),
        _ => panic!("Invalid layout"),
    };
    for x in matrix.iter_mut() {
        *x = F::rand();
    }
    return matrix;
}

/* #endregion */

/* #region Sized subatrix */

pub fn slice(nrow: usize, ncol: usize, srow: usize, scol: usize) -> SliceInfo<[SliceInfoElem; 2], Ix2, Ix2> {
    s![5..(5+nrow*srow);srow, 10..(10+ncol*scol);scol]
}

/* #endregion */

/* #region Basic matrix operations */

pub fn gemm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> Array2<F>
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
    return c;
}

pub fn transpose<F>(a: &ArrayView2<F>, trans: BLASTrans) -> Array2<F>
where 
    F: BLASFloat
{
    match trans {
        BLASTrans::NoTrans => a.into_owned(),
        BLASTrans::Trans => a.t().into_owned(),
        BLASTrans::ConjNoTrans => match F::is_complex() {
            true => {
                let a = a.mapv(|x| F::conj(x));
                a.into_owned()
            },
            false => a.into_owned(),
        }
        BLASTrans::ConjTrans => match F::is_complex() {
            true => {
                let a = a.t().into_owned();
                a.mapv(|x| F::conj(x))
            },
            false => a.t().into_owned(),
        }
        _ => panic!("Invalid BLASTrans"),
    }
}

pub fn symmetrize<F>(a: &ArrayView2<F>, uplo: char) -> Array2<F>
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
    return a;
}

pub fn hermitianize<F>(a: &ArrayView2<F>, uplo: char) -> Array2<F>
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
    return a;
}

pub fn tril_assign<F>(c: &mut ArrayViewMut2<F>, a: &ArrayView2<F>, uplo: char)
where 
    F: BLASFloat
{
    if uplo == 'L' {
        for i in 0..a.dim().0 {
            for j in 0..=i {
                c[[i, j]] = a[[i, j]];
            }
        }
    } else if uplo == 'U' {
        for i in 0..a.dim().0 {
            for j in i..a.dim().1 {
                c[[i, j]] = a[[i, j]];
            }
        }
    }
}

/* #endregion */
