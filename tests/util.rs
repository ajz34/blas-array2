use approx::*;
use blas_array2::util::*;
use cblas_sys::*;
use ndarray::{prelude::*, SliceInfo, SliceInfoElem};
use num_complex::*;
use num_traits::*;
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
    F: RandomNumber<F> + BLASFloat,
{
    let mut matrix = match layout {
        BLASRowMajor => Array2::zeros((row, col)),
        BLASColMajor => Array2::zeros((row, col).f()),
        _ => panic!("Invalid layout"),
    };
    for x in matrix.iter_mut() {
        *x = F::rand();
    }
    return matrix;
}

pub fn random_array<F>(size: usize) -> Array1<F>
where
    F: RandomNumber<F> + BLASFloat,
{
    let mut array = Array1::zeros(size);
    for x in array.iter_mut() {
        *x = F::rand();
    }
    return array;
}

/* #endregion */

/* #region Sized subatrix */

pub fn slice(nrow: usize, ncol: usize, srow: usize, scol: usize) -> SliceInfo<[SliceInfoElem; 2], Ix2, Ix2> {
    s![5..(5+nrow*srow);srow, 10..(10+ncol*scol);scol]
}

pub fn slice_1d(n: usize, s: usize) -> SliceInfo<[SliceInfoElem; 1], Ix1, Ix1> {
    s![50..(50+n*s);s]
}

/* #endregion */

/* #region Basic matrix operations */

pub fn gemm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> Array2<F>
where
    F: BLASFloat,
{
    let (m, k) = a.dim();
    let n = b.len_of(Axis(1));
    assert_eq!(b.len_of(Axis(0)), k);
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

pub fn gemv<F>(a: &ArrayView2<F>, x: &ArrayView1<F>) -> Array1<F>
where
    F: BLASFloat,
{
    let (m, n) = a.dim();
    assert_eq!(x.len(), n);
    let mut y = Array1::<F>::zeros(m);
    for i in 0..m {
        let mut sum = F::zero();
        for j in 0..n {
            sum += a[[i, j]] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

pub fn transpose<F>(a: &ArrayView2<F>, trans: BLASTranspose) -> Array2<F>
where
    F: BLASFloat,
{
    match trans {
        BLASNoTrans => a.into_owned(),
        BLASTrans => a.t().into_owned(),
        BLASTranspose::ConjNoTrans => match F::is_complex() {
            true => {
                let a = a.mapv(|x| F::conj(x));
                a.into_owned()
            },
            false => a.into_owned(),
        },
        BLASConjTrans => match F::is_complex() {
            true => {
                let a = a.t().into_owned();
                a.mapv(|x| F::conj(x))
            },
            false => a.t().into_owned(),
        },
        _ => panic!("Invalid BLASTrans"),
    }
}

pub fn symmetrize<F>(a: &ArrayView2<F>, uplo: char) -> Array2<F>
where
    F: BLASFloat,
{
    let mut a = a.into_owned();
    if uplo == 'L' {
        for i in 0..a.len_of(Axis(0)) {
            for j in 0..i {
                a[[j, i]] = a[[i, j]];
            }
        }
    } else if uplo == 'U' {
        for i in 0..a.len_of(Axis(0)) {
            for j in i + 1..a.len_of(Axis(1)) {
                a[[j, i]] = a[[i, j]];
            }
        }
    }
    return a;
}

pub fn hermitianize<F>(a: &ArrayView2<F>, uplo: char) -> Array2<F>
where
    F: BLASFloat + ComplexFloat + From<<F as ComplexFloat>::Real>,
{
    let mut a = a.into_owned();
    if uplo == 'L' {
        for i in 0..a.len_of(Axis(0)) {
            a[[i, i]] = a[[i, i]].re().into();
            for j in 0..i {
                a[[j, i]] = a[[i, j]].conj();
            }
        }
    } else if uplo == 'U' {
        for i in 0..a.len_of(Axis(0)) {
            a[[i, i]] = a[[i, i]].re().into();
            for j in i + 1..a.len_of(Axis(1)) {
                a[[j, i]] = a[[i, j]].conj();
            }
        }
    }
    return a;
}

pub fn tril_assign<F>(c: &mut ArrayViewMut2<F>, a: &ArrayView2<F>, uplo: char)
where
    F: BLASFloat,
{
    if uplo == 'L' {
        for i in 0..a.len_of(Axis(0)) {
            for j in 0..=i {
                c[[i, j]] = a[[i, j]];
            }
        }
    } else if uplo == 'U' {
        for i in 0..a.len_of(Axis(0)) {
            for j in i..a.len_of(Axis(1)) {
                c[[i, j]] = a[[i, j]];
            }
        }
    }
}

pub fn unpack_tril<F>(ap: &ArrayView1<F>, a: &mut ArrayViewMut2<F>, layout: char, uplo: char)
where
    F: BLASFloat,
{
    let n = a.len_of(Axis(0));
    if layout == 'R' {
        let mut k = 0;
        if uplo == 'L' {
            for i in 0..n {
                for j in 0..=i {
                    a[[i, j]] = ap[k];
                    k += 1;
                }
            }
        } else if uplo == 'U' {
            for i in 0..n {
                for j in i..n {
                    a[[i, j]] = ap[k];
                    k += 1;
                }
            }
        }
    } else if layout == 'C' {
        let mut k = 0;
        if uplo == 'U' {
            for j in 0..n {
                for i in 0..=j {
                    a[[i, j]] = ap[k];
                    k += 1;
                }
            }
        } else if uplo == 'L' {
            for j in 0..n {
                for i in j..n {
                    a[[i, j]] = ap[k];
                    k += 1;
                }
            }
        }
    }
}

pub fn check_same<F, D>(a: &ArrayView<F, D>, b: &ArrayView<F, D>, eps: <F::RealFloat as AbsDiffEq>::Epsilon)
where
    F: BLASFloat,
    D: Dimension,
    <F as BLASFloat>::RealFloat: approx::AbsDiffEq,
{
    let err: F::RealFloat = (a - b).mapv(F::abs).sum();
    let acc: F::RealFloat = a.mapv(F::abs).sum();
    let err_div = err / acc;
    assert_abs_diff_eq!(err_div, F::RealFloat::zero(), epsilon = eps);
}

/* #endregion */

/* #region array alignment */

pub fn ndarray_to_colmajor<A, D>(arr: Array<A, D>) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    let arr = arr.reversed_axes(); // data not copied
    if arr.is_standard_layout() {
        // arr is f-contiguous = reversed arr is c-contiguous
        // CowArray `into_owned` will not copy if own data, but will copy if it represents view
        // So, though `arr.as_standard_layout().reversed_axes().into_owned()` works, it clones data instead of move it
        return arr.reversed_axes(); // data not copied
    } else {
        // arr is not f-contiguous
        // make reversed arr c-contiguous, then reverse arr again
        return arr.as_standard_layout().reversed_axes().into_owned();
    }
}

pub fn ndarray_to_rowmajor<A, D>(arr: Array<A, D>) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    if arr.is_standard_layout() {
        return arr;
    } else {
        return arr.as_standard_layout().into_owned();
    }
}

pub fn ndarray_to_layout<A, D>(arr: Array<A, D>, layout: char) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    match layout {
        'R' => ndarray_to_rowmajor(arr),
        'C' => ndarray_to_colmajor(arr),
        _ => panic!("invalid layout"),
    }
}

/* #endregion */

/* #region cblas enums */

pub fn to_cblas_layout(layout: char) -> CBLAS_LAYOUT {
    match layout {
        'R' => CblasRowMajor,
        'C' => CblasColMajor,
        _ => panic!("Invalid layout"),
    }
}

pub fn to_cblas_trans(trans: char) -> CBLAS_TRANSPOSE {
    match trans {
        'N' => CblasNoTrans,
        'T' => CblasTrans,
        'C' => CblasConjTrans,
        _ => panic!("Invalid trans"),
    }
}

pub fn to_cblas_uplo(uplo: char) -> CBLAS_UPLO {
    match uplo {
        'L' => CblasLower,
        'U' => CblasUpper,
        _ => panic!("Invalid uplo"),
    }
}

pub fn to_cblas_side(side: char) -> CBLAS_SIDE {
    match side {
        'L' => CblasLeft,
        'R' => CblasRight,
        _ => panic!("Invalid side"),
    }
}

pub fn to_cblas_diag(diag: char) -> CBLAS_DIAG {
    match diag {
        'N' => CblasNonUnit,
        'U' => CblasUnit,
        _ => panic!("Invalid diag"),
    }
}

/* #endregion */
