#[cfg(test)]
mod test_gemm {
    use blas_array2::blas3::symm::*;
    use blas_array2::util::*;
    use ndarray::prelude::*;
    use num_complex::*;

    #[test]
    fn lu_f_contiguous() {
        let arr = Array1::<f32>::linspace(1.0, 120.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f32>::linspace(2.0, 240.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let b = Array2::from_shape_vec((12, 10).f(), arr.to_vec()).unwrap();

        let a = a.slice(s![1..6, 3..8]);
        let b = b.slice(s![2..7, 0..9]);

        // output not defined
        let c = CSYMM::default().a(a.view()).b(b.view()).side('L').uplo('U').run().unwrap().into_owned();
        let c_naive = naive_symm(&a.view(), &b.view(), 'L', 'U');
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);
        
        // output defined, c-contiguous
        let mut c = Array2::zeros((20, 40));
        let c_view = c.slice_mut(s![6..11, 23..32]);
        CSYMM::default().a(a.view()).b(b.view()).c(c_view).side('L').uplo('U').run().unwrap();
        let c_naive = naive_symm(&a.view(), &b.view(), 'L', 'U');
        let c_view = c.slice(s![6..11, 23..32]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
        
        // output defined, arbitary slice
        let mut c = Array2::zeros((20, 40));
        let c_view = c.slice_mut(s![6..16;2, 5..32;3]);
        CSYMM::default().a(a.view()).b(b.view()).c(c_view).side('L').uplo('U').run().unwrap();
        let c_naive = naive_symm(&a.view(), &b.view(), 'L', 'U');
        let c_view = c.slice(s![6..16;2, 5..32;3]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    #[test]
    fn ll_c_contiguous() {
        let arr = Array1::<f32>::linspace(1.0, 120.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let a = Array2::from_shape_vec((10, 12), arr.to_vec()).unwrap();
        let arr = Array1::<f32>::linspace(2.0, 240.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();

        let a = a.slice(s![1..6, 3..8]);
        let b = b.slice(s![2..7, 0..9]);

        // output not defined
        let c = CSYMM::default().a(a.view()).b(b.view()).side('L').uplo('L').run().unwrap().into_owned();
        let c_naive = naive_symm(&a.view(), &b.view(), 'L', 'L');
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);
        
        // output defined, f-contiguous
        let mut c = Array2::zeros((20, 40).f());
        let c_view = c.slice_mut(s![6..11, 23..32]);
        CSYMM::default().a(a.view()).b(b.view()).c(c_view).side('L').uplo('L').run().unwrap();
        let c_naive = naive_symm(&a.view(), &b.view(), 'L', 'L');
        let c_view = c.slice(s![6..11, 23..32]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
        
        // output defined, arbitary slice
        let mut c = Array2::zeros((20, 40));
        let c_view = c.slice_mut(s![6..16;2, 5..32;3]);
        CSYMM::default().a(a.view()).b(b.view()).c(c_view).side('L').uplo('L').run().unwrap();
        let c_naive = naive_symm(&a.view(), &b.view(), 'L', 'L');
        let c_view = c.slice(s![6..16;2, 5..32;3]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    #[test]
    fn rl_hybrid_contiguous_hemm() {
        let arr = Array1::<f32>::linspace(1.0, 120.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f32>::linspace(2.0, 240.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();

        let a = a.slice(s![1..6, 3..8]);
        let b = b.slice(s![0..9, 2..7]);

        // output not defined
        let c = CHEMM::default().a(a.view()).b(b.view()).side('R').uplo('L').run().unwrap().into_owned();
        let c_naive = naive_hemm(&a.view(), &b.view(), 'R', 'L');
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);
        
        // output defined, arbitary slice
        let mut c = Array2::zeros((40, 20));
        let c_view = c.slice_mut(s![5..32;3, 6..16;2]);
        CHEMM::default().a(a.view()).b(b.view()).c(c_view).side('R').uplo('L').run().unwrap();
        let c_naive = naive_hemm(&a.view(), &b.view(), 'R', 'L');
        let c_view = c.slice(s![5..32;3, 6..16;2]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    fn naive_symm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>, side: char, uplo: char) -> Array2<F>
    where 
        F: BLASFloat,
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

        let (m, n) = if side == 'L' {
            (a.shape()[0], b.shape()[1])
        } else {
            (b.shape()[0], a.shape()[1])
        };
        let mut c = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..a.shape()[1] {
                    if side == 'L' {
                        sum += a[[i, k]] * b[[k, j]];
                    } else if side == 'R' {
                        sum += b[[i, k]] * a[[k, j]];
                    }
                }
                c[[i, j]] = sum;
            }
        }
        c
    }

    fn naive_hemm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>, side: char, uplo: char) -> Array2<F>
    where 
        F: BLASFloat + ComplexFloat + From<<F as ComplexFloat>::Real>,
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

        let (m, n) = if side == 'L' {
            (a.shape()[0], b.shape()[1])
        } else {
            (b.shape()[0], a.shape()[1])
        };
        let mut c = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for k in 0..a.shape()[1] {
                    if side == 'L' {
                        sum += a[[i, k]] * b[[k, j]];
                    } else if side == 'R' {
                        sum += b[[i, k]] * a[[k, j]];
                    }
                }
                c[[i, j]] = sum;
            }
        }
        c
    }
}