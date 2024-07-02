#[cfg(test)]
mod test_gemm {
    use ndarray::prelude::*;
    use blas_array2::util::*;
    use blas_array2::blas3::gemm::GEMM;
    use num_complex::*;

    #[test]
    fn nn_f_contiguous()
    {
        let arr = Array1::<f64>::linspace(1.0, 120.0, 120);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f64>::linspace(2.0, 240.0, 120);
        let b = Array2::from_shape_vec((12, 10).f(), arr.to_vec()).unwrap();
        let a = a.slice(s![1..4, 5..10]);
        let b = b.slice(s![3..8, 4..8]);
        let arr = Array1::<f64>::zeros(120);

        // output not defined
        let c = GEMM::default().a(a.view()).b(b.view())
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view(), &b.view());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, f-contiguous
        let mut c = Array2::from_shape_vec((6, 20).f(), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![2..5, 12..16]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .run().unwrap();
        let c_view = c.slice(s![2..5, 12..16]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, c-contiguous
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![2..5, 12..16]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .run().unwrap();
        let c_view = c.slice(s![2..5, 12..16]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, arbitary slice
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![0..6;2, 4..16;3]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .run().unwrap();
        let c_view = c.slice(s![0..6;2, 4..16;3]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    #[test]
    fn nn_c_contiguous()
    {
        let arr = Array1::<f64>::linspace(1.0, 120.0, 120);
        let a = Array2::from_shape_vec((10, 12), arr.to_vec()).unwrap();
        let arr = Array1::<f64>::linspace(2.0, 240.0, 120);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();
        let a = a.slice(s![1..4, 5..10]);
        let b = b.slice(s![3..8, 4..8]);
        let arr = Array1::<f64>::zeros(120);
        
        // output not defined
        let c = GEMM::default().a(a.view()).b(b.view())
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view(), &b.view());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, f-contiguous
        let mut c = Array2::from_shape_vec((6, 20).f(), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![2..5, 12..16]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .run().unwrap();
        let c_view = c.slice(s![2..5, 12..16]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, c-contiguous
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![2..5, 12..16]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .run().unwrap();
        let c_view = c.slice(s![2..5, 12..16]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, arbitary slice
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![0..6;2, 4..16;3]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .run().unwrap();
        let c_view = c.slice(s![0..6;2, 4..16;3]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    #[test]
    fn nt_c_contiguous()
    {
        let arr = Array1::<f64>::linspace(1.0, 120.0, 120);
        let a = Array2::from_shape_vec((10, 12), arr.to_vec()).unwrap();
        let arr = Array1::<f64>::linspace(2.0, 240.0, 120);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();
        let a = a.slice(s![1..4, 5..10]);
        let b = b.slice(s![4..8, 3..8]);
        let arr = Array1::<f64>::zeros(120);
        
        // output not defined
        let c = GEMM::default().a(a.view()).b(b.view())
            .try_transb('T').unwrap()
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view(), &b.view().t());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, f-contiguous
        let mut c = Array2::from_shape_vec((6, 20).f(), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![2..5, 12..16]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .try_transb('T').unwrap()
            .run().unwrap();
        let c_view = c.slice(s![2..5, 12..16]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, c-contiguous
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![2..5, 12..16]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .try_transb('T').unwrap()
            .run().unwrap();
        let c_view = c.slice(s![2..5, 12..16]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, arbitary slice
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![0..6;2, 4..16;3]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .try_transb('T').unwrap()
            .run().unwrap();
        let c_view = c.slice(s![0..6;2, 4..16;3]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    #[test]
    fn tt_hybrid_contiguous()
    {
        let arr = Array1::<f64>::linspace(1.0, 120.0, 120);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f64>::linspace(2.0, 240.0, 120);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();
        let a = a.slice(s![5..10, 1..4]);
        let b = b.slice(s![4..8, 3..8]);
        let arr = Array1::<f64>::zeros(120);
        
        // output not defined
        let c = GEMM::default().a(a.view()).b(b.view())
            .transa(BLASTrans::Trans)
            .try_transb('T').unwrap()
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view().t(), &b.view().t());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);

        // output defined, arbitary slice
        let mut c = Array2::from_shape_vec((6, 20), arr.to_vec()).unwrap();
        let c_view = c.slice_mut(s![0..6;2, 4..16;3]);
        GEMM::default().a(a.view()).b(b.view()).c(c_view)
            .transa(BLASTrans::Trans)
            .try_transb('T').unwrap()
            .run().unwrap();
        let c_view = c.slice(s![0..6;2, 4..16;3]);
        assert_eq!((&c_naive - &c_view).mapv(|x| x.abs()).sum(), 0.0);
    }

    #[test]
    fn various_float_types() {
        let arr = Array1::<f32>::linspace(1.0, 120.0, 120);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f32>::linspace(2.0, 240.0, 120);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();
        let a = a.slice(s![0..6, 0..8]);
        let b = b.slice(s![0..8, 0..9]);
        let c = GEMM::default().a(a.view()).b(b.view())
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view(), &b.view());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);

        let arr = Array1::<f32>::linspace(1.0, 120.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f32>::linspace(2.0, 240.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c32::i() * x);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();
        let a = a.slice(s![0..6, 0..8]);
        let b = b.slice(s![0..8, 0..9]);
        let c = GEMM::default().a(a.view()).b(b.view())
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view(), &b.view());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);

        let arr = Array1::<f64>::linspace(1.0, 120.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c64::i() * x);
        let a = Array2::from_shape_vec((10, 12).f(), arr.to_vec()).unwrap();
        let arr = Array1::<f64>::linspace(2.0, 240.0, 120);
        let arr = arr.mapv(|x| x + 2.0 * c64::i() * x);
        let b = Array2::from_shape_vec((12, 10), arr.to_vec()).unwrap();
        let a = a.slice(s![0..6, 0..8]);
        let b = b.slice(s![0..8, 0..9]);
        let c = GEMM::default().a(a.view()).b(b.view())
            .run().unwrap().into_owned();
        let c_naive = naive_gemm(&a.view(), &b.view());
        assert_eq!((&c_naive - &c).mapv(|x| x.abs()).sum(), 0.0);
    }

    fn naive_gemm<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> Array2<F>
    where
        F: BLASFloat
    {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut c = Array2::zeros((m, n).f());
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c[[i, j]] += a[[i, l]] * b[[l, j]];
                }
            }
        }
        c
    }
}
