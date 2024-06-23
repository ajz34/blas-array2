#[cfg(test)]
mod unit_asum {
    use ndarray::prelude::*;
    use blas_array2::blas1::asum::*;
    use blas_array2::util::*;
    
    #[test]
    fn test_f32() {
        let x = Array1::<f32>::linspace(-1.0, 2.0, 4);
        let mut asum = ASUM::default()
            .x(x.view())
            .build().unwrap();
        assert_eq!(asum.run().unwrap(), 4.0);
    }
    
    #[test]
    fn test_f64() {
        let x = Array1::<f64>::linspace(-1.0, 2.0, 4);
        let mut asum = ASUM::default()
            .x(x.view())
            .build().unwrap();
        assert_eq!(asum.run().unwrap(), 4.0);
    }
    
    #[test]
    fn test_c32() {
        let x_re = Array1::<f32>::linspace(-1.0, 2.0, 4).mapv(|x| c32::from(x));
        let x_im = Array1::<f32>::linspace(-2.0, 1.0, 4).mapv(|x| c32::from(x));
        let x = x_re * c32::i() + x_im;
        let mut asum = ASUM::default()
            .x(x.view())
            .build().unwrap();
        assert_eq!(asum.run().unwrap(), 8.0);
    }
    
    #[test]
    fn test_c64() {
        let x_re = Array1::<f64>::linspace(-1.0, 2.0, 4).mapv(|x| c64::from(x));
        let x_im = Array1::<f64>::linspace(-2.0, 1.0, 4).mapv(|x| c64::from(x));
        let x = x_re * c64::i() + x_im;
        let mut asum = ASUM::default()
            .x(x.view())
            .build().unwrap();
        assert_eq!(asum.run().unwrap(), 8.0);
    }
}