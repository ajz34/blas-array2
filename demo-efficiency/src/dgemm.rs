#![allow(non_upper_case_globals)]

use ndarray::prelude::*;
use blas_array2::prelude::*;
use std::time::Instant;
use rand::{thread_rng, Rng};

pub fn bench_dgemm() {
    println!("AMD 7945HX estimated efficiency: {max_flops} GFLOPS");
    let mut a = Array3::<f64>::zeros((nset, n, n));
    let mut b = Array3::<f64>::zeros((nset, n, n));
    let mut c = Array2::<f64>::zeros((n, n));

    // random initialize
    a.mapv_inplace(| _ | thread_rng().gen());
    b.mapv_inplace(| _ | thread_rng().gen());
    println!("random initialize finished");

    // inplace test
    bench_blas_array2_inplace(2, &a, &b, &mut c);
    bench_blas_array2_inplace(10, &a, &b, &mut c);
    bench_blas_array2_outplace(2, &a, &b, &mut c);
    bench_blas_array2_outplace(10, &a, &b, &mut c);
    bench_ndarray(2, &a, &b, &mut c);
    bench_ndarray(10, &a, &b, &mut c);
    bench_faer(2, &a, &b, &mut c);
    bench_faer(10, &a, &b, &mut c);
}

/// AMD 7945HX CPU info

/// turbo frequency: 4400 MHz on average (not exact value)
static f_turbo: f64 = 4400. * 1000. * 1000.;
/// FMA double count
static n_fma: usize = 2;
/// AVX-512 SIMD (512 / 64 byte)
static n_avx512: usize = 8;
/// throughput of FMA double (numbers of AVX-512 unit on one physical core)
static n_throughput: usize = 1;
/// physical cores in AMD 7945HX
static n_core: usize = 16;
/// maximum f64 FLOPS estimation of AMD 7945HX (in GFLOPS or G/sec)
static max_flops: f64 = f_turbo * (n_fma * n_avx512 * n_throughput * n_core) as f64 / 1.0e9;

/// matrix dimension (nset x n x n)
static nset: usize = 10;
static n: usize = 2560;

/// FLOP count of mat-mul (in GFLOP)
static flop_count: f64 = (2 * nset * n * n * n) as f64 / 1.0e9;

fn bench_blas_array2_inplace(
    ntest: usize,
    a: &Array3<f64>,
    b: &Array3<f64>,
    c: &mut Array2<f64>,
) {
    let mut time_list: Vec<f64> = vec![];

    for _ in 0..ntest {
        let now = Instant::now();

        c.fill(0.0);
        for iset in 0..nset {
            let _ = DGEMM::default()
                .a(a.index_axis(Axis(0), iset))
                .b(b.index_axis(Axis(0), iset))
                .c(c.view_mut())
                .beta(1.0)
                .run();
        }

        let elapsed = now.elapsed();
        time_list.push(elapsed.as_secs_f64());
    }

    let time_sum: f64 = time_list.iter().sum();
    let time_avg: f64 = time_sum / ntest as f64;
    let time_std: f64 = (time_list.iter().map(|x| (x - time_avg).powi(2)).sum::<f64>() / ntest as f64).sqrt();
    if ntest > 2 {
        println!("== bench_blas_array2_inplace ==");
        println!("time for all tests: {:.3} sec, {:.1} GFLOPS", time_sum, ntest as f64 * flop_count / time_sum);
        println!("time for one test : {:.3} ± {:.3} msec", time_avg * 1000., time_std * 1000.);
        println!("c_sum: {:.6}", c.sum());
    }
}

fn bench_blas_array2_outplace(
    ntest: usize,
    a: &Array3<f64>,
    b: &Array3<f64>,
    c: &mut Array2<f64>,
) {
    let mut time_list: Vec<f64> = vec![];

    for _ in 0..ntest {
        let now = Instant::now();

        c.fill(0.0);
        for iset in 0..nset {
            let c_to_add = DGEMM::default()
                .a(a.index_axis(Axis(0), iset))
                .b(b.index_axis(Axis(0), iset))
                .run().unwrap().into_owned();
            *c += &c_to_add;
        }

        let elapsed = now.elapsed();
        time_list.push(elapsed.as_secs_f64());
    }

    let time_sum: f64 = time_list.iter().sum();
    let time_avg: f64 = time_sum / ntest as f64;
    let time_std: f64 = (time_list.iter().map(|x| (x - time_avg).powi(2)).sum::<f64>() / ntest as f64).sqrt();
    if ntest > 2 {
        println!("== bench_blas_array2_outplace ==");
        println!("time for all tests: {:.3} sec, {:.1} GFLOPS", time_sum, ntest as f64 * flop_count / time_sum);
        println!("time for one test : {:.3} ± {:.3} msec", time_avg * 1000., time_std * 1000.);
        println!("c_sum: {:.6}", c.sum());
    }
}

fn bench_ndarray(
    ntest: usize,
    a: &Array3<f64>,
    b: &Array3<f64>,
    c: &mut Array2<f64>,
) {
    let mut time_list: Vec<f64> = vec![];

    for _ in 0..ntest {
        let now = Instant::now();

        c.fill(0.0);
        for iset in 0..nset {
            *c += &a.index_axis(Axis(0), iset).dot(&b.index_axis(Axis(0), iset));
        }

        let elapsed = now.elapsed();
        time_list.push(elapsed.as_secs_f64());
    }

    let time_sum: f64 = time_list.iter().sum();
    let time_avg: f64 = time_sum / ntest as f64;
    let time_std: f64 = (time_list.iter().map(|x| (x - time_avg).powi(2)).sum::<f64>() / ntest as f64).sqrt();
    if ntest > 2 {
        println!("== bench_ndarray ==");
        println!("time for all tests: {:.3} sec, {:.1} GFLOPS", time_sum, ntest as f64 * flop_count / time_sum);
        println!("time for one test : {:.3} ± {:.3} msec", time_avg * 1000., time_std * 1000.);
        println!("c_sum: {:.6}", c.sum());
    }
}

fn bench_faer(
    ntest: usize,
    a: &Array3<f64>,
    b: &Array3<f64>,
    c: &mut Array2<f64>,
) {
    use faer::Parallelism;
    use faer::linalg::matmul::matmul;
    use faer_ext::*;
    
    let mut time_list: Vec<f64> = vec![];

    for _ in 0..ntest {
        let now = Instant::now();

        c.fill(0.0);
        for iset in 0..nset {
            let a_view = a.index_axis(Axis(0), iset);
            let b_view = b.index_axis(Axis(0), iset);
            let a_faer = a_view.view().into_faer();
            let b_faer = b_view.view().into_faer();
            let mut c_faer = c.view_mut().into_faer();
            matmul(c_faer.as_mut(), a_faer.as_ref(), b_faer.as_ref(), Some(1.0), 1.0, Parallelism::Rayon(n_core));
        }

        let elapsed = now.elapsed();
        time_list.push(elapsed.as_secs_f64());
    }

    let time_sum: f64 = time_list.iter().sum();
    let time_avg: f64 = time_sum / ntest as f64;
    let time_std: f64 = (time_list.iter().map(|x| (x - time_avg).powi(2)).sum::<f64>() / ntest as f64).sqrt();
    if ntest > 2 {
        println!("== bench_faer ==");
        println!("time for all tests: {:.3} sec, {:.1} GFLOPS", time_sum, ntest as f64 * flop_count / time_sum);
        println!("time for one test : {:.3} ± {:.3} msec", time_avg * 1000., time_std * 1000.);
        println!("c_sum: {:.6}", c.sum());
    }
}
