#![allow(non_upper_case_globals)]

mod dgemm;
mod dsyrk;

fn main() {
    println!(">>> bench_dgemm <<<");
    dgemm::bench_dgemm();
    println!(">>> bench_dsyrk <<<");
    dsyrk::bench_dsyrk();
}

