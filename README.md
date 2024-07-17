# blas-array2

[![codecov](https://codecov.io/gh/ajz34/blas-array2/graph/badge.svg?token=n1ucRtIupr)](https://codecov.io/gh/ajz34/blas-array2)
[![crates.io](https://img.shields.io/crates/v/blas-array2.svg)](https://crates.io/crates/blas-array2)

Implementation of parameter-optional BLAS wrapper by `ndarray::Array` (`Ix1` or `Ix2`) in rust.

## Features

- **BLAS2/BLAS3 Functionality**: All (legacy) BLAS2/BLAS3 functions have been implemented.
- **Optional Parameters**: `scipy.linalg.blas` convention. Shape of matrix, and information of leading dimension will be checked properly. These values are automatically parsed in program, so users do not need to give these values.
- **Row-major Layout**: Row-major support to Fortran77 API (CBLAS functionality without CBLAS functions). You can safely use the default `libopenblas.so` shipped by debian with `blas-sys`, where CBLAS is not automatically integrated, for example.
- **Generics**: For example, `GEMM<F> where F: BLASFloat` for `f32`, `f64`, `Complex<f32>`, `Complex<f64>` types, in one generic (template) class. The same to `SYRK` or `GEMV`, etc.
- **View instead of copy if possible**: All input in row-major (or col-major) should not involve unnecessary transpositions with explicit copy (a few exceptions occurs for complex BLAS2 functions). Further more, for some BLAS3 functions (GEMM, SYRK, TRMM, TRSM), if transposition does not involve `BLASConjTrans`, then mixed row-major or col-major also does not involve explicit transposition. Also note that in many cases, sub-matrices (sliced matrix) are also considered as row-major (or col-major), if data is stored contiguously in any dimension.
- **Arbitary Layout**: Supports strides that `ndarray` allows (for correctness and API convenience only, but may require explicit raw data copy among DRAM if data is not stored contiguously in either dimension).
- **`no_std`**: This crate is `no_std` by default. Feature `std` will only enhance error handling.

## Simple example
For simple illustration to this package, we perform $C = A B$ (`dgemm`):
```rust
use blas_array2::prelude::*;
use ndarray::prelude::*;
let a = array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
let b = array![[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]];
let c_out = DGEMM::default().a(a.view()).b(b.view()).run().unwrap().into_owned();
println!("{:7.3?}", c_out);
```
Important points are
- using `::default()` to initialize struct;
- `.a`, `.b` are setter functions;
- `.run().unwrap()` will perform computation;
- `.into_owned()` will return result matrix as `Array2<f64>`.

## Complicated example

For complicated situation, we perform $C = A B^\dagger$ by `SGEMM = GEMM<f32>`:
```rust
use blas_array2::prelude::*;
use ndarray::prelude::*;

let a = array![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
let b = array![[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]];
let mut c = Array::ones((3, 3).f());

let c_out = GEMM::<f32>::default()
    .a(a.slice(s![.., ..2]))
    .b(b.view())
    .c(c.slice_mut(s![0..3;2, ..]))
    .transb('T')
    .beta(1.5)
    .run()
    .unwrap();
// one can get the result as an owned array
// but the result may not refer to the same memory location as `c`
println!("{:4.3?}", c_out.into_owned());
// this modification on `c` is actually performed in-place
// so if `c` is pre-defined, not calling `into_owned` could be more efficient
println!("{:4.3?}", c);
```
Important points are
- `.c` is (optional) output setter, which consumes `ArrayViewMut2<f64>`; this matrix will be modified in-place;
- `.transb`, `.beta` are optional setters; default of `transb` is `'N'`, while default of `beta` is zero, which are the same convention to scipy's implementation to python interface of BLAS. You may change these default values by feeding values into optional setters.
- There are three ways to utilize output:
    - `c_out.into_owned()` returns output (submatrix if `c` was sliced when passed into setter) as `Array2<f64>`. Note that this output does not share the same memory address to `mut c`.
    - `c_out.view()` or `c_out.view_mut()` returns view of `c`; these views share the same memory address to `mut c`.
    - Or you may use `c` directly. DGEMM operation is performed inplace if output matrix `c` is given.

To make clear of the code above, this code spinnet performs matrix multiplication in-place
```output
c = alpha * a * transpose(b) + beta * c
where
alpha = 1.0 (by default)
beta = 1.5
a = [[1.0, 2.0, ___],
     [3.0, 4.0, ___]]
        (sliced by `s![.., ..2]`)
b = [[-1.0, -2.0],
     [-3.0, -4.0],
     [-5.0, -6.0]]
c = [[1.0, 1.0, 1.0],
     [___, ___, ___],
     [1.0, 1.0, 1.0]]
        (Column-major, sliced by `s![0..3;2, ..]`)
```
Output of `c` is
```
[[-3.500,  -9.500, -15.500],
 [ 1.000,   1.000,   1.000],
 [-9.500, -23.500, -37.500]]
```

## Installation

This crate is available on crates.io.

Currently, this crate depends on `blas-sys`. If there's any difficulties encountered in installation, then please check if `blas-sys` installed and linked properly. May be resolved by declaring
```
RUSTFLAGS="-lopenblas"
```
if using OpenBLAS as backend.

## Limitation of this crate

Though I believe this crate provides many functionalities that interests audiences in scientific computation, there are also some points that this crate is lack of, or is not going to support with by design.

Known problems and will be fixed is
- **Layout**. Layout rules has been implemented in all BLAS3 functions, but not all BLAS2 functions. Implementation details to layout is slightly different to CBLAS, and this will be documented in detail in future.

For the features that will not support with,
- **Supports to other types of matrices**. Currently crates such as `ndarray`, `nalgebra`, `matrix`, `faer-rs`, `rulinalg`, `rest_tensors` represent typical matrix implementations in rust. Though some crates are extremely fast (comparable to MKL or OpenBLAS) in linalgs, it seems that `ndarray` could better support features in high-dimension tensors and sub-matrices, as well as advanced slicing/view. This is also related to concept of "leading dimension" in BLAS. So `ndarray` is choosen to represent matrix in this crate. Matrices defined in other crates should be able to easily casted to `ndarray`, without explicit memory copy (by rust's moving or by tensor view of slice from raw-data), and thus not providing BLAS to other types of matrices.
- **Other targets (GPU)**. This may well require a new data structure (such as numpy v.s. torch/jax). Though machine learning frameworks (such as `candle`, `burn`, etc.) seem to be promising, an API-stable tensor structure that accepts various targets, with advanced slicing/stride support has probably yet existed in rust.
- **Arbitary data types**. Currently, this crate supports f32/f64/c32/c64, which should be supported by legacy BLAS standard. However, this crate will not implement something like int8/rug. To address this issue, a BLAS reference implementation to any types is required, and is out of scope for this crate, which is only a high-level wrapper to BLAS (or BLAS-like) functions.
- **Fn instead of struct**. A common sense for using BLAS functions with matrices, is function with optional parameters. However, this is not possible in rust, syntactically. So we choose to use struct (with `derive_build`) to pass optional parameters. In this way, there is at least one additional drawback: no IDE-time/compile-time check to non-optional parameters, so errors may occur in runtime; this require programmers to use this crate with caution.
- **Lapack wrapper**. This is surely important, but will probably be implemented in a new crate.

For the features that can be added, but currently haven't been added (probably I'm not interested in for the moment I'm writing this crate) and may be implemented in a later time, or may be implemented after someone giving feature requests in issues/PRs.
- **BLAS1 functions**. I think that in many cases, functionality of BLAS1 can be simply covered by iterators, or features in matrix libraries such as `ndarray`. Efficiency of BLAS1 can be achieved by compiler optimization (especially for serial/single-thread usage), using BLAS may not significantly improve efficiency if your code is mostly computation-bounded instead of memory-bounded.
- **Other kinds of floats**. With development of machine learning nowadays, demands of low-precision BLAS is increasing; MKL and OpenBLAS has already implemented some `BF16` features.
- **BLAS extensions**. There are some important BLAS extensions, such as `omatcopy`, `imatcopy`, `gemmt`, `gemm3m`, that has already been implemented in both OpenBLAS, MKL and BLIS. These can be implemented by features with cargo, and may be one of the first things to be implemented.
- **ILP64**. Currently, this crate uses `blas-sys` for FFI to BLAS, which FFI integer is `libc::c_int`. In most cases it is LP64; so currently ILP64 is not supported. I'm not sure if there exist some tasks that LP64 is not enough. This problem also relates to blas-sys.
- **Documents**. I hope that this crate is clear and simple to use by itself, especially to programmers from C/C++/F77. This crate want to be `scipy.linalg.blas` replacement in rust. So documentation may not be of that important (<https://netlib.org/lapack/explore-html/> is farely well documented), but probably it's still required to newcommers.
- **Tests**. It works, but it is messy. Boundary conditions have not been fully checked.

## Acknowledges

This project is developed as a side project from [REST](https://github.com/igor-1982/rest).
