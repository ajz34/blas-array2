# blas-array2

[![codecov](https://codecov.io/gh/ajz34/blas-array2/graph/badge.svg?token=n1ucRtIupr)](https://codecov.io/gh/ajz34/blas-array2)

Implementation of parameter-optional BLAS wrapper by `ndarray::Array` (`Ix1` or `Ix2`)

**This is a package under construction.**

Currently,
- BLAS3 functions have been implemented. BLAS1/2 functions implementation is on-going work.
- Optional parameters (`scipy.linalg.blas` convention), complex numbers, arbitary layouts (row-major, col-major, strided) supported.
- Shape of matrix, and information of leading dimension will be checked properly. These values are automatically parsed in program, so users do not need to give these values (like calling to Fortran BLAS).
- All input in row-major (or col-major) should not involve unnecessary transpositions with explicit clone. So all-row-major or all-col-major is preferred.
- Generics is able for some blas functions: you can use `GEMM<F> where F: BLASFloat` for `f32`, `f64`, `Complex<f32>`, `Complex<f64>` types, in one generic (template) class.

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
```
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

This package haven't been deployed. After implementing all functions in BLAS1/2/3, a version 0.1 will be released.

For development usage, if there's any difficulties encountered in installation, then please check if `blas-sys` installed and linked properly. May be resolved by declaring
```
RUSTFLAGS="-lopenblas"
```
if using OpenBLAS as backend.

