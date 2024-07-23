# Document for Develop

This document is written at v0.3.0 version.

## Limitation of this crate

Though I believe this crate provides many functionalities that interests audiences in scientific computation, there are also some points that this crate is lack of, or is not going to support with by design.

For the features that will not support with,
- **Supports to other types of matrices**. Currently crates such as `ndarray`, `nalgebra`, `matrix`, `faer-rs`, `rulinalg`, `rest_tensors` represent typical matrix implementations in rust. Though some crates are extremely fast (comparable to MKL or OpenBLAS, especially `faer-rs`) in linalgs, it seems that `ndarray` could better support features in high-dimension tensors and sub-matrices, as well as advanced slicing/view. This is also related to concept of "leading dimension" in BLAS. So `ndarray` is choosen to represent matrix in this crate. Matrices defined in other crates should be able to easily casted to `ndarray`, without explicit memory copy (by rust's moving or by tensor view of slice from raw-data), and thus not providing BLAS to other types of matrices.
- **Other targets (GPU)**. This may well require a new data structure (such as numpy v.s. torch/jax). Though machine learning frameworks (such as `candle`, `burn`, etc.) seem to be promising, an API-stable tensor structure that accepts various targets, with advanced slicing/stride support has probably yet existed in rust.
- **Arbitary data types**. Currently, this crate supports f32/f64/c32/c64, which should be supported by legacy BLAS standard. However, this crate will not implement something like int8/rug. To address this issue, a BLAS reference implementation to any types is required, and is out of scope for this crate, which is only a high-level wrapper to BLAS (or BLAS-like) functions.
- **Fn instead of struct**. A common sense for using BLAS functions with matrices, is function with optional parameters. However, this is not possible in rust, syntactically. So we choose to use struct (with `derive_build`) to pass optional parameters. In this way, there is at least one additional drawback: no IDE-time/compile-time check to non-optional parameters, so errors may occur in runtime; this require programmers to use this crate with caution.
- **Lapack wrapper**. This is surely important, but will probably be implemented in a new crate.

For the features that can be added, but currently haven't been added (probably I'm not interested in for the moment I'm writing this crate) and may be implemented in a later time, or may be implemented after someone giving feature requests in issues/PRs.
- **BLAS1 functions**. I think that in many cases, functionality of BLAS1 can be simply covered by iterators, or features in matrix libraries such as `ndarray`. Efficiency of BLAS1 can be achieved by compiler optimization (especially for serial/single-thread usage), using BLAS may not significantly improve efficiency if your code is mostly computation-bounded instead of memory-bounded.
- **Other kinds of floats**. With development of machine learning nowadays, demands of low-precision BLAS is increasing; MKL and OpenBLAS has already implemented some `BF16` features.
- **BLAS extensions**. There are some important BLAS extensions, such as `omatcopy`, `imatcopy`, `gemmt`, `gemm3m`, that has already been implemented in both OpenBLAS, MKL and BLIS. Currently, `gemmt` has been implemented. Others are on-going work.
- **Documents**. I hope that this crate is clear and simple to use by itself, especially to programmers from C/C++/F77. This crate want to be `scipy.linalg.blas` replacement in rust. So documentation may not be of that important (<https://netlib.org/lapack/explore-html/> is farely well documented), but probably it's still required to newcommers.
- **Tests**. It works, but it is messy. Boundary conditions have not been fully checked.

## Structure of Wrapper

For example of GEMM:

- `trait GEMMNum: BLASFloat`

    This trait directly wraps FFI function `?gemm_`, and implemented directly to float types (`f32`, `f64`, `Complex<f32>`, `Complex<f64>` if appropriate). With this trait, one can call FFI function `dgemm_` by `f64::gemm`.

- `struct GEMM_Driver` (with trait bound `BLASDriver`)

    This struct is a (not so safe) wrapper to FFI wrapper `GEMMFunc`. This struct takes rust types, and return result by calling function `run_blas` function, which calls FFI function.

    On calling function `run_blas`, all fields in `GEMM_Driver` will be consumed (dropped), and only result matrix $\mathbf{C}$ will be returned.

- `struct GEMM_` (with trait bound `BLASBuilder_`)

    This struct acts as inner driver, and will be wrapped by `GEMM_Builder` by crate `derive_builder`.

    Trait function `driver` (inner driver, trait function) of `BLASBuilder_` is not the same to struct function `driver` (outer driver, automatically derived) of `GEMM_Builder`.
    
    This struct is not designed to be directly called. When calling `driver` function in `GEMM_`,
    - **All input (except output) should be column-major**;
    - All fields should be filled;
    - `layout` must be `Some(BLASColMajor)`;
    - Quick return if zero dimension input;
    - Output `c` can be arbitary major, or simply `None`.
    
    Inner driver will perform
    - Obtaining dimension and leading dimension that will be needed in FFI functions,
    - Checking validability of parameters (all in column-major),
    - Allocating output buffer if output `c` is not column-major (to be `ArrayOut::ToBeCloned`), or not defined (to be `ArrayOut::Owned`).
    - Returning FFI `GEMM_Driver` for future computation.

- `struct GEMM_Builder` (with trait bound `BLASBuilder`)

    This struct is builder, automatically derived by crate `derive_builder`, which is designed to be accessible by API user.

    **This struct accepts input with arbitary stride**, though this is not recommanded.

    Trait function `run` will perform the following works:
    - Check layout (will be further explained in [Layout Convention](#layout-convention)),
    - Perform trans / side / uplo flag flip, and matrix transposition to proper layout;
    - Generate `GEMM_` struct, and perform FFI computation;
    - If layout is row-major, reverse axis of output matrix from FFI (FFI only accepts column-major matrix).

## Layout Convention

### Row-major and Col-major

For this crate, row-major and col-major is not the same meaning to C-contiguous and F-contiguous (flags of numpy).

Contiguous means that **all** data in memory is contiguous. Difference of C-contiguous and F-contiguous in that how the matrix is presented in memory.

```output
       C-contiguous  F-contiguous  Non-contiguous  Sequential
        (by row)      (by col)        (by row)      (onedim)
         +++++         +++--           -----         +++++
         +++++         +++--           -+++-
         +++++         +++--           -+++-
         -----         +++--           -+++-
         -----         +++--           -----
  shape [3, 5]        [5, 3]          [3, 3]
strides [5, 1]        [1, 5]          [5, 1]
```

Major does not require all data in memory is contiguous. However, it require data in one dimension to be contiguous.

```output
       row-major  col-major  row/col-major  row/col-major  custom-layout
        (by row)   (by col)    (by row)       (by col)       (by row)
         ++++-      -----       -----          -----          +-+-+
         -----      +-+-+       -+++-          -+++-          -----
         ++++-      +-+-+       -+++-          -+++-          +-+-+
         -----      +-+-+       -+++-          -+++-          -----
         ++++-      +-+-+       -----          -----          +-+-+
  shape [ 3, 4]    [4,  3]     [3, 3]         [3, 3]         [ 3, 3]
strides [10, 1]    [1, 10]     [5, 1]         [1, 5]         [10, 2]
```

In this crate, we check row/col-major if stride of second/first dimension is one.

### BLAS3

For all BLAS3 functions, specifying row-major or col-major to keyword `layout` will **not change result**; however, efficiency may vary.

Layout is defined by this way:
- Use layout provided by user; otherwise (user does not explicitly provide layout)
- If output matrix provided by user is either row-major or col-major, then specify layout by output matrix layout; otherwise (output matrix is custom-layout, or no output matrix provided)
- If **all** input is col-major, then specify layout as col-major; otherwise, specify layout as row-major.

### BLAS2

For packed and banded operations, row-major and col-major could incur **different** results. For more information, we refer to BLAST document of [C Interface to the Legacy BLAS](https://netlib.org/blas/blast-forum/cinterface.pdf).

In other cases, result is unrelated to layout, as the same case in BLAS3.


