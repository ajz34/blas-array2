# Document for Develop

This document is written at v0.2.0 version.

## Structure of Wrapper

For example of GEMM:

- `trait GEMMFunc` (implemented for `BLASFunc`)

    This trait directly wraps FFI function `?gemm_`, with generic support by `<F> where F: BLASFloat`.

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


