# Efficiency Demonstration

Notes of efficiency when programming (general advice, not only for `blas-array2`):
- Row/col-major layout. One of two dimensions in matrix should be contiguous (stride of the dimension is one).
    > This note may seems to be common sense, but will be useful when applying matrix multiplication to sub-tensors, or problem of tensor products that can be reduced to matrix multiplication. For example, $A_{ijkl} \rightarrow_{ik} A_{jl}$ is row-major, $A_{ijkl} \rightarrow_{jl} A_{ik}$ is col-major, but $A_{ijkl} \rightarrow_{il} A_{jk}$ is nither row/col-major. So it is important to avoid such cases when deriving formulas involving tensor products.
- Avoid memory allocation and outplace add-assign.
    > This will not affect efficiency a lot, but could be avoided if calling API correctly. To be more specific, it is better to call
    > ```
    > gemm(A, B, out = C, beta = 1)
    > ```
    > instead to call the following form, though more intutive but involves unnecessary allocation and add-assign of `C`
    > ```
    > C += gemm(A, B)
    > ```

We benchmark efficiency of `blas-array2`, `ndarray` and `faer` on [AMD 7945HX](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7945hx.html). Estimated theoretical performance is 1126.4 GFLOPS.

Related code is in [`demo-efficiency`](https://github.com/ajz34/blas-array2/tree/main/demo-efficiency) directory in repository.

The problem to be benchmarked is

$$
C_{ij} = \sum_{mk} A_{mik} B_{mkj} \quad (\texttt{DGEMM}, 10 \times 2560 \times 2560)
$$
and
$$
C_{ij} = \sum_{mk} A_{mik} A_{mkj} \quad (\texttt{DSYRK}, 10 \times 2560 \times 2560)
$$

Main purpose of this benchmark, is to show that cost of `blas-array2` itself (manipulation and generation of `ndarray` views, flags, stride length) is considerably small. Efficiency is mostly affected by BLAS backend, instead of representation of data.

## DGEMM

| crate | API call | backend | GFLOPS |
|--|--|--|--|
| `blas-array2` | inplace | AOCL | 964.1 |
|  |  | OpenBLAS | 947.6 |
|  |  | MKL | 768.0 |
|  | outplace add-assign | AOCL | 790.3 |
|  |  | OpenBLAS | 641.3 |
|  |  | MKL | 630.3 |
| `ndarray` | outplace add-assign | AOCL | 780.8 |
|  |  | OpenBLAS | 636.5 |
|  |  | MKL | 626.6 |
| `faer` | inplace | faer | 844.7 |

## DSYRK

| crate | API call | backend | GFLOPS |
|--|--|--|--|
| `blas-array2` | inplace | AOCL | 735.6 |
|  |  | OpenBLAS | 669.2 |
|  |  | MKL | 594.1 |
|  | outplace add-assign | AOCL | 541.5 |
|  |  | OpenBLAS | 456.0 |
|  |  | MKL | 487.9 |
| `ndarray` | outplace add-assign | AOCL (GEMM) | 408.3 |
|  |  | OpenBLAS (GEMM) | 324.9 |
|  |  | MKL (GEMM) | 340.4 |
| `faer` | inplace | faer (GEMMT) | 670.0 |

## Version info

- AOCL 4.2
- OpenBLAS 0.3.27
- MKL 2024.0
