# # Bindgen of `f77blas.h`

import subprocess

# ## Pre-process

with open("blas.h", "r") as f:
    token = f.read()
token = token.replace("MKL", "BLAS")
with open("blas.h", "w") as f:
    f.write(token)

with open("blas.h", "r") as f:
    token = f.read()

# to be changed
token = token.replace("BLAS_Complex8", "Complex64")
token = token.replace("BLAS_Complex16", "Complex128")

with open("blas_bindgen.h", "w") as f:
    f.write(token)

# ## Bindgen

subprocess.run([
    "bindgen", "blas_bindgen.h",
    "-o", "blas.rs",
    "--allowlist-function", "^.*_$",
    "--no-layout-tests",
    "--use-core"
])

# ## Post-process

with open("blas.rs", "r") as f:
    token = f.read()

# hardcode blasint
token = token.replace("pub type blasint = ::core::ffi::c_int;\n", "")
token = token.replace("::core::ffi::c_char", "c_char")
token = """
#![allow(non_camel_case_types)]

use num_complex::*;
use core::ffi::c_char;

#[cfg(not(feature = "ilp64"))]
pub type blasint = i32;
#[cfg(feature = "ilp64")]
pub type blasint = i64;

pub type c32 = Complex<f32>;
pub type c64 = Complex<f64>;
""".strip() + "\n\n" + token

# __BindgenComplex<T> -> Complex<T>
token = token.replace(
    """
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct c32 {
    pub real: f32,
    pub imag: f32,
}""".strip(), "")
token = token.replace(
    """
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct c64 {
    pub real: f64,
    pub imag: f64,
}""".strip(), "")

with open("blas.rs", "w") as f:
    f.write(token)

subprocess.run([
    "rustfmt", "blas.rs"
])

subprocess.run([
    "mv", "blas.rs", "../src/ffi"
])
