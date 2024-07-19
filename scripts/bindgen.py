# # Bindgen of `f77blas.h`

import subprocess

# ## Pre-process

# +
# change original file

with open("blas.h", "r") as f:
    token = f.read()

# I hope it is more proper that MKL marker is not in original header file?
token = token.replace("MKL", "BLAS")
# remove unnecessary markers, new lines
token = token.replace(" NOTHROW;", ";")
token = token.replace("\n    ", " ")
token = token.replace("__", "_")
for _ in range(10):
    token = token.replace("  ", " ")
# add function suffix
token_list = []
for l in token.split("\n"):
    if any([l.startswith(k) for k in ("void", "double", "float", "BLAS_INT")]):
        l_ = l.split("(")
        if l_[0][-1] != "_":
            l_[0] = l_[0] + "_"
        l = "(".join(l_)
    token_list.append(l)
token = "\n".join(token_list)

with open("blas.h", "w") as f:
    f.write(token)
# -

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
token = token.replace("pub type blas_int = ::core::ffi::c_int;\n", "")
token = token.replace("::core::ffi::c_char", "c_char")
token = """
#![allow(non_camel_case_types)]

use num_complex::*;
use core::ffi::c_char;

#[cfg(not(feature = "ilp64"))]
pub type blas_int = i32;
#[cfg(feature = "ilp64")]
pub type blas_int = i64;

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
