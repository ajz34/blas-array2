import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid_owned():
    # test_macro!(test_000: inline, f32, (7, 5, 1, 1), 'C', 'L', 'N', SYRK, 'T', f32);
    list_num = [
        ("f32", "SYRK", "T", "f32"),
        ("f64", "SYRK", "T", "f64"),
        ("c32", "SYRK", "T", "c32"),
        ("c64", "SYRK", "T", "c64"),
        ("c32", "HERK", "C", "f32"),
        ("c64", "HERK", "C", "f64"),
    ]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_uplo = ["L", "U"]
    list_trans = ["T", "N"]

    set_inp = [
        list_num,
        list_stride, list_stride,
        list_layout, list_uplo, list_trans
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            (num, blas, blas_trans, blas_ty),
            as0, as1,
            al, uplo, trans
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(7, 5, as0, as1)}, "
            f"'{al}', "
            f"'{uplo}', '{trans if (trans == 'N' or blas == 'SYRK') else 'C'}', "
            f"{blas}, '{blas_trans}', {blas_ty});"
        )
        tokens.append(token)
    return tokens


def gen_valid_view():
    # test_macro!(test_000: inline, f32, (7, 5, 1, 1), 'C', 'L', 'N', SYRK, 'T', f32);
    list_num = [
        ("f32", "SYRK", "T", "f32"),
        ("f64", "SYRK", "T", "f64"),
        ("c32", "SYRK", "T", "c32"),
        ("c64", "SYRK", "T", "c64"),
        ("c32", "HERK", "C", "f32"),
        ("c64", "HERK", "C", "f64"),
    ]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_uplo = ["L", "U"]
    list_trans = [
        ("T", (5, 5)),
        ("N", (7, 7)),
    ]

    set_inp = [
        list_num,
        list_stride, list_stride, list_stride, list_stride,
        list_layout, list_layout,
        list_uplo, list_trans
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            (num, blas, blas_trans, blas_ty),
            as0, as1, cs0, cs1,
            al, cl,
            uplo, (trans, (cd0, cd1))
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(7, 5, as0, as1)}, "
            f"{(cd0, cd1, cs0, cs1)}, "
            f"'{al}', '{cl}', "
            f"'{uplo}', '{trans if (trans == 'N' or blas == 'SYRK') else 'C'}', "
            f"{blas}, '{blas_trans}', {blas_ty});"
        )
        tokens.append(token)
    return tokens

if __name__ == "__main__":
    print("\n".join(gen_valid_owned()))
    print("\n".join(gen_valid_view()))
