import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid_owned():
    # test_macro!(test_000: inline, f32, (7, 7, 1, 1), (7, 9, 1, 1), 'R', 'R', 'L', 'L', SYMM, symmetrize);
    list_num = [
        ("f32", "SYMM", "symmetrize"),
        ("f64", "SYMM", "symmetrize"),
        ("c32", "SYMM", "symmetrize"),
        ("c64", "SYMM", "symmetrize"),
        ("c32", "HEMM", "hermitianize"),
        ("c64", "HEMM", "hermitianize"),
    ]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_shape_a = [(7, 7, "L"), (9, 9, "R")]
    list_uplo = ["L", "U"]

    set_inp = [
        list_num,
        list_shape_a,
        list_stride, list_stride, list_stride, list_stride,
        list_layout, list_layout, list_uplo,
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            (num, blas, symm),
            (ad0, ad1, side),
            as0, as1, bs0, bs1,
            al, bl, uplo
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(ad0, ad1, as0, as1)}, "
            f"{(7, 9, bs0, bs1)}, "
            f"'{al}', '{bl}', "
            f"'{side}', '{uplo}', "
            f"{blas}, {symm});"
        )
        tokens.append(token)
    return tokens


def gen_valid_view():
    # test_macro!(test_000: inline, f32, (7, 7, 1, 1), (7, 9, 1, 1), (7, 9, 1, 3), 'R', 'R', 'R', 'L', 'L', SYMM, symmetrize);
    list_num = [
        ("f32", "SYMM", "symmetrize"),
        ("f64", "SYMM", "symmetrize"),
        ("c32", "SYMM", "symmetrize"),
        ("c64", "SYMM", "symmetrize"),
        ("c32", "HEMM", "hermitianize"),
        ("c64", "HEMM", "hermitianize"),
    ]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_shape_a = [(7, 7, "L"), (9, 9, "R")]
    list_uplo = ["L", "U"]

    set_inp = [
        list_num,
        list_shape_a,
        list_stride, list_stride, list_stride, list_stride, list_stride, list_stride,
        list_layout, list_layout, list_layout, list_uplo,
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size, nkeep=1000)):
        (
            (num, blas, symm),
            (ad0, ad1, side),
            as0, as1, bs0, bs1, cs0, cs1,
            al, bl, cl, uplo
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(ad0, ad1, as0, as1)}, "
            f"{(7, 9, bs0, bs1)}, "
            f"{(7, 9, cs0, cs1)}, "
            f"'{al}', '{bl}', '{cl}', "
            f"'{side}', '{uplo}', "
            f"{blas}, {symm});"
        )
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    # print("\n".join(gen_valid_owned()))
    print("\n".join(gen_valid_view()))
