import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid_view():
    # test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    list_num = ["f32", "f64", "c32", "c64"]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_uplo = ["U", "L"]
    list_trans = ["N", "T", "C"]
    list_diag = ["N", "U"]

    set_inp = [
        list_num,
        list_stride, list_stride, list_stride,
        list_layout, list_uplo, list_trans, list_diag
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            num,
            a_stride_0, a_stride_1, incx,
            a_layout, uplo, trans, diag
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(4, 8, a_stride_0, a_stride_1)}, "
            f"({8}, {incx}), "
            f"'{a_layout}', '{uplo}', '{trans}', '{diag}');"
        )
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    print("\n".join(gen_valid_view()))
