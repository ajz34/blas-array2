import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid_view():
    # test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    list_num = ["f32", "f64", "c32", "c64"]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_uplo = ["U", "L"]

    set_inp = [
        list_num,
        list_stride, list_stride, list_stride, list_stride,
        list_layout, list_uplo,
    ]
    run_size = 16

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            num,
            a_stride_0, a_stride_1, incx, incy,
            a_layout, uplo,
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(4, 8, a_stride_0, a_stride_1)}, "
            f"({8}, {incx}), ({8}, {incy}), "
            f"'{a_layout}', '{uplo}');"
        )
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    print("\n".join(gen_valid_view()))
