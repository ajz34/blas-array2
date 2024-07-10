import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid_view():
    list_num = ["f32", "f64", "c32", "c64"]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_trans = ["N", "T", "C"]

    set_inp = [
        list_num,
        list_stride, list_stride, list_stride, list_stride,
        list_layout, list_trans,
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            num,
            a_stride_0, a_stride_1, incx, incy,
            a_layout, trans,
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{a_stride_0}, {a_stride_1}, {incx}, {incy}, "
            f"'{a_layout}', '{trans}');"
        )
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    print("\n".join(gen_valid_view()))
