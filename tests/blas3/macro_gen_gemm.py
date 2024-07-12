import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid_owned():
    # test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 9, 1, 1), 'R', 'R', 'N', 'N');
    list_num = ["f32", "f64", "c32", "c64"]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_shape_a = [(7, 8, "N"), (8, 7, "T")]
    list_shape_b = [(8, 9, "N"), (9, 8, "T")]

    set_inp = [
        list_num,
        list_shape_a, list_shape_b,
        list_stride, list_stride, list_stride, list_stride,
        list_layout, list_layout,
    ]
    run_size = 16

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            num,
            a_shape, b_shape,
            a_stride_0, a_stride_1, b_stride_0, b_stride_1,
            a_layout, b_layout,
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(a_shape[0], a_shape[1], a_stride_0, a_stride_1)}, "
            f"{(b_shape[0], b_shape[1], b_stride_0, b_stride_1)}, "
            f"'{a_layout}', '{b_layout}', "
            f"'{a_shape[2]}', '{b_shape[2]}');"
        )
        tokens.append(token)
    return tokens


def gen_valid_view():
    # test_macro!(test_000: inline, f32, (7, 8, 1, 1), (8, 9, 1, 1), (7, 9, 1, 1), 'R', 'R', 'R', 'N', 'N');
    list_num = ["f32", "f64", "c32", "c64"]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_shape_a = [(7, 8, "N"), (8, 7, "T")]
    list_shape_b = [(8, 9, "N"), (9, 8, "T")]

    set_inp = [
        list_num,
        list_shape_a, list_shape_b,
        list_stride, list_stride, list_stride, list_stride, list_stride, list_stride,
        list_layout, list_layout, list_layout,
    ]
    run_size = 16

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            num,
            a_shape, b_shape,
            a_stride_0, a_stride_1, b_stride_0, b_stride_1, c_stride_0, c_stride_1,
            a_layout, b_layout, c_layout,
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(a_shape[0], a_shape[1], a_stride_0, a_stride_1)}, "
            f"{(b_shape[0], b_shape[1], b_stride_0, b_stride_1)}, "
            f"{(7, 9, c_stride_0, c_stride_1)}, "
            f"'{a_layout}', '{b_layout}', '{c_layout}', "
            f"'{a_shape[2]}', '{b_shape[2]}');"
        )
        tokens.append(token)
    return tokens


def gen_valid_cblas():
    list_layout = ["R", "C"]
    list_shape_a = [(7, 8, "N"), (8, 7, "T"), (8, 7, "C")]
    list_shape_b = [(8, 9, "N"), (9, 8, "T"), (9, 8, "C")]

    set_inp = [
        list_shape_a, list_shape_b,
        list_layout
    ]
    run_size = 12

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size, strength=1)):
        (
            (ad0, ad1, transa), (bd0, bd1, transb),
            layout
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, c32, cblas_cgemm, "
            f"{(ad0, ad1, 1, 1)}, "
            f"{(bd0, bd1, 1, 1)}, "
            f"{(7, 9, 1, 1)}, "
            f"'{layout}', '{layout}', '{layout}', "
            f"'{transa}', '{transb}', '{layout}');"
        )
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    # print("\n".join(gen_valid_owned()))
    # print("\n".join(gen_valid_view()))
    print("\n".join(gen_valid_cblas()))
