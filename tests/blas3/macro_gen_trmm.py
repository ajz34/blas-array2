import sys
sys.path.append("../")

from taguchi import taguchi_by_list


def gen_valid():
    list_num = ["f32", "f64", "c32", "c64"]
    list_layout = ["R", "C"]
    list_stride = [1, 3]
    list_side = [
        ("L", (8, 8)),
        ("R", (9, 9)),
    ]
    list_uplo = ["L", "U"]
    list_trans = ["N", "T", "C"]
    list_diag = ["N", "U"]

    set_inp = [
        list_num,
        list_stride, list_stride, list_stride, list_stride,
        list_layout, list_layout,
        list_side,
        list_uplo, list_trans, list_diag,
    ]
    run_size = 24

    tokens = []
    for n, list_taguchi in enumerate(taguchi_by_list(set_inp, run_size)):
        (
            num,
            as0, as1, bs0, bs1,
            al, bl,
            (side, (ad0, ad1)),
            uplo, trans, diag
        ) = list_taguchi
        token = (
            f"test_macro!(test_{n:03d}: inline, {num}, "
            f"{(ad0, ad1, as0, as1)}, {(8, 9, bs0, bs1)}, "
            f"'{al}', '{bl}', "
            f"'{side}', '{uplo}', '{trans}', '{diag}')"
        )
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    print("\n".join(gen_valid()))
