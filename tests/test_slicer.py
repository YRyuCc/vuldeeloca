from vuldee.source2slice.slicer import backward_slice


IR = [
    "%1 = alloca i32, align 4",
    "%2 = load i32, i32* %1, align 4",
    "%3 = add nsw i32 %2, 1",
    "store i32 %3, i32* %1, align 4",
]


def test_backward_slice_includes_data_dependencies():
    sliced = backward_slice(IR, [2])
    assert any("%2 = load" in line for line in sliced)
    assert any("%3 = add" in line for line in sliced)
    assert any("%1 = alloca" in line for line in sliced)


def test_backward_slice_orders_by_source_line():
    sliced = backward_slice(IR, [2])
    assert sliced == sorted(sliced, key=IR.index)
