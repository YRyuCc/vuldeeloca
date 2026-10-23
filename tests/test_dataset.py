from vuldee.data.dataset import VulDataset
from vuldee.data.vectorize import build_place


def test_build_place_marks_vulnerable_line_range():
    place = build_place([0, 3, 6], [3], maxlen=10)
    assert place[0:3] == [0, 0, 0]
    assert place[3:6] == [1, 1, 1]


def test_build_place_empty_is_all_ones():
    assert build_place([0, 3], [], maxlen=10) == [1] * 10


def test_dataset_shapes():
    sample = {
        "id": "a",
        "vectors": [[0.1] * 30 for _ in range(5)],
        "linetokens": [0, 3],
        "vlinenum": [3],
        "label": 1,
    }
    ds = VulDataset([sample], maxlen=8, vector_dim=30)
    item = ds[0]
    assert tuple(item["vectors"].shape) == (8, 30)
    assert tuple(item["place"].shape) == (8,)
    assert item["length"].item() == 5
