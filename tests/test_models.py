import torch

from vuldee.models import BGRULocator, locate


def test_locator_forward_shapes():
    model = BGRULocator(vector_dim=4, hidden_size=8, n_layers=2, dropout=0.1)
    x = torch.randn(2, 6, 4)
    lengths = torch.tensor([6, 4])
    place = torch.zeros(2, 6)
    place[:, 2] = 1
    probs, score = model(x, lengths, place)
    assert tuple(probs.shape) == (2, 6)
    assert tuple(score.shape) == (2,)


def test_locate_returns_expected_lines():
    probs = [0.1, 0.2, 0.9, 0.8, 0.1, 0.1]
    assert locate(probs, [0, 2, 4], k=1, threshold=0.5) == [1]
