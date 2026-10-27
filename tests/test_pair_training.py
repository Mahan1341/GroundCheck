"""CPU-only loss gradients and extractive token alignment; torch is optional in CI."""

import pytest

from grounded_hallucination_lab.research.pair_training import pair_objective, slot_mask


def test_slot_alignment_and_prefix_exclusion():
    e = {"answer_indices": [-1, -1, 0, 1, 2, -1], "labels": [-100, -100, 0, 1, 0, -100]}
    offsets = [(0, 3), (4, 6), (7, 12)]
    assert slot_mask(e, offsets, 4, 6) == [False, False, False, True, False, False]
    with pytest.raises(ValueError):
        slot_mask(e, offsets, 20, 22)
    with pytest.raises(ValueError):
        slot_mask(e, offsets, 4, 4)
    e["labels"][3] = -100
    with pytest.raises(ValueError):
        slot_mask(e, offsets, 4, 6)


def test_pair_direction_and_gradients():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((2, 4, 2), requires_grad=True)
    masks = [[False, True, False, False]] * 2
    loss = pair_objective(logits, masks)
    loss.backward()
    assert logits.grad[0, 1, 1] > 0  # gradient descent lowers supported risk
    assert logits.grad[1, 1, 1] < 0  # raises contradicted risk
    assert torch.count_nonzero(logits.grad[:, [0, 2, 3], :]) == 0
    good = logits.detach().clone()
    good[0, 1, 1] = -2
    good[1, 1, 1] = 2
    assert pair_objective(good, masks) < loss
    assert pair_objective(good.flip(0), masks) > loss
    shifted = good + 30
    assert torch.allclose(pair_objective(good, masks), pair_objective(shifted, masks))
    common_risk_shift = good.clone()
    common_risk_shift[:, :, 1] += 3
    assert torch.allclose(pair_objective(good, masks), pair_objective(common_risk_shift, masks))
    with pytest.raises(ValueError):
        pair_objective(good, [[False] * 4] * 2)


def test_lambda_zero_preserves_ce_gradients():
    torch = pytest.importorskip("torch")
    torch.manual_seed(42)
    a = torch.randn(2, 3, 2, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    targets = torch.tensor([[0, 0, 0], [0, 1, 0]])

    def ce(x):
        return torch.nn.functional.cross_entropy(x.reshape(-1, 2), targets.flatten())

    ce(a).backward()
    (ce(b) + 0 * pair_objective(b, [[False, True, False]] * 2)).backward()
    assert torch.equal(a.grad, b.grad)
