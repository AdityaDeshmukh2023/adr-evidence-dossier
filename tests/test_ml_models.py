from __future__ import annotations

import pytest
torch = pytest.importorskip('torch')
from adr_system.ml.models import PairClassifier, MeanSage, mean_neighbors


def test_sparse_neighbor_mean_matches_direct_definition_and_gradient():
    x = torch.tensor([[1., 4.], [3., 6.], [9., 12.]], requires_grad=True)
    edge_index = torch.tensor([[0, 2, 1], [1, 1, 0]])
    actual = mean_neighbors(x, edge_index)
    assert torch.allclose(actual, torch.tensor([[3., 6.], [5., 8.], [0., 0.]]))
    actual.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([[.5, .5], [1., 1.], [.5, .5]]))


def test_linear_projection_before_mean_preserves_values_and_gradients():
    torch.manual_seed(17)
    layer = MeanSage(16, 5)
    x = torch.rand(6, 16, requires_grad=True)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    gates = torch.tensor([1., .3, 0., .7, 1., .5], requires_grad=True)
    original = layer.root(x) + layer.neighbor(mean_neighbors(x, edges, gates))
    optimized = layer(x, edges, gates)
    assert torch.allclose(original, optimized, atol=1e-6)
    variables = (x, gates, layer.neighbor.weight, layer.root.weight, layer.root.bias)
    reference = torch.autograd.grad(original.square().sum(), variables, retain_graph=True)
    actual = torch.autograd.grad(optimized.square().sum(), variables)
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(reference, actual))


@pytest.mark.parametrize('name', ['fingerprint', 'graphsage', 'fusion', 'role_fusion'])
def test_pair_decoder_is_symmetric_and_trainable(name):
    torch.manual_seed(17)
    model = PairClassifier(name, role_dim=2, fingerprint_dim=8, hidden=12, embedding=6, dropout=0)
    x, roles = torch.rand(4, 8), torch.rand(4, 2)
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    pairs = torch.tensor([[0, 2], [1, 3]])
    logits = model(x, edges, pairs, roles)
    reversed_logits = model(x, edges, pairs.flip(1), roles)
    assert torch.allclose(logits, reversed_logits)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
