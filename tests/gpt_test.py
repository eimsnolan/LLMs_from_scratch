from types import SimpleNamespace

import torch

from ..models import gpt


def test_MLP():
    # setting up a simple config
    config = SimpleNamespace(n_embd=768, bias=True, dropout=0.1)

    # instantiate the model
    mlp = gpt.MLP(config)

    # create a random tensor of size [10, 768]
    x = torch.randn(10, 768)

    # forward pass
    output = mlp(x)

    # the output should have the same size as input for MLP
    assert output.shape == x.shape, "The output shape is not as expected"

    # make sure no element in output tensor is NaN or infinite
    assert not torch.any(torch.isnan(output)), "The output tensor contains NaN values"
    assert not torch.any(torch.isinf(output)), "The output tensor contains Inf values"
