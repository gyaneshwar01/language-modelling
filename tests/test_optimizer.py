import torch

from language_model.optimizer import AdamW


def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw(numpy_snapshot):
    """
    Our reference implementation yields slightly different results than the
    PyTorch AdamW, since there are a couple different ways that you can apply
    weight decay that are equivalent in principle, but differ in practice due to
    floating point behavior. So, we test that the provided implementation matches
    _either_ our reference implementation's expected results or those from the PyTorch AdamW.
    """
    # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(AdamW)

    # Might need to exit early if the weights match pytorch, since that should also be valid
    matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
    if matches_pytorch:
        return

    numpy_snapshot.assert_match(
        actual_weights,
        atol=1e-4,
    )
