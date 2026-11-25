import numpy
import torch
import torch.nn.functional as F

from language_model.utils import cross_entropy


def test_cross_entropy():
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1)).detach().numpy(),
        expected.detach().numpy(),
        atol=1e-4,
    )

    # Test that cross-entropy handles numerical overflow issues
    large_inputs = 1000.0 * inputs
    large_expected_cross_entropy = F.cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)).detach().numpy(),
        large_expected_cross_entropy.detach().numpy(),
        atol=1e-4,
    )
