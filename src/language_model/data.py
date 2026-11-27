import numpy as np
import numpy.typing as npt
import torch


def get_batch(x: npt.NDArray, batch_size: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    start_idxs = torch.randint(len(x) - seq_len, (batch_size,))  # high = len(x) - seq_len is exclusive.
    inputs = torch.stack([torch.from_numpy((x[i : i + seq_len]).astype(np.int64)) for i in start_idxs])
    outputs = torch.stack([torch.from_numpy((x[i + 1 : i + seq_len + 1]).astype(np.int64)) for i in start_idxs])

    inputs = inputs.to(device)
    outputs = outputs.to(device)
    return inputs, outputs
