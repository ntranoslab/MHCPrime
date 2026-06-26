import numpy as np
import torch
from torch.utils.data import TensorDataset

from mhcprime.samplers import make_simple_negative_loader


def test_make_simple_negative_loader_num_workers_zero_with_persistent_workers_true():
    dataset = TensorDataset(torch.arange(6))
    labels = np.array([1, 1, 1, 0, 0, 0])

    loader, sampler = make_simple_negative_loader(
        dataset=dataset,
        labels=labels,
        batch_size=2,
        num_pos_per_epoch=2,
        neg_pos_ratio=1,
        num_workers=0,
        collate_fn=None,
        persistent_workers=True,
        seed=42,
    )

    batch = next(iter(loader))

    assert batch is not None
    assert len(loader) >= 1