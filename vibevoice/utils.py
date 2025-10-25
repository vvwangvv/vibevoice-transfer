from typing import List

import torch


def make_pad_mask(lengths: List[int], max_len: int = None) -> torch.Tensor:
    if not lengths:
        return None
    if max_len is None:
        max_len = max(lengths)
    batch_size = len(lengths)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return mask
