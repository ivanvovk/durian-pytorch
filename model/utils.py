import torch


def get_mask_from_lengths(lengths, expand_multiple):
    max_len = lengths.max().item()
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len)).to(lengths)
    mask = (ids < lengths.unsqueeze(1)).bool().unsqueeze(-1)
    mask = torch.cat([mask for _ in range(expand_multiple)], dim=-1)
    return mask
