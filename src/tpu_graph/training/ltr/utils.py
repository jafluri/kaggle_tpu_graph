from typing import Optional

import torch as _torch


def mask_padded_values(
    xs: _torch.FloatTensor, n: _torch.LongTensor, mask_value: float = -float("inf"), mutate: bool = False
):
    """Turns padded values into given mask value.

    Args:
        xs: A tensor of size (batch_size, list_size, 1) containing padded
            values.
        n: A tensor of size (batch_size) containing list size of each query.
        mask_value: The value to mask with (default: -inf).
        mutate: Whether to mutate the values of xs or return a copy.
    """
    mask = _torch.repeat_interleave(
        _torch.arange(xs.shape[1], device=xs.device).reshape((1, xs.shape[1])), xs.shape[0], dim=0
    )
    n_mask = _torch.repeat_interleave(n.reshape((n.shape[0], 1)), xs.shape[1], dim=1)
    if not mutate:
        xs = xs.clone()
    xs[mask >= n_mask] = mask_value
    return xs


def tiebreak_argsort(
    x: _torch.FloatTensor, descending: bool = True, generator: Optional[_torch.Generator] = None
) -> _torch.LongTensor:
    """Computes a per-row argsort of matrix x with random tiebreaks.

    Args:
        x: A 2D tensor where each row will be argsorted.
        descending: Whether to sort in descending order.

    Returns:
        A 2D tensor of the same size as x, where each row is the argsort of x,
        with ties broken randomly.
    """
    rng_kwargs = {"generator": generator} if generator is not None else {}
    p = _torch.randperm(x.shape[1], device=x.device, **rng_kwargs)
    return p[_torch.argsort(x[:, p], descending=descending)]


def rank_by_score(
    scores: _torch.FloatTensor, n: _torch.LongTensor, generator: Optional[_torch.Generator] = None
) -> _torch.LongTensor:
    """Sorts scores in decreasing order.

    This method ensures that padded documents are placed last and ties are
    broken randomly.

    Args:
        scores: A tensor of size (batch_size, list_size, 1) or
                (batch_size, list_size) containing scores.
        n: A tensor of size (batch_size) containing list size of each query.
    """
    if scores.dim() == 3:
        scores = scores.reshape((scores.shape[0], scores.shape[1]))
    return tiebreak_argsort(mask_padded_values(scores, n), generator=generator)


def batch_pairs(x: _torch.Tensor) -> _torch.Tensor:
    """Returns a pair matrix

    This matrix contains all pairs (i, j) as follows:
        p[_, i, j, 0] = x[_, i]
        p[_, i, j, 1] = x[_, j]

    Args:
        x: The input batch of dimension (batch_size, list_size) or
            (batch_size, list_size, 1).

    Returns:
        Two tensors of size (batch_size, list_size ^ 2, 2) containing
        all pairs.
    """

    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    # Construct broadcasted x_{:,i,0...list_size}
    x_ij = _torch.repeat_interleave(x, x.shape[1], dim=2)

    # Construct broadcasted x_{:,0...list_size,i}
    x_ji = _torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return _torch.stack([x_ij, x_ji], dim=3)
