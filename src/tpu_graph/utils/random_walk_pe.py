from typing import Any, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    get_self_loop_attr,
)


def add_node_attr(data: Data, value: Any, attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if "x" in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


class AddRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """

    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = "random_walk_pe",
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight

        adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float("inf")] = 0
        adj = adj * deg_inv.view(-1, 1)

        out = adj
        row, col, value = out.coo()
        pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            row, col, value = out.coo()
            pe = out.sum(dim=0) - get_self_loop_attr((row, col), value, num_nodes)
            pe_list.append(pe)
        pe = torch.stack(pe_list, dim=-1)

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data
