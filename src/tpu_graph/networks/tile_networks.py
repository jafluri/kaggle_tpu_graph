import torch
from torch import nn

import numpy as np
from igraph import Graph


class TileNetwork(nn.Sequential):
    """
    A simple network used for the tile predictions
    """

    def __init__(self, *args):
        """
        Inits the network in the same way as the sequential network
        The network should take as an input a list of tensors with shape (n_nodes, n_features) and output a tensor with
        shape (n_nodes, 1) containing the predicted runtime in nanoseconds
        :param args: args for the sequential network, e.g. layers
        """

        # init the sequential network
        super().__init__(*args)

    def forward(self, features: list[torch.Tensor]):
        """
        Forward pass of the network
        :param features: A list of tensors with shape (n_nodes, n_features)
        :return: The predicted runtime in nanoseconds
        """

        # apply the sequential network for each element in the list
        node_runtimes = []
        for f in features:
            node_runtimes.append(super().forward(f))
        return node_runtimes

    def accumulate_runtime(self, features: list[torch.Tensor], edge_indices: list[torch.Tensor]):
        """
        Calls the network on the features and accumulates the runtime over the graph defined by the edge_indices
        :param features: A list of tensors with shape (n_nodes, n_features)
        :param edge_indices: A list of tensors with shape (n_edges, 2)
        :return: A tensor with length of the input features containing the accumulated runtime
        """

        # get the runtimes
        node_runtimes = self(features)

        # accumulate the runtimes
        accumulated_runtimes = []
        for runtimes, feat, edges in zip(node_runtimes, features, edge_indices):
            # we need to flip the edges because of the different definition of the edge index
            edges = np.fliplr(edges.cpu().numpy())
            runtime_numpy = runtimes.cpu().detach().numpy()

            # create the graph
            graph = Graph(n=len(runtime_numpy) + 1, edges=edges, directed=True)

            # get the output node (this is always an imaginary node added by the dataset loader)
            output_node = len(runtime_numpy)
            # The weight for an edge is the runtime of the source node
            weights = -runtime_numpy[edges[:, 0]]

            # get the shortest path
            dists = graph.distances(source=output_node, mode="in", weights=weights)[0]
            min_dist = np.argmin(dists)
            shortest_path = graph.get_shortest_paths(v=output_node, to=min_dist, weights=weights, mode="in")[0]
            # truncate the shortest path to avoid the output node
            shortest_path = shortest_path[1:]

            # now we create the accumulation vector
            accumulate_vec = np.zeros_like(runtime_numpy)
            accumulate_vec[shortest_path] = 1.0
            accumulate_vec = torch.tensor(accumulate_vec, dtype=torch.float32).to(runtimes.device)

            # accumulate the runtimes
            accumulated_runtimes.append((runtimes * accumulate_vec).sum())

        return accumulated_runtimes
