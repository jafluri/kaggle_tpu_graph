import torch
from torch import nn

import numpy as np
from igraph import Graph


class EmeddingInputLayer(nn.Module):
    """
    This is just a layer that splits of the first column of the input and embeds it
    """

    def __init__(self, emb_size: int = 32, num_embeddings: int = 128):
        """
        Inits the layer
        :param emb_size: The size of the embedding
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the layer
        :param x: The input tensor
        :return: The input tensor with the first column embedded
        """

        # get the first column and convert to int
        op_code = x[:, 0].long()

        # embed the first column
        embedding = self.emb(op_code)

        return torch.concatenate([embedding, x[:, 1:]], dim=1)


class TPUGraphNetwork(nn.Sequential):
    """
    A simple network used for the tile predictions
    """

    def __init__(self, *args, exp: bool = False, **kwargs):
        """
        Inits the network in the same way as the sequential network
        The network should take as an input a list of tensors with shape (n_nodes, n_features) and output a tensor with
        shape (n_nodes, 1) containing the predicted runtime in nanoseconds
        :param args: args for the sequential network, e.g. layers
        :param exp: If True, the runtime is taken as the exponential of the output of the network
        """

        # dict for the paths
        self.path_dict = dict()

        # save attributes
        self.exp = exp

        # init the sequential network
        super().__init__(*args)

    def forward(self, features: list[torch.Tensor]):
        """
        Forward pass of the network
        :param features: A list of tensors with shape (n_nodes, n_features)
        :return: The predicted runtime in nanoseconds
        """

        # we concat everything and split again for efficiency
        lengths = [f.shape[0] for f in features]
        features = torch.cat(features, dim=0)
        runtimes = super().forward(features)
        if self.exp:
            runtimes = torch.exp(runtimes)
        runtimes = torch.split(runtimes, lengths, dim=0)

        return runtimes

    def accumulate_runtime(
        self,
        features: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        graphs: list[Graph],
        p_update_path: float = 1.0,
    ):
        """
        Calls the network on the features and accumulates the runtime over the graph defined by the edge_indices
        :param features: A list of tensors with shape (n_nodes, n_features)
        :param edge_indices: A list of tensors with shape (n_edges, 2)
        :param graphs: A list of graphs
        :param p_update_path: The probability to update the longest path for a given graph. Defaults to 1.0 (always).
                              Reducing this value can speed up the runtime but also reduces the accuracy as
                              the longest path might not be correct for each run.
        :return: A tensor with length of the input features containing the accumulated runtime
        """

        # get the runtimes
        node_runtimes = self(features)

        # accumulate the runtimes
        accumulated_runtimes = []
        for runtimes, feat, edges, graph in zip(node_runtimes, features, edge_indices, graphs):
            if graph["name"] not in self.path_dict or np.random.uniform() < p_update_path:
                # detach and copy everything to cpu
                edges = edges.cpu().numpy()
                runtime_numpy = runtimes.cpu().detach().numpy()

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

                # save the path
                self.path_dict[graph["name"]] = accumulate_vec

            # get the accumulate vec
            accumulate_vec = self.path_dict[graph["name"]]

            # accumulate the runtimes
            accumulated_runtimes.append((runtimes * accumulate_vec).sum())

        return accumulated_runtimes
