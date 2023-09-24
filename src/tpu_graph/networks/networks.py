import torch
import torch_scatter
from igraph import Graph
from torch import nn


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


class BatchedSemiAttention(nn.Module):
    """
    Implements the scaled dot product attention with batching over one large tensor using the scatter library
    """

    def __init__(self, inp_dim: int, emb_dim: int):
        """
        Init the layer
        :param inp_dim: The input dimension
        :param emb_dim: The embedding dimension
        """

        # init the super class
        super().__init__()

        # save attributes
        self.inp_dim = inp_dim
        self.emb_dim = emb_dim

        # init the layers
        self.k = nn.Linear(inp_dim, emb_dim)
        self.v = nn.Linear(inp_dim, emb_dim)
        self.outp = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor, input_lengths: list[int]):
        """
        Forward pass of the layer
        :param x: The input where all batches are concatenated
        :param input_lengths: The lengths of the individual batches
        :return: The attention output
        """

        # create and index array for the scatter operation
        index = torch.cat([torch.ones(l) * i for i, l in enumerate(input_lengths)]).long().to(x.device)

        # get the keys and values
        keys = self.k(x)
        values = self.v(x)

        # sum all the keys and softmax
        softmax = torch_scatter.scatter_softmax(torch.sum(keys, dim=1, keepdim=True), index=index, dim=0)

        # multiply with the values
        output = torch_scatter.scatter_add(softmax * values, index=index, dim=0)

        # project to output
        output = self.outp(output)

        return output


class TPUGraphNetwork(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        embedding_layer: EmeddingInputLayer,
        projection_network: nn.Sequential,
        graph_embedding_network: nn.Sequential,
        batch_semi_attention: BatchedSemiAttention,
        exp: bool = False,
        **kwargs,
    ):
        """
        Init the network
        :param embedding_layer: A layer that embeds the first column of the input for the op code
        :param projection_network: The network that projects the input to the correct size
        :param graph_embedding_network: Network that takes the projections and features and embeds them
        :param batch_semi_attention: The attention for the final output
        :param exp: Exponentiate the output (makes stuff additive in the loss)
        :param kwargs: Additional arguments for the super class
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_layer = embedding_layer
        self.projection_network = projection_network
        self.graph_embedding_network = graph_embedding_network
        self.batch_semi_attention = batch_semi_attention
        self.exp = exp

        # dict for the paths
        self.path_dict = dict()

    def forward(self, features: torch.Tensor, connection_matrix: torch.Tensor, lengths: list[int]):
        """
        Forward pass of the network
        :param features: The input features (multiple graphs concatenated)
        :param connection_matrix: The connection matrix for the graphs
        :param lengths: The lengths of the individual graphs
        :return: The predicted runtime in nanoseconds
        """

        # embed the first column
        emb_features = self.embedding_layer(features)

        # project the features
        pro_features = self.projection_network(emb_features)

        # matrix dense multiplication
        features = torch.mm(connection_matrix, pro_features)

        # embed the graph to rutimes
        features = torch.cat([features, emb_features], dim=1)
        runtimes = self.graph_embedding_network(features)

        # attention
        runtimes = self.batch_semi_attention(runtimes, lengths)

        if self.exp:
            runtimes = torch.exp(runtimes)

        return runtimes.reshape(-1)

    def accumulate_runtime(
        self,
        features: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        connection_matrices: list[torch.Tensor],
        graphs: list[Graph],
        p_update_path: float = 1.0,
    ):
        """
        Calls the network on the features and accumulates the runtime over the graph defined by the edge_indices
        :param features: A list of tensors with shape (n_nodes, n_features)
        :param edge_indices: A list of tensors with shape (n_edges, 2)
        :param connection_matrices: A list of tensors with shape (n_nodes, n_nodes)
        :param graphs: A list of graphs
        :param p_update_path: The probability to update the longest path for a given graph. Defaults to 1.0 (always).
                              Reducing this value can speed up the runtime but also reduces the accuracy as
                              the longest path might not be correct for each run.
        :return: A tensor with length of the input features containing the accumulated runtime
        """

        # get the runtimes
        node_runtimes = self(features, connection_matrices)

        return node_runtimes
