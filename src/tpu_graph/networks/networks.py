import numpy as np
import torch
import torch_scatter
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
        op_code = x[..., 0].long()

        # embed the first column
        embedding = self.emb(op_code)

        return torch.concatenate([embedding, x[..., 1:]], dim=-1)


class BatchedSemiAttention(nn.Module):
    """
    Implements the scaled dot product attention with batching over one large tensor using the scatter library
    """

    def __init__(self, inp_dim: int, val_dim: int, key_dim: int):
        """
        Init the layer
        :param inp_dim: The input dimension
        :param val_dim: The embedding dimension
        :param key_dim: The key dimension
        """

        # init the super class
        super().__init__()

        # save attributes
        self.inp_dim = inp_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        # init the layers
        self.k = nn.Linear(inp_dim, key_dim)
        self.v = nn.Linear(inp_dim, val_dim)
        self.q = nn.Linear(inp_dim, key_dim)
        self.out = nn.Linear(val_dim, val_dim)
        self.silu = nn.SiLU()
        self.layernorm = nn.LayerNorm(val_dim)

    def forward(self, inp_tensors: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):
        """
        Forward pass of the layer
        :param inp_tensors: The input tensors (features, connection_matrix)
        :return: The attention output
        """

        # unpack the input tensors
        x, connection_matrix = inp_tensors
        row_indices, col_indices = connection_matrix

        # get the keys and values
        keys = self.k(x)
        values = self.v(x)
        queries = self.q(x)

        # we need to reshape the keys for the embedding lookup (list, graph, features) -> (graph, -1)
        list_dim, graph_dim, key_dim = keys.shape
        keys = keys.transpose(0, 1).reshape(graph_dim, -1)
        keys = nn.functional.embedding(col_indices, keys).reshape(col_indices.shape[0], list_dim, key_dim)

        # same for the queries
        queries = queries.transpose(0, 1).reshape(graph_dim, -1)
        queries = nn.functional.embedding(col_indices, queries).reshape(col_indices.shape[0], list_dim, key_dim)

        # we multiply the keys with the queries (graph, list, features) * (graph, list, features)
        weights = keys * queries
        # sum over the features dimension
        weights = torch.sum(weights, dim=2, keepdim=True)
        # softmax with the row indices as the graph dimension
        weights = torch_scatter.scatter_softmax(weights, row_indices, dim=0)

        # look up the values
        values = values.transpose(0, 1).reshape(graph_dim, -1)
        values = nn.functional.embedding(col_indices, values).reshape(col_indices.shape[0], list_dim, self.val_dim)

        # now we do values times weights and sum over the graph dimension
        values = values * weights

        # sum over the lep dimension
        output = torch_scatter.scatter_sum(values, row_indices, dim=0)

        # permute to list, graph, features
        output = output.transpose(0, 1)

        # activation and layer norm
        output = self.silu(output)
        output = self.layernorm(output)

        # we output the connection matrix for the next layer
        return output, connection_matrix


class TPUGraphNetwork(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        embedding_layer: EmeddingInputLayer,
        local_transformer_network: nn.Sequential,
        projection_network: nn.Module,
        exp: bool = False,
        **kwargs,
    ):
        """
        Init the network
        :param embedding_layer: A layer that embeds the first column of the input for the op code
        :param local_transformer_network: A list of batched semi attention layers
        :param projection_network: A network that projects the output of the transformer network to the output dimension
        :param exp: Exponentiate the output (makes stuff additive in the loss) if False, take the absolute value
        :param kwargs: Additional arguments for the super class
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_layer = embedding_layer
        self.local_transformer_network = local_transformer_network
        self.projection_network = projection_network
        self.exp = exp

    def forward(
        self,
        features: torch.Tensor,
        connection_matrix: tuple[torch.Tensor, torch.Tensor],
        lengths: list[int],
    ):
        """
        Forward pass of the network
        :param features: The input features (multiple graphs concatenated)
        :param connection_matrix: The connection matrix for the graphs
        :param lengths: The lengths of the individual graphs
        :return: The predicted runtime in nanoseconds
        """

        # create and index for the scatter sum
        index = torch.Tensor(np.concatenate([np.ones(l) * i for i, l in enumerate(lengths)])).long().to("cuda")

        # embed the first column
        emb_features = self.embedding_layer(features)

        # apply the transformer networks
        graph_embedding, _ = self.local_transformer_network((emb_features, connection_matrix))
        runtimes = self.projection_network(graph_embedding)

        # exp the output if necessary
        if self.exp:
            runtimes = torch.exp(runtimes)
        else:
            runtimes = torch.abs(runtimes)

        # sum over the graphs
        runtimes = torch_scatter.scatter_sum(runtimes, index=index, dim=1)

        return torch.squeeze(runtimes.transpose(0, 1))
