import numpy as np
import torch
import torch_scatter
from torch import nn
from tpu_graph.constants import MAX_OP_CODE


class EmbeddingInputLayer(nn.Module):
    """
    This is just a layer that splits of the first column of the input and embeds it
    """

    def __init__(self, in_channels: int, out_channels: int, emb_size: int = 32, num_embeddings: int = 128):
        """
        Inits the layer
        :param in_channels: The number of input channels without the embedding
        :param out_channels: The number of output channels after the projection
        :param emb_size: The size of the embedding
        :param num_embeddings: The number of embeddings
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)

        # the other layers
        self.linear = nn.Linear(in_channels + emb_size - 1, out_channels, bias=True)
        self.silu = nn.SiLU()
        self.layernorm = nn.LayerNorm(out_channels)

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
        x = torch.concatenate([embedding, x[..., 1:]], dim=-1)

        # project
        x = self.linear(x)
        x = self.silu(x)
        x = self.layernorm(x)

        return x


class SAGEConv(nn.Module):
    """
    Implements a simple SAGE convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        Inits the layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """

        # init the super class
        super().__init__()

        # save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # init the layers
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.agg_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.silu = nn.SiLU()
        self.layernorm = nn.LayerNorm(out_channels)

    def forward(self, inp_tensors: tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass of the layer
        :param inp_tensors: The input tensors, a tuple of (features, connection_matrix)
        :return: The output of the layer and the connection matrix
        """

        # unpack the input tensors
        x, connection_matrix = inp_tensors

        # the normal projection
        projection = self.linear(x)

        # get the input dimension
        list_dim, graph_dim, inp_dim = x.shape

        # (list, graph, inp) -> (graph, inp * list)
        x = x.transpose(0, 1).reshape(graph_dim, -1)

        # apply the connection matrix
        x = torch.sparse.mm(connection_matrix, x)

        # back to (list, graph, inp)
        x = x.reshape(graph_dim, list_dim, inp_dim).transpose(0, 1)

        # the aggregation projection
        agg_projection = self.agg_linear(x)

        # the output
        output = self.silu(projection + agg_projection)
        output = self.layernorm(output)

        return output, connection_matrix


class RetentiveAttention(nn.Module):
    """
    Implements a retentive attention layer
    """

    def __init__(
        self, in_channels: int, out_channels: int, key_dim: int = 16, n_iterations: int = 3, decay: float = 0.7
    ):
        """
        Inits the layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param key_dim: The dimension of the keys
        :param n_iterations: The number of iterations
        :param decay: The decay factor for the attention
        """

        # init the super class
        super().__init__()

        # save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_dim = key_dim
        self.n_iterations = n_iterations
        self.decay = decay

        # the linear layers
        self.key_embedding = nn.Linear(in_channels, key_dim, bias=False)
        self.query_embedding = nn.Linear(in_channels, key_dim, bias=False)
        self.value_embedding = nn.Linear(in_channels, out_channels, bias=False)
        self.layernorm = nn.LayerNorm(out_channels)

    def forward(self, inp_tensors: tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass of the layer
        :param inp_tensors: The input tensors, a tuple of (features, connection_matrix)
        :return: The output of the layer and the connection matrix
        """

        # unpack the input tensors
        x, connection_matrix = inp_tensors

        # the initial weights
        key = self.key_embedding(x)
        query = self.query_embedding(x)
        values = self.value_embedding(x)
        weights = (key * query).mean(dim=-1, keepdim=True)

        # now the recursive retention, weights are now (graph, list)
        iter_weights = torch.squeeze(weights).T
        for i in range(1, self.n_iterations):
            # apply the connection matrix with the decay (apply it directly to the weights because everything is linear)
            iter_weights = iter_weights * self.decay
            iter_weights = torch.sparse.mm(connection_matrix, iter_weights)

            # reshape and add to weights
            weights = weights + iter_weights.T.unsqueeze(-1)

        # apply the values
        output = self.layernorm(values * weights)

        return output, connection_matrix


class GPSConv(nn.Module):
    """
    Implements the scaled dot product attention with batching over one large tensor using the scatter library
    """

    def __init__(self, inp_dim: int, out_dim: int):
        """
        Init the layer
        :param inp_dim: The input dimension
        :param out_dim: The output dimension
        """

        # init the super class
        super().__init__()

        # save attributes
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        # init the layers
        self.sage_conv = SAGEConv(inp_dim, out_dim)
        self.attention = RetentiveAttention(inp_dim, out_dim)
        self.linear1 = nn.Linear(2 * out_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.silu = nn.SiLU()
        self.layernorm1 = nn.LayerNorm(out_dim)
        self.layernorm2 = nn.LayerNorm(out_dim)

    def forward(self, inp_tensors: tuple[torch.Tensor, torch.sparse.Tensor]):
        """
        Forward pass of the layer
        :param inp_tensors: The input tensors (features, connection_matrix)
        :return: The attention output
        """

        # unpack the input tensors
        x_orig, connection_matrix = inp_tensors

        # apply the layers
        sage_output, _ = self.sage_conv(inp_tensors)
        attention_output, _ = self.attention(inp_tensors)

        # add and project
        x = self.linear1(torch.concatenate([sage_output, attention_output], dim=-1))

        # activation and layer norm
        x = self.silu(x)
        x = self.layernorm1(x)

        # second layer with skip connection
        x = self.linear2(x)
        output = self.silu(x + x_orig)
        output = self.layernorm2(output)

        # we output the connection matrix for the next layer
        return output, connection_matrix


class TPUGraphNetwork(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_network: nn.Sequential,
        projection_network: nn.Module,
        graph_embedding_dim: int,
        key_dim: int = 32,
        op_embedding_dim: int = 32,
        **kwargs,
    ):
        """
        Init the network
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels for the embedding layer
        :param message_network: A network that performs the message passing
        :param projection_network: A network that projects the output of the transformer network to the output dimension
        :param graph_embedding_dim: The dimension of the graph embedding
        :param op_embedding_dim: The dimension of the op embedding
        :param kwargs: Additional arguments for the super class
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_layer = EmbeddingInputLayer(in_channels, out_channels, op_embedding_dim, MAX_OP_CODE)
        self.message_network = message_network
        self.projection_network = projection_network

        # for the final global attention
        self.k_dim = key_dim
        self.k = nn.Linear(graph_embedding_dim, 1, bias=False)
        self.silu = nn.SiLU()

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        lengths: list[int],
    ):
        """
        Forward pass of the network
        :param features: The input features (multiple graphs concatenated)
        :param edge_index: The indices of the connection matrix
        :param lengths: The lengths of the individual graphs
        :return: The predicted runtime in nanoseconds
        """

        # create and index for the scatter sum
        index = torch.Tensor(np.concatenate([np.ones(l) * i for i, l in enumerate(lengths)])).long().to("cuda")

        # build the connection matrix
        with torch.no_grad():
            n_nodes = features.shape[1]
            values = torch.ones(edge_index.shape[1]).to(edge_index.device)
            norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
            norm = norm.clamp(min=1.0)[edge_index[0]]
            values = values / norm
            connection_matrix = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))

        # embed the first column
        emb_features = self.embedding_layer(features)

        # apply the transformer networks
        graph_embedding, _ = self.message_network((emb_features, connection_matrix))

        # to key, query, value
        key = self.k(graph_embedding)

        # the weights are just the sum of the key
        weights = self.silu(key)

        # now we can apply the weights
        graph_embedding = torch_scatter.scatter_sum(weights * graph_embedding, index=index, dim=1)

        # now we do the final projection
        runtimes = self.projection_network(graph_embedding)

        return torch.squeeze(runtimes.transpose(0, 1))
