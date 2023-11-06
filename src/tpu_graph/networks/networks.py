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


class EmbeddingInputLayerV2(nn.Module):
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
        full_dim = in_channels + emb_size - 1
        self.mlp = nn.Sequential(
            nn.Linear(full_dim, full_dim, bias=True),
            nn.SiLU(),
            nn.Linear(full_dim, out_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(out_channels),
        )

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
        x = self.mlp(x)

        return x


class EmbeddingInputLayerV3(nn.Module):
    """
    This is just a layer that splits of the first column of the input and embeds it
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_size: int = 32,
        num_embeddings: int = 128,
        n_configs: int = 18,
        n_projections: int = 8,
    ):
        """
        Inits the layer
        :param in_channels: The number of input channels without the embedding
        :param out_channels: The number of output channels after the projection
        :param emb_size: The size of the embedding
        :param num_embeddings: The number of embeddings
        :param n_configs: The number of configurations
        :param n_projections: The number of projections
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings
        self.n_configs = n_configs
        self.n_projections = n_projections

        # some dims
        self.full_dim = in_channels + emb_size + n_projections - 1
        self.n_features = in_channels - 1 - n_configs

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)
        self.projections = nn.Linear(n_configs, self.n_features * n_projections, bias=True)

        # the other layers
        self.mlp = nn.Sequential(
            nn.Linear(self.full_dim, out_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(out_channels),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the layer
        :param x: The input tensor
        :return: The input tensor with the first column embedded
        """

        # get the first column and convert to int
        op_code, features, configs = torch.split(x, [1, self.n_features, self.n_configs], dim=-1)
        op_code = torch.squeeze(op_code, dim=-1).long()

        # embed the first column
        embedding = self.emb(op_code)

        # project the configs
        configs_emb = self.projections(configs)
        configs_emb = nn.functional.gelu(configs_emb)
        list_dim, graph_dim, _ = configs_emb.shape
        configs_emb = configs_emb.reshape(list_dim, graph_dim, self.n_features, self.n_projections)

        # project the features
        weights = torch.einsum("lgf,lgfp->lgp", features, configs_emb)

        # concatenate
        x = torch.concatenate([embedding, features, configs, weights], dim=-1)

        # project
        x = self.mlp(x)

        return x


class EmbeddingInputLayerV4(nn.Module):
    """
    This is just a layer that splits of the first column of the input and embeds it
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_size: int = 32,
        num_embeddings: int = 128,
        n_configs: int = 18,
        n_projections: int = 18,
        n_dim_features: int = 74,
    ):
        """
        Inits the layer
        :param in_channels: The number of input channels without the embedding
        :param out_channels: The number of output channels after the projection
        :param emb_size: The size of the embedding
        :param num_embeddings: The number of embeddings
        :param n_configs: The number of configurations
        :param n_projections: The number of projections
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings
        self.n_configs = n_configs
        self.n_projections = n_projections
        self.n_dim_features = n_dim_features

        # some dims
        self.full_dim = in_channels + emb_size + n_projections - n_configs - n_dim_features - 1
        self.n_features = in_channels - 1 - n_configs - n_dim_features

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)
        self.projections = nn.Linear(n_configs, self.n_dim_features * n_projections, bias=True)

        # the other layers
        self.mlp = nn.Sequential(
            nn.Linear(self.full_dim, out_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(out_channels),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the layer
        :param x: The input tensor
        :return: The input tensor with the first column embedded
        """

        # get the first column and convert to int
        op_code, features, dim_features, configs = torch.split(
            x, [1, self.n_features, self.n_dim_features, self.n_configs], dim=-1
        )
        op_code = torch.squeeze(op_code, dim=-1).long()

        # embed the first column
        embedding = self.emb(op_code)

        # project the configs
        configs_emb = self.projections(configs)
        configs_emb = nn.functional.gelu(configs_emb)
        list_dim, graph_dim, _ = configs_emb.shape
        configs_emb = configs_emb.reshape(list_dim, graph_dim, self.n_dim_features, self.n_projections)

        # project the features
        weights = torch.einsum("lgf,lgfp->lgp", dim_features, configs_emb)

        # concatenate
        x = torch.concatenate([embedding, features, weights], dim=-1)

        # project
        x = self.mlp(x)

        return x


class EmbeddingInputLayerV5(nn.Module):
    """
    This is just a layer that splits of the first column of the input and embeds it
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_size: int = 32,
        num_embeddings: int = 128,
        n_configs: int = 18,
        n_projections: int = 18,
        n_dim_features: int = 74,
    ):
        """
        Inits the layer
        :param in_channels: The number of input channels without the embedding
        :param out_channels: The number of output channels after the projection
        :param emb_size: The size of the embedding
        :param num_embeddings: The number of embeddings
        :param n_configs: The number of configurations
        :param n_projections: The number of projections
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings
        self.n_configs = n_configs
        self.n_projections = n_projections
        self.n_dim_features = n_dim_features

        # some dims
        self.full_dim = in_channels + emb_size + n_projections - n_configs - n_dim_features - 1
        self.n_features = in_channels - 1 - n_configs - n_dim_features

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)
        self.projections = nn.Linear(n_configs, self.n_dim_features * n_projections, bias=True)

        # config mlp
        self.config_mlp = nn.Sequential(
            nn.Linear(n_configs, n_configs, bias=True),
            nn.SiLU(),
            nn.Linear(n_configs, n_configs, bias=True),
            nn.SiLU(),
        )
        self.dim_mlp = nn.Sequential(
            nn.Linear(n_dim_features, n_dim_features, bias=True),
            nn.SiLU(),
            nn.Linear(n_dim_features, n_dim_features, bias=True),
            nn.SiLU(),
        )

        # the other layers
        self.mlp = nn.Sequential(
            nn.Linear(self.full_dim, out_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(out_channels),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the layer
        :param x: The input tensor
        :return: The input tensor with the first column embedded
        """

        # get the first column and convert to int
        op_code, features, dim_features, configs = torch.split(
            x, [1, self.n_features, self.n_dim_features, self.n_configs], dim=-1
        )
        op_code = torch.squeeze(op_code, dim=-1).long()

        # embed the first column
        embedding = self.emb(op_code)

        # apply the MLPs
        configs = self.config_mlp(configs)
        dim_features = self.dim_mlp(dim_features)

        # project the configs
        configs_emb = self.projections(configs)
        configs_emb = nn.functional.gelu(configs_emb)
        list_dim, graph_dim, _ = configs_emb.shape
        configs_emb = configs_emb.reshape(list_dim, graph_dim, self.n_dim_features, self.n_projections)

        # project the features
        weights = torch.einsum("lgf,lgfp->lgp", dim_features, configs_emb)

        # concatenate
        x = torch.concatenate([embedding, features, weights], dim=-1)

        # project
        x = self.mlp(x)

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


class SAGEConvV2(nn.Module):
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

        # for the output MLP with layer norm
        self.mlp_out = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU(),
            nn.LayerNorm(out_channels),
        )

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

        # apply the MLP
        output = self.mlp_out(output)

        return output, connection_matrix


class SAGEConvV3(nn.Module):
    """
    Implements a simple SAGE convolution
    """

    def __init__(self, in_channels: int, out_channels: int, in_and_out: bool = True):
        """
        Inits the layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param in_and_out: If True, use the in and out connection matrices separately
        """

        # init the super class
        super().__init__()

        # save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_and_out = in_and_out

        if in_and_out:
            # init the layers
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
            self.agg_linear_in = nn.Linear(in_channels, out_channels, bias=True)
            self.agg_linear_out = nn.Linear(in_channels, out_channels, bias=True)
            self.silu = nn.SiLU()
            self.layernorm = nn.LayerNorm(3 * out_channels)

            # for the output MLP with layer norm
            self.mlp_out = nn.Sequential(
                nn.Linear(3 * out_channels, out_channels),
                nn.SiLU(),
                nn.LayerNorm(out_channels),
            )
        else:
            self.sage_conv = SAGEConv(in_channels, out_channels)

    def forward(self, inp_tensors: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]):
        """
        Forward pass of the layer
        :param inp_tensors: The input tensors, a tuple of (features, connection_matrix)
        :return: The output of the layer and the connection matrix
        """

        # unpack the input tensors
        x, connection_matrix = inp_tensors
        connection_matrix_in, connection_matrix_out = connection_matrix

        if not self.in_and_out:
            output, _ = self.sage_conv((x, connection_matrix_in))
            return output, connection_matrix

        # the normal projection
        projection = self.linear(x)

        # get the input dimension
        list_dim, graph_dim, inp_dim = x.shape

        # (list, graph, inp) -> (graph, inp * list)
        x = x.transpose(0, 1).reshape(graph_dim, -1)

        # apply the connection matrix
        in_coming = torch.sparse.mm(connection_matrix_in, x)
        out_going = torch.sparse.mm(connection_matrix_out, x)

        # back to (list, graph, inp)
        in_coming = in_coming.reshape(graph_dim, list_dim, inp_dim).transpose(0, 1)
        out_going = out_going.reshape(graph_dim, list_dim, inp_dim).transpose(0, 1)

        # the aggregation projection
        agg_projection_in = self.agg_linear_in(in_coming)
        agg_projection_out = self.agg_linear_out(out_going)

        # the output
        output = torch.concatenate([projection, agg_projection_in, agg_projection_out], dim=-1)
        output = self.silu(output)
        output = self.layernorm(output)

        # apply the MLP
        output = self.mlp_out(output)

        return output, connection_matrix


class RetentiveAttention(nn.Module):
    """
    Implements a retentive attention layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        key_dim: int = 16,
        n_iterations: int = 3,
        decay: list[float] | None = None,
    ):
        """
        Inits the layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param key_dim: The dimension of the keys
        :param n_iterations: The number of iterations
        :param decay: A list of decay values for the different attention heads (defaults to 8 linearly spaced values),
                      note that the number of heads is inferred from the length of the list and needs to evently
                      divide the output dimension
        """

        # init the super class
        super().__init__()

        # save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_dim = key_dim
        self.n_iterations = n_iterations

        if decay is None:
            decay = np.linspace(0.1, 0.5, 4)
        self.decay = decay

        # the linear layers
        self.key_embedding = nn.Linear(in_channels, key_dim * len(decay), bias=False)
        self.query_embedding = nn.Linear(in_channels, key_dim * len(decay), bias=False)
        self.value_embedding = nn.Linear(in_channels, out_channels, bias=False)
        self.layernorm = nn.LayerNorm(out_channels // len(decay))

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

        # activation
        key = nn.functional.elu(key) + 1
        query = nn.functional.elu(query) + 1

        # reshape the key (list, graph, key_dim * n_heads) -> (graph, list, n_heads, key_dim)
        key = key.reshape(x.shape[0], x.shape[1], -1, self.key_dim).transpose(0, 1)

        # do the retentive attention
        decay_tensor = torch.Tensor(self.decay)[None, None, :, None].to(key.device)
        iter_key = key
        for i in range(1, self.n_iterations):
            # apply the decay
            iter_key = iter_key * decay_tensor

            # shape to matrix (graph, list, n_heads, key_dim) -> (graph , n_heads * list * key_dim)
            iter_key = iter_key.reshape(x.shape[1], -1)
            iter_key = torch.sparse.mm(connection_matrix, iter_key)

            # back to (graph, list, n_heads)
            iter_key = iter_key.reshape(x.shape[1], x.shape[0], -1, self.key_dim)
            key += iter_key

        # weights are now (graph, list, n_heads)
        key = key.transpose(0, 1)

        # get the weights
        weights = key * query.reshape(x.shape[0], x.shape[1], len(self.decay), -1)

        # sum over the key dimension
        weights = weights.mean(dim=-1, keepdim=True)

        # reshape the values
        values = values.reshape(x.shape[0], x.shape[1], len(self.decay), -1)

        # apply the weights
        values = values * weights

        # apply normalization
        values = self.layernorm(values)

        # reshape back to (list, graph, out)
        output = values.reshape(x.shape[1], x.shape[0], -1).transpose(0, 1)

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
        op_embedding_dim: int = 32,
        embedding_version: str = "v1",
        dropout: float = 0.25,
        undirected: bool = False,
        in_and_out: bool = False,
        **kwargs,
    ):
        """
        Init the network
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels for the embedding layer
        :param message_network: A network that performs the message passing
        :param projection_network: A network that projects the output of the transformer network to the output dimension
        :param op_embedding_dim: The dimension of the op embedding
        :param dropout: The dropout to use
        :param undirected: If True, the connection matrix is symmetrized
        :param in_and_out: If True, The in and out edges connection matrices are calculated separately and are
                           fed as a tuple to the message network
        :param kwargs: Additional arguments for the super class
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.undirected = undirected
        self.in_and_out = in_and_out
        if embedding_version == "v1":
            self.embedding_layer = EmbeddingInputLayer(in_channels, out_channels, op_embedding_dim, MAX_OP_CODE)
        elif embedding_version == "v2":
            self.embedding_layer = EmbeddingInputLayerV2(in_channels, out_channels, op_embedding_dim, MAX_OP_CODE)
        elif embedding_version == "v3":
            self.embedding_layer = EmbeddingInputLayerV3(in_channels, out_channels, op_embedding_dim, MAX_OP_CODE)
        elif embedding_version == "v4":
            self.embedding_layer = EmbeddingInputLayerV4(in_channels, out_channels, op_embedding_dim, MAX_OP_CODE)
        elif embedding_version == "v5":
            self.embedding_layer = EmbeddingInputLayerV5(in_channels, out_channels, op_embedding_dim, MAX_OP_CODE)
        else:
            raise ValueError(f"Unknown embedding version {embedding_version}")
        self.message_network = message_network
        self.projection_network = projection_network
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        lengths: list[int],
        drop_mask: torch.Tensor | None = None,
    ):
        """
        Forward pass of the network
        :param features: The input features (multiple graphs concatenated)
        :param edge_index: The indices of the connection matrix
        :param lengths: The lengths of the individual graphs
        :param drop_mask: The dropout mask to use for the nodes (optional), if None, random dropout is used
        :return: The predicted runtime in nanoseconds
        """

        # create and index for the scatter sum
        index = torch.Tensor(np.concatenate([np.ones(l) * i for i, l in enumerate(lengths)])).long().to(features.device)

        # build the connection matrix
        with torch.no_grad():
            # add the reverse edges if necessary
            if self.undirected:
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)

            # create the connection matrix
            n_nodes = features.shape[1]
            values = torch.ones(edge_index.shape[1]).to(edge_index.device)
            norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
            norm = norm.clamp(min=1.0)[edge_index[0]]
            values = values / norm
            connection_matrix = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))

            # the other connection matrix if necessary
            if self.in_and_out:
                # create the connection matrix
                edge_index = edge_index.flip(0)
                values = torch.ones(edge_index.shape[1]).to(edge_index.device)
                norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
                norm = norm.clamp(min=1.0)[edge_index[1]]
                values = values / norm
                connection_matrix_out = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))
                connection_matrix = (connection_matrix, connection_matrix_out)

        # embed the first column
        emb_features = self.embedding_layer(features)

        # apply dropout
        if drop_mask is None:
            _, graph_dim, _ = emb_features.shape
            drop_mask = torch.ones((1, graph_dim, 1), device=emb_features.device)
            drop_mask = self.dropout(drop_mask)
        emb_features = emb_features * drop_mask

        # apply the transformer networks
        graph_embedding, _ = self.message_network((emb_features, connection_matrix))

        # now we can apply the weights
        graph_embedding = torch_scatter.scatter_sum(graph_embedding, index=index, dim=1)

        # now we do the final projection
        runtimes = self.projection_network(graph_embedding)

        return graph_embedding, torch.squeeze(runtimes.transpose(0, 1), dim=-1)
