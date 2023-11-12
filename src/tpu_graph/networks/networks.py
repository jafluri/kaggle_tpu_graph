import numpy as np
import torch
import torch_scatter
from torch import nn
from tpu_graph.constants import MAX_OP_CODE


class EmbeddingInputLayer(nn.Module):
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
        self.full_dim = in_channels + emb_size + n_projections

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

    def forward(self, op_code: torch.Tensor, features: torch.Tensor, configs: torch.Tensor, dim_features: torch.Tensor):
        """
        Forward pass of the layer
        :param op_code: The op code
        :param features: The features
        :param configs: The configs
        :param dim_features: The dim features
        :return: The output of the layer
        """

        # get the first column and convert to int
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


class EmbeddingInputLayerV2(nn.Module):
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
        n_dim_features: int = 74,
        layer_norm: bool = True,
        **kwargs,
    ):
        """
        Inits the layer
        :param in_channels: The number of input channels without the embedding
        :param out_channels: The number of output channels after the projection
        :param emb_size: The size of the embedding
        :param num_embeddings: The number of embeddings
        :param n_configs: The number of configurations
        :param n_dim_features: The number of dimension features
        :param layer_norm: Whether to use layer norm or not
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings
        self.n_configs = n_configs
        self.n_dim_features = n_dim_features

        # some dims
        self.full_dim = in_channels + emb_size + n_configs + n_dim_features

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)

        # the other layers
        self.mlp = nn.Sequential(
            nn.Linear(self.full_dim, out_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(out_channels) if layer_norm else nn.Identity(),
        )

    def forward(self, op_code: torch.Tensor, features: torch.Tensor, configs: torch.Tensor, dim_features: torch.Tensor):
        """
        Forward pass of the layer
        :param op_code: The op code
        :param features: The features
        :param configs: The configs
        :param dim_features: The dim features
        :return: The output of the layer
        """

        # get the first column and convert to int
        op_code = torch.squeeze(op_code, dim=-1).long()

        # embed the first column
        embedding = self.emb(op_code)

        # concatenate
        x = torch.concatenate([embedding, features, dim_features, configs], dim=-1)

        # project
        x = self.mlp(x)

        return x


class SAGEConv(nn.Module):
    """
    Implements a simple SAGE convolution
    """

    def __init__(self, in_channels: int, out_channels: int, message_dim=32, lpe_conv=False):
        """
        Inits the layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param message_dim: The dimension of the messages
        :param lpe_conv: Whether this is an LPE convolution or not, if True, no layer norm is applied and the
                         activation is tanh
        """

        # init the super class
        super().__init__()

        # save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_dim = message_dim
        self.lpe_conv = lpe_conv

        # init the layers
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.agg_linear_in = nn.Linear(in_channels, message_dim, bias=True)
        self.agg_linear_out = nn.Linear(in_channels, message_dim, bias=True)

        # for the output MLP with layer norm
        if lpe_conv:
            self.mlp_out = nn.Sequential(
                nn.Tanh(),
                nn.Linear(out_channels + 2 * message_dim, out_channels),
                nn.Tanh(),
            )
        else:
            self.mlp_out = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(out_channels + 2 * message_dim),
                nn.Linear(out_channels + 2 * message_dim, out_channels),
                nn.SiLU(),
                nn.LayerNorm(out_channels),
            )

    def forward(
        self, x: torch.Tensor, connection_matrix_in: torch.sparse.Tensor, connection_matrix_out: torch.sparse.Tensor
    ):
        """
        Forward pass of the layer
        :param x: The input features
        :param connection_matrix_in: The connection matrix for the incoming edges
        :param connection_matrix_out: The connection matrix for the outgoing edges
        :return: The output of the layer
        """

        # project everything
        projection = self.linear(x)
        projection_in = self.agg_linear_in(x)
        projection_out = self.agg_linear_out(x)

        # get the input dimension
        list_dim, graph_dim, _ = x.shape

        # (list, graph, inp) -> (graph, inp * list)
        projection_in = projection_in.transpose(0, 1).reshape(graph_dim, -1)
        projection_out = projection_out.transpose(0, 1).reshape(graph_dim, -1)

        # apply the connection matrix
        in_coming = torch.sparse.mm(connection_matrix_in, projection_in)
        out_going = torch.sparse.mm(connection_matrix_out, projection_out)

        # back to (list, graph, inp)
        agg_projection_in = in_coming.reshape(graph_dim, list_dim, self.message_dim).transpose(0, 1)
        agg_projection_out = out_going.reshape(graph_dim, list_dim, self.message_dim).transpose(0, 1)

        # the output
        output = torch.concatenate([projection, agg_projection_in, agg_projection_out], dim=-1)

        # apply the MLP
        output = self.mlp_out(output)

        return output


class LinFormer(nn.Module):
    """
    A linear attention transformer
    """

    def __init__(self, in_channels, out_channels, key_dim=64, query_dim=64):
        """
        Init the layer
        :param in_channels: The number of input channels
        :param out_channels: The number of output channels
        :param key_dim: The dimension of the key
        :param query_dim: The dimension of the query
        """

        # init the super class
        super().__init__()

        # layers
        self.key_layer = nn.Linear(in_channels, key_dim, bias=True)
        self.query_layer = nn.Linear(in_channels, query_dim, bias=True)
        self.value_layer = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, features: torch.Tensor, lengths: list[int]):
        """
        Forward pass of the layer
        :param features: The input features
        :param lengths: The lengths of the individual graphs
        :return: The output of the layer
        """

        # embed the keys, queries and values
        keys = self.key_layer(features)
        queries = self.query_layer(features)
        values = self.value_layer(features)

        # activate the queries and keys
        keys = nn.functional.elu(keys) + 1
        queries = nn.functional.elu(queries) + 1

        # split everything
        keys = torch.split(keys, lengths, dim=1)
        queries = torch.split(queries, lengths, dim=1)
        values = torch.split(values, lengths, dim=1)

        # k.T@val
        ktv = [torch.matmul(k.transpose(1, 2), v) for k, v in zip(keys, values)]

        # now key @ qtv
        qktv = [torch.matmul(q, ktv_i) for q, ktv_i in zip(queries, ktv)]

        # stack everything
        qktv = torch.concatenate(qktv, dim=1)

        return qktv


class TPUGraphNetwork(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        embedding_out: int,
        message_network_dims: list[int],
        n_normal_features: int,
        n_dim_features: int,
        n_lpe_features: int,
        n_configs: int = 18,
        embedding_dim: int = 32,
        lpe_embedding_dim: int = 32,
        message_dim: int = 32,
        linformer_dim: int = 32,
        embedding_version: str = "v2",
        **kwargs,
    ):
        """
        Init the network
        :param embedding_out: Output dimension of the embedding layer
        :param message_network_dims: The dimensions of the message network (output dimensions of each layer)
        :param n_normal_features: The number of normal features
        :param n_dim_features: The number of dimension features
        :param n_lpe_features: The number of LPE features
        :param n_configs: The number of configurations
        :param kwargs: Additional arguments
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_out = embedding_out
        self.message_network_dims = message_network_dims
        self.n_normal_features = n_normal_features
        self.n_dim_features = n_dim_features
        self.n_lpe_features = n_lpe_features
        self.n_configs = n_configs
        self.in_channels = n_normal_features + n_dim_features + n_lpe_features + n_configs + 1
        self.embedding_dim = embedding_dim
        self.lpe_embedding_dim = lpe_embedding_dim
        self.message_dim = message_dim
        self.linformer_dim = linformer_dim

        # the embedding layer
        if embedding_version == "v1":
            emb_layer_class = EmbeddingInputLayer
        elif embedding_version == "v2":
            emb_layer_class = EmbeddingInputLayerV2
        else:
            raise ValueError(f"Unknown embedding version {embedding_version}")

        self.embedding_layer = emb_layer_class(
            in_channels=n_normal_features,
            out_channels=embedding_out,
            num_embeddings=MAX_OP_CODE,
            emb_size=embedding_dim,
            n_configs=n_configs,
            n_dim_features=n_dim_features,
            n_projections=n_configs,
        )

        # lpe initial projection
        self.lpe_projection = nn.Sequential(
            nn.Linear(n_lpe_features, lpe_embedding_dim),
            nn.Tanh(),
        )

        # the message network
        self.feature_sage_convs = nn.ModuleList()
        self.lpe_sage_convs = nn.ModuleList()
        self.linformers = nn.ModuleList()
        self.combination_nets = nn.ModuleList()
        message_network_dims = [embedding_out] + message_network_dims
        for i, (in_dim, out_dim) in enumerate(zip(message_network_dims[:-1], message_network_dims[1:])):
            self.feature_sage_convs.append(SAGEConv(lpe_embedding_dim + in_dim, out_dim, message_dim=message_dim))
            self.lpe_sage_convs.append(
                SAGEConv(lpe_embedding_dim, lpe_embedding_dim, lpe_conv=True, message_dim=lpe_embedding_dim // 2)
            )
            self.linformers.append(
                LinFormer(lpe_embedding_dim + in_dim, out_dim, key_dim=linformer_dim, query_dim=linformer_dim)
            )
            self.combination_nets.append(
                nn.Sequential(
                    nn.Linear(2 * out_dim, out_dim),
                    nn.SiLU(),
                    nn.LayerNorm(out_dim),
                )
            )

        # last mlp to fold in the final lpe features
        in_dim = message_network_dims[-1] + lpe_embedding_dim
        out_dim = message_network_dims[-1]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )

        # final projection
        self.projection_network = nn.Linear(out_dim, 1, bias=False)

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
        index = torch.Tensor(np.concatenate([np.ones(l) * i for i, l in enumerate(lengths)])).long().to(features.device)

        # build the connection matrix
        with torch.no_grad():
            # create the connection matrix
            n_nodes = features.shape[1]
            values = torch.ones(edge_index.shape[1]).to(edge_index.device)
            norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
            norm = norm.clamp(min=1.0)[edge_index[0]]
            values = values / norm
            connection_matrix_in = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))

            # the other connection matrix if necessary
            edge_index = edge_index.flip(0)
            values = torch.ones(edge_index.shape[1]).to(edge_index.device)
            norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
            norm = norm.clamp(min=1.0)[edge_index[1]]
            values = values / norm
            connection_matrix_out = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))

        # split the input features
        op_code, features, dim_features, lpe_features, configs = torch.split(
            features, [1, self.n_normal_features, self.n_dim_features, self.n_lpe_features, self.n_configs], dim=-1
        )

        # embed the first column
        emb_features = self.embedding_layer(op_code, features, configs, dim_features)

        # project the LPE features
        lpe_features = self.lpe_projection(lpe_features)

        # cycle through all layers
        for feature_sage_conv, lpe_sage_conv, linformer, combi_net in zip(
            self.feature_sage_convs, self.lpe_sage_convs, self.linformers, self.combination_nets
        ):
            # add the current LPE features to the features
            emb_features = torch.cat([lpe_features, emb_features], dim=-1)

            # apply the feature sage conv
            sage_features = feature_sage_conv(emb_features, connection_matrix_in, connection_matrix_out)

            # apply the LPE sage conv
            lpe_features = lpe_sage_conv(lpe_features, connection_matrix_in, connection_matrix_out)

            # apply the linformer
            linformer_features = linformer(emb_features, lengths)

            # concatenate and combine
            emb_features = torch.cat([sage_features, linformer_features], dim=-1)
            emb_features = combi_net(emb_features)

        # final mlp
        emb_features = torch.cat([lpe_features, emb_features], dim=-1)
        emb_features = self.mlp(emb_features)

        # sum over the graphs
        graph_embedding = torch_scatter.scatter_sum(emb_features, index=index, dim=1)

        # now we do the final projection
        runtimes = self.projection_network(graph_embedding)

        return graph_embedding, torch.squeeze(runtimes.transpose(0, 1), dim=-1)


class TPUGraphNetworkV2(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        embedding_out: int,
        message_network_dims: list[int],
        n_normal_features: int,
        n_dim_features: int,
        n_lpe_features: int,
        n_configs: int = 18,
        n_projections: int = 15,
        embedding_dim: int = 32,
        lpe_embedding_dim: int = 32,
        message_dim: int = 32,
        linformer_dim: int = 32,
        embedding_version: str = "v2",
        **kwargs,
    ):
        """
        Init the network
        :param embedding_out: Output dimension of the embedding layer
        :param message_network_dims: The dimensions of the message network (output dimensions of each layer)
        :param n_normal_features: The number of normal features
        :param n_dim_features: The number of dimension features
        :param n_lpe_features: The number of LPE features
        :param n_configs: The number of configurations
        :param kwargs: Additional arguments
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_out = embedding_out
        self.message_network_dims = message_network_dims
        self.n_normal_features = n_normal_features
        self.n_dim_features = n_dim_features
        self.n_lpe_features = n_lpe_features
        self.n_configs = n_configs
        self.n_projections = n_projections
        self.in_channels = n_normal_features + n_dim_features + n_lpe_features + n_configs + 1
        self.embedding_dim = embedding_dim
        self.lpe_embedding_dim = lpe_embedding_dim
        self.message_dim = message_dim
        self.linformer_dim = linformer_dim

        # the embedding layer
        if embedding_version == "v1":
            emb_layer_class = EmbeddingInputLayer
        elif embedding_version == "v2":
            emb_layer_class = EmbeddingInputLayerV2
        else:
            raise ValueError(f"Unknown embedding version {embedding_version}")

        self.embedding_layer = emb_layer_class(
            in_channels=n_normal_features,
            out_channels=embedding_out,
            num_embeddings=MAX_OP_CODE,
            emb_size=embedding_dim,
            n_configs=n_configs,
            n_dim_features=n_dim_features,
            n_projections=n_projections,
        )

        # lpe initial projection
        self.lpe_projection = nn.Sequential(
            nn.Linear(n_lpe_features, lpe_embedding_dim),
            nn.Tanh(),
        )

        # the message network
        self.feature_sage_convs = nn.ModuleList()
        self.lpe_sage_convs = nn.ModuleList()
        message_network_dims = [embedding_out] + message_network_dims
        for i, (in_dim, out_dim) in enumerate(zip(message_network_dims[:-1], message_network_dims[1:])):
            self.feature_sage_convs.append(SAGEConv(lpe_embedding_dim + in_dim, out_dim, message_dim=message_dim))
            self.lpe_sage_convs.append(
                SAGEConv(lpe_embedding_dim, lpe_embedding_dim, lpe_conv=True, message_dim=lpe_embedding_dim // 2)
            )

        # for the last layer we add a linformer and a combination net
        in_dim = message_network_dims[-2]
        out_dim = message_network_dims[-1]
        self.linformer = LinFormer(lpe_embedding_dim + in_dim, out_dim, key_dim=linformer_dim, query_dim=linformer_dim)
        self.combination_net = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )

        # fold in the last lpe
        in_dim = message_network_dims[-1] + lpe_embedding_dim
        out_dim = message_network_dims[-1]
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )

        # final projection
        out_dim = message_network_dims[-1]
        self.projection_network = nn.Linear(out_dim, 1, bias=False)

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
        index = torch.Tensor(np.concatenate([np.ones(l) * i for i, l in enumerate(lengths)])).long().to(features.device)

        # build the connection matrix
        with torch.no_grad():
            # create the connection matrix
            n_nodes = features.shape[1]
            values = torch.ones(edge_index.shape[1]).to(edge_index.device)
            norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
            norm = norm.clamp(min=1.0)[edge_index[0]]
            values = values / norm
            connection_matrix_in = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))

            # the other connection matrix if necessary
            edge_index = edge_index.flip(0)
            values = torch.ones(edge_index.shape[1]).to(edge_index.device)
            norm = torch_scatter.scatter_sum(values, index=edge_index[0], dim=0, dim_size=n_nodes)
            norm = norm.clamp(min=1.0)[edge_index[1]]
            values = values / norm
            connection_matrix_out = torch.sparse_coo_tensor(edge_index, values, (n_nodes, n_nodes))

        # split the input features
        op_code, features, dim_features, lpe_features, configs = torch.split(
            features, [1, self.n_normal_features, self.n_dim_features, self.n_lpe_features, self.n_configs], dim=-1
        )

        # embed the first column
        sage_features = self.embedding_layer(op_code, features, configs, dim_features)

        # project the LPE features
        lpe_features = self.lpe_projection(lpe_features)

        # cycle through all layers
        for feature_sage_conv, lpe_sage_conv in zip(self.feature_sage_convs, self.lpe_sage_convs):
            # add the current LPE features to the features
            emb_features = torch.cat([lpe_features, sage_features], dim=-1)

            # apply the feature sage conv
            sage_features = feature_sage_conv(emb_features, connection_matrix_in, connection_matrix_out)

            # apply the LPE sage conv
            lpe_features = lpe_sage_conv(lpe_features, connection_matrix_in, connection_matrix_out)

        # apply the linformer
        linformer_features = self.linformer(emb_features, lengths)

        # concatenate and combine
        emb_features = torch.cat([sage_features, linformer_features], dim=-1)
        emb_features = self.combination_net(emb_features)

        # fold in the last LPE features
        emb_features = torch.cat([lpe_features, emb_features], dim=-1)
        emb_features = self.mlp(emb_features)

        # now we can apply the weights
        graph_embedding = torch_scatter.scatter_sum(emb_features, index=index, dim=1)

        # now we do the final projection
        runtimes = self.projection_network(graph_embedding)

        return graph_embedding, torch.squeeze(runtimes.transpose(0, 1), dim=-1)


class TPUGraphNetworkSimple(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        embedding_out: int,
        message_network_dims: list[int],
        n_normal_features: int,
        n_dim_features: int,
        n_lpe_features: int,
        n_configs: int = 18,
        embedding_dim: int = 128,
        embedding_version: str = "v2",
        num_embeddings: int | None = None,
        layer_norm: bool = True,
        **kwargs,
    ):
        """
        Init the network
        :param embedding_out: Output dimension of the embedding layer
        :param message_network_dims: The dimensions of the message network (output dimensions of each layer)
        :param n_normal_features: The number of normal features
        :param n_dim_features: The number of dimension features
        :param n_lpe_features: The number of LPE features
        :param n_configs: The number of configurations
        :param kwargs: Additional arguments
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_out = embedding_out
        self.message_network_dims = message_network_dims
        self.n_normal_features = n_normal_features
        self.n_dim_features = n_dim_features
        self.n_lpe_features = n_lpe_features
        self.n_configs = n_configs
        self.in_channels = n_normal_features + n_dim_features + n_lpe_features + n_configs + 1
        self.embedding_dim = embedding_dim

        # the embedding layer
        if embedding_version == "v1":
            emb_layer_class = EmbeddingInputLayer
        elif embedding_version == "v2":
            emb_layer_class = EmbeddingInputLayerV2
        else:
            raise ValueError(f"Unknown embedding version {embedding_version}")

        self.embedding_layer = emb_layer_class(
            in_channels=n_normal_features,
            out_channels=embedding_out,
            num_embeddings=MAX_OP_CODE if num_embeddings is None else num_embeddings,
            emb_size=embedding_dim,
            n_configs=n_configs,
            n_dim_features=n_dim_features,
            n_projections=n_configs,
            layer_norm=layer_norm,
        )

        # the message network is a simple MLP
        message_network_dims = [embedding_out] + message_network_dims
        self.message_network = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(message_network_dims[:-1], message_network_dims[1:])):
            self.message_network.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.SiLU(),
                    nn.LayerNorm(out_dim) if layer_norm else nn.Identity(),
                )
            )

        # final projection
        out_dim = message_network_dims[-1]
        self.projection_network = nn.Linear(out_dim, 1, bias=False)

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
        index = torch.Tensor(np.concatenate([np.ones(l) * i for i, l in enumerate(lengths)])).long().to(features.device)

        # split the input features
        op_code, features, dim_features, lpe_features, configs = torch.split(
            features, [1, self.n_normal_features, self.n_dim_features, self.n_lpe_features, self.n_configs], dim=-1
        )

        # embed the first column
        emb_features = self.embedding_layer(op_code, features, configs, dim_features)

        # cycle through all layers
        for layer in self.message_network:
            # apply the layer
            emb_features = layer(emb_features)

        # now we can apply the weights
        graph_embedding = torch_scatter.scatter_sum(emb_features, index=index, dim=1)

        # now we do the final projection
        runtimes = self.projection_network(graph_embedding)

        return graph_embedding, torch.squeeze(runtimes.transpose(0, 1), dim=-1)
