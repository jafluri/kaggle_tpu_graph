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
        layer_norm: bool = True,
    ):
        """
        Inits the layer
        :param in_channels: The number of input channels without the embedding
        :param out_channels: The number of output channels after the projection
        :param emb_size: The size of the embedding
        :param num_embeddings: The number of embeddings
        :param layer_norm: Whether to use layer norm or not
        """

        # this line is mandatory for all subclasses
        super().__init__()

        # save the attributes
        self.emb_size = emb_size
        self.num_embeddings = num_embeddings
        self.in_channels = in_channels
        self.out_channels = out_channels
        # some dims
        self.full_dim = in_channels + emb_size - 1

        # init the embedding
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_size)

        # the other layers
        self.mlp = nn.Sequential(
            nn.Linear(self.full_dim, out_channels, bias=True),
            nn.SiLU(),
            nn.LayerNorm(out_channels) if layer_norm else nn.Identity(),
        )

    def forward(self, op_code: torch.Tensor, features: torch.Tensor):
        """
        Forward pass of the layer
        :param op_code: The op code
        :param features: The features
        :return: The output of the layer
        """

        # get the first column and convert to int
        op_code = torch.squeeze(op_code, dim=-1).long()

        # embed the first column
        embedding = self.emb(op_code)

        # concatenate
        x = torch.concatenate([embedding, features], dim=-1)

        # project
        x = self.mlp(x)

        return x


class SAGEConv(nn.Module):
    """
    Implements a simple SAGE convolution
    """

    def __init__(self, in_channels: int, out_channels: int, message_dim=32, pe_conv=False):
        """
        Inits the layer
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param message_dim: The dimension of the messages
        :param pe_conv: Whether this is an PE convolution or not, if True, no layer norm is applied and the
                        activation is tanh
        """

        # init the super class
        super().__init__()

        # save attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_dim = message_dim
        self.lpe_conv = pe_conv

        # init the layers
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.agg_linear_in = nn.Linear(in_channels, message_dim, bias=True)
        self.agg_linear_out = nn.Linear(in_channels, message_dim, bias=True)

        # for the output MLP with layer norm
        if pe_conv:
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
        n_pe_features: int,
        embedding_dim: int = 32,
        pe_embedding_dim: int = 32,
        message_dim: int = 32,
        linformer_dim: int = 32,
        dropout: float = 0.0,
    ):
        """
        Inits the network
        :param embedding_out: The number of output channels after the embedding layer
        :param message_network_dims: The dimensions of the message network
        :param n_normal_features: The number of normal features
        :param n_pe_features: The number of PE features
        :param embedding_dim: The dimension of the embedding
        :param pe_embedding_dim: The dimension of the PE embedding (used to learn the PE features)
        :param message_dim: The dimension of the messages
        :param linformer_dim: The dimension of the linformer
        """

        # init the super class
        super().__init__()

        # save attributes
        self.embedding_out = embedding_out
        self.message_network_dims = message_network_dims
        self.n_normal_features = n_normal_features
        self.n_lpe_features = n_pe_features
        self.in_channels = n_normal_features + n_pe_features + 1
        self.embedding_dim = embedding_dim
        self.lpe_embedding_dim = pe_embedding_dim
        self.message_dim = message_dim
        self.linformer_dim = linformer_dim
        self.dropout = dropout

        # the dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # embedding layer
        self.embedding_layer = EmbeddingInputLayer(
            in_channels=n_normal_features,
            out_channels=embedding_out,
            num_embeddings=MAX_OP_CODE,
            emb_size=embedding_dim,
        )

        # lpe initial projection
        self.lpe_projection = nn.Sequential(
            nn.Linear(n_pe_features, pe_embedding_dim),
            nn.Tanh(),
        )

        # the message network
        self.feature_sage_convs = nn.ModuleList()
        self.lpe_sage_convs = nn.ModuleList()
        self.linformers = nn.ModuleList()
        self.combination_nets = nn.ModuleList()
        message_network_dims = [embedding_out] + message_network_dims
        for i, (in_dim, out_dim) in enumerate(zip(message_network_dims[:-1], message_network_dims[1:])):
            self.feature_sage_convs.append(SAGEConv(pe_embedding_dim + in_dim, out_dim, message_dim=message_dim))
            self.lpe_sage_convs.append(
                SAGEConv(pe_embedding_dim, pe_embedding_dim, pe_conv=True, message_dim=pe_embedding_dim // 2)
            )
            self.linformers.append(
                LinFormer(pe_embedding_dim + in_dim, out_dim, key_dim=linformer_dim, query_dim=linformer_dim)
            )
            self.combination_nets.append(
                nn.Sequential(
                    nn.Linear(2 * out_dim, out_dim),
                    nn.SiLU(),
                    nn.LayerNorm(out_dim),
                )
            )

        # last mlp to fold in the final lpe features
        in_dim = message_network_dims[-1] + pe_embedding_dim
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
            features, [1, self.n_normal_features, self.n_lpe_features, self.n_configs], dim=-1
        )

        # glue the configs to the features
        features = torch.cat([features, configs], dim=-1)

        # embed the first column
        emb_features = self.embedding_layer(op_code, features)

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
