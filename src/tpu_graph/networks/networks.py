import numpy as np
import torch
import torch_scatter
from torch import nn
from tpu_graph.constants import MAX_OP_CODE


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


class BatchedMessagePassing(nn.Module):
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
        self.linear = nn.Linear(inp_dim, out_dim)
        self.silu = nn.SiLU()
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, inp_tensors: tuple[torch.Tensor, torch.sparse.Tensor]):
        """
        Forward pass of the layer
        :param inp_tensors: The input tensors (features, connection_matrix)
        :return: The attention output
        """

        # unpack the input tensors
        x, connection_matrix = inp_tensors

        # get the input dimension
        list_dim, graph_dim, inp_dim = x.shape

        # (list, graph, inp) -> (graph, inp * list)
        x = x.transpose(0, 1).reshape(graph_dim, -1)

        # apply the connection matrix
        x = torch.mm(connection_matrix, x)

        # back to (list, graph, inp)
        x = x.reshape(graph_dim, list_dim, inp_dim).transpose(0, 1)

        # the forward pass
        x = self.linear(x)

        # activation and layer norm
        output = self.silu(x)
        output = self.layernorm(output)

        # we output the connection matrix for the next layer
        return output, connection_matrix


class TPUGraphNetwork(nn.Module):
    """
    A simple network used for the tile predictions
    """

    def __init__(
        self,
        message_network: nn.Sequential,
        projection_network: nn.Module,
        op_embedding_dim: int = 32,
        n_edge_types: int = 512,
        decay: float = 0.5,
        exp: bool = False,
        **kwargs,
    ):
        """
        Init the network
        :param embedding_layer: A layer that embeds the first column of the input for the op code
        :param message_network: A network that performs the message passing
        :param projection_network: A network that projects the output of the transformer network to the output dimension
        :param exp: Exponentiate the output (makes stuff additive in the loss) if False, take the absolute value
        :param kwargs: Additional arguments for the super class
        """

        # init the super class
        super().__init__(**kwargs)

        # save attributes
        self.embedding_layer = EmeddingInputLayer(op_embedding_dim, MAX_OP_CODE)
        self.message_network = message_network
        self.projection_network = projection_network
        self.exp = exp

        # create the dists
        dists = decay ** (np.arange(n_edge_types) % MAX_OP_CODE)
        self.dists = torch.Tensor(dists).reshape(-1, 1)

    def forward(
        self,
        features: torch.Tensor,
        connection_matrix: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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

        # build the connection matrix
        with torch.no_grad():
            row_indices, col_indices, edge_codes = connection_matrix
            indices = torch.stack([row_indices, col_indices], dim=0)
            edge_types = torch.squeeze(nn.functional.embedding(edge_codes, self.dists.to(edge_codes.device)))
            connection_matrix = torch.sparse_coo_tensor(
                indices, edge_types, size=(features.shape[1], features.shape[1])
            )

        # embed the first column
        emb_features = self.embedding_layer(features)

        # apply the transformer networks
        graph_embedding, _ = self.message_network((emb_features, connection_matrix))
        runtimes = self.projection_network(graph_embedding)

        # exp the output if necessary
        if self.exp:
            runtimes = torch.exp(runtimes)
        else:
            runtimes = torch.abs(runtimes)

        # sum over the graphs
        runtimes = torch_scatter.scatter_sum(runtimes, index=index, dim=1)

        return torch.squeeze(runtimes.transpose(0, 1))
