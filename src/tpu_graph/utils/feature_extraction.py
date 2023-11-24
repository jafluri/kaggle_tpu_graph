import numpy as np

from tpu_graph.proto.hlo_pb2 import HloModuleProto


def get_additional_features(proto_module: HloModuleProto, data_dict: dict[str, np.ndarray], padding: int = -1):
    """
    Extracts features from the protobuf and extracts the dimensions of the inputs of the conv and dot operations
    :param proto_module: The protobuf module matching the data_dict
    :param data_dict: The data_dict of the npz file
    :param padding: The padding value to use
    :return: The input features and the new features
    """

    # define the type of the features to add
    dt = np.dtype(
        [
            ("has_dynamic_com", np.int64, (1,)),
            ("is_root_of_com", np.int64, (1,)),
            ("lhs_contracting_dimensions", np.int64, 3),
            ("rhs_contracting_dimensions", np.int64, 3),
            ("lhs_batch_dimensions", np.int64, 3),
            ("rhs_batch_dimensions", np.int64, 3),
            ("offset_dims", np.int64, 3),
            ("collapsed_slice_dims", np.int64, 3),
            ("start_index_map", np.int64, 3),
            ("index_vector_dim", np.int64, (1,)),
            ("gather_slice_sizes", np.int64, 5),
            ("indices_are_sorted", np.int64, (1,)),
        ]
    )
    # the names, will be important for the merge
    field_names = sorted(dt.fields)

    # create the map dict from id to index of the features vector and get the number of instructions
    id2index = {}
    i_id = 0
    n_i = 0
    for com in proto_module.computations:
        n_i += len(com.instructions)
        for i in com.instructions:
            id2index[i.id] = i_id
            i_id += 1

    # readout some of the data
    features = data_dict["node_feat"]
    op_codes = data_dict["node_opcode"]
    n_nodes = len(op_codes)
    assert n_nodes == n_i

    # init the new features (add one for the imaginary node)
    new_features = np.full(n_nodes, padding, dtype=dt)
    # the data for the input features (the input shapes of the operations)
    input_features = padding * np.ones((n_nodes, 16))

    # fill the global
    new_features["has_dynamic_com"] = proto_module.is_dynamic

    # extract
    i_id = 0
    for com in proto_module.computations:
        for i in com.instructions:
            # extract the inputs
            inputs = [id2index[node_id] for node_id in i.operand_ids]
            # if we have a conv or dot operation
            if op_codes[i_id] == 26 or op_codes[i_id] == 34:
                for n_in, in_index in enumerate(inputs):
                    input_features[i_id, 8 * (n_in) : 8 * (n_in + 1)] = features[in_index, 21:29]
            # otherwise we indicate with -2
            else:
                input_features[i_id, :] = -2

            # root of a computation
            new_features["is_root_of_com"][i_id] = com.root_id == i.id

            # the dot stuff
            for dot_attr in [
                "lhs_contracting_dimensions",
                "rhs_contracting_dimensions",
                "lhs_batch_dimensions",
                "rhs_batch_dimensions",
            ]:
                for id_x, val in enumerate(getattr(i.dot_dimension_numbers, dot_attr)):
                    new_features[dot_attr][i_id, id_x] = val

            # the gather stuff
            for gat_attr in ["offset_dims", "collapsed_slice_dims", "start_index_map"]:
                for id_x, val in enumerate(getattr(i.gather_dimension_numbers, gat_attr)):
                    new_features[gat_attr][i_id, id_x] = val
            new_features["index_vector_dim"][i_id] = i.gather_dimension_numbers.index_vector_dim
            for id_x, val in enumerate(i.gather_slice_sizes):
                new_features["gather_slice_sizes"][i_id, id_x] = val

            # the sorted indices
            new_features["indices_are_sorted"][i_id] = i.indices_are_sorted

            # increment
            i_id += 1

    # merge the new features
    feature_stack = np.concatenate([new_features[n] for n in field_names], axis=1, dtype=np.float64)

    return input_features, feature_stack
