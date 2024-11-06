from nets.code2.SAN_NodeLPE import GraphTransformer

def GraphTransformerLoad(net_params, node_encoder):
    return GraphTransformer(net_params, node_encoder)


def gnn_model(MODEL_NAME, net_params, node_encoder):
    model = {
        'GraphTransformer': GraphTransformerLoad,
    }
        
    return model[MODEL_NAME](net_params, node_encoder)
