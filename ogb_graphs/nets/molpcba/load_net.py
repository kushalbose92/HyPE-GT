from nets.molpcba.SAN_NodeLPE import GraphTransformer

def GraphTransformerLoad(net_params):
    return GraphTransformer(net_params)


def gnn_model(MODEL_NAME, net_params):
    model = {
        'GraphTransformer': GraphTransformerLoad,
    }
        
    return model[MODEL_NAME](net_params)
