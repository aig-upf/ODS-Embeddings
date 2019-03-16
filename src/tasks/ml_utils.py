import numpy as np


merge_functions = {
    'concat': lambda x, y: np.concatenate([x, y]),
    'product': lambda x, y: x * y,
    'average': lambda x, y: 0.5 * (x + y),
    'l1': lambda x, y: np.abs(x - y),
    'l2': lambda x, y: np.abs(x - y) ** 2,
}


class NetworkFactory():
    def __init__(self, args):
        self.hidden_size       = args.hidden_size
        self.hidden_number     = args.hidden_number
        self.hidden_activation = args.hidden_activation
        self.output_activation = args.output_activation
        self.loss              = args.loss
        self.optimizer         = args.optimizer

    def make_network(self, input_size, output_size):
        from keras import Model
        from keras.layers import Input, Dense

        input_node = Input((input_size,), name='input_layer')
        hidden_cur = input_node
        for i in range(1, self.hidden_number + 1):
            hidden_cur = Dense(self.hidden_size, 
                               activation=self.hidden_activation, 
                               name='hidden_layer_{}'.format(i))(hidden_cur)
        output_node = Dense(output_size, 
                            activation=self.output_activation, 
                            name='output_layer')(hidden_cur)
        model = Model(inputs=input_node, 
                      outputs=output_node)
        model.compile(optimizer=self.optimizer,
                      loss=self.loss, 
                      metrics=['accuracy'] if 'entropy' in self.loss else []) 
        return model


def node_features(n, model_fn, feat_fn=None):
    '''
    Extract node features from the model and an arbitrary feature function.
    '''
    w = model_fn(n)
    if feat_fn is not None:
        f = feat_fn(n)
        return np.concatenate([w, f])
    return w


def edge_features(e, 
                  model_fn,
                  merge_fn, 
                  edge_feat_fn=None, 
                  node_feat_fn=None, 
                  order_independent=False):
    '''
    Extract edge features. The edge embeddings are given by:

        1. a model embedding function,
        2. a merge function of node features,
        3. additional feature functions for nodes and edges,

    Additionally, sort the node indices if index order matters for the model. 
    '''
    u, v = e
    u_w = node_features(u, model_fn, node_feat_fn)
    v_w = node_features(v, model_fn, node_feat_fn)
    m_w = merge_fn(u_w, v_w) if order_independent or u < v else merge_fn(v_w, u_w)
    if edge_feat_fn is not None:
        f = edge_feat_fn(e)
        return np.concatenate([m_w, f])
    return m_w

