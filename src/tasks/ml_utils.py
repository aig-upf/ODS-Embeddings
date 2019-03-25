import numpy as np
from sklearn.preprocessing import StandardScaler


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


def make_label_vectorizer(labels, skip_none=True):
    '''Vectorize labels to automagically manage classification tasks.'''
    try:
        targets = {i for v in labels.values() for i in v}
    except TypeError:
        targets = {v for v in labels.values()}
    values = {v: i for i, v in enumerate(sorted(targets))}
    total_classes = len(values)

    if total_classes <= 2:
        return lambda l: values[l], 1

    def vectorizer(l):
        if skip_none and l is None:
            return None
        vector = np.zeros(total_classes)
        vector[values[l]] = 1.0
        return vector
    return vectorizer, total_classes


def prepare_data(G, mapping, model, labels, feat_fn=None, remove_unlabeled=True, vectorize=True):
    '''Prepare the full input features out of a given graph'''
    to_mapping = lambda v: mapping.get(v, None)
    raw_nodes  = [(v['name'], to_mapping(v['name'])) for v in G.vs]
    raw_labels = [labels.get(v['name'], None) for v in G.vs]

    # prepare the label vectorizer and the loss function
    if vectorize:
        vector_fn, vector_size = make_label_vectorizer(labels, skip_none=remove_unlabeled)
    else:
        vector_fn, vector_size = lambda l: l, np.asarray(labels.values())[0].size

    # get the raw data
    instances  = [(n, vector_fn(l)) for (n, l) in zip(raw_nodes, raw_labels) 
                                    if not remove_unlabeled or l is not None]
    full_data  = [(node_features(n, model, feat_fn), l) for n, l in instances]
    X, y = map(np.asarray, zip(*full_data))
    return X, y.reshape(-1) if y.shape[-1] == 1 else y


def node_features(n_t, model_fn, feat_fn=None):
    '''
    Extract node features from the model and an arbitrary feature function.
    '''
    n, l = n_t # a node is the node name/id (n) and its structural label (l)
    w = model_fn(l)
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

