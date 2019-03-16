import random
import numpy as np
from sklearn.metrics import fbeta_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

from ml_utils import node_features


def make_label_vectorizer(labels):
    try:
        targets = {i for v in labels.values() for i in v}
    except TypeError:
        targets = {v for v in labels.values()}
    values = {v: i for i, v in enumerate(sorted(targets))}
    total_classes = len(values)

    if total_classes <= 2:
        return lambda l: values[l], 1

    def vectorizer(l):
        vector = np.zeros(total_classes)
        vector[values[l]] = 1.0
        return vector
    return vectorizer, total_classes


def train(G, 
          mapping, 
          model, 
          labels,
          seed=None, 
          feat_fn=None,
          train_split=0.8,
          remove_unlabeled=True, 
          vectorize=True,
          network_factory=None,
          epochs=50, 
          verbose=0):
    if seed is not None:
        random.seed(seed)
    to_mapping = lambda v: mapping.get(v, None)
    raw_nodes  = [to_mapping(v['name']) for v in G.vs]
    raw_labels = [labels.get(v['name'], None) for v in G.vs]

    # prepare the label vectorizer and the loss function
    if vectorize:
        vector_fn, vector_size = make_label_vectorizer(labels)
    else:
        vector_fn, vector_size = lambda l: l, np.asarray(labels.values())[0].size

    # get the raw data
    instances  = [(m, vector_fn(l)) for (m, l) in zip(raw_nodes, raw_labels) 
                                    if not remove_unlabeled or l is not None]
    full_data  = [(node_features(n, model, feat_fn), l) for n, l in instances]
    random.shuffle(full_data)

    # prepare train and validation sets
    train_samples = int(round(len(full_data) * train_split))
    train_split = full_data[:train_samples]
    valid_split = full_data[train_samples:]

    # prepare the input in vector form
    Xt_t, yt_t = map(np.asarray, zip(*train_split))
    yt_t = yt_t.reshape(-1) if yt_t.shape[-1] == 1 else yt_t
    input_size = len(Xt_t[0])
    output_size = len(yt_t[0]) if hasattr(yt_t[0], '__len__') else 1

    # validation data in the same format
    Xv_t, yv_t = map(np.asarray, zip(*valid_split))
    yv_t = yv_t.reshape(-1) if yv_t.shape[-1] == 1 else yv_t

    # train the model
    m = network_factory.make_network(input_size, output_size)
    if verbose:
        m.summary()
    m.fit(Xt_t, yt_t, validation_data=(Xv_t, yv_t), epochs=epochs)

    # evaluate the model -- finally
    yv_p = m.predict(Xv_t)

    # try to compute metrics 'automagically'
    try:
        return m, fbeta_score(yv_t, yv_p, 1, average='macro'), 'f1-score'
    except ValueError:
        return m, mean_squared_error(yv_t, yv_p), 'mse'

