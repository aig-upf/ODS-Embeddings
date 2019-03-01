import random
import numpy as np
from sklearn.metrics import fbeta_score
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
        return lambda l: l

    def vectorizer(l):
        vector = np.zeros(total_classes)
        vector[l] = 1.0
        return vector
    return vectorizer


def train(G, 
          mapping, 
          model, 
          labels,
          seed=None, 
          feat_fn=None,
          train_split=0.8,
          remove_unlabeled=True):
    if seed is not None:
        random.seed(seed)
    to_mapping = lambda v: mapping.get(v, None)
    raw_nodes  = [to_mapping(v['name']) for v in G.vs]
    raw_labels = [labels.get(v['name'], None) for v in G.vs]

    # prepare the label vectorizer and the loss function
    vector_fn = make_label_vectorizer(labels)

    # get the raw data
    instances  = [(m, vector_fn(l)) for (m, l) in zip(raw_nodes, raw_labels) 
                                    if not remove_unlabeled or l is not None]
    full_data  = [(node_features(n, model, feat_fn), l) for n, l in instances]
    random.shuffle(full_data)

    # prepare train and validation sets
    train_samples = int(round(len(full_data) * train_split))
    train_split = full_data[:train_samples]
    valid_split = full_data[train_samples:]

    # train a logistic regression
    Xt_t, yt_t = zip(*train_split)
    lr = LogisticRegression().fit(Xt_t, yt_t)

    # evaluate the logistic regression
    Xv_t, yv_t = zip(*valid_split)
    yv_p = lr.predict(Xv_t)
    f1 = fbeta_score(yv_t, yv_p, 1, average='macro')

    # return the classifier and the area under the curve
    return lr, f1

