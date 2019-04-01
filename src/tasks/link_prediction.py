import random
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from .ml_utils import edge_features


def get_labelled_edges(G, mapping):
    to_mapping = lambda v: (v, mapping.get(G.vs[v]['name'], None))

    # get the edges that have mappings for both nodes
    raw_edges = [tuple(map(to_mapping, t.tuple)) for t in G.es]
    pos_edges = {t for t in raw_edges if None not in t}

    # get the negative, randomly sampled edges
    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        v1 = random.choice(G.vs).index
        v2 = random.choice(G.vs).index
        if not G.are_connected(v1, v2):
            r1 = to_mapping(v1)
            r2 = to_mapping(v2)
            if r1 is not None and r2 is not None:
                neg_edges.append((r1, r2))

    # return positive and negative sets, labelled
    return [(e, 1) for e in pos_edges], [(e, 0) for e in neg_edges]


def train(G, 
          mapping, 
          model, 
          seed=None, 
          merge_fn=lambda x, y: np.concatenate([x, y]),
          train_split=0.8):
    if seed is not None:
        random.seed(seed)
    pos, neg  = get_labelled_edges(G, mapping)
    full_data = [(edge_features(e, model, merge_fn), l) for (e, l) in (pos + neg)]
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
    auc = roc_auc_score(yv_t, yv_p)

    # return the classifier and the area under the curve
    return lr, auc

