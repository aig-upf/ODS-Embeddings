import random
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from ml_utils import edge_features


def get_labelled_edges(G, mapping):
    to_mapping = lambda v: mapping.get(v, None)
    raw_nodes = [to_mapping(v) for v in G.nodes()]
    val_nodes = {t for t in raw_nodes if t is not None}
    nodes_as_list = list(val_nodes)

    # get the edges that have mappings for both nodes
    raw_edges = [tuple(map(to_mapping, t)) for t in G.edges()]
    pos_edges = {t for t in raw_edges if None not in t}

    # get the negative, randomly sampled edges
    neg_edges = set()
    while len(neg_edges) < len(pos_edges):
        u = random.choice(nodes_as_list)
        v = random.choice(nodes_as_list)
        if u != v and (u, v) not in pos_edges:
            neg_edges.add((u, v))

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

