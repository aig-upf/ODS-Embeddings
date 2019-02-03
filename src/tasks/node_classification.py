import random
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LogisticRegression

from ml_utils import node_features


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
    raw_nodes  = [to_mapping(v) for v in G.nodes()]
    raw_labels = [labels.get(v, None) for v in G.nodes()]

    # get the raw data
    instances  = [(m, l) for (m, l) in zip(raw_nodes, raw_labels) 
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

