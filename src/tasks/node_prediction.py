import random
import numpy as np
from scipy.stats import kendalltau, spearmanr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, mean_squared_error, median_absolute_error, mean_squared_log_error

from ml_utils import prepare_data

SKLEARN_AVERAGES = ['binary', 'micro', 'macro', 'samples', 'weighted']


def train(G, 
          mapping, 
          model, 
          labels,
          network_factory,
          feat_fn=None,
          remove_unlabeled=True, 
          vectorize=True,
          epochs=50, 
          verbose=0,
          use_scaler=True):
    X, y = prepare_data(G, mapping, model, labels, feat_fn, remove_unlabeled, vectorize)
    input_size = len(X[0])
    output_size = len(y[0]) if hasattr(y[0], '__len__') else 1

    # train the model
    m = network_factory.make_network(input_size, output_size)
    if verbose:
        m.summary()

    # use a scaler if specified
    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    m.fit(X, y, epochs=epochs, verbose=verbose)
    return m, scaler


def evaluate(G, 
             mapping, 
             model, 
             labels,
             network,
             eval_func,
             feat_fn=None,
             remove_unlabeled=True,
             vectorize=True,
             scaler=None):
    X, y = prepare_data(G, mapping, model, labels, feat_fn, remove_unlabeled, vectorize)

    # use a scaler if specified
    if scaler is not None:
        X = scaler.transform(X)

    # evaluate the model -- finally
    y_p = network.predict(X)

    # try to compute metrics 'automagically'
    all_avgs = [a for a in SKLEARN_AVERAGES if a in eval_func]
    average = all_avgs[0] if all_avgs else 'macro'
    if 'label' in eval_func:
        return fbeta_score(y, y_p.round(), 1, average=average)
    elif 'categorical' in eval_func:
        return fbeta_score(y.argmax(axis=-1), y_p.argmax(axis=-1), 1, average=average)
    elif 'kendalltau' in eval_func:
        s = -1 if 'rev' in eval_func else 1
        limit = int(eval_func.split('@')[-1]) if '@' in eval_func else y.size
        y_rnk = rankdata(s * y, method='min')
        y_p_rnk = rankdata(s * y_p, method='min')
        return kendalltau(y_rnk[:limit], y_p_rnk[:limit])[0]
    elif 'spearmanr' in eval_func:
        return spearmanr(y, y_p)[0]
    elif 'mse' in eval_func:
        return mean_squared_error(y, y_p)
    elif 'medianae' in eval_func:
        return median_absolute_error(y, y_p)
    elif 'msle' in eval_func:
        return mean_squared_log_error(y, y_p)
    elif 'rse' in eval_func: # relative squared error
        return (np.abs(y - y_p) / y).mean()
    return 'UNKNOWN: {}'.format(eval_func)
