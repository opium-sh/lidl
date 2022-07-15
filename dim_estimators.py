import pickle

import numpy as np
from scipy.spatial import distance_matrix
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from dimensions import (
    intrinsic_dim_sample_wise_double_mle,
)
from likelihood_estimators import LLGaussianMixtures, LLFlow


def mle_skl(data, k):
    print("Computing the KNNs")
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(data)
    dist = nn.kneighbors(data)[0]
    mle, invmle = intrinsic_dim_sample_wise_double_mle(k, dist)
    return mle


def mle_inv(data, k):
    print("Computing the KNNs")
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(data)
    dist = nn.kneighbors(data)[0]
    mle, invmle = intrinsic_dim_sample_wise_double_mle(k, dist)
    return [1.0 / invmle.mean()] * data.shape[0]


def corr_dim(data, l_perc=0.000001, u_perc=0.01):
    N = len(data)
    distances = distance_matrix(data, data, p=2)[np.triu_indices(N, k=1)]
    r_low, r_high = np.quantile(distances, [l_perc, u_perc])
    C_r_list = []
    r_list = np.linspace(r_low, r_high, 3)

    for r in tqdm(r_list):
        distances_r = distances <= r
        # print(f'total, r = {r}, percenttrue: {(distances_r.sum())/distances_r.size}')
        C_r = 2 * distances_r.sum() / N / (N - 1)
        C_r_list.append(C_r)

    regr = linear_model.LinearRegression()
    regr.fit(np.log10(r_list).reshape(-1, 1), np.log10(C_r_list))
    return [regr.coef_[0]] * N


class LIDL:
    def __init__(self, model_type, **model_args):
        if model_type == "gm":
            self.model = LLGaussianMixtures(**model_args)
        elif model_type == "rqnsf":
            self.model = LLFlow(flow_type="rqnsf", **model_args)
        elif model_type == "maf":
            self.model = LLFlow(flow_type="maf", **model_args)
        else:
            raise ValueError(f"incorrect model type: {model_type}")

    def __call__(self, deltas, train_dataset, test):
        total_dim = train_dataset.shape[1]
        sort_deltas = np.argsort(np.array(deltas))
        lls = list()
        losses = list()
        tq = tqdm(deltas, position=0, leave=False, unit='delta')
        for delta in tq:
            tq.set_description(f"delta: {delta}")
            ll, score = self.model(delta=delta, dataset=train_dataset, test=test)
            lls.append(ll)
            losses.append(score)
        lls = np.array(lls)
        print(f"loss: {sum(losses)/len(losses)}")

        lls = lls[sort_deltas]
        deltas = np.array(deltas)[sort_deltas]

        lls = lls.transpose()
        dims = list()
        for i in range(lls.shape[0]):
            good_inds = ~np.logical_or(np.isnan(lls[i]), np.isinf(lls[i]))
            if ~good_inds.all():
                print(f"[WARNING] some log likelihoods are incorrect, deltas: {deltas}")
            ds = np.log(deltas[good_inds])
            ll = lls[i][good_inds]
            if ll.size < 2:
                dims.append(np.nan)
            else:
                regr = linear_model.LinearRegression()
                regr.fit(ds.reshape(-1, 1), ll)
                regr.predict(ds.reshape(-1, 1))
                dims.append(total_dim - regr.coef_[0])

        return np.array(dims)
