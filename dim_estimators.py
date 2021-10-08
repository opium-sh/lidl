import pickle

import numpy as np
from scipy.spatial import distance_matrix
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from dimensions import (
    intrinsic_dim_sample_wise_double_mle,
)
from likelihood_estimators import LLGaussianMixtures, LLFlow, LLGlow


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
    def __init__(self, model_type):
        if model_type == "gaussian_mixture":
            self.model = LLGaussianMixtures()
        elif model_type == "rqnsf":
            self.model = LLFlow("rqnsf")
        elif model_type == "maf":
            self.model = LLFlow("maf")
        elif model_type == "glow":
            self.model = LLGlow()
        else:
            assert False, "incorrect model type"

    def run_on_deltas(self, deltas, **model_args):
        for delta in deltas:
            self.model.run(delta=delta, **model_args)

    def dims_on_deltas(self, deltas, epoch, total_dim):
        indsort = np.argsort(np.array(deltas))
        if isinstance(epoch, dict):
            lls = np.array(
                [self.model.results[delta][epoch[delta]] for delta in deltas]
            )
        else:
            lls = np.array([self.model.results[delta][epoch] for delta in deltas])
        lls = lls[indsort]
        deltas = np.array(deltas)[indsort]

        lls = lls.transpose()
        dims = list()
        for i in range(lls.shape[0]):
            good_inds = ~np.logical_or(np.isnan(lls[i]), np.isinf(lls[i]))
            if ~good_inds.all():
                print(
                    f"[WARNING] some log likelihoods are incorrect, deltas: {deltas}, epochs: {epoch}"
                )
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

    def save(self, name):
        with open(f"{name}_lidl.obj", "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, name):
        with open(f"{name}_lidl.obj", "rb") as f:
            self.__dict__ = pickle.load(f)
