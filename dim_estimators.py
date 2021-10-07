import os
import sys
import torch
from torch.utils.data import DataLoader
from dimensions import update_nn, KNNComputerNoCheck, intrinsic_dim_sample_wise_double_mle
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix
import numpy as np
from sklearn import linear_model
from likelihood_estimators import LLGaussianMixtures, LLFlow, LLGlow
import pickle


def mle(data, k, batch_size=1024, device='cuda'):
    data_loader = torch.utils.data.DataLoader(list(zip(data, [0] * data.shape[0])))
    print("Computing the KNNs")
    nn_computer = KNNComputerNoCheck(data.shape[0], K=k + 1).to(device)
    update_nn(data_loader, 0, data_loader, 0, nn_computer, device)
    dist = nn_computer.min_dists.cpu().numpy()
    _, invmle = intrinsic_dim_sample_wise_double_mle(k, dist)

    return invmle


def corr_dim_per_sample(data, l_perc=0.000001, u_perc=0.01):
    N = data.shape[0]
    distances = distance_matrix(data, data,p=2)
    r_low, r_high = np.quantile(distances[np.triu_indices(N, k=1)], [l_perc, u_perc])

    rs_samples = []
    r_list = np.linspace(r_low, r_high, 10) 

    for r in r_list:
        distances_r = distances <= r
        #print(f'persample, r = {r}, percenttrue: {(distances_r.sum())/distances_r.size}')
        likelihoods = (distances_r.sum(axis=1) - 1)/(N-1)
        rs_samples.append(likelihoods)

    samples_rs = np.transpose(np.array(rs_samples))
    dims_per_sample = list()
    for i in range(samples_rs.shape[0]):
        sample_likelihoods = samples_rs[i]
        regr = linear_model.LinearRegression()
        regr.fit(np.log10(r_list).reshape(-1, 1), np.log10(sample_likelihoods))
        dims_per_sample.append(regr.coef_[0])

    return dims_per_sample


def corr_dim(data, l_perc=0.000001, u_perc=0.01):
    N = len(data)
    distances = distance_matrix(data, data,p=2)[np.triu_indices(N, k=1)]
    r_low, r_high = np.quantile(distances, [l_perc, u_perc])

    C_r_list = []
    r_list = np.linspace(r_low, r_high, 3)

    for r in r_list:
        distances_r = distances <= r
        #print(f'total, r = {r}, percenttrue: {(distances_r.sum())/distances_r.size}')
        C_r = 2 * distances_r.sum()/N/(N-1)
        C_r_list.append(C_r)

    regr = linear_model.LinearRegression()
    regr.fit(np.log10(r_list).reshape(-1,1), np.log10(C_r_list))
    return regr.coef_[0]


class LIDL():
    def __init__(self, model_type):
        if model_type == 'gaussian_mixture':
            self.model = LLGaussianMixtures()
        elif model_type == 'rqnsf':
            self.model = LLFlow('rqnsf')
        elif model_type == 'maf':
            self.model = LLFlow('maf')
        elif model_type == 'glow':
            self.model = LLGlow()
        else:
            assert False, 'incorrect model type'

    def run_on_deltas(self, deltas, **model_args):
        for delta in deltas:
            self.model.run(delta=delta, **model_args)

    def dims_on_deltas(self, deltas, epochs, total_dim):
        indsort = np.argsort(np.array(deltas))
        if isinstance(epochs, dict):
            lls = np.array([self.model.results[delta][epochs[delta]] for delta in deltas])
        else:
            lls = np.array([self.model.results[delta][epochs] for delta in deltas])
        lls = lls[indsort]
        deltas = np.array(deltas)[indsort]


        lls = lls.transpose()
        dims = list()
        for i in range(lls.shape[0]):
            good_inds = ~np.logical_or(np.isnan(lls[i]), np.isinf(lls[i]))
            if ~good_inds.all():
                print(f'[WARNING] some log likelihoods are incorrect, deltas: {deltas}, epochs: {epochs}')
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
        with open(f'{name}_lidl.obj', 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, name):
        with open(f'{name}_lidl.obj', 'rb') as f:
            self.__dict__ = pickle.load(f)
