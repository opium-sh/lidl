import pickle
from collections import defaultdict

import numpy as np
import torch
import tqdm
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from sklearn.mixture import GaussianMixture
from torch import optim
from torch.utils.data import DataLoader


class LLGaussianMixtures:
    def __init__(self):
        self.models = defaultdict(list)
        self.results = defaultdict(list)

    def run(
        self,
        delta,
        data,
        samples,
        runs=10,
        test_size=0.25,
        max_components=200,
        covariance_type="full",
    ):
        train_size = int(round(data.shape[0] * (1 - test_size)))
        inds = np.arange(data.shape[0])
        np.random.shuffle(inds)
        train = data[inds[:train_size]]
        test = data[inds[train_size:]]
        data_with_noise = data + np.random.randn(*data.shape) * delta

        # train = data[:train_size, :]
        # test = data[train_size:, :]

        for _ in range(runs):
            best_score = -np.inf
            best_comps = 0
            n_comps = list(range(1, max_components + 1))
            # Find the optimal number of components from the given range
            ll = list()
            train_with_noise = train + np.random.randn(*train.shape) * delta
            test_with_noise = test + np.random.randn(*test.shape) * delta
            for n_comp in tqdm.tqdm(n_comps):
                model = GaussianMixture(
                    n_components=n_comp, covariance_type=covariance_type
                )
                model.fit(train_with_noise)
                score = model.score(test_with_noise)
                ll.append(score)
                if score > best_score:
                    best_score = score
                    best_comps = n_comp
                if (n_comp - best_comps) > 10:
                    break
            print(f"Best number of components: {best_comps}")

            model = GaussianMixture(
                n_components=best_comps, covariance_type=covariance_type
            )
            model.fit(data_with_noise)

            self.models[delta].append(model)
            self.results[delta].append(-model.score_samples(samples))

    def save(self, name):
        with open(f"{name}_gaussian_mixture.obj", "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, name):
        with open(f"{name}_gaussian_mixture.obj", "rb") as f:
            self.__dict__ = pickle.load(f)

    def ll(self, samples, run, delta):
        return self.models[delta][run].score_samples(samples)


class LLFlow:
    def __init__(self, flow_type):
        self.models = defaultdict(list)
        self.results = defaultdict(list)
        self.losses = defaultdict(list)
        self.flow_type = flow_type

    def save(self, name):
        with open(f"{name}_{self.flow_type}.obj", "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, name):
        with open(f"{name}_{self.flow_type}.obj", "rb") as f:
            self.__dict__ = pickle.load(f)

    def ll(self, delta, epoch):
        return self.results[delta][epoch]

    # def run(self, data, samples, delta=0.05, test_size = 0.1, num_layers=10, lr=0.0001, epochs=10_000, device='cpu'):
    def run(
        self,
        delta,
        data,
        test_size=0.1,
        num_layers=10,
        lr=0.0001,
        epochs=3,
        device="cpu",
        hidden=5,
        batch_size=256,
        num_blocks=5,
        report_test_losses=True,
    ):
        train_size = int(round(data.shape[0] * (1 - test_size)))

        inds = np.arange(data.shape[0])
        np.random.shuffle(inds)
        train = data[inds[:train_size]]
        test = data[inds[train_size:]]

        # train = data[:train_size, :]
        # test = data[train_size:, :]

        base_dist = StandardNormal(shape=[data.shape[1]])
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=data.shape[1]))
            if self.flow_type == "maf":
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=data.shape[1], hidden_features=int(round(hidden * data.shape[1]))
                    )
                )
            elif self.flow_type == "rqnsf":
                transforms.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=data.shape[1],
                        hidden_features=int(round(hidden * data.shape[1])),
                        num_bins=5,
                        num_blocks=5,
                        tails="linear",
                        tail_bound=5,
                        use_batch_norm=False,
                    )
                )
            else:
                assert False, "Give a flow type: rqnsf or maf"

        transform = CompositeTransform(transforms)

        flow = Flow(transform, base_dist)
        flow.to(device)
        optimizer = optim.Adam(flow.parameters(), lr=lr)

        if report_test_losses:
            losses_file = open(f"{self.flow_type}_{delta}_losses.txt", "w+")
        best_loss = np.inf
        best_epoch = 0
        for epoch in tqdm.tqdm(range(epochs)):
            dloader = DataLoader(train, batch_size=batch_size)
            for x in dloader:
                x = x + np.random.randn(*x.shape) * delta
                x = x.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=x).mean()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                inp = torch.tensor(data, dtype=torch.float32).to(device)
                ll = -flow.log_prob(inp)
                self.results[delta].append(ll.detach().cpu().numpy())

                test_inp = torch.tensor(test, dtype=torch.float32).to(device)
                test_loss = -flow.log_prob(inputs=test_inp).mean()
                self.losses[delta].append(test_loss.detach().cpu().numpy())
                if report_test_losses:
                    print(test_loss.item(), file=losses_file)
                    losses_file.flush()

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_epoch = epoch

                if (epoch - best_epoch) > round(epochs * 2 / 100):
                    print(f"Stopping after {best_epoch} epochs")
                    return best_epoch
        return best_epoch


class LLGlow:
    def __init__(self):
        self.results = dict()

    def _lls_from_file(self, path):
        test_lls = list()
        train_lls = list()
        with open(path, "r") as f:
            for line in f:
                _, lpv, ldv, lptv, ldtv = line.strip().split(" ")
                test_ll = -(float(lpv) + float(ldv))
                train_ll = -(float(lptv) + float(ldtv))
                test_lls.append(test_ll)
                train_lls.append(train_ll)

        return test_lls, train_lls

    def run(
        self,
        delta,
        dataset,  # = 'mnist',
        epochs,  # = 200,
        lr,  # = 5e-05,
        img_size=32,
        device="cpu",
        results_path="/home/rm360179/glow-pytorch/",
        ll_batch=64,
        n_channels=1,
        n_flow=32,
        n_block=4,
        no_lu=False,
        affine=True,
        tr_dq=False,
        te_dq=False,
        te_noise=False,
        n_bits=8,
        temp=0.7,
        n_sample=20,
    ):

        filename = (
            f"ll_batch#{ll_batch};"
            f"n_channels#{n_channels};"
            f"epochs#{epochs};"
            f"n_flow#{n_flow};"
            f"n_block#{n_block};"
            f"no_lu#{no_lu};"
            f"affine#{affine};"
            f"tr_dq#{tr_dq};"
            f"te_dq#{te_dq};"
            f"te_noise#{te_noise};"
            f"n_bits#{n_bits};"
            f"lr#{lr};"
            f"img_size#{img_size};"
            f"temp#{temp};"
            f"n_sample#{n_sample};"
            f"dataset#{dataset};"
            f"device#{device};"
            f"delta#{delta}"
        )

        self.results[delta] = list()
        for run in tqdm.tqdm(range(1, 40)):  # epochs + 1)):
            fpath = f"{results_path}/ll/{filename}_{run}.txt"
            # try:
            test_lls, train_lls = self._lls_from_file(fpath)
            self.results[delta].append(test_lls)
            # except FileNotFoundError:
            #    print(f'There are some files missing, please run the experiment using srun_train.sh.\n'\
            #          f'Missing file: {fpath}')

    def save(self, name):
        with open(f"{name}_gaussian_mixture.obj", "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, name):
        with open(f"{name}_gaussian_mixture.obj", "rb") as f:
            self.__dict__ = pickle.load(f)

    def ll(self, delta, epoch):
        return self.results[delta][epoch]
