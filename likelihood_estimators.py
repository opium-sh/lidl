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


def split_dataset(dataset, val_size):
    train_size = int(round(dataset.shape[0] * (1 - val_size)))
    inds = np.arange(dataset.shape[0])
    np.random.shuffle(inds)
    train = dataset[inds[:train_size]]
    val = dataset[inds[train_size:]]

    return train, val


class LLGaussianMixtures:
    def __init__(self, val_size=0.1, runs=3, max_components=200, covariance_type="full"):
        self.val_size = val_size
        self.runs = runs
        self.max_components = max_components
        self.covariance_type = covariance_type

    def __call__(self, delta, dataset, test):
        train, val = split_dataset(dataset, self.val_size)
        # we'll pick the best run
        best_score_per_run = -np.inf
        tq1 = tqdm.tqdm(range(self.runs), position=0, leave=False, unit='run')
        for run in tq1:
            tq1.set_description(f"run: {run + 1}")
            best_score = -np.inf
            best_comps = 0
            n_comps = list(range(1, self.max_components + 1))

            # Find the optimal number of components from the given range
            train_with_noise = train + np.random.randn(*train.shape) * delta
            val_with_noise = val + np.random.randn(*val.shape) * delta
            #train_val_with_noise = np.concatenate((train_with_noise, val_with_noise), dim=0)

            tq2 = tqdm.tqdm(n_comps, position=0, leave=False, unit='num_comp')
            for n_comp in tq2:
                tq2.set_description(f"components: {n_comp}")
                model = GaussianMixture(n_components=n_comp, covariance_type=self.covariance_type)
                model.fit(train_with_noise)
                score = model.score(val_with_noise)
                if score > best_score:
                    best_score = score
                    best_comps = n_comp
                if (n_comp - best_comps) > 10:
                    break
            #tq1.set_postfix_str(f"Best number of components: {best_comps}")

            model = GaussianMixture(n_components=best_comps, covariance_type=self.covariance_type)
            model.fit(train_with_noise)

            score_per_run = model.score(val_with_noise)
            if score_per_run > best_score_per_run:
                best_model = model
                best_score_per_run = score_per_run

        return -best_model.score_samples(test), -best_score_per_run


class LLFlow:
    def __init__(self, flow_type, val_size=0.1, num_layers=10,
                 lr=0.0001, epochs=30, device="cpu", hidden=5, batch_size=256, num_blocks=5):
        self.flow_type = flow_type
        self.val_size = val_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.hidden = hidden
        self.batch_size = batch_size
        self.num_blocks = num_blocks

    def __create_model(self, features):
        base_dist = StandardNormal(shape=[features])
        transforms = []
        for _ in range(self.num_layers):
            transforms.append(ReversePermutation(features=features))
            if self.flow_type == "maf":
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=features, hidden_features=int(round(self.hidden * features))
                    )
                )
            elif self.flow_type == "rqnsf":
                transforms.append(
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                        features=features,
                        hidden_features=int(round(self.hidden * features)),
                        num_bins=5,
                        num_blocks=self.num_blocks,
                        tails="linear",
                        tail_bound=5,
                        use_batch_norm=False,
                    )
                )
            else:
                raise ValueError(f"flow type is incorrect: {self.flow_type}")

        transform = CompositeTransform(transforms)
        flow = Flow(transform, base_dist)
        flow.to(self.device)

        return flow

    # def run(self, data, samples, delta=0.05, test_size = 0.1, num_layers=10, lr=0.0001, epochs=10_000, device='cpu'):
    def __call__(self, delta, dataset, test):
        train, val = split_dataset(dataset, self.val_size)
        if test.shape[1] != dataset.shape[1]:
            raise ValueError(f"train and test datasets have different number of features: \
            train features: {dataset.shape[1]}, test features: {test.shape[1]}")

        flow = self.__create_model(train.shape[1])
        optimizer = optim.Adam(flow.parameters(), lr=self.lr)

        train_tensor = torch.tensor(train, dtype=torch.float32)
        val_tensor = torch.tensor(val, dtype=torch.float32)
        test_tensor = torch.tensor(test, dtype=torch.float32)

        best_loss = np.inf
        best_epoch = 0

        losses = list()
        results = list()
        tq1 = tqdm.tqdm(range(self.epochs), position=0, leave=False)
        for epoch in tq1:
            tq1.set_description(f"epoch: {epoch + 1}")
            tq2 = tqdm.tqdm(DataLoader(train_tensor, batch_size=self.batch_size), position=0, leave=False)
            for x in tq2:
                tq2.set_description("batch")
                x = x + torch.randn_like(x) * delta
                x = x.to(self.device)
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=x).mean()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                # validation loss for early stopping
                val_loss = -flow.log_prob(inputs=val_tensor).mean()
                losses.append(val_loss.detach().cpu().numpy())

                # remember the results for early stopping
                ll = -flow.log_prob(test_tensor)
                results.append(ll.detach().cpu().numpy())

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch

                if (epoch - best_epoch) > round(self.epochs * 2 / 100):
                    print(f"Stopping after {best_epoch} epochs")
                    return results[best_epoch], losses[best_epoch]
            tq1.set_postfix_str(f"loss: {losses[best_epoch]}")

        return results[best_epoch], losses[best_epoch]
