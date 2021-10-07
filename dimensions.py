#############################################################
# Functions copied from https://github.com/ppope/dimensions #
#############################################################


import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random


class KNNComputerNoCheck(nn.Module):
    """
    Using this hack for data parallel
    without checking for the sample itself
    """
    def __init__(self, sample_num, K=1, cosine_dist=False):
        super(KNNComputerNoCheck, self).__init__()

        self.K = K
        self.cosine_dist = cosine_dist
        self.register_buffer("num_computed", torch.zeros([]))

        if K == 1:
            self.register_buffer("min_dists", torch.full((sample_num, ), float('inf')))
            self.register_buffer("nn_indices", torch.full((sample_num,), 0, dtype=torch.int64))
        else:
            self.register_buffer("min_dists", torch.full((sample_num, K), float('inf')))
            self.register_buffer("nn_indices", torch.full((sample_num, K), 0, dtype=torch.int64))

    def forward(self, x, x_idx_start, y, y_idx_start):
        # update the min dist for existing examples...
        x_bsize, y_bsize = x.size(0), y.size(0)
        x = x.view(x_bsize, -1)
        y = y.view(y_bsize, -1)
        if self.cosine_dist:

            x = x / x.norm(dim=1, keepdim=True)
            y = y / y.norm(dim=1, keepdim=True)
            dist = x.mm(y.t())

        else:
            # dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)
            dist = torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        if self.K == 1:
            new_min_dist, nn_idxes = torch.min(dist, dim=1)

            self.min_dists[x_idx_start:x_idx_start + x_bsize] = torch.min(new_min_dist,
                                                                  self.min_dists[x_idx_start:x_idx_start+x_bsize])

            self.nn_indices[x_idx_start:x_idx_start + x_bsize] = nn_idxes + y_idx_start
        else:
            comp = torch.cat([dist, self.min_dists[x_idx_start:x_idx_start+x_bsize]], dim=1)
            # updated_min_dist, nn_idxes = torch.topk(comp, self.K, dim=1, largest=False)
            # check for repeated images
            sorted_dists, sorted_idxes = torch.sort(comp, dim=1, descending=False)
            updated_dist_list, nn_idx_list = [], []
            for row in range(sorted_dists.shape[0]):
                sidx = 1
                while sidx < sorted_dists.shape[1]:
                    if sorted_dists[row, sidx] == 0:
                        sidx += 1
                    else:
                        break
                updated_dist_list.append(sorted_dists[row, sidx-1:sidx-1+self.K])
                nn_idx_list.append(sorted_idxes[row, sidx-1:sidx-1+self.K])
            updated_min_dist = torch.stack(updated_dist_list)
            nn_idxes = torch.stack(nn_idx_list)

            self.min_dists[x_idx_start:x_idx_start + x_bsize] = updated_min_dist

            sample_idxes = (nn_idxes < y_bsize).int() * (nn_idxes + y_idx_start) \
                           + (nn_idxes >= y_bsize).int() * self.nn_indices[x_idx_start:x_idx_start + x_bsize]
            self.nn_indices[x_idx_start:x_idx_start + x_bsize] = sample_idxes

    def get_mean_nn_dist(self, sidx, eidx):
        if self.K == 1:
            return torch.mean(self.min_dists[sidx:eidx])


def update_nn(anchor_loader, anchor_start_idx, new_img_loader, new_start_idx, nn_computer, device):
    anchor_counter = anchor_start_idx
    # ignoring the labels
    with torch.no_grad():
        for n, (abatch, _) in enumerate(anchor_loader):
            abatch = abatch.to(device)

            new_img_counter = new_start_idx
            for newbatch, _ in new_img_loader:
                newbatch = newbatch.to(device)

                nn_computer(abatch, anchor_counter, newbatch, new_img_counter)

                new_img_counter += newbatch.size(0)

                equiv_flag = (nn_computer.min_dists[anchor_start_idx:anchor_start_idx+abatch.size(0), 0] == 0) & (nn_computer.min_dists[anchor_start_idx:anchor_start_idx+abatch.size(0), 1] == 0)
                if torch.any(equiv_flag):
                    raise Exception("Identical data detected!")

            anchor_counter += abatch.size(0)

            #if n % 50 == 0 or n == len(anchor_loader) - 1:
            #    #print("Finished {} images".format(anchor_counter))


def intrinsic_dim_sample_wise_double_mle(k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation and the correction by MacKay-Ghahramani

    Input parameters:
    X    - data
    k    - number of nearest neighbours (Default = 5)
    dist - matrix of distances to the k (or more) nearest neighbors of each point (Optional)

    Returns:
    two dimensionality estimates
    """
    dist = dist[:, 1:(k + 1)]
    if not np.all(dist > 0):
        # trying to catch the bug
        np.save("error_dist.npy", dist)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    inv_mle = d.copy()

    d = 1. / d
    mle = d
    return mle, np.reciprocal(inv_mle)
