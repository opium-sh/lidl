import sys

sys.path.append("../../../glow-pytorch")
sys.path.append("../../")
import numpy as np
from samplers import memory_mnist
import dim_estimators


def print_dims_to_file(filename, dims):
    with open(filename, "w") as f:
        for dim in dims:
            print(dim, file=f)


train, val, train_val = memory_mnist(100_000, 32, 1)
train = [x for x in train][0].reshape(-1, 32 * 32)
val = [x for x in val][0].reshape(-1, 32 * 32)
# train_val = [x for x in train_val][0].reshape(-1, 32 * 32)

all_samples = np.vstack((train, val))


ks = [3, 10]
for k in ks:
    dims = dim_estimators.mle_skl(all_samples, 3)
    train_dims = dims[: train.shape[0]]
    val_dims = dims[train.shape[0] :]

    print_dims_to_file(f"mnist_dimensions_train_k={k}", train_dims)
    print_dims_to_file(f"mnist_dimensions_val_k={k}", val_dims)
