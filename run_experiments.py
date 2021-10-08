import argparse

import datasets
from dim_estimators import mle_skl, corr_dim, LIDL

size = 1000
inputs = {
    "uniform-1": datasets.uniform_N(1, size),
    "uniform-10": datasets.uniform_N(10, size),
    "uniform-100": datasets.uniform_N(100, size),
    "uniform-1000": datasets.uniform_N(1000, size),
    "uniform-10000": datasets.uniform_N(10000, size),
    "gaussian-1": datasets.gaussian(1, size),
    "gaussian-10": datasets.gaussian(10, size),
    "gaussian-100": datasets.gaussian(100, size),
    "gaussian-1000": datasets.gaussian(1000, size),
    "gaussian-10000": datasets.gaussian(10000, size),
    "sphere-7": datasets.sphere_7(size),
    "uniform-helix-r3": datasets.uniform_helix_r3(size),
    "swiss-roll-r3": datasets.swiss_roll_r3(size),
    "sin": datasets.sin(size),
    "sin-quant": datasets.sin_quant(size),
    "sin-dequant": datasets.sin_dequant(size),
    "gaussian-1-2": datasets.N_1_2(size),
    "gaussian-10-20": datasets.N_10_20(size),
    "lollipop": datasets.lollipop_dataset(size)
}

parser = argparse.ArgumentParser(description="LIDL experiments")
parser.add_argument(
    "--algorithm",
    default="mle",
    type=str,
    choices=["mle", "gm", "rqnsf", "maf", "corrdim"],
    help="name of the algorithm",
)
parser.add_argument(
    "--dataset",
    default="uniform-1",
    type=str,
    choices=list(inputs.keys()),
    help="dataset which id will be estimated",
)
parser.add_argument(
    "--k",
    default="3",
    type=int,
    help="number of neighbours in mle (does nothing with other algorithms)",
)

parser.add_argument(
    "--delta",
    default=None,
    type=float,
    help="delta for density estimator models (does nothing with other algorithms)",
)

parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    help="torch device to run the algorithm on (cpu/cuda) - works only for maf and rqnsf",
)

parser.add_argument(
    "--layers",
    default="2",
    type=int,
    help="number of layers in maf/reqnsf"
)

args = parser.parse_args()

report_filename = (
    f"report_dim_estimate_{args.algorithm}_{args.dataset}_{args.delta}.csv"
)
f = open(report_filename, "w")

if args.delta is None:
    deltas = [
        0.010000,
        0.013895,
        0.019307,
        0.026827,
        0.037276,
        0.051795,
        0.071969,
        0.100000,
    ]
else:
    assert args.delta > 0, "delta must be greater than 0"
    deltas = [
        args.delta / 2.0,
        args.delta / 1.41,
        args.delta,
        args.delta * 1.41,
        args.delta * 2.0,
    ]


data = inputs[args.dataset]
data -= data.mean(axis=0)
data /= data.std() + 0.001

print(args)

if args.algorithm == "gm":
    gm = LIDL("gaussian_mixture")
    print(f"gm", file=f)
    gm.run_on_deltas(deltas, data=data, samples=data, runs=1, covariance_type="full")
    results = gm.dims_on_deltas(deltas, epoch=0, total_dim=data.shape[1])
    gm.save(f"{args.dataset}")
elif args.algorithm == "corrdim":
    print("corrdim", file=f)
    results = corr_dim(data)
elif args.algorithm == "maf":
    maf = LIDL("maf")
    best_epochs = maf.run_on_deltas(
        deltas, data=data, epochs=1500, device=args.device, num_layers=args.layers, lr=0.0002
    )
    print("maf", file=f)
    results = maf.dims_on_deltas(deltas, epoch=best_epochs, total_dim=data.shape[1])
    #maf.save(f"{args.algorithm}_{args.dataset}")
elif args.algorithm == "rqnsf":
    rqnsf = LIDL("rqnsf")
    rqnsf.run_on_deltas(
        deltas, data=data, epochs=1500, device=args.device, num_layers=args.layers, lr=0.0002
    )
    print("rqnsf", file=f)
    results = rqnsf.dims_on_deltas(deltas, epoch=1499, total_dim=data.shape[1])
    #rqnsf.save(f"{args.algorithm}_{args.dataset}")
elif args.algorithm == "mle":
    print(f"mle:k={args.k}", file=f)
    # results = mle(data, k=args.k)
    results = mle_skl(data, k=args.k)

print("\n".join(map(str, results)), file=f)
