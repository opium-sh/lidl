import argparse

import datasets
from dim_estimators import mle, corr_dim, LIDL

size = 1000
inputs = {
    "uniform-1": (datasets.uniform_N(1, size), 1),
    "uniform-10": (datasets.uniform_N(10, size), 10),
    "uniform-100": (datasets.uniform_N(100, size), 100),
    "uniform-1000": (datasets.uniform_N(1000, size), 1000),
    "uniform-10000": (datasets.uniform_N(10000, size), 10000),
    "gaussian-1": (datasets.gaussian(1, size), 1),
    "gaussian-10": (datasets.gaussian(10, size), 10),
    "gaussian-100": (datasets.gaussian(100, size), 100),
    "gaussian-1000": (datasets.gaussian(1000, size), 1000),
    "gaussian-10000": (datasets.gaussian(10000, size), 10000),
    "sphere-7": (datasets.sphere_7(size), 8),
    "uniform-helix-r3": (datasets.uniform_helix_r3(size), 3),
    "swiss-roll-r3": (datasets.swiss_roll_r3(size), 3),
    "sin": (datasets.sin(50), 2),
    "sin-quant": (datasets.sin_quant(50), 2),
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
    help="number of neighbours in mle (does nothing with different algorithms)",
)

args = parser.parse_args()

report_filename = f"report_dim_estimate_{args.algorithm}_{args.dataset}.txt"
f = open(report_filename, "w")
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


data, total_dim = inputs[args.dataset]
if args.algorithm == "gm":
    gm = LIDL("gaussian_mixture")
    print(f"gm", file=f)
    gm.run_on_deltas(deltas, data=data, samples=data, runs=1)
    print(gm.dims_on_deltas(deltas, epoch=0, total_dim=total_dim), file=f)
    gm.save(f"{args.dataset}")
elif args.algorithm == "corrdim":
    print("corrdim", file=f)
    print(corr_dim(data), file=f)
elif args.algorithm == "maf":
    maf = LIDL("maf")
    maf.run_on_deltas(deltas, data=data, epochs=200)
    print("maf", file=f)
    print(maf.dims_on_deltas(deltas, epoch=199, total_dim=total_dim), file=f)
    maf.save(f"{args.dataset}")
elif args.algorithm == "rqnsf":
    rqnsf = LIDL("rqnsf")
    rqnsf.run_on_deltas(deltas, data=data, epochs=200)
    print("rqnsf", file=f)
    print(rqnsf.dims_on_deltas(deltas, epoch=199, total_dim=total_dim), file=f)
    rqnsf.save(f"{args.dataset}")
elif args.algorithm == "mle":
    print(f"mle, k={args.k}", file=f)
    print(mle(data, k=args.k, device="cpu"), file=f)
