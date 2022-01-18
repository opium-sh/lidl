import argparse
import datasets
from datasets import normalize
from dim_estimators import mle_skl, corr_dim, LIDL, mle_inv
import numpy as np
import neptune.new as neptune
import skdim
import json

inputs = {
    "uniform-1": lambda size, seed: datasets.uniform_N(1, size, seed=seed),
    "uniform-10": lambda size, seed: datasets.uniform_N(10, size, seed=seed),
    "uniform-12": lambda size, seed: datasets.uniform_N(12, size, seed=seed),
    "uniform-100": lambda size, seed: datasets.uniform_N(100, size, seed=seed),
    "uniform-1000": lambda size, seed: datasets.uniform_N(1000, size, seed=seed),
    "uniform-10000": lambda size, seed: datasets.uniform_N(10000, size, seed=seed),
    "uniform_N_0_1-12": lambda size, seed: datasets.uniform_N_0_1(12, size, seed=seed),
    "gaussian-1": lambda size, seed: datasets.gaussian(1, size, seed=seed),
    "gaussian-5": lambda size, seed: datasets.gaussian(5, size, seed=seed),
    "gaussian-10": lambda size, seed: datasets.gaussian(10, size, seed=seed),
    "gaussian-100": lambda size, seed: datasets.gaussian(100, size, seed=seed),
    "gaussian-1000": lambda size, seed: datasets.gaussian(1000, size, seed=seed),
    "gaussian-10000": lambda size, seed: datasets.gaussian(10000, size, seed=seed),
    "sphere-7": lambda size, seed: datasets.sphere_7(size, seed=seed),
    "uniform-helix-r3": lambda size, seed: datasets.uniform_helix_r3(size, seed=seed),
    "swiss-roll-r3": lambda size, seed: datasets.swiss_roll_r3(size, seed=seed),
    "sin": lambda size, seed: datasets.sin(size, seed=seed),
    "sin-quant": lambda size, seed: datasets.sin_quant(size, seed=seed),
    "sin-dequant": lambda size, seed: datasets.sin_dequant(size, seed=seed),
    "gaussian-1-2": lambda size, seed: datasets.gaussian_N_2N(size, N=1, seed=seed),
    "gaussian-10-20": lambda size, seed: datasets.gaussian_N_2N(size, N=10, seed=seed),
    "gaussian-100-200": lambda size, seed: datasets.gaussian_N_2N(size, N=100, seed=seed),
    "gaussian-500-1000": lambda size, seed: datasets.gaussian_N_2N(size, N=500, seed=seed),
    "gaussian-1000-2000": lambda size, seed: datasets.gaussian_N_2N(size, N=1000, seed=seed),
    "gaussian-10000-20000": lambda size, seed: datasets.gaussian_N_2N(size, N=10000, seed=seed),
    "lollipop": lambda size, seed: datasets.lollipop_dataset(size, seed=seed),
    "lollipop-0": lambda size, seed: datasets.lollipop_dataset_0(size, seed=seed),
    "sin-01": lambda size, seed: datasets.sin_freq(size, freq=0.1, seed=seed),
    "sin-02": lambda size, seed: datasets.sin_freq(size, freq=0.2, seed=seed),
    "sin-05": lambda size, seed: datasets.sin_freq(size, freq=0.5, seed=seed),
    "sin-10": lambda size, seed: datasets.sin_freq(size, freq=1.0, seed=seed),
    "sin-20": lambda size, seed: datasets.sin_freq(size, freq=2.0, seed=seed),
    "sin-30": lambda size, seed: datasets.sin_freq(size, freq=3.0, seed=seed),
    "sin-50": lambda size, seed: datasets.sin_freq(size, freq=5.0, seed=seed),
    "sin-80": lambda size, seed: datasets.sin_freq(size, freq=8.0, seed=seed),
    "sin-160": lambda size, seed: datasets.sin_freq(size, freq=16.0, seed=seed),
    "sin-320": lambda size, seed: datasets.sin_freq(size, freq=16.0, seed=seed),
    "sin-dens-1": lambda size, seed: datasets.sin_dens(size, freq=1.0, seed=seed),
    "sin-dens-2": lambda size, seed: datasets.sin_dens(size, freq=2.0, seed=seed),
    "sin-dens-3": lambda size, seed: datasets.sin_dens(size, freq=3.0, seed=seed),
    "sin-dens-4": lambda size, seed: datasets.sin_dens(size, freq=4.0, seed=seed),
    "sin-dens-6": lambda size, seed: datasets.sin_dens(size, freq=6.0, seed=seed),
    "sin-dens-8": lambda size, seed: datasets.sin_dens(size, freq=8.0, seed=seed),
    "sin-dens-10": lambda size, seed: datasets.sin_dens(size, freq=10.0, seed=seed),
    "sin-dens-12": lambda size, seed: datasets.sin_dens(size, freq=12.0, seed=seed),
    "sin-dens-14": lambda size, seed: datasets.sin_dens(size, freq=14.0, seed=seed),
    "sin-dens-16": lambda size, seed: datasets.sin_dens(size, freq=16.0, seed=seed),
    "boston": lambda size, seed: datasets.csv_dataset(
        # path to boston_housing dataset here
        "~/datasets/boston_housing.txt"
    ),
    "protein": lambda size, seed: datasets.csv_dataset(
        # path to protein dataset here
        "~/datasets/protein.txt"
    ),
    "wine": lambda size, seed: datasets.csv_dataset(
        # path to wine dataset here
        "~/datasets/wine.txt"
    ),
    "power": lambda size, seed: datasets.csv_dataset(
        # path to power dataset here
        "~/datasets/power.txt"
    ),
    "yacht": lambda size, seed: datasets.csv_dataset(
        # path to yacht dataset here
        "~/datasets/yacht.txt"
    ),
    "concrete": lambda size, seed: datasets.csv_dataset(
        # path to concrete dataset here
        "~/datasets/concrete.txt"
    ),
    "energy": lambda size, seed: datasets.csv_dataset(
        # path to enery_heating_load dataset here
        "~/datasets/energy_heating_load.txt"
    ),
    "kin8nm": lambda size, seed: datasets.csv_dataset(
        # path to kin8nm dataset here
        "~/datasets/kin8nm.txt"
    ),
    "naval": lambda size, seed: datasets.csv_dataset(
        # path to naval_compressor_decay dataset here
        "~/datasets/naval_compressor_decay.txt"
    ),
    "year": lambda size, seed: datasets.csv_dataset(
        # path to year_prediction_msd dataset here
        "~/datasets/year_prediction_msd.txt"
    )
}

skdim_algorithms = {
        'skdim_corrint': skdim.id.CorrInt,
        'skdim_danco': skdim.id.DANCo,
        'skdim_ess': skdim.id.ESS,
        'skdim_fishers': skdim.id.FisherS,
        'skdim_knn': skdim.id.KNN,
        'skdim_lpca': skdim.id.lPCA,
        'skdim_mada': skdim.id.MADA,
        'skdim_mind_ml':skdim.id.MiND_ML,
        'skdim_mle':skdim.id.MLE,
        'skdim_mom':skdim.id.MOM,
        'skdim_tle': skdim.id.TLE,
        'skdim_twonn': skdim.id.TwoNN,
}

parser = argparse.ArgumentParser(description="LIDL experiments")
parser.add_argument(
    "--algorithm",
    default="mle",
    type=str,
    choices=["mle", "mle-inv", "gm", "rqnsf", "maf", "corrdim"] + list(skdim_algorithms.keys()),
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
    "--covariance",
    default="diag",
    type=str,
    choices=['spherical', 'tied', 'diag', 'full'],
    help="covariance_type for GaussianMixture",
)
parser.add_argument(
    "--k",
    default="3",
    type=int,
    help="number of neighbours in mle and mle_inv (does nothing with other algorithms)",
)

parser.add_argument(
    "--delta",
    default=None,
    type=float,
    help="delta for density estimator models (does nothing with other algorithms)",
)

parser.add_argument(
    "--num_deltas",
    default=None,
    type=int,
    help="number of deltas for density estimator models (does nothing with other algorithms)",
)

parser.add_argument(
    "--gdim",
    default=False,
    type=bool,
    help="should skdim try to estimate global dim?",
)

parser.add_argument(
    "--neptune_name",
    default=None,
    type=str,
    help="name of the project you want to log to <YOUR_WORKSPACE>/<YOUR_PROJECT>",
)

parser.add_argument(
    "--neptune_token",
    default=None,
    type=str,
    help="token to your project",
)

parser.add_argument(
    "--deltas",
    required=False,
    default=None,
    type=str,
    help="all deltas for density estimator models separated by a comma (does nothing with other algorithms)",
)

parser.add_argument(
    "--ground_truth_const",
    required=False,
    default=None,
    type=int,
    help="if the dimension is constant in every item, you can estimate mse by adding this argument",
)

parser.add_argument(
    "--device",
    default="cpu",
    type=str,
    help="torch device to run the algorithm on (cpu/cuda) - works only for maf and rqnsf",
)

parser.add_argument(
    "--layers",
    default="4",
    type=int,
    help="number of layers in maf/reqnsf"
)

parser.add_argument(
    "--size",
    default="1000",
    type=int,
    help="number of samples in each dataset (number of rows)"
)

parser.add_argument(
    "--seed",
    default="0",
    type=int,
    help="seed for each dataset generator"
)

parser.add_argument(
    "--hidden",
    default="5",
    type=float,
    help="number of hidden features in maf"
)

parser.add_argument(
    "--lr",
    default=0.0001,
    type=float,
    help="learning rate"
)

parser.add_argument(
    "--epochs",
    default=10000,
    type=int,
    help="number of epochs"
)

parser.add_argument(
    "--bs",
    default=256,
    type=int,
    help="batch_size"
)

parser.add_argument(
    "--blocks",
    default=5,
    type=int,
    help="number of blocks in rqnsf"
)

parser.add_argument(
    "--json_params",
    default=None,
    type=str,
    help="arguments to skdim"
)

args = parser.parse_args()

not_in_filename = [
        'neptune_token',
        'neptune_name',
        'ground_truth_const',
        'gdim']
argname = "_".join([f"{k}:{v}" for k, v in vars(args).items() if not k in not_in_filename])

report_filename = (
    f"report_dim_estimate_{argname}.csv"
)
f = open(report_filename, "w")

if args.deltas is not None:
    ldeltas = args.deltas.split(',')
    deltas = list()
    assert len(ldeltas) >= 2
    for delta in ldeltas:
        fdelta = float(delta)
        assert fdelta > 0
        deltas.append(fdelta)
elif args.delta is not None:
    assert args.delta > 0, "delta must be greater than 0"
    if args.num_deltas is None:
        deltas = [
            args.delta / 2.0,
            args.delta / 1.41,
            args.delta,
            args.delta * 1.41,
            args.delta * 2.0,
        ]
    else:
        deltas = np.geomspace(args.delta/2, args.delta*2, args.num_deltas)
else:
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


data = inputs[args.dataset](size=args.size, seed=args.seed)
data = normalize(data)
print(args)

if not (args.neptune_name is None or args.neptune_token is None):
    run = neptune.init(
            project=args.neptune_name,
            api_token=args.neptune_token,
            source_files=['datasets.py', 'dim_estimators.py', 'likelihood_estimators.py', 'run_experiments.py', 's3.sh'],
    )
    for key, value in vars(args).items():
        run[key] = value

if args.algorithm in skdim_algorithms:
    print(args.algorithm, file=f)
    if args.json_params is not None:
        with open(args.json_params) as f:
            params = json.load(f)
    else:
        params = dict()
    if not (args.neptune_name is None or args.neptune_token is None):
        run['skdim_params'] = params

    model = skdim_algorithms[args.algorithm](**params)
    ldims = model.fit_transform_pw(data)

    if args.gdim:
        model = skdim_algorithms[args.algorithm](**params)
        gdim = model.fit_transformw(data)

    results = ldims

elif args.algorithm == "gm":
    gm = LIDL("gaussian_mixture")
    print(f"gm", file=f)
    gm.run_on_deltas(deltas, data=data, samples=data, runs=1, covariance_type="diag")
    results = gm.dims_on_deltas(deltas, epoch=0, total_dim=data.shape[1])
    gm.save(f"{args.dataset}")

elif args.algorithm == "corrdim":
    print("corrdim", file=f)
    results = corr_dim(data)

elif args.algorithm == "maf":
    maf = LIDL("maf")
    best_epochs = maf.run_on_deltas(
        deltas, data=data, device=args.device, num_layers=args.layers, lr=args.lr, hidden=args.hidden, epochs=args.epochs, batch_size=args.bs, test_losses_name=argname
    )
    print("maf", file=f)
    results = maf.dims_on_deltas(deltas, epoch=best_epochs, total_dim=data.shape[1])
    #maf.save(f"{args.algorithm}_{args.dataset}")

elif args.algorithm == "rqnsf":
    rqnsf = LIDL("rqnsf")
    best_epochs = rqnsf.run_on_deltas(
        deltas, data=data, device=args.device, num_layers=args.layers, lr=args.lr, hidden=args.hidden, epochs=args.epochs, batch_size=args.bs, num_blocks=args.blocks
    )
    print("rqnsf", file=f)
    results = rqnsf.dims_on_deltas(deltas, epoch=best_epochs, total_dim=data.shape[1])
    #rqnsf.save(f"{args.algorithm}_{args.dataset}")

elif args.algorithm == "mle":
    print(f"mle:k={args.k}", file=f)
    # results = mle(data, k=args.k)
    results = mle_skl(data, k=args.k)

elif args.algorithm == "mle-inv":
    print(f"mle-inv:k={args.k}", file=f)
    results = mle_inv(data, k=args.k)

print("\n".join(map(str, results)), file=f)

if not (args.neptune_name is None or args.neptune_token is None):
    for lid in results:
        run['lids'].log(lid)
    if args.ground_truth_const is not None:
        def mse(a, b):
            return ((a - b) ** 2).mean()
        mse_val = mse(np.array(results), np.full(len(results), args.ground_truth_const))
        run['mse'] = mse_val
    if args.algorithm in skdim_algorithms and args.gdim:
        run['gdim'] = gdim
    run.stop()

