import json
import time
from pathlib import Path

from dim_estimators import mle_skl, corr_dim, LIDL, mle_inv
import numpy as np
import neptune.new as neptune

from src.experiments_parser import make_experiment_argument_parser
from src.script_args.datasets import inputs
from src.script_args.algorithms import skdim_algorithms

parser = make_experiment_argument_parser()
args = parser.parse_args()

not_in_filename = [
        'json_params',
        'gm_max_components',
        'neptune_token',
        'neptune_name',
        'ground_truth_const',
        'gdim']
argname = "_".join([f"{k}:{v}" for k, v in vars(args).items() if not k in not_in_filename])

report_folder = Path('./results')
report_folder.mkdir(exist_ok=True, parents=True)
report_filename = report_folder.joinpath(f"report_dim_estimate_{argname}.csv")
figure_filename = report_folder.joinpath(f"report_dim_estimate_{argname}.csv")
print(report_filename)

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
#data = normalize(data)
#print(args)

run = None
if not (args.neptune_name is None or args.neptune_token is None):
    run = neptune.init(
            project=args.neptune_name,
            api_token=args.neptune_token,
            source_files=['datasets.py', 'dim_estimators.py', 'likelihood_estimators.py', 'run_experiments.py', 's3.sh'],
    )
    for key, value in vars(args).items():
        run[key] = value
    start_time = time.time()


report_file = open(report_filename, "w")
if args.algorithm in skdim_algorithms:
    print(args.algorithm, file=report_file)
    if args.json_params is not None:
        with open(args.json_params) as f_skdim_args:
            params = json.load(f_skdim_args)
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
    #TODO(from original LIDL) fix arguments (covariance)
    gm = LIDL(
        model_type="gm",
        runs=1,
        covariance_type="diag",
        max_components=args.gm_max_components)
    print(f"gm", file=report_file)
    results = gm(deltas=deltas, train_dataset=data, test=data)
    #gm.save(f"{args.dataset}")

elif args.algorithm == "corrdim":
    print("corrdim", file=report_file)
    results = corr_dim(data)

elif args.algorithm == "maf":
    maf = LIDL(
        model_type="maf",
        device=args.device,
        num_layers=args.layers,
        lr=args.lr,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.bs)
    print("maf", file=report_file)
    results = maf(
        deltas=deltas,
        train_dataset=data,
        test=data)
    #maf.save(f"{args.algorithm}_{args.dataset}")

elif args.algorithm == "rqnsf":
    rqnsf = LIDL(
        model_type="rqnsf",
        device=args.device,
        num_layers=args.layers,
        lr=args.lr,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.bs,
        num_blocks=args.blocks)
    results = rqnsf(
        deltas=deltas,
        train_dataset=data,
        test=data)
    print("rqnsf", file=report_file)
    #results = rqnsf.dims_on_deltas(deltas, epoch=best_epochs, total_dim=data.shape[1])
    #rqnsf.save(f"{args.algorithm}_{args.dataset}")

elif args.algorithm == "mle":
    print(f"mle:k={args.k}", file=report_file)
    # results = mle(data, k=args.k)
    results = mle_skl(data, k=args.k)

elif args.algorithm == "mle-inv":
    print(f"mle-inv:k={args.k}", file=report_file)
    results = mle_inv(data, k=args.k)


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
    ## End measuring time
    end_time = time.time()
    run['running_time'] = end_time - start_time
    run.stop()


print("\n".join(map(str, results)), file=report_file)
