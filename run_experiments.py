from dim_estimators import mle, corr_dim, LIDL
from datasets import lollipop_dataset, spirals_dataset, swiss_roll_r3, N_100_200, generate_datasets
import argparse
import datasets

size = 100
inputs = {
    'uniform_1': (datasets.uniform_N(1, size), 1),
    'uniform_10': (datasets.uniform_N(10, size), 10),
    'uniform_100': (datasets.uniform_N(100, size), 100),
    'uniform_1000': (datasets.uniform_N(1000, size), 1000),
    'uniform_10000': (datasets.uniform_N(10000, size), 10000),
    'gaussian_1': (datasets.gaussian(1, size), 1),
    'gaussian_10': (datasets.gaussian(10, size), 10),
    'gaussian_100': (datasets.gaussian(100, size), 100),
    'gaussian_1000': (datasets.gaussian(1000, size), 1000),
    'gaussian_10000': (datasets.gaussian(10000, size), 10000),
    'sphere_7': (datasets.sphere_7(size), 6),
    'uniform_helix_r3': (datasets.uniform_helix_r3(size), 1),
    'swiss_roll_r3': (datasets.swiss_roll_r3(size), 2),
    'sin': (datasets.sin(50), 1),
    'sin_quant': (datasets.sin(50), 1)
    }

parser = argparse.ArgumentParser(description="LIDL experiments")
parser.add_argument(
    "--algorithm",
    default="mle",
    type=str,
    choices=["mle", "gaussian_mixture", "rqnsf", "maf", "corr_dim"],
    help="name of the algorithm"
)
parser.add_argument(
    "--dataset",
    default="uniform_1",
    type=str,
    choices=list(inputs.keys()),
    help="dataset whose id will be estimated"
)
parser.add_argument(
    "--k",
    default="3",
    type=int,
    help="number of neighbours in mle (does nothing with different algorithms"
)

args = parser.parse_args()


report_filename = 'report_dim_estimate.txt'
f = open(report_filename, 'w+')
deltas = [0.010000, 0.013895, 0.019307, 0.026827, 0.037276, 0.051795, 0.071969, 0.100000]



data, total_dim = inputs[args.dataset]
if args.algorithm == 'gaussian_mixture':
    gm = LIDL('gaussian_mixture')
    print(f'Gaussian mixture', file=f)
    gm.run_on_deltas(deltas, data=data, samples=data, runs=1)
    print(gm.dims_on_deltas(deltas, epochs=0, total_dim=total_dim), file=f)
    gm.save(f'{args.dataset}')
elif args.algorithm == 'corr_dim':
    print('Corrdim', file=f)
    print(corr_dim(data), file=f)
elif args.algorithm == 'maf':
    maf = LIDL('maf')
    maf.run_on_deltas(deltas, data=data, epochs=200)
    print('maf', file=f)
    print(maf.dims_on_deltas(deltas, epoch=199, total_dim=total_dim), file=f)
    maf.save(f'{args.dataset}')
elif args.algorithm == 'rqnsf':
    rqnsf = LIDL('rqnsf')
    rqnsf.run_on_deltas(deltas, data=data, epochs=200)
    print('rqnsf', file=f)
    print(rqnsf.dims_on_deltas(deltas, epoch=199, total_dim=total_dim), file=f)
    rqnsf.save(f'{args.dataset}')
elif args.algorithm == 'mle':
    print(f'MLE, k={args.k}', file=f)
    print(mle(data, k=args.k, device='cpu'), file=f)
