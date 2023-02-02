import argparse

from src.script_args.algorithms import skdim_algorithms
from src.script_args.datasets import inputs


def make_experiment_argument_parser():
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

    parser.add_argument(
        "--gm_max_components",
        default=200,
        type=int,
        help="number of components in gaussian mixture"
    )
    return parser
