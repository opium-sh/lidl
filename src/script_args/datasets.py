import datasets

inputs = {
    "uniform-1": lambda size, seed: datasets.uniform_N(1, size, seed=seed),
    "uniform-10": lambda size, seed: datasets.uniform_N(10, size, seed=seed),
    "uniform-12": lambda size, seed: datasets.uniform_N(12, size, seed=seed),
    "uniform-100": lambda size, seed: datasets.uniform_N(100, size, seed=seed),
    "uniform-500": lambda size, seed: datasets.uniform_N(500, size, seed=seed),
    "uniform-1000": lambda size, seed: datasets.uniform_N(1000, size, seed=seed),
    "uniform-2000": lambda size, seed: datasets.uniform_N(2000, size, seed=seed),
    "uniform-4000": lambda size, seed: datasets.uniform_N(4000, size, seed=seed),
    "uniform-10000": lambda size, seed: datasets.uniform_N(10000, size, seed=seed),
    "uniform_N_0_1-12": lambda size, seed: datasets.uniform_N_0_1(12, size, seed=seed),
    "gaussian-1": lambda size, seed: datasets.gaussian(1, size, seed=seed),
    "gaussian-5": lambda size, seed: datasets.gaussian(5, size, seed=seed),
    "gaussian-10": lambda size, seed: datasets.gaussian(10, size, seed=seed),
    "gaussian-100": lambda size, seed: datasets.gaussian(100, size, seed=seed),
    "gaussian-500": lambda size, seed: datasets.gaussian(500, size, seed=seed),
    "gaussian-1000": lambda size, seed: datasets.gaussian(1000, size, seed=seed),
    "gaussian-2000": lambda size, seed: datasets.gaussian(2000, size, seed=seed),
    "gaussian-4000": lambda size, seed: datasets.gaussian(4000, size, seed=seed),
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
    "gaussian-2000-4000": lambda size, seed: datasets.gaussian_N_2N(size, N=2000, seed=seed),
    "gaussian-10000-20000": lambda size, seed: datasets.gaussian_N_2N(size, N=10000, seed=seed),
    "lollipop": lambda size, seed: datasets.lollipop_dataset(size, seed=seed),
    "lollipop-0": lambda size, seed: datasets.lollipop_dataset_0(size, seed=seed),
    "lollipop-0-dense-head": lambda size, seed: datasets.lollipop_dataset_0_dense_head(size, seed=seed),
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
