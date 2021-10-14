import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def normalize(data):
    data -= data.mean(axis=0)
    data /= data.std() + 0.001
    return data


def moons_dataset(bs, seed=0):
    np.random.seed(seed)
    x, y = datasets.make_moons(bs)
    return x


def moons_dataset_in_4d(bs, seed=0):
    np.random.seed(seed)
    x = moons_dataset(bs)
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def moons_2d_dataset_in_4d(bs, seed=0):
    np.random.seed(seed)
    x = moons_dataset(bs)
    x = x + np.random.randn(*x.shape) * 0.05
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def line_dataset(bs, seed=0):
    np.random.seed(seed)
    x = np.random.randn(bs, 2) / 2
    x[:, 1] = x[:, 0]
    return x


def parabola_dataset(bs, seed=0):
    np.random.seed(seed)
    x = np.random.randn(bs, 2) / 2
    x[:, 1] = x[:, 0] ** 2
    return x


def parabola_3d_dataset(bs, seed=0):
    np.random.seed(seed)
    x = np.random.randn(bs, 3) / 2
    x[:, 1] = x[:, 0] ** 2
    x[:, 2] = 1.0
    return x


def parabola_2d_dataset_in_4d(bs, seed=0):
    np.random.seed(seed)
    x = np.random.randn(bs, 2)
    x[:, 1] = x[:, 0] ** 2
    x = x + np.random.randn(*x.shape) * 0.5
    x_2 = np.ones_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def parabola_2d_dataset_in_10d(bs, seed=0):
    np.random.seed(seed)
    x = np.random.randn(bs, 2)
    x[:, 1] = x[:, 0] ** 2
    x = x + np.random.randn(*x.shape) * 0.5
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2, x_2, x_2, x_2], axis=1)
    return x


def parabola_6d_dataset_in_18d(bs, seed=0):
    np.random.seed(seed)
    x = np.random.randn(bs, 6)
    x[:, 1] = x[:, 0] ** 2
    x[:, 2] = x[:, 0] ** 2
    x[:, 3] = np.abs(x[:, 0]) ** 0.5
    x[:, 4] = x[:, 0]
    x = x + np.random.randn(*x.shape) * 0.5
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2, x_2], axis=1)
    return x


def s_dataset_in_6d(bs, seed=0):
    np.random.seed(seed)
    x = datasets.make_s_curve(bs)[0]
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def spirals_dataset(bs, seed=0):
    np.random.seed(seed)
    n = np.sqrt(np.random.rand(bs // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n
    d1y = np.sin(n) * n
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    return x


def lollipop_dataset(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.95 * bs)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x = np.zeros((bs, 2))
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x += 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(bs - cs))
    x[cs:, 0] = stick
    x[cs:, 1] = stick
    return x

def lollipop_dataset_0(bs, seed=0):
    np.random.seed(seed)
    cs = int(0.94 * bs)
    cp = int(0.99 * bs)
    r = np.random.uniform(size=cs)
    fi = np.random.uniform(0, 2 * np.pi, size=cs)
    x = np.zeros((bs, 2))
    x[:cs, 0] = r ** 0.5 * np.sin(fi)
    x[:cs, 1] = r ** 0.5 * np.cos(fi)
    x += 2
    stick = np.random.uniform(high=2 - 1 / np.sqrt(2), size=(cp - cs))
    x[cs:cp, 0] = stick
    x[cs:cp, 1] = stick
    x[cp:] = np.random.normal(loc=(-.5, -.5), scale=1e-3, size=(bs-cp, 2))
    x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    return x


def uniform_helix_r3(bs, seed=0):
    np.random.seed(seed)
    x = np.zeros((bs, 3))
    t = np.random.uniform(-1, 1, size=bs)
    x[:, 0] = np.sin(np.pi * t)
    x[:, 1] = np.cos(np.pi * t)
    x[:, 2] = t
    return x


def swiss_roll_r3(bs, seed=0):
    np.random.seed(seed)
    x = sklearn.datasets.make_swiss_roll(bs)[0]
    x -= x.mean(axis=0).reshape(1, 3)
    x /= x.std(axis=0).reshape(1, 3)
    return x


def N_5(bs, seed=0):
    np.random.seed(seed)
    return np.random.normal(size=(bs, 5))


def uniform_sphere_S7(bs, seed=0):
    np.random.seed(seed)
    pass


def uniform_12(bs, seed=0):
    np.random.seed(seed)
    return np.random.uniform(low=-0.5, high=0.5, size=(bs, 12))


def uniform_N(N, bs, seed=0):
    np.random.seed(seed)
    return np.random.uniform(low=-0.5, high=0.5, size=(bs, N))


def sphere_7(bs, seed=0):
    np.random.seed(seed)
    x = np.random.normal(size=(bs, 8))
    lam = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    x = x / lam
    return x


def gaussian_N_2N(bs, N=1, seed=0):
    np.random.seed(seed)
    x = np.random.normal(size=(bs, N))
    x = np.concatenate([x, x], axis=1)
    return x


def gaussian(N, bs, seed=0):
    np.random.seed(seed)
    return np.random.randn(bs, N)


def sin(bs, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, bs)
    return np.stack([x, (10 * np.sin(x / 3))], axis=1)


def sin_freq(bs, freq=5, seed=0):
    np.random.seed(seed)
    x = np.random.uniform(0, 1, bs)
    return np.stack([x, np.sin(freq*x*(2 * np.pi))], axis=1)


def sin_quant(bs, seed=0):
    np.random.seed(seed)
    return np.round(sin(bs))


def sin_dequant(bs, seed=0):
    np.random.seed(seed)
    x = sin_quant(bs)
    x += np.random.rand(*x.shape)
    return x


def generate(generator, bs, seed):
    np.random.seed(seed)
    return generator(bs)


def generate_datasets(seed, size):
    np.random.seed(seed)
    datasets = list()
    for dim in (1, 10, 100, 1000, 10000):
        datasets.append(uniform_N(dim, size))
    for dim in (1, 10, 100, 1000, 10000):
        datasets.append(gaussian(dim, size))
    datasets += [sphere_7(size), uniform_helix_r3(size), swiss_roll_r3(size)]

    return datasets


def sin_dens(bs, freq=5, offset=2.1, seed=0):
    def fun(x, y, freq, offset):
        return np.cos(freq * x) + np.cos(freq * y) + offset
    np.random.seed(seed)
    sample = np.random.rand(10*bs, 3) * np.array([[2*np.pi, 2*np.pi, offset + 2]]) + np.array([[-np.pi, -np.pi, 0]])
    resampled = np.array([[point[0], point[1]]
                          for point in sample
                          if point[2] < fun(point[0], point[1], freq, offset)
                         ])
    assert len(resampled) >= bs
    resampled = resampled[:bs]
    resampled = np.concatenate([resampled, np.zeros_like(resampled)], axis=1)
    return resampled

def csv_dataset(path):
    df = pd.read_csv(path, header=None, delim_whitespace=True)
    x = df.iloc[:, :-1].values
    print(x.shape)
    x_scaler = StandardScaler()
    x_scaler.fit(x)
    x = x_scaler.transform(x)
    return x
