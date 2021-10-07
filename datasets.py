from sklearn import datasets
import numpy as np
import sklearn


def moons_dataset(bs):
    x, y = datasets.make_moons(bs)
    return x


def moons_dataset_in_4d(bs):
    x = moons_dataset(bs)
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def moons_2d_dataset_in_4d(bs):
    x = moons_dataset(bs)
    x = x + np.random.randn(*x.shape) * 0.05
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def line_dataset(bs):
    x = np.random.randn(bs, 2) / 2
    x[:, 1] = x[:, 0]
    return x


def parabola_dataset(bs, seed=4):
    x = np.random.randn(bs, 2) / 2
    x[:, 1] = x[:, 0] ** 2
    return x


def parabola_3d_dataset(bs, seed=5):
    x = np.random.randn(bs, 3) / 2
    x[:, 1] = x[:, 0] ** 2
    x[:, 2] = 1.0
    return x


def parabola_2d_dataset_in_4d(bs, seed=6):
    x = np.random.randn(bs, 2)
    x[:, 1] = x[:, 0] ** 2
    x = x + np.random.randn(*x.shape) * 0.5
    x_2 = np.ones_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def parabola_2d_dataset_in_10d(bs):
    x = np.random.randn(bs, 2)
    x[:, 1] = x[:, 0] ** 2
    x = x + np.random.randn(*x.shape) * 0.5
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2, x_2, x_2, x_2], axis=1)
    return x


def parabola_6d_dataset_in_18d(bs):
    x = np.random.randn(bs, 6)
    x[:, 1] = x[:, 0] ** 2
    x[:, 2] = x[:, 0] ** 2
    x[:, 3] = np.abs(x[:, 0]) ** 0.5
    x[:, 4] = x[:, 0]
    x = x + np.random.randn(*x.shape) * 0.5
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2, x_2], axis=1)
    return x


def s_dataset_in_6d(bs):
    x = datasets.make_s_curve(bs)[0]
    x_2 = np.zeros_like(x)
    x = np.concatenate([x, x_2], axis=1)
    return x


def spirals_dataset(bs):
    n = np.sqrt(np.random.rand(bs // 2, 1)) * 540 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n
    d1y = np.sin(n) * n
    x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
    return x


def lollipop_dataset(bs):
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


def uniform_helix_r3(bs):
    x = np.zeros((bs, 3))
    t = np.random.uniform(-1, 1, size=bs)
    x[:, 0] = np.sin(np.pi * t)
    x[:, 1] = np.cos(np.pi * t)
    x[:, 2] = t
    return x


def swiss_roll_r3(bs):
    x = sklearn.datasets.make_swiss_roll(bs)[0]
    x -= x.mean(axis=0).reshape(1, 3)
    x /= x.std(axis=0).reshape(1, 3)
    return x


def N_5(bs):
    return np.random.normal(size=(bs, 5))


def uniform_sphere_S7(bs):
    pass


def uniform_12(bs):
    return np.random.uniform(low=-0.5, high=0.5, size=(bs, 12))


def uniform_N(N, bs):
    return np.random.uniform(low=-0.5, high=0.5, size=(bs, N))


def sphere_7(bs):
    x = np.random.normal(size=(bs, 8))
    lam = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    x = x / lam
    return x


def N_1_2(bs):
    x = np.random.normal(size=(bs, 1))
    x = np.concatenate([x, x], axis=1)
    return x


def N_10_20(bs):
    x = np.random.normal(size=(bs, 10))
    x = np.concatenate([x, x], axis=1)
    return x


def N_100_200(bs):
    x = np.random.normal(size=(bs, 100))
    x = np.concatenate([x, x], axis=1)
    return x


def N_1000_2000(bs):
    x = np.random.normal(size=(bs, 1000))
    x = np.concatenate([x, x], axis=1)
    return x


def N_10000_20000(bs):
    x = np.random.normal(size=(bs, 10000))
    x = np.concatenate([x, x], axis=1)
    return x


def gaussian(N, bs):
    return np.random.randn(bs, N)


def sin(bs):
    x = np.linspace(0, 10, bs)
    return (10 * np.sin(x / 3)).reshape(-1, 1)


def sin_quant(bs):
    return np.round(sin(bs))


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
