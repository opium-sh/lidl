from dim_estimators import mle, corr_dim, LIDL
from datasets import lollipop_dataset, spirals_dataset, swiss_roll_r3, N_100_200


report_filename = 'report_dim_estimate.txt'
N = 5000
datasets = [lollipop_dataset(N), spirals_dataset(N), swiss_roll_r3(N), N_100_200(N)]

f = open(report_filename, 'w+')


for i, data in enumerate(datasets):
    for k in range(2, 6):
        print(f'MLE, k={k}', file=f)
        print(mle(data, k), file=f)

    print('Corrdim', file=f)
    print(corr_dim(data), file=f)

    deltas = [0.010000, 0.013895, 0.019307, 0.026827, 0.037276, 0.051795, 0.071969, 0.100000]
    gm = LIDL('gaussian_mixture')
    gm.run_on_deltas(deltas, data=data, samples=data)
    print('Gaussian mixture', file=f)
    print(gm.dims_on_deltas(deltas, epoch=0, total_dim=0), file=f)
    gm.save(f'dataset_{i}')

    rqnsf = LIDL('rqnsf')
    rqnsf.run_on_deltas(deltas, data=data, epochs=200)
    print('rqnsf', file=f)
    print(rqnsf.dims_on_deltas(deltas, epoch=199, total_dim=0), file=f)
    rqnsf.save(f'dataset_{i}')

    maf = LIDL('maf')
    maf.run_on_deltas(deltas, data=data, epochs=200)
    print('maf', file=f)
    print(maf.dims_on_deltas(deltas, epoch=199, total_dim=0), file=f)
    maf.save(f'dataset_{i}')
