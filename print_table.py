import argparse
import re
import numpy as np
import pandas as pd
from collections import defaultdict


def extract_datasets(filenames):
    result = list()
    dataset_pattern = re.compile(r'.*dataset:([^_]*)_.*')
    for filename in filenames:
        matches = dataset_pattern.findall(filename)
        if len(matches) == 1:
            result.append(matches[0])

    return result


def print_table(results, rownames, colnames, row_width=20):
    print("-----------------------------")
    print(f"{'':30}", end=" ")
    for colname in colnames:
        print(f"{colname:>{row_width}}", end=" ")
    print()

    for rowname in rownames:
        print(f"{rowname:30}", end=" ")
        for colname in colnames:
            value = results[((rowname, colname))]
            s_value = str(value)
            if len(s_value) > row_width:
                s_value = s_value[:row_width-3] + "..."
            print(f'{s_value:>{row_width}}', end=" ")
        print()
    print("-----------------------------")

parser = argparse.ArgumentParser(description="LIDL experiments")
parser.add_argument("files", type=argparse.FileType("r"), nargs="+")
args = parser.parse_args()

filenames = [f.name.strip() for f in args.files]
datasets = np.unique(extract_datasets(filenames))
algorithms = "mle mle-inv corrdim gm maf rqnsf".split()

counter = defaultdict(lambda: 0)
experiment_results = defaultdict(lambda: 'empty')
for dataset in datasets:
    for algorithm in algorithms:
        matched_files = list()
        for filename in filenames:
            if f'algorithm:{algorithm}_' in filename and f'dataset:{dataset}_' in filename:
                matched_files.append(filename)

        if len(matched_files) == 0:
            continue
        try:
            means = list()
            for matched_file in matched_files:
                df = pd.read_csv(matched_file)
                means.append(df.mean(numeric_only=True).values[0])
                counter[(dataset, algorithm)] += 1

            means = np.array(means)
            total_mean = means.mean()
            total_std = means.std()

            s_mean = f"{total_mean:.2f}"
            s_std = f"{total_std:.2f}"
            result = f"{s_mean}\u00B1{s_std}"
            experiment_results[(dataset, algorithm)] = result
        except Exception as e:
            experiment_results[(dataset, algorithm)] = 'error'
            counter[(dataset, algorithm)] = 'X'


print("Results")
print_table(experiment_results, datasets, algorithms)

print('Number of files used to estimate the values above:')
print_table(counter, datasets, algorithms, row_width=9)
