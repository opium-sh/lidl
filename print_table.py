import argparse
import re

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="LIDL experiments")
parser.add_argument("files", type=argparse.FileType("r"), nargs="+")

args = parser.parse_args()

algorithms = "mle corrdim gm maf rqnsf".split()
datasets = np.unique([file.name.split("_")[4] for file in args.files])
filenames = [file.name.strip() for file in args.files]


print("-----")
print(f"{'':30}", end=" ")
for algorithm in algorithms:
    print(f"{algorithm:>9}", end=" ")
print()

for dataset in datasets:
    print(f"{dataset:30}", end=" ")
    for algorithm in algorithms:
        pattern = r".*" + f"_{algorithm}_{dataset}" + r"_.*"
        p = re.compile(pattern)
        matched = [filename for filename in filenames if p.match(filename) is not None]
        if len(matched) == 0:
            print(f"{'empty':>9}", end=" ")
        elif len(matched) == 1:
            try:
                df = pd.read_csv(matched[0])
                mean = df.mean()
                string = f"{mean.values[0]:9.2f}"
                if len(string) > 9:
                    string = string[:6] + "..."
                print(string, end=" ")
            except:
                print(f"{'error':>9}", end=" ")
        else:
            raise ValueError("Too many matching files")
    print()
print("-----")
