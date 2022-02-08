# LIDL
## Quick Start

```sh
pip install -r requirements.txt
bash run.sh
```
Dimension estimations of all samples will appear in a simple csv file in main directory. Running the default experiment can take 2-6h.

## Parameters
You may want to change dataset or algorithm in run.sh (they are described in run_experiments.py).

The algorithms based on probability models are: "rqnsf", "gm", "maf".

num_deltas increases the accuracy at the cost of running time.

Details of the algorithm that are independent on the probability model (and use only log likelihoods for a set of samples on given deltas) are implemented in dim_estimators.py in dims_on_deltas(line 71).
