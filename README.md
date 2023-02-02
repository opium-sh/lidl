# not LIDL

Based on [LIDL: Local Intrinsic Dimension Estimation Using Approximate Likelihood](https://arxiv.org/abs/2206.14882).

## Method

# Repository

## Quick Start

```sh
git clone ....
cd ....
pip install -r requirements.txt
python run_experiments.py --dataset=lollipop-0 --algorithm=rqnsf --size=1000 --delta 0.05 --num_deltas 12 --device cuda
```

Dimension estimations of samples of the lollipop dataset will appear in a simple csv file in main directory. Running the default experiment should take less than 10 minutes.

For more details run
```
python run_experiments.py -h
```


[//]: # (## Parameters)

[//]: # ()
[//]: # (The safest parameters to experiment with are:)

[//]: # (- dataset: there are multiple datasets available &#40;implemented in datasets.py, you can list them with `python run_experiments.py -h`&#41;)

[//]: # (- algorithm: &#40;rqnsf, gm or maf if you want to use LIDL&#41;. We've also added some of the most common algorithms used for LID estimation &#40;mle, corrdim&#41;)

[//]: # (- size: smaller dataset size can increase speed, but reduce accuracy)

[//]: # ()
[//]: # ()
[//]: # (num_deltas increases the accuracy at the cost of running time.)


## Citation
```
...
```
