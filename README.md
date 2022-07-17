# LIDL: Local Intrinsic Dimension Estimation Using Approximate Likelihood

Piotr Tempczyk, Rafał Michaluk, Łukasz Garncarek, Przemysław Spurek, Jacek Tabor, Adam Goliński

in ICML2022: https://icml.cc/Conferences/2022/Schedule?showEvent=18215

https://arxiv.org/abs/2206.14882

## Method

We propose Local Intrinsic Dimension estimation using approximate Likelihood (LIDL). Our method relies on an arbitrary density estimation method as its subroutine, and hence tries to sidestep the dimensionality challenge by making use of the recent progress in parametric neural methods for likelihood estimation. We carefully investigate the empirical properties of the proposed method, compare them with our theoretical predictions, show that LIDL yields competitive results on the standard benchmarks for this problem, and that it scales to thousands of dimensions. What is more, we anticipate this approach to improve further with the continuing advances in the density estimation literature.




| <img width="792" alt="Screenshot 2022-07-17 at 03 54 26" src="https://user-images.githubusercontent.com/15023195/179380725-0b669d9a-e69c-45f8-b82d-1d2491f533f0.png"> <img width="361" alt="Screenshot 2022-07-17 at 03 58 33" src="https://user-images.githubusercontent.com/15023195/179380806-adf6345e-9255-475e-8b93-6641ea3f871f.png"> | 
|:--:| 
| Comparison of relative MAE of LID estimates of different algorithms |

| <img width="403" alt="Screenshot 2022-07-17 at 03 52 08" src="https://user-images.githubusercontent.com/15023195/179380727-f497f8b0-d6cf-4f09-a32c-f55ac79ce87f.png"> | 
|:--:| 
|  Illustration of LIDL’s core insight. [Top] Three uniform distributions pS supported respectively on a square, interval, and a point, with intrinsic dimensions 2, 1, 0. [Middle/bottom] Perturbed densities $ρ_δ$ and $ρ_{2δ}$ resulting from addition of Gaussian noise with different noise magnitudes: $δ$ and $2δ$. Our core insight is that the difference between the densities $ρ_δ(x)$ and $ρ_{2δ}(x)$ at any point x depends on the local intrinsic dimension (LID) at that point. Consider point x = (0, 0). For the left column, that difference is zero; for the middle one, the density is halved; for the right one, it is quartered. We leverage this mechanism to estimate LID. |



| <img width="407" alt="Screenshot 2022-07-17 at 03 52 50" src="https://user-images.githubusercontent.com/15023195/179380726-2a1de0aa-9ba2-4b8e-a3c1-e7107195e082.png"> | 
|:--:| 
| LIDL can very accurately predict local intrinsic dimension of the points from the lollipop benchmark dataset |


| <img width="385" alt="Screenshot 2022-07-17 at 03 55 39" src="https://user-images.githubusercontent.com/15023195/179380762-3c7d7ec8-5636-4e2a-83ec-ddeb067c239e.png"> | 
|:--:| 
| Samples from different image datasets (MNIST, Celab-A, FMNIST from left to right) presented according to their LIDL estimates (top to bottom). Those results are highly correlated with the complexity of an image. |




# Repository

## Examples

Look at examples/swiss_roll.ipynb and examples/lollipop.ipynb

## Quick Start

```sh
pip install -r requirements.txt
python run_experiments.py dataset=lollipop-0 algorithm=rqnsf size=1000 --delta 0.05 --num_deltas 12 --device cuda
```

Dimension estimations of samples of the lollipop dataset will appear in a simple csv file in main directory. Running the default experiment should take less than 10 minutes.

For more details run
```
python run_experiments.py -h
```


## Parameters

The safest parameters to experiment with are:
- dataset: there are multiple datasets available (implemented in datasets.py, you can list them with `python run_experiments.py -h`)
- algorithm: (rqnsf, gm or maf if you want to use LIDL). We've also added some of the most common algorithms used for LID estimation (mle, corrdim)
- size: smaller dataset size can increase speed, but reduce accuracy


num_deltas increases the accuracy at the cost of running time.


