#/bin/bash

source ~/glow-pytorch/venv/bin/activate
sstatus=$?
if [ "$sstatus" -ne 0 ]; then
    exit 2
fi

#srun --time 3-0 --qos=32gpu7d  --gres=gpu:1 python run_experiments.py --dataset gaussian-10-20 --algorithm maf --delta 0.05 --num_deltas 2 --neptune_name rm360179/lidl-loss-mse --neptune_token "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==" --device cuda --size 100 --epochs 2 --ground_truth_const 10

algorithms=("skdim_corrint"
        "skdim_danco"
        "skdim_ess"
        "skdim_fishers"
        "skdim_knn"
        "skdim_lpca"
        "skdim_mada"
        "skdim_mind_ml"
        "skdim_mle"
        "skdim_mom"
        "skdim_tle"
        "skdim_twonn")
dataset="swiss-roll-r3"
for seed in {1..10}; do
        for algorithm in "${algorithms[@]}"; do
		srun --time 3-0 --qos=32gpu7d  --gres=gpu:1 python run_experiments.py \
			--dataset $dataset \
			--algorithm $algorithm \
			--seed $seed \
			--neptune_name rm360179/lidl-loss-mse \
			--neptune_token "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==" \
			--device cuda \
			--ground_truth_const 2 &
                done;
	done;


deactivate
