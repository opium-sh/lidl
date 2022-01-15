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
        "skdim_twonn"
	"maf"
	"rqnsf")
datasets=("uniform_N_0_1-12"
          "gaussian-5"
	  "sphere-7"
	  "uniform-helix-r3"
	  "swiss-roll-r3"
	  "gaussian-10-20"
	  "gaussian-100-200")

for seed in {1..5}; do
        for algorithm in "${algorithms[@]}"; do
		for dataset in "${datasets[@]}"; do
			echo $seed-$algorithm-$dataset
			srun --time 3-0 --qos=32gpu7d  --gres=gpu:1 python run_experiments.py \
				--dataset $dataset \
				--algorithm $algorithm \
				--seed $seed \
			        --size 10000 \
			        --delta 0.05 \
		        	--num_deltas 11 \
				--neptune_name rm360179/lidl-loss-mse \
				--neptune_token "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MmVjN2EzYS04Y2FmLTRkYjItOTkyMi1mNmEwYWQzM2I3Y2UifQ==" \
				--device cuda &
			done;
                done;
	done;


deactivate
