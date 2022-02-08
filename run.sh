#/bin/bash

dataset=lollipop-0
algorithm=rqnsf #maf #gm
size=10000
python run_experiments.py \
	--dataset $dataset \
	--algorithm $algorithm \
	--size $size \
	--delta 0.05 \
	--num_deltas 12 \
	--device cuda &
