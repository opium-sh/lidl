#/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 params_file seed"
    exit 1;
fi

params=$1
seed=$2
while IFS= read -r line; do
    srun --time 3-0 --qos=32gpu7d --gres=gpu python ../../run_experiments.py $line --seed=$seed >> experiment_log &
done < $params
