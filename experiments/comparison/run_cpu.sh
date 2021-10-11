#/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 params_file seed"
    exit 1;
fi

params=$1
seed=$2
while IFS= read -r line; do
    python ../../run_experiments.py $line --seed=$seed >> experiment_log &
done < $params
