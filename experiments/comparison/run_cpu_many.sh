#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 params_file"
    exit 1;
fi

params=$1
for seed in {0..9}; do
  bash run_cpu.sh $params $seed
done;
