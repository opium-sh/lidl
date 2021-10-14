for d in sin-01 sin-02 sin-05 sin-10 sin-20 sin-30 sin-50 sin-80  sin-160 sin-320;
do python ../../run_experiments.py --algorithm rqnsf --dataset $d --delta 0.1 \
--device cuda --size 10000 --layers 4 --epochs 10000 --hidden 15;
done
