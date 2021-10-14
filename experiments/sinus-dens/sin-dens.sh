#for d in sin-dens-1 sin-dens-2 sin-dens-4 sin-dens-8 sin-dens-16;
for d in sin-dens-3 sin-dens-6 sin-dens-10 sin-dens-12 sin-dens-14;
do python ../../run_experiments.py --algorithm rqnsf --dataset $d --delta 0.1 \
--device cuda --size 10000 --layers 4 --epochs 10000 --hidden 15;
done