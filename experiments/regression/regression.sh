for d in "boston" "protein" "wine" "power" "yacht" "concrete" "energy" "kin8nm" "naval" "year";
  do python ../../run_experiments.py --dataset $d --algorithm rqnsf --delta 0.2 --hidden 15 --layers 4 --device cuda;
done;