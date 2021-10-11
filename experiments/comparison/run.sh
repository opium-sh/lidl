for seed in {0..9}; do
  bash run_gpu.sh corrdim.params $seed;
  bash run_gpu.sh maf.params $seed;
  bash run_gpu.sh mle-inv.params $seed;
  bash run_gpu.sh mle.params $seed;
  bash run_gpu.sh rqnsf.params $seed;
  bash run_gpu.sh run.params $seed;
done;
