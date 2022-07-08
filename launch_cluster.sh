
#! /bin/bash

for tau in 0.1 0.05 0.2; do

  for frq in 0.5 0.7; do
    for gamma in 1 2 5 10; do
      sbatch --export=LT=0,F=$frq,T=$tau,M=1.,E=2,L=1,G=$gamma,SEED=0 cluster_sbatch.sbatch
      sleep 1
    done
  done
  sleep 1000
done

for m in 0.05 0.01 0.005; do

  for frq in 0.5 0.7; do
    for gamma in 1 2 5 10; do
      sbatch --export=LT=0,F=$frq,T=1.,M=$m,E=2,L=1,G=$gamma,SEED=0 cluster_sbatch.sbatch
      sleep 1
    done
  done
  sleep 1000
done