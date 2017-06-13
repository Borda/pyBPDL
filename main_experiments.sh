#!/bin/bash
# launcher for all experiments

# OUR method - BPDL

python experiments/run_experiments_bpdl.py \
    -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v0 \
    -out ~/Medical-drosophila/TEMPORARY/experiments_APDL_synth

# STATE-OF-THE-ART methods - SPCA, ICA, DL, NMF, BPDL

python experiments/run_experiments_all.py \
     -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v0 \
     -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth


for i in {1..50}
do

    python experiments/run_dataset_add_noise.py -p ~/Medical-drosophila/synthetic_data \
        -d  atomicPatternDictionary_v0 \
            atomicPatternDictionary_v1 \
            atomicPatternDictionary_v2

    python experiments/run_experiments_all.py \
         -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v0 \
         -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

    python experiments/run_experiments_all.py \
         -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v1 \
         -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

    python experiments/run_experiments_all.py \
         -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v2 \
         -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

done

# EVALUATE experiments

python experiments/run_parse_experiments_results.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APD_synth \
    --fname_results results.csv --fname_config config.json --func_stat none

python experiments/run_recompute_experiments_results.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

python experiments/run_parse_experiments_results.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APD_synth \
    --fname_results results_NEW.csv --fname_config config.json --func_stat none


# REAL IMAGES by folds

for i in {1..4}
do
      cmd="python experiments/run_experiments_all.py --type real \
            -in ~/Medical-drosophila/TEMPORARY/type_${i}_segm_reg_binary \
            -out ~/Medical-drosophila/TEMPORARY/experiments_APD_real_folds \
            --dataset gene_ssmall"
      echo $cmd
      $cmd
done

for i in {1..4}
do
   for j in {1..5}
   do
      cmd="python experiments/run_experiments_all.py --type real \
            -in ~/Medical-drosophila/TEMPORARY/type_${i}_segm_reg_binary \
            -out ~/Medical-drosophila/TEMPORARY/experiments_APD_real_folds \
            --dataset gene_ssmall \
            --list_images ../list_images_subset_${j}of5.csv"
      echo $cmd
      $cmd
   done
done