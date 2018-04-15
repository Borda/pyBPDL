#!/bin/bash
# launcher for all experiments

# STATE-OF-THE-ART methods - SPCA, ICA, DL, NMF, BPDL

python experiments/run_experiments.py \
     -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v0 \
     -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth


for i in {1..50}
do

    python experiments/run_dataset_add_noise.py \
        -p ~/Medical-drosophila/synthetic_data \
        -d  synthDataset_v0 \
            synthDataset_v1 \
            synthDataset_v2

    python experiments/run_experiments.py \
         -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v0 \
         -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

    python experiments/run_experiments.py \
         -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v1 \
         -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

    python experiments/run_experiments.py \
         -in ~/Medical-drosophila/synthetic_data/atomicPatternDictionary_v2 \
         -out ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

done

# EVALUATE experiments

python experiments/run_parse_experiments_result.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APD_synth \
    --name_results results.csv --name_config config.json --func_stat none

python experiments/run_recompute_experiments_result.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APD_synth

python experiments/run_parse_experiments_result.py \
    -p ~/Medical-drosophila/TEMPORARY/experiments_APD_synth \
    --name_results results_NEW.csv --name_config config.json --func_stat none


# REAL IMAGES by folds

for i in {1..4}
do
      cmd="python experiments/run_experiments.py --type real \
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
      cmd="python experiments/run_experiments.py --type real \
            -in ~/Medical-drosophila/TEMPORARY/type_${i}_segm_reg_binary \
            -out ~/Medical-drosophila/TEMPORARY/experiments_APD_real_folds \
            --dataset gene_ssmall \
            --list_images ../list_images_subset_${j}of5.csv"
      echo $cmd
      $cmd
   done
done