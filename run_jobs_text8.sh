#!/bin/bash

sbatch -p ai job_candi_text8.sh; 
sbatch -p ai job_mdlm_text8.sh;
sbatch -p ai job_pure_cont_corrupt_text8.sh;
sbatch -p ai job_pure_cont_percentile_text8.sh;
sbatch -p ai job_pure_cont_embed_text8.sh;
