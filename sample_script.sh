#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N pythia-160m
#$ -o /exports/eddie/scratch/UUN/pythia-160m_$JOB_ID.log
#$ -e /exports/eddie/scratch/UUN/pythia-160m_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00

# Create / activate conda env if it doesn't exist

source /exports/eddie/scratch/UUN/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/UUN/
#conda remove --name extract --all

conda create -n extract python=3.9 
conda activate extract

git clone https://github.com/dmcgrath19/base_extraction.git

cd base_extraction

pip install -r requirements.txt

# Run the main script
python main.py /
    --N 1000 
    --batch-size 10 /
    --model1 EleutherAI/pythia-2.8b /
    --model2 EleutherAI/pythia-160m /
    --corpus-path monology/pile-uncopyrighted /
    --outfile pythia-160m

conda deactivate 
