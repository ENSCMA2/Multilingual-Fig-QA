#!/bin/bash
#SBATCH --job-name=anlp
#SBATCH --output=anlp.out
#SBATCH --error=anlp.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00

# We currently have a bunch of shell scripts that run the experiments. 
# This basically provides an easier interface to access all experiments and run multiple times

# TODO: average over non-degenerate seeds
LANGS=("en_dev" "hi" "id" "jv" "kn" "su" "sw")

for model in "Qwen/Qwen1.5-7B" "mistralai/Mixtral-8x7B-v0.1" "meta-llama/Llama-2-7b-hf";
do
    echo $model;
    for lang in "en_dev" "hi" "id" "jv" "kn" "su" "sw";
    do
        echo $lang;
        python run_llm_baselines.py --test_file $lang --test_dir langdata --n 0 --model_name_or_path $model
    done
    for lang in "hi" "id" "jv" "kn" "su" "sw";
    do
        echo $lang;
        python run_llm_baselines.py --test_file $lang --test_dir translate-test --n 0 --model_name_or_path $model
    done
    for lang in "hi" "id" "jv" "kn" "su" "sw";
    do
        echo $lang;
        for n in 2 4 6 8 10;
        do
           echo $n;
           python run_llm_baselines.py --test_file $lang_$n --test_dir data/addition_data_winogrand/test/$lang --n $n --model_name_or_path $model 
        done
    done
done
