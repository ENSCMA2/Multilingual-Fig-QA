#!/bin/bash
#SBATCH --job-name=anlp
#SBATCH --output=anlp.out
#SBATCH --error=anlp.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=16
#SBATCH --time=100:00:00

# TODO: average over non-degenerate seeds
LANGS=("en_dev" "hi" "id" "jv" "kn" "su" "sw")

for model in "meta-llama/Llama-2-7b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.2" "Qwen/Qwen1.5-7B-Chat"
do
    echo $model;
    for lang in "jv" "kn" "su" "sw";
    do
        echo $lang;
        python main.py --lang $lang --testset langdata --generator $model
    done
    for lang in "jv" "kn" "su" "sw";
    do
        echo $lang;
        python main.py --lang $lang --testset translate-test --generator $model
    done
done
