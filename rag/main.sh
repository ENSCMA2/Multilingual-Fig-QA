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

for lang in "jv" "kn" "su" "sw";
do
    echo $lang;
    for model in "mistralai/Mistral-7B-Instruct-v0.2"; # "Qwen/Qwen1.5-7B-Chat"; # "meta-llama/Llama-2-7b-chat-hf"
    do
        echo $model;
        python main.py --lang $lang --testset langdata --generator $model --retriever wikidatatok
        python main.py --lang $lang --testset translate-test --generator $model --retriever wikidatatok
        python main.py --lang $lang --testset langdata --generator $model --retriever bm25tok
        python main.py --lang $lang --testset translate-test --generator $model --retriever bm25tok
    done
done
