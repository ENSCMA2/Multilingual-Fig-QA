#!/bin/bash
MODEL=${1}
MODEL_NAME_OR_PATH=${2}
BASE_DIR=`pwd`
LR=${3:-5e-5}
SEED=${4:-10}
declare -a languages=( "hi" "id" "jv" "kn" "su" "sw" )
declare -a number=( "2" "4" "6" "8" "10" )

for lang in "${!languages[@]}"
do
    for num in "${!number[@]}"
    do
        echo "${languages[lang]}: ${number[num]}"

        python run_baselines.py \
        --model_name_or_path ./${MODEL_NAME_OR_PATH}/${languages[lang]}_${number[num]}/ckpts_seed${SEED}_lr${LR} \
        --test_file data/addition_data_winogrand/test/${languages[lang]}/${languages[lang]}_${number[num]}.csv \
        --do_predict \
        --per_device_eval_batch_size 32 \
        --test_runner_mode 
    done
done
