#!/bin/bash
MODEL=${1}
BASE_DIR=`pwd`
LR=${3:-5e-5}
SEED=${2}
declare -a languages=( "hi" "id" "jv" "kn" "su" "sw" )
declare -a number=( "2" "4" "6" "8" "10" )
echo "ARGS"
echo ${MODEL}
echo ${MODEL_NAME_OR_PATH}
echo ${BASE_DIR}
echo ${LR}
echo ${SEED}
for lang in "${!languages[@]}"
do
    for num in "${!number[@]}"
    do
        echo "${languages[lang]}: ${number[num]}"
	echo "TEST PATH: ./${MODEL}/${MODEL_NAME_OR_PATH}${languages[lang]}_${number[num]}/ckpts_seed${SEED}_lr${LR}"
        python run_baselines.py \
        --model_name_or_path ./${MODEL}/${MODEL_NAME_OR_PATH}${languages[lang]}_${number[num]}/ckpts_seed${SEED}_lr${LR} \
        --test_file data/addition_data_winogrand/test/${languages[lang]}/${languages[lang]}_${number[num]}.csv \
        --do_predict \
        --per_device_eval_batch_size 32 \
        --test_runner_mode 
    done
done
