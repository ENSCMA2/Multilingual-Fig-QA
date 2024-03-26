#!/bin/bash

# We currently have a bunch of shell scripts that run the experiments. 
# This basically provides an easier interface to access all experiments and run multiple times

# TODO: average over non-degenerate seeds
SEEDS=( 10 31 32 33 81 )
# LANGS=( "en_dev" )
LANGS=( "hi" "id" "jv" "kn" "su" "sw")
print_help() {
    echo "Usage: $0 -e <experiment setting> -o <output file> -m <model name> "
    echo -e "\t-e: Experiment setting. One of [train, zero2hero]"
    echo -e "\t-o: Output file. The file to write the results to"
    echo -e "\t-m: Model name. One of [xlm-roberta-large, xlm-roberta-base, bert-base-multilingual-cased]"
    echo -e "Example usages: ./test_runner.sh -e train -o xlm_roberta_large -m xlm-roberta-large"
    echo -e "                ./test_runner.sh -e zero2hero -o xlm_roberta_large -m xlm-roberta-large"
    exit 1
}

while getopts "e:o:m:" opt; do
    case $opt in
        e)
            EXPERIMENT=$OPTARG
            ;;
        o)
            OUTPUT=$OPTARG
            ;;
        m)
            MODEL=$OPTARG
            ;;
        \?)
            print_help
            ;;
    esac
done

if [ -z "$EXPERIMENT" ] || [ -z "$OUTPUT" ] || [ -z "$MODEL" ]; then
    print_help
fi

if [ "$MODEL" == "xlm-roberta-large" ]; then
    LR=5e-6
    BATCH_SIZE=32
    NUM_EPOCHS=20
elif [ "$MODEL" == "xlm-roberta-base" ]; then
    LR=2e-5
    BATCH_SIZE=64
    NUM_EPOCHS=30
elif [ "$MODEL" == "bert-base-multilingual-cased" ]; then
    LR=5e-5
    BATCH_SIZE=64
    NUM_EPOCHS=30
else
    echo "Invalid model name"
    exit 1
fi

if [ "$EXPERIMENT" == "train" ]; then
    for SEED in "${SEEDS[@]}"; do
        if [ ! -d "${OUTPUT}/ckpts_seed${SEED}_lr${LR}" ]; then
            echo "INFO: Haven't trained yet. Training now..."
            ./run_baselines_train.sh "${OUTPUT}" "${MODEL}" "${SEED}" "${LR}" "${BATCH_SIZE}" "${NUM_EPOCHS}"
            echo "INFO: Done training. Information about this run is in ${OUTPUT}/ckpts_seed${SEED}_lr${LR}/"
        fi
        OUTPUT_FILE="${OUTPUT}/ckpts_seed${SEED}_lr${LR}/results.txt"
        echo "INFO: Outputting results to ${OUTPUT_FILE}"
        > "${OUTPUT_FILE}"
        echo "Seed: ${SEED}" >> "${OUTPUT_FILE}"
        ./run_baselines_test.sh "${OUTPUT}" "${LR}" "${SEED}" | tee -a "${OUTPUT_FILE}"

        OUTPUT_TRANSLATED_FILE="${OUTPUT}/ckpts_seed${SEED}_lr${LR}/results_translated.txt"
        echo "INFO: Outputting translated results to ${OUTPUT_TRANSLATED_FILE}"
        > "${OUTPUT_TRANSLATED_FILE}"
        echo "Seed: ${SEED}" >> "${OUTPUT_TRANSLATED_FILE}"
        ./run_baselines_test.sh "${OUTPUT}" "${LR}" "${SEED}" "translate-test" | tee -a "${OUTPUT_TRANSLATED_FILE}"
    done
elif [ "$EXPERIMENT" == "zero2hero" ]; then
    for SEED in "${SEEDS[@]}"; do
        # TODO: test this one
        # just check for one language since the z2h script trains all of them
        if [ ! -d "${OUTPUT}/hi_2/ckpts_seed${SEED}_lr${LR}" ]; then
            echo "INFO: Haven't trained yet. Training now..."
            PRETRAINED_EN_MODEL="${OUTPUT}/ckpts_seed${SEED}_lr${LR}/"
            if [ ! -d "${PRETRAINED_EN_MODEL}" ]; then
                echo "ERROR: ${PRETRAINED_EN_MODEL} does not exist. Please train the English model first."
                exit 1
            fi
            ./run_baselines_zero2hero.sh "${MODEL}" "${PRETRAINED_EN_MODEL}" "${SEED}" "${LR}" "${BATCH_SIZE}" 
            echo "INFO: Done training. Information about these runs is in ${OUTPUT}/<lang>_<num>"
        fi
        OUTPUT_FILE="${OUTPUT}/z2h_results.txt"
        echo "INFO: Outputting results to ${OUTPUT_FILE}"
        > "${OUTPUT_FILE}"
        echo "Seed: ${SEED}" >> "${OUTPUT_FILE}"
        ./test_z2h.sh "${PRETRAINED_EN_MODEL}" "${SEED}" | tee "${OUTPUT_FILE}"
    done
else
    print_help
fi
