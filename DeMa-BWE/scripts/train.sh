#!/bin/bash

# activate pytorch environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

src_lang="pt"
tgt_lang="es"
MODEL_NAME=id_${src_lang}_${tgt_lang}

# path to save saved params/setting/logs
exp_path="../saved_exps/${MODEL_NAME}"

# path to data sets including monolingual embeddings and evaluation data sets
data_path="../data"

if [[ ! -e $exp_path ]]; then
    mkdir $exp_path
fi

mkdir -p ${exp_path}
cp $0 ${exp_path}/run.sh

s_var=0.01
s2t_t_var=0.01
t_var=0.01
t2s_s_var=0.01

# number of identical words used as supervisions; some languages might need more seeds for better performance
# but is not a sensitive hyperparameter
sup_seeds=2000

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u ../main_e2e.py \
	--cuda \
    --model_name ${MODEL_NAME} \
    --export_emb 1 \
    --supervise_id ${sup_seeds} \
    --valid_option unsup \
    --src_train_most_frequent 20000 \
    --tgt_train_most_frequent 20000 \
    --src_base_batch_size 20000 \
    --tgt_base_batch_size 20000 \
    --batch_size 2048 \
    --sup_s_weight 10. \
    --sup_t_weight 10. \
    --s_var ${s_var} \
    --s2t_t_var ${s2t_t_var} \
    --t_var ${t_var} \
    --t2s_s_var ${t2s_s_var} \
    --src_emb_path $data_path/fasttext/cc.${src_lang}.300.bin \
    --tgt_emb_path $data_path/fasttext/cc.${tgt_lang}.300.bin \
    --n_steps 150000 \
    --display_steps 100 \
    --valid_steps 5000 \
    --sup_dict_path $data_path/crosslingual/dictionaries/${src_lang}-${tgt_lang}.0-5000.txt \
    --dico_eval $data_path/crosslingual/dictionaries/${src_lang}-${tgt_lang}.5000-6500.txt 2>&1 | tee ${exp_path}/train.log
