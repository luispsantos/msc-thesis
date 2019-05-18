#!/bin/bash

# activate pytorch environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

src_lang="pt"
tgt_lang="es"
load_path="../saved_exps/id_${src_lang}_${tgt_lang}"
data_path="../data"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u ../evaluation/export_embs.py \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --s2t_map_path ${load_path}/best_s2t_params.bin \
    --t2s_map_path ${load_path}/best_t2s_params.bin \
    --src_emb_path $data_path/fasttext/cc.${src_lang}.300.bin \
    --tgt_emb_path $data_path/fasttext/cc.${tgt_lang}.300.bin \
    --vocab_size 50000
