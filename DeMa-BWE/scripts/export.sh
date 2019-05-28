#!/bin/bash

# activate pytorch environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

src_lang="es"
tgt_lang="pt"
load_path="../saved_exps/id_${src_lang}_${tgt_lang}"
data_path="../data"

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -u ../evaluation/export_embs.py \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --s2t_map_path ${load_path}/best_s2t_params.bin \
    --src_emb_path $data_path/embeddings/${src_lang}.fasttext.oov.vec.gz \
    --tgt_emb_path $data_path/embeddings/${tgt_lang}.fasttext.oov.vec.gz \
    --max_vocab -1
