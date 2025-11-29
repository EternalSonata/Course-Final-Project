#!/usr/bin/env bash
set -e


PY=python                  
SRC=en
TGT=de
OUTDIR="runs_mt"           
MAX_LEN=80                
EPOCHS=40                  
BATCH=384
VOCAB=20000
MIN_FREQ=2
GPUS="4,5,6,7"
SPM_PREFIX="${OUTDIR}/spm_${SRC}_${TGT}"
BEAM_SIZE=5
LENGTH_PENALTY=0.7


mkdir -p "${OUTDIR}"


run_exp () {
  MODEL=$1
  TOK=$2
  RUNNAME=$3
  SEED=$4
  LS=$5

  echo "========================================================"
  echo "Running: model=${MODEL}, tok=${TOK}, run=${RUNNAME}, seed=${SEED}, ls=${LS}"
  echo "========================================================"

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS} \
  $PY MT_baselines.py \
    --model "${MODEL}" \
    --tokenizer "${TOK}" \
    --src_lang "${SRC}" \
    --tgt_lang "${TGT}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH} \
    --max_len ${MAX_LEN} \
    --vocab_size ${VOCAB} \
    --min_freq ${MIN_FREQ} \
    --output_dir "${OUTDIR}" \
    --run_name "${RUNNAME}" \
    --seed ${SEED} \
    --label_smoothing ${LS} \
    --early_stop_patience 20 \
    --lr_scheduler plateau \
    --lr_factor 0.5 \
    --lr_patience 20 \
    --eval_decode beam \
    --beam_size ${BEAM_SIZE} \
    --length_penalty ${LENGTH_PENALTY} \
    --amp \
    --sp_model_prefix "${SPM_PREFIX}"
}

########################################
# A. 模型 × tokenizer 基础对比（6 个实验）
#   - GRU / LSTM / Transformer
#   - Whitespace vs SentencePiece
#   - 统一 seed=42, label_smoothing=0.1
########################################

# GRU + Attention
# run_exp rnn        whitespace    "gru_ws_seed42"        42 0.1
# run_exp rnn        sentencepiece "gru_sp_seed42"        42 0.1

# # LSTM + Attention
# run_exp rnn_lstm   whitespace    "lstm_ws_seed42"       42 0.1
# run_exp rnn_lstm   sentencepiece "lstm_sp_seed42"       42 0.1

# Transformer
run_exp transformer whitespace   "tfm_ws_seed42"        42 0.0
# run_exp transformer sentencepiece "tfm_sp_seed42_ls01"  42 0.1

########################################
# B. Transformer + SentencePiece 的多 seed 重复实验
########################################

# run_exp transformer whitespace "tfm_sp_seed0_ls01"  0 0.1
# run_exp transformer whitespace "tfm_sp_seed2025_ls01"  2025 0.1

########################################
# C. Transformer + SentencePiece 的 label smoothing 消融
#    -> 对比 ls = 0.0 / 0.1 / 0.2 对 BLEU 的影响
#    注意：ls=0.1 的那条已经在 A 里跑过 (tfm_sp_seed42_ls01)
########################################

# run_exp transformer whitespace "tfm_sp_seed42_ls00"  42 0.0
# run_exp transformer whitespace "tfm_sp_seed42_ls02"  42 0.2

########################################
# D. BPE vocab size 消融
########################################

# echo "======== BPE vocab_size = 8000 (Transformer + SentencePiece) ========"
# VOCAB=8000
# run_exp transformer whitespace "tfm_sp_vocab8k_seed42" 42 0.1
# VOCAB=20000

# echo "所有计划内实验运行完毕。结果保存在 ${OUTDIR}/*/ 中："
# echo "- 每个 run 有 loss_curves.png, train.log, metrics.json 和 best checkpoint。"