#!/usr/bin/env bash
set -e


PY=python                  
SRC=en
TGT=de
OUTDIR="runs_mt_3"           
MAX_LEN=80                
EPOCHS=30                  
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
    --early_stop_patience 5 \
    --lr_scheduler plateau \
    --lr_factor 0.5 \
    --lr_patience 2 \
    --eval_decode beam \
    --beam_size ${BEAM_SIZE} \
    --length_penalty ${LENGTH_PENALTY} \
    --amp \
    --sp_model_prefix "${SPM_PREFIX}"
}


run_exp rnn        char    "gru_char_seed42_ls0"    42 0.0
run_exp rnn_lstm   char    "lstm_char_seed42_ls0"       42 0.0
run_exp rnn        whitespace    "gru_ws_seed42_ls0"    42 0.0
run_exp rnn_lstm   whitespace    "lstm_ws_seed42_ls0"       42 0.0
run_exp rnn        sentencepiece    "gru_sp_seed42_ls0"    42 0.0
run_exp rnn_lstm   sentencepiece    "lstm_sp_seed42_ls0"       42 0.0

# GRU + Attention

# run_exp rnn        whitespace    "gru_ws_seed42_ls02"    42 0.2
# run_exp rnn        whitespace    "gru_ws_seed42_ls04"    42 0.4
# run_exp rnn        whitespace    "gru_ws_seed42"        42 0.1
# run_exp rnn        sentencepiece "gru_sp_seed42"        42 0.1

# LSTM + Attention

# run_exp rnn_lstm   whitespace    "lstm_ws_seed42_ls02"       42 0.2
# run_exp rnn_lstm   whitespace    "lstm_ws_seed42_ls04"       42 0.4
# run_exp rnn_lstm   whitespace    "lstm_ws_seed42"       42 0.1
# run_exp rnn_lstm   sentencepiece "lstm_sp_seed42"       42 0.1