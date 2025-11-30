#!/usr/bin/env bash
# 多语言训练脚本：训练时使用 en->de + en->fr，评估只看 en->de
# set -e

PY=python
SRC=en
TGT=de                 
EXTRA_TGT=fr           

OUTDIR="runs_mt_multi_tfm"
MAX_LEN=80
EPOCHS=50
BATCH=384
VOCAB=20000
MIN_FREQ=2
GPUS="4,5,6,7"

SPM_PREFIX="${OUTDIR}/spm_${SRC}_${TGT}_${EXTRA_TGT}"

BEAM_SIZE=5
LENGTH_PENALTY=0.7

mkdir -p "${OUTDIR}"


run_exp_plateau_multi () {
  MODEL=$1     
  TOK=$2       
  RUNNAME=$3
  SEED=$4
  LS=$5        

  echo "========================================================"
  echo "Running MULTILINGUAL (plateau):"
  echo "  model=${MODEL}, tok=${TOK}, run=${RUNNAME}, seed=${SEED}, ls=${LS}"
  echo "  langs: ${SRC}->${TGT} + ${SRC}->${EXTRA_TGT}"
  echo "========================================================"

  if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS} \
     $PY MT_baselines.py \
      --model "${MODEL}" \
      --tokenizer "${TOK}" \
      --src_lang "${SRC}" \
      --tgt_lang "${TGT}" \
      --extra_tgt_lang "${EXTRA_TGT}" \
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
      --lr_patience 3 \
      --eval_decode beam \
      --beam_size ${BEAM_SIZE} \
      --length_penalty ${LENGTH_PENALTY} \
      --amp \
      --sp_model_prefix "${SPM_PREFIX}"
  then
      echo "[OK] Multilingual (plateau) experiment completed: ${RUNNAME}"
  else
      echo "[ERROR] Multilingual (plateau) experiment FAILED: ${RUNNAME}" >&2
      echo "[ERROR] Skipping this run and continuing..." >&2
  fi

  echo ""
  echo ""
}


run_exp_noam_multi () {
  MODEL=$1
  TOK=$2
  RUNNAME=$3
  SEED=$4
  LS=$5
  WARMUP=$6

  echo "========================================================"
  echo "Running MULTILINGUAL (noam):"
  echo "  model=${MODEL}, tok=${TOK}, run=${RUNNAME}, seed=${SEED}, ls=${LS}, warmup=${WARMUP}"
  echo "  langs: ${SRC}->${TGT} + ${SRC}->${EXTRA_TGT}"
  echo "========================================================"

  if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS} \
     $PY MT_baselines.py \
      --model "${MODEL}" \
      --tokenizer "${TOK}" \
      --src_lang "${SRC}" \
      --tgt_lang "${TGT}" \
      --extra_tgt_lang "${EXTRA_TGT}" \
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
      --lr_scheduler noam \
      --warmup_steps ${WARMUP} \
      --eval_decode beam \
      --beam_size ${BEAM_SIZE} \
      --length_penalty ${LENGTH_PENALTY} \
      --amp \
      --sp_model_prefix "${SPM_PREFIX}"
  then
      echo "[OK] Multilingual (noam) experiment completed: ${RUNNAME}"
  else
      echo "[ERROR] Multilingual (noam) experiment FAILED: ${RUNNAME}" >&2
      echo "[ERROR] Skipping this run and continuing..." >&2
  fi

  echo ""
  echo ""
}

###########################################################
# A. Tokenizer 对比（whitespace / sp / char），scheduler = plateau
#    ==> 全部使用多语言训练（en->de + en->fr），但 eval 只看 en->de
###########################################################

# 1) whitespace
run_exp_plateau_multi transformer whitespace "tfm_multi_ws_ls0_seed42"   42 0.0
run_exp_plateau_multi transformer whitespace "tfm_multi_ws_ls02_seed42"  42 0.2

# 2) sentencepiece
run_exp_plateau_multi transformer sentencepiece "tfm_multi_sp_ls0_seed42"  42 0.0
run_exp_plateau_multi transformer sentencepiece "tfm_multi_sp_ls02_seed42" 42 0.2

# 3) char-level
run_exp_plateau_multi transformer char "tfm_multi_char_ls0_seed42"   42 0.0
run_exp_plateau_multi transformer char "tfm_multi_char_ls02_seed42"  42 0.2

###########################################################
# B. Scheduler 对比（whitespace + ls=0.0），多语言 + noam
###########################################################
# run_exp_noam_multi transformer whitespace "tfm_multi_ws_noam_w1000_seed42" 42 0.0 1000
# run_exp_noam_multi transformer whitespace "tfm_multi_ws_noam_w1500_seed42" 42 0.0 1500
run_exp_noam_multi transformer whitespace "tfm_multi_ws_noam_w2500_seed42" 42 0.0 2500

###########################################################
# C. （可选）不同随机种子，多语言 + whitespace + plateau
###########################################################
# run_exp_plateau_multi transformer whitespace "tfm_multi_ws_ls0_seed0"     0   0.0
# run_exp_plateau_multi transformer whitespace "tfm_multi_ws_ls0_seed128"   128 0.0
# run_exp_plateau_multi transformer whitespace "tfm_multi_ws_ls0_seed512"   512 0.0
