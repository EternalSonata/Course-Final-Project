#!/usr/bin/env bash
# 不要用 set -e，否则某个实验失败会导致整个脚本退出
# set -e

PY=python
SRC=en
TGT=de
OUTDIR="runs_mt_2"
MAX_LEN=80
EPOCHS=50
BATCH=384
VOCAB=20000
MIN_FREQ=2
GPUS="4,5,6,7"
SPM_PREFIX="${OUTDIR}/spm_${SRC}_${TGT}"
BEAM_SIZE=5
LENGTH_PENALTY=0.7

mkdir -p "${OUTDIR}"

########################################
# 1) plateau scheduler 版本的通用函数
########################################
run_exp_plateau () {
  MODEL=$1
  TOK=$2
  RUNNAME=$3
  SEED=$4
  LS=$5

  echo "========================================================"
  echo "Running (plateau): model=${MODEL}, tok=${TOK}, run=${RUNNAME}, seed=${SEED}, ls=${LS}"
  echo "========================================================"

  if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS} \
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
      --lr_patience 3 \
      --eval_decode beam \
      --beam_size ${BEAM_SIZE} \
      --length_penalty ${LENGTH_PENALTY} \
      --amp \
      --sp_model_prefix "${SPM_PREFIX}"
  then
      echo "[OK] Experiment completed: ${RUNNAME}"
  else
      echo "[ERROR] Experiment FAILED: ${RUNNAME}" >&2
      echo "[ERROR] Skipping this run and continuing..." >&2
  fi

  echo ""
  echo ""
}

########################################
# 2) Noam scheduler 版本的通用函数
########################################
run_exp_noam () {
  MODEL=$1
  TOK=$2
  RUNNAME=$3
  SEED=$4
  LS=$5
  WARMUP=$6  # 例如 4000

  echo "========================================================"
  echo "Running (noam): model=${MODEL}, tok=${TOK}, run=${RUNNAME}, seed=${SEED}, ls=${LS}, warmup=${WARMUP}"
  echo "========================================================"

  if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS} \
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
      --lr_scheduler noam \
      --warmup_steps ${WARMUP} \
      --eval_decode beam \
      --beam_size ${BEAM_SIZE} \
      --length_penalty ${LENGTH_PENALTY} \
      --amp \
      --sp_model_prefix "${SPM_PREFIX}"
  then
      echo "[OK] Noam experiment completed: ${RUNNAME}"
  else
      echo "[ERROR] Noam experiment FAILED: ${RUNNAME}" >&2
      echo "[ERROR] Skipping this run and continuing..." >&2
  fi

  echo ""
  echo ""
}

###########################################################
# A. Tokenizer 对比（ws / sp / char），scheduler = plateau
###########################################################

# 1) whitespace, ls=0.1, 0.4
run_exp_plateau transformer whitespace "tfm_ws_ls01_seed42" 42 0.1
run_exp_plateau transformer whitespace "tfm_ws_ls04_seed42" 42 0.4

# 2) sentencepiece, ls=0.1, 0.2, 0.4
run_exp_plateau transformer sentencepiece "tfm_sp_ls01_seed42" 42 0.1
run_exp_plateau transformer sentencepiece "tfm_sp_ls02_seed42" 42 0.2
run_exp_plateau transformer sentencepiece "tfm_sp_ls04_seed42" 42 0.4

# 3) char-level, ls=0.1
# 注意：char 句子更长，如果显存吃不消可以把 BATCH 调小
run_exp_plateau transformer char "tfm_char_ls0_seed42" 42 0.0
run_exp_plateau transformer char "tfm_char_ls01_seed42" 42 0.1
run_exp_plateau transformer char "tfm_char_ls02_seed42" 42 0.2
run_exp_plateau transformer char "tfm_char_ls04_seed42" 42 0.4

###########################################################
# B. Scheduler 对比（sp + ls=0.1）
###########################################################

# plateau 版本（其实就是上面的 tfm_sp_ls01_seed42，可以不重复）
# 再跑一个 noam 对比
run_exp_noam transformer whitespace "tfm_ws_ls0_noam_w1000_seed42" 42 0.0 1000
run_exp_noam transformer whitespace "tfm_ws_ls0_noam_w1500_seed42" 42 0.0 1500
run_exp_noam transformer whitespace "tfm_ws_ls0_noam_w2500_seed42" 42 0.0 2500

###########################################################
# C. （可选）vocab size sweep，whitespace + plateau
###########################################################

VOCAB=8000
run_exp_plateau transformer whitespace "tfm_ws_vocab8k_ls0_seed42" 42 0.0

VOCAB=16000
run_exp_plateau transformer whitespace "tfm_ws_vocab16k_ls0_seed42" 42 0.0

VOCAB=20000
###########################################################
# D. （可选）Different Random seed
###########################################################
run_exp_plateau transformer whitespace "tfm_ws_ls0_seed0"     0 0.0
run_exp_plateau transformer whitespace "tfm_ws_ls0_seed128" 128 0.0
run_exp_plateau transformer whitespace "tfm_ws_ls0_seed512" 512 0.0