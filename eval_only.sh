#!/usr/bin/env bash
set -u

PY=python

EXPERIMENTS=(
    # "./runs_mt/gru_sp_seed42"
    # "./runs_mt/gru_ws_seed42"
    # "./runs_mt/lstm_sp_seed42"
    # "./runs_mt/lstm_ws_seed42"
    # "./runs_mt/tfm_sp_seed42_ls0"
    # "./runs_mt/tfm_ws_seed42_ls02"
    # "./runs_mt/tfm_ws_seed42"
    # "./runs_mt_2/tfm_ws_ls0_seed0"
    # "./runs_mt_2/tfm_ws_ls0_seed128"
    # "./runs_mt_2/tfm_ws_ls0_seed512"
    # "./runs_mt_2/tfm_char_ls0_seed42"
    # "./runs_mt_2/tfm_char_ls01_seed42"
    # "./runs_mt_2/tfm_char_ls02_seed42"
    # "./runs_mt_2/tfm_char_ls04_seed42"
    # "./runs_mt_2/tfm_sp_ls01_seed42"
    # "./runs_mt_2/tfm_sp_ls02_seed42"
    # "./runs_mt_2/tfm_sp_ls04_seed42"
    # "./runs_mt_2/tfm_ws_ls0_noam_w1000_seed42"
    # "./runs_mt_2/tfm_ws_ls0_noam_w1500_seed42"
    # "./runs_mt_2/tfm_ws_ls0_noam_w2500_seed42"
    "./runs_mt_2/tfm_ws_ls01_seed42"
    "./runs_mt_2/tfm_ws_ls04_seed42"
    "./runs_mt_2/tfm_ws_vocab8k_ls0_seed42"
    "./runs_mt_2/tfm_ws_vocab16k_ls0_seed42"
)

for EXP_DIR in "${EXPERIMENTS[@]}"; do
    echo "============================================================"
    echo " Evaluating best checkpoint in: ${EXP_DIR}"
    echo "============================================================"

    BEST_CKPT=$(ls "${EXP_DIR}"/*_best.pt 2>/dev/null | head -n 1)

    if [ -z "$BEST_CKPT" ]; then
        echo "[WARN] No *_best.pt found in ${EXP_DIR}, skipping."
        echo ""
        continue
    fi

    echo "Found checkpoint: ${BEST_CKPT}"

    ${PY} MT_baselines.py \
        --eval_only \
        --resume "${BEST_CKPT}" \
        --gpus "4,5,6,7" \
        --output_dir "run_eval"
    echo ""
done