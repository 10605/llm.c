#!/usr/bin/env bash
# Sweep batch size x learning rate for llm.c train_gpt2cu, weight decay fixed at 0.1.
# Run with:  nohup ./run_lr_sweep.sh > my_logs_output/lr_sweep_progress.txt 2>&1 &

set -u

MODEL="mup_tinier_2L"

case "$MODEL" in
  *"mup_tiniest_1L"*) PARAMS=3332096 ;;
  *"mup_tinier_2L"*)  PARAMS=6960768 ;;
  *"mup_tiny_3L"*)    PARAMS=11180928 ;;
  *)
    echo "Error: Unknown model '$MODEL'" >&2
    exit 1
    ;;
esac

TOTAL_TOKENS=$((20 * PARAMS))

# BATCH_SIZES=(32768 16384 8192 4096 2048)
BATCH_SIZES=(65536 131072 262144 524288)
#(65536 131072 262144 524288)
#(32768 16384 8192 4096 2048)

# Learning rates (doubling grid) and their run-name labels ("p" stands for "0.")
LR_VALUES=(0.0384 0.0768 0.1536)
# LR_VALUES=(0.0003 0.0006 0.0012 0.0024 0.0048 0.0096 0.0192)
# LR_VALUES=(0.00000029296875 0.0000005859375 0.000001171875 0.00000234375 0.0000046875 0.000009375 0.00001875 0.0000375 0.000075 0.00015 0.0003 0.0006 0.0012 0.0024 0.0048 0.0096 0.0192 0.0384 0.0768 0.1536 0.3072 0.6144 1.2288 2.4576 4.9152 9.8304 19.6608)

WEIGHT_DECAY=0.1
WD_LABEL=p1

mkdir -p model_log_loss
mkdir -p verbose_outputs

for batch_size in "${BATCH_SIZES[@]}"; do
    step_size=$(( TOTAL_TOKENS / batch_size ))
    warm_up=$(( step_size / 10 ))       # 0.10 * step_size
    micro_b=$(( batch_size / 1024 ))    # B*T must divide total batch size
    if [ "$micro_b" -gt 32 ]; then micro_b=32; fi

    for i in "${!LR_VALUES[@]}"; do
	lr=${LR_VALUES[$i]}
	
        # Format label: '0.05' -> 'p05', '1.6' -> '1p6'
        if [[ "$lr" == 0.* ]]; then
            lr_label="p${lr#0.}"
        else
            lr_label="${lr/./p}"
        fi

        run_name="B${batch_size}_LR${lr_label}_WD_${WD_LABEL}"

	mkdir -p "model_log_loss/${MODEL}/${run_name}"

        if [ -f "model_log_loss/${MODEL}/${run_name}/main.log" ]; then
            echo "=== ${run_name} already has main.log, skipping ==="
            continue
        fi

        echo "=== ${run_name} | steps=${step_size} warmup=${warm_up} lr=${lr} wd=${WEIGHT_DECAY} micro_b=${micro_b} ==="

        ./train_gpt2cu \
            -i "/home/ubuntu/mydata/fineweb10B/fineweb_train_*.bin" \
            -j "/home/ubuntu/mydata/fineweb10B/fineweb_val_*.bin" \
            -o "model_log_loss/${MODEL}/${run_name}" \
            -e "my_models/${MODEL}.bin" \
            -b "${micro_b}" -t 1024 \
            -d "${batch_size}" \
            -r 1 -z 1 \
            -v 5 \
            -s 0 \
            -x "${step_size}" \
            -u "${warm_up}" \
            -l "${lr}" \
            -c "${WEIGHT_DECAY}" \
            -me 1 \
            -mm 1 \
            -mw 64 &> "verbose_outputs/${MODEL}___${run_name}.txt"

        status=$?
        if [ $status -ne 0 ]; then
            echo "    WARNING: ${run_name} exited with status ${status} (see verbose_outputs/${MODEL}___${run_name}.txt)"
        fi
    done
done

echo "Sweep complete."
