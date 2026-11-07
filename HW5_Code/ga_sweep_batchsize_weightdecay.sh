#!/usr/bin/env bash
# Sweep over batch size and weight decay for llm.c train_gpt2cu.
# Runs are executed sequentially, one after another.

set -u

MODEL="mup_tiniest_1L"

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

BATCH_SIZES=(2048 65536 131072 262144 524288)
#(32768 16384 8192 4096 2048)

# Weight decay values
WD_VALUES=(0.00078125 0.0015625 0.003125 0.00625 0.0125 0.025 0.05 0.1 0.2 0.4 0.8 1.6)
#WD_VALUES=(0.000390625 0.0001953125 0.00009765625 0.000048828125 0.0000244140625 0.00001220703125 0.000006103515625 0.0000030517578125 0.00000152587890625 0.000000762939453125 0.00078125 0.0015625 0.003125 0.00625 0.0125 0.025 0.05 0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8)

LR_VALUE=0.0006
LR_LABEL="p0006"

mkdir -p model_log_loss
mkdir -p verbose_outputs

for batch_size in "${BATCH_SIZES[@]}"; do
    step_size=$(( TOTAL_TOKENS / batch_size ))
    warm_up=$(( step_size / 10 ))   # 0.10 * step_size
    micro_b=$(( batch_size / 1024 ))   # so B*T == total batch size (single GPU)

    for i in "${!WD_VALUES[@]}"; do
        weight_decay=${WD_VALUES[$i]}

	# Format label: '0.05' -> 'p05', '1.6' -> '1p6'
        if [[ "$weight_decay" == 0.* ]]; then
            wd_label="p${weight_decay#0.}"
        else
            wd_label="${weight_decay/./p}"
        fi

        run_name="B${batch_size}_LRp${LR_LABEL}_WD_${wd_label}"

	mkdir -p "model_log_loss/${MODEL}/${run_name}"

	if [ -f "model_log_loss/${MODEL}/${run_name}/main.log" ]; then
            echo "=== ${run_name} already has main.log, skipping ==="
            continue
        fi

        echo "=== ${run_name} | steps=${step_size} warmup=${warm_up} wd=${weight_decay} ==="

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
            -l ${LR_VALUE} \
            -c "${weight_decay}" \
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
