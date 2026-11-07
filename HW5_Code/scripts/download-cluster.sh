#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: bash scripts/download-cluster.sh cluster_01" >&2
  exit 2
fi
CLUSTER="$1"
case "$CLUSTER" in
  cluster_0[1-9]|cluster_1[0-3]|cluster_1[6-9]|cluster_20) ;;
  *) echo "Choose your assigned cluster: 01–13 or 16–20." >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
command -v hf >/dev/null || { echo "Run: conda activate hw5-sft" >&2; exit 1; }
hf download jrast/climbmix-students-tokenized --repo-type dataset \
  --revision f30c5f5ea5085e7cc39212be709521effafd6003 \
  --include "llmc/$CLUSTER/*.bin" --local-dir data/downloads
printf '\nTraining files: data/downloads/llmc/%s/climbmix_students_train_*.bin\n' "$CLUSTER"
printf 'Cluster validation: data/downloads/llmc/%s/climbmix_students_val_*.bin\n' "$CLUSTER"
