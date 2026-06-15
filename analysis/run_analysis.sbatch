#!/usr/bin/env bash
#SBATCH --job-name=analysis_noOT
#SBATCH -p mt
#SBATCH -A mt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=/home/usuaris.new/adria.capdevila.zurita/lsc-parlament/logs/analysis_noOT_%j.log

set -e
source ~/lsc-parlament/venv/bin/activate

# Set SMOKE=1 for a quick sanity run on 50 samples; SMOKE=0 for the full test set.
SMOKE=1

BASE=/home/usuaris.new/adria.capdevila.zurita/lsc-parlament
REPO=$BASE/multimodalhugs
TEST=/home/usuaris.new/carlos.escolano/lsc-parlament/ca_filtered/test.tsv

if [ "$SMOKE" = "1" ]; then
    EVAL_EXTRA="--max_samples 50"
    GEN_EXTRA="--max_predict_samples 50"
else
    EVAL_EXTRA=""
    GEN_EXTRA=""
fi

cd "$REPO"
git fetch origin && git checkout fix/whisper-audio-enc-mask && git pull
python -c "import sacrebleu, sklearn, scipy; print('deps ok')"

# ---- Analysis 2+3: embedding-space alignment (no-OT model, no setup needed) ----
echo "==== space alignment ===="
python analysis/eval_space_alignment.py \
    --checkpoint_path $BASE/checkpoints_video_from_speech_noOT/train/checkpoint-best \
    --metadata_file $TEST \
    --skip_frames_stride 3 \
    --device cuda \
    $EVAL_EXTRA

# ---- Analysis 1: generate predictions for both models on the test set ----
echo "==== generate baseline (V2T) ===="
multimodalhugs-generate --task translation \
    --config_path $BASE/v5_video2text_config.yaml \
    --model_name_or_path $BASE/checkpoints_video_stride3/train/checkpoint-best \
    --setup_path $BASE/setup_video/setup/ \
    --metric_name sacrebleu,chrf \
    --generate_output_dir $BASE/gen_baseline \
    $GEN_EXTRA

echo "==== generate proposed (no-OT) ===="
multimodalhugs-generate --task translation \
    --config_path $BASE/ft_speech_checkpoint_noOT.yaml \
    --model_name_or_path $BASE/checkpoints_video_from_speech_noOT/train/checkpoint-best \
    --setup_path $BASE/setup_video_from_speech_noOT/setup/ \
    --metric_name sacrebleu,chrf \
    --generate_output_dir $BASE/gen_noot \
    $GEN_EXTRA

# ---- Analysis 1: paired bootstrap significance ----
echo "==== bleu bootstrap ===="
python analysis/bleu_significance.py \
    --baseline $BASE/gen_baseline/predictions_labels.txt \
    --proposed $BASE/gen_noot/predictions_labels.txt \
    --metric bleu --n_samples 1000
